/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*
 * Successive-cancellation decoder of polar_decoder_int16() (rate recovery, SC decoding, information bit extraction).
 *
 * Same arithmetic as OAI's original tree decoder (per-node alpha/beta buffers walked through a linearised list of
 * F / G / beta operations), bit for bit, including its saturation and tie behaviour:
 *  - F (left child LLRs), node half size sz >= 4 (the SIMD paths): sign(min(|a|,|b|), sign(a, b)) with SSSE3 semantics
 *    (wrapping abs and negation, 0 when a or b is 0); sz 1 and 2 (the scalar path): (a ^ b) < 0 ? -min : min.
 *  - G (right child LLRs): left child frozen: a + b saturated to [-32768, 32767] for sz >= 8, clipped to
 *    [-32767, 32767] for sz <= 4; otherwise b -s (beta ? a : -a) (saturating subtraction, wrapping negation).
 *  - hard decision u = (llr <= 0), beta = 2u - 1; beta of a node = (beta_l XNOR beta_r, beta_r).
 * The decoder is "compiled" once per parameter set into a flat list of operations on one alpha buffer per tree level
 * and one in-place partial-sum array (node (level l, first leaf f) owns beta[f, f + 2^l)); the two lowest tree levels
 * are fused into one register-only operation per 4 leaves (QUAD, specialised per frozen pattern). The rate recovery
 * (bit de-interleaving, de-repetition / de-puncturing / de-shortening, sub-block de-interleaving) works on blocks of
 * N/32 LLRs instead of single LLRs, and the decoded information bits are written straight to their position in the
 * CRC-ordered word.
 */

#include "PHY/CODING/nrPolar_tools/nr_polar_defs.h"
#include "assertions.h"
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

/* operation codes: one per (operation, tree level 3..10) so that every kernel runs with a constant size */
enum {
  OP_F = 0, // F to the left child
  OP_GA = 8, // G to the right child, left child frozen
  OP_GS = 16, // G to the right child, left child decoded
  OP_BX = 24, // partial sums, both children decoded
  OP_BLF = 32, // partial sums, left child frozen
  OP_BRF = 40, // partial sums, right child frozen
  OP_QUAD = 48, // level-2 node (4 leaves), fused, + frozen pattern (0..14)
  OP_END = 63,
};

/* ------------------------------------------------------------------------------------------------ scalar helpers */

/* applyFtoleft, scalar branch (sz 1 and 2) */
static inline int16_t f_sc(int16_t a, int16_t b)
{
  int16_t ma = a >> 15, mb = b >> 15;
  int16_t absa = (a + ma) ^ ma, absb = (b + mb) ^ mb;
  int16_t m = absa < absb ? absa : absb;
  return (ma ^ mb) == 0 ? m : (int16_t)-m;
}
/* applyGtoright without left beta, scalar branch (sz 1, 2 and 4) */
static inline int16_t g_add_clip(int16_t a, int16_t b)
{
  int t = a + b;
  return t > SHRT_MAX ? SHRT_MAX : t < -SHRT_MAX ? -SHRT_MAX : t;
}
/* applyGtoright with left beta: subs(b, sign(a, beta)), beta = +-1 */
static inline int16_t g_sub(int16_t a, int16_t b, int8_t beta)
{
  int16_t x = beta > 0 ? a : (int16_t)-a;
  int t = b - x;
  return t > SHRT_MAX ? SHRT_MAX : t < SHRT_MIN ? SHRT_MIN : t;
}

/* level-1 node (leaves 0, 1 of b/u), frozen flags p0, p1 (not both). Only the signs of the leaf LLRs are needed:
 *  - f_sc(a0, a1) <= 0  <=>  signs differ, or a0 or a1 is 0 or -32768 (min(|a0|,|a1|) wraps to -32768 then)
 *  - saturation / clipping of G does not change its sign: g_sub(a0, a1, beta) <= 0 <=> a1 <= (beta > 0 ? a0 : -a0)
 *    (negation wrapping in int16), g_add_clip(a0, a1) <= 0 <=> a0 + a1 <= 0 */
__attribute__((always_inline)) static inline void sc_pair(int16_t a0, int16_t a1, int8_t *b, uint8_t *u, const int p0, const int p1)
{
  int d0 = 0, d1;
  if (!p0) {
    d0 = ((a0 ^ a1) < 0) | ((a0 & 0x7fff) == 0) | ((a1 & 0x7fff) == 0);
    u[0] = d0;
  }
  if (p1) { // beta = (beta0, -1)
    b[0] = 2 * d0 - 1;
    b[1] = -1;
    return;
  }
  if (p0)
    d1 = a0 + a1 <= 0;
  else
    d1 = a1 <= (d0 ? a0 : (int16_t)-a0);
  u[1] = d1;
  b[0] = 2 * (d0 ^ d1) - 1; // left frozen: d0 = 0, beta0 = beta1
  b[1] = 2 * d1 - 1;
}

/* level-2 node: 4 LLRs a[0..3] -> 4 decisions and the node's partial sums, frozen pattern pat (bit i: leaf i frozen) */
__attribute__((always_inline)) static inline void sc_quad(const int16_t *a, int8_t *b, uint8_t *u, const unsigned pat)
{
  const int p0 = pat & 1, p1 = (pat >> 1) & 1, p2 = (pat >> 2) & 1, p3 = (pat >> 3) & 1;
  const int lf = p0 && p1, rf = p2 && p3;
  const int16_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  if (!lf)
    sc_pair(f_sc(a0, a2), f_sc(a1, a3), b, u, p0, p1);
  if (!rf) {
    int16_t r0, r1;
    if (lf) {
      r0 = g_add_clip(a0, a2);
      r1 = g_add_clip(a1, a3);
    } else {
      r0 = g_sub(a0, a2, b[0]);
      r1 = g_sub(a1, a3, b[1]);
    }
    sc_pair(r0, r1, b + 2, u + 2, p2, p3);
  }
  if (lf) {
    b[0] = b[2];
    b[1] = b[3];
  } else if (rf) {
    b[2] = -1;
    b[3] = -1;
  } else {
    b[0] = ~(b[0] ^ b[2]);
    b[1] = ~(b[1] ^ b[3]);
  }
}

/* ------------------------------------------------------------------------------------------------ vector kernels */

static inline void beta_xnor(int8_t *bv, const int8_t *br, int n) // bv[i] = ~(bv[i] ^ br[i]), n multiple of 4
{
  for (int i = 0; i < n; i += 8) {
    if (n - i >= 8) {
      uint64_t x, y;
      __builtin_memcpy(&x, bv + i, 8);
      __builtin_memcpy(&y, br + i, 8);
      x = ~(x ^ y);
      __builtin_memcpy(bv + i, &x, 8);
    } else {
      uint32_t x, y;
      __builtin_memcpy(&x, bv + i, 4);
      __builtin_memcpy(&y, br + i, 4);
      x = ~(x ^ y);
      __builtin_memcpy(bv + i, &x, 4);
    }
  }
}

#if defined(__aarch64__)

/* F, SSSE3 semantics: sign(min(abs(a), abs(b)), sign(a, b)) */
static inline int16x8_t f_v8(int16x8_t a, int16x8_t b)
{
  int16x8_t m = vminq_s16(vabsq_s16(a), vabsq_s16(b));
  int16x8_t s = vshrq_n_s16(veorq_s16(a, b), 15);
  int16x8_t r = vsubq_s16(veorq_s16(m, s), s);
  uint16x8_t z = vminq_u16(vreinterpretq_u16_s16(a), vreinterpretq_u16_s16(b)); // 0 where a or b is 0
  return vandq_s16(r, vreinterpretq_s16_u16(vtstq_u16(z, z)));
}
static inline int16x4_t f_v4(int16x4_t a, int16x4_t b)
{
  int16x4_t m = vmin_s16(vabs_s16(a), vabs_s16(b));
  int16x4_t s = vshr_n_s16(veor_s16(a, b), 15);
  int16x4_t r = vsub_s16(veor_s16(m, s), s);
  uint16x4_t z = vmin_u16(vreinterpret_u16_s16(a), vreinterpret_u16_s16(b));
  return vand_s16(r, vreinterpret_s16_u16(vtst_u16(z, z)));
}
static inline void op_f4(const int16_t *av, int16_t *ac)
{
  int16x8_t v = vld1q_s16(av);
  vst1_s16(ac, f_v4(vget_low_s16(v), vget_high_s16(v)));
}
static inline void op_fv(const int16_t *av, int16_t *ac, int sz) // sz multiple of 8
{
  for (int i = 0; i < sz; i += 8)
    vst1q_s16(ac + i, f_v8(vld1q_s16(av + i), vld1q_s16(av + sz + i)));
}
static inline void op_ga4(const int16_t *av, int16_t *ac)
{
  int16x8_t v = vld1q_s16(av);
  vst1_s16(ac, vmax_s16(vqadd_s16(vget_low_s16(v), vget_high_s16(v)), vdup_n_s16(-SHRT_MAX)));
}
static inline void op_gav(const int16_t *av, int16_t *ac, int sz)
{
  for (int i = 0; i < sz; i += 8)
    vst1q_s16(ac + i, vqaddq_s16(vld1q_s16(av + i), vld1q_s16(av + sz + i)));
}
static inline void op_gs4(const int16_t *av, int16_t *ac, const int8_t *bl)
{
  int16x8_t v = vld1q_s16(av);
  int16x4_t x = vmul_s16(vget_low_s16(v), vget_low_s16(vmovl_s8(vld1_s8(bl))));
  vst1_s16(ac, vqsub_s16(vget_high_s16(v), x));
}
static inline void op_gsv(const int16_t *av, int16_t *ac, const int8_t *bl, int sz)
{
  for (int i = 0; i < sz; i += 8) {
    int16x8_t x = vmulq_s16(vld1q_s16(av + i), vmovl_s8(vld1_s8(bl + i)));
    vst1q_s16(ac + i, vqsubq_s16(vld1q_s16(av + sz + i), x));
  }
}
static inline void op_bxv(int8_t *bv, const int8_t *br, int sz) // sz multiple of 16
{
  for (int i = 0; i < sz; i += 16)
    vst1q_s8(bv + i, vmvnq_s8(veorq_s8(vld1q_s8(bv + i), vld1q_s8(br + i))));
}

#else /* x86 (native SSE/AVX2 through SIMDe) and other architectures (SIMDe portable code) */

static inline simde__m128i f_v8(simde__m128i a, simde__m128i b)
{
  simde__m128i m = simde_mm_min_epi16(simde_mm_abs_epi16(a), simde_mm_abs_epi16(b));
  return simde_mm_sign_epi16(m, simde_mm_sign_epi16(a, b));
}
static inline void op_f4(const int16_t *av, int16_t *ac)
{
  simde__m128i v = simde_mm_loadu_si128((const simde__m128i *)av);
  simde_mm_storel_epi64((simde__m128i *)ac, f_v8(v, simde_mm_srli_si128(v, 8)));
}
static inline void op_fv(const int16_t *av, int16_t *ac, int sz)
{
  int i = 0;
#if defined(__AVX2__)
  for (; i + 16 <= sz; i += 16) {
    simde__m256i a = simde_mm256_loadu_si256((const simde__m256i *)(av + i));
    simde__m256i b = simde_mm256_loadu_si256((const simde__m256i *)(av + sz + i));
    simde__m256i m = simde_mm256_min_epi16(simde_mm256_abs_epi16(a), simde_mm256_abs_epi16(b));
    simde_mm256_storeu_si256((simde__m256i *)(ac + i), simde_mm256_sign_epi16(m, simde_mm256_sign_epi16(a, b)));
  }
#endif
  for (; i < sz; i += 8)
    simde_mm_storeu_si128((simde__m128i *)(ac + i),
                          f_v8(simde_mm_loadu_si128((const simde__m128i *)(av + i)),
                               simde_mm_loadu_si128((const simde__m128i *)(av + sz + i))));
}
static inline void op_ga4(const int16_t *av, int16_t *ac)
{
  simde__m128i v = simde_mm_loadu_si128((const simde__m128i *)av);
  simde__m128i s = simde_mm_max_epi16(simde_mm_adds_epi16(v, simde_mm_srli_si128(v, 8)), simde_mm_set1_epi16(-SHRT_MAX));
  simde_mm_storel_epi64((simde__m128i *)ac, s);
}
static inline void op_gav(const int16_t *av, int16_t *ac, int sz)
{
  int i = 0;
#if defined(__AVX2__)
  for (; i + 16 <= sz; i += 16)
    simde_mm256_storeu_si256((simde__m256i *)(ac + i),
                             simde_mm256_adds_epi16(simde_mm256_loadu_si256((const simde__m256i *)(av + sz + i)),
                                                    simde_mm256_loadu_si256((const simde__m256i *)(av + i))));
#endif
  for (; i < sz; i += 8)
    simde_mm_storeu_si128((simde__m128i *)(ac + i),
                          simde_mm_adds_epi16(simde_mm_loadu_si128((const simde__m128i *)(av + sz + i)),
                                              simde_mm_loadu_si128((const simde__m128i *)(av + i))));
}
static inline void op_gs4(const int16_t *av, int16_t *ac, const int8_t *bl)
{
  simde__m128i v = simde_mm_loadu_si128((const simde__m128i *)av);
  simde__m128i x = simde_mm_sign_epi16(v, simde_mm_cvtepi8_epi16(simde_mm_loadl_epi64((const simde__m128i *)bl)));
  simde_mm_storel_epi64((simde__m128i *)ac, simde_mm_subs_epi16(simde_mm_srli_si128(v, 8), x));
}
static inline void op_gsv(const int16_t *av, int16_t *ac, const int8_t *bl, int sz)
{
  int i = 0;
#if defined(__AVX2__)
  for (; i + 16 <= sz; i += 16) {
    simde__m256i x = simde_mm256_sign_epi16(simde_mm256_loadu_si256((const simde__m256i *)(av + i)),
                                            simde_mm256_cvtepi8_epi16(simde_mm_loadu_si128((const simde__m128i *)(bl + i))));
    simde_mm256_storeu_si256((simde__m256i *)(ac + i),
                             simde_mm256_subs_epi16(simde_mm256_loadu_si256((const simde__m256i *)(av + sz + i)), x));
  }
#endif
  for (; i < sz; i += 8) {
    simde__m128i x = simde_mm_sign_epi16(simde_mm_loadu_si128((const simde__m128i *)(av + i)),
                                         simde_mm_cvtepi8_epi16(simde_mm_loadl_epi64((const simde__m128i *)(bl + i))));
    simde_mm_storeu_si128((simde__m128i *)(ac + i),
                          simde_mm_subs_epi16(simde_mm_loadu_si128((const simde__m128i *)(av + sz + i)), x));
  }
}
static inline void op_bxv(int8_t *bv, const int8_t *br, int sz)
{
  const simde__m128i ones = simde_mm_set1_epi8(-1);
  for (int i = 0; i < sz; i += 16)
    simde_mm_storeu_si128((simde__m128i *)(bv + i),
                          simde_mm_xor_si128(simde_mm_xor_si128(simde_mm_loadu_si128((const simde__m128i *)(bv + i)),
                                                                simde_mm_loadu_si128((const simde__m128i *)(br + i))),
                                             ones));
}
#endif

/* ------------------------------------------------------------------------------------------------ program */

static int frozen_range(const uint8_t *ibp, int f, int len)
{
  for (int i = 0; i < len; i++)
    if (ibp[f + i])
      return 0;
  return 1;
}

static void sc_emit(nr_polar_sc_t *d, int code, int arg, int f)
{
  d->ops[d->nops++] = (nr_polar_sc_op_t){.code = code, .arg = arg, .f = f};
}

/* the operations of a non-frozen node of level l (>= 2), in the order of OAI's recursive decoder */
static void sc_build(nr_polar_sc_t *d, const uint8_t *ibp, int l, int f, int root)
{
  if (l == 2) {
    unsigned pat = 0;
    for (int i = 0; i < 4; i++)
      pat |= (ibp[f + i] == 0) << i;
    sc_emit(d, OP_QUAD + pat, 2, f);
    return;
  }
  const int sz = 1 << (l - 1);
  const int lf = frozen_range(ibp, f, sz), rf = frozen_range(ibp, f + sz, sz);
  if (!lf) {
    sc_emit(d, OP_F + l - 3, l, f);
    sc_build(d, ibp, l - 1, f, 0);
  }
  if (!rf) {
    sc_emit(d, (lf ? OP_GA : OP_GS) + l - 3, l, f);
    sc_build(d, ibp, l - 1, f + sz, 0);
  }
  if (root) // nobody reads the root's partial sums
    return;
  sc_emit(d, (lf ? OP_BLF : rf ? OP_BRF : OP_BX) + l - 3, l, f);
}

void nr_polar_build_sc_decoder(t_nrPolar_params *pp)
{
  nr_polar_sc_t *d = &pp->sc;
  const int N = pp->N, n = pp->n;
  AssertFatal(n >= 5 && n <= nmax, "N = %d not supported\n", N);
  /* memory: operations, alpha per level (at least 8 values each, 64-byte aligned), partial sums, decisions,
   * information bit positions and targets */
  size_t sz_ops = sizeof(nr_polar_sc_op_t) * N, sz_alpha = 0;
  for (int l = 0; l <= n; l++)
    sz_alpha += ((sizeof(int16_t) * (l < 3 ? 8 : 1 << l)) + 63) & ~63;
  size_t total = sz_ops + sz_alpha + 2 * (N + 64) + 2 * N + 16 + 64;
  uint8_t *mem = memalign(64, total);
  AssertFatal(mem, "polar decoder memory\n");
  memset(mem, 0, total);
  d->mem = mem;
  uint8_t *p = mem;
  for (int l = 0; l <= n; l++) {
    d->alpha[l] = (int16_t *)p;
    p += ((sizeof(int16_t) * (l < 3 ? 8 : 1 << l)) + 63) & ~63;
  }
  d->beta = (int8_t *)p;
  memset(d->beta, -1, N + 64);
  p += N + 64;
  d->u = p;
  p += N + 64;
  d->info_src = (uint16_t *)p;
  p += 2 * N + 16;
  d->ops = (nr_polar_sc_op_t *)p;

  d->nops = 0;
  sc_build(d, pp->information_bit_pattern, n, 0, 1);
  sc_emit(d, OP_END, 0, 0);
  AssertFatal(d->nops <= N, "polar SC program too long: %d\n", d->nops);

  /* information bits (parity-check bits excluded): the k-th in leaf order is bit K - 1 - interleaving_pattern[k] of the
   * decoded word; info_src[t] = leaf of bit t (a permutation of [0, K)), padded to a multiple of 8 with leaf N (u[N] = 0) */
  int k = 0;
  for (int i = 0; i < N; i++)
    if (pp->information_bit_pattern[i] == 1 && (pp->n_pc == 0 || pp->parity_check_bit_pattern[i] == 0)) {
      const int t = pp->K - 1 - pp->interleaving_pattern[k];
      AssertFatal(t >= 0 && t < 128, "information bit %d\n", t);
      d->info_src[t] = i;
      k++;
    }
  AssertFatal(k == pp->K, "%d information bits, K = %d\n", k, pp->K);
  d->n_info = k;
  for (; k < ((d->n_info + 7) & ~7); k++)
    d->info_src[k] = N;

  /* rate recovery: s = de-repeated / de-punctured / de-shortened LLRs in sub-block interleaver order (5.4.1.2),
   * then d[J(m)] = s[m] with J a permutation of 32 blocks of N/32 */
  const int E = pp->encoderLength;
  d->rm_mode = E >= N ? 0 : (pp->K / (double)E) <= (7.0 / 16) ? 1 : 2;
  const int bs = N / 32;
  for (int b = 0; b < 32; b++)
    d->blk_perm[b] = nr_polar_subblock_interleaver_pattern[b];
  /* consistency with the rate matching pattern used by the encoder */
  for (int i = 0; i < E; i++) {
    int m = d->rm_mode == 0 ? i % N : d->rm_mode == 1 ? i + N - E : i;
    AssertFatal(pp->rate_matching_pattern[i] == d->blk_perm[m / bs] * bs + m % bs, "rate matching pattern\n");
  }
  /* coded bit interleaving without repetition: bit de-interleaving and rate recovery composed into one gather,
   * root LLR j = input[rr_src[j]], index E = the fill value (0 punctured, INT16_MAX shortened) */
  d->rr_src = NULL;
  if (pp->i_bil && E <= N) {
    d->rr_src = malloc(sizeof(uint16_t) * N);
    AssertFatal(d->rr_src, "malloc\n");
    for (int j = 0; j < N; j++)
      d->rr_src[j] = E;
    for (int i = 0; i < E; i++)
      d->rr_src[pp->rate_matching_pattern[i]] = pp->i_bil_pattern[i];
  }
}

void nr_polar_free_sc_decoder(t_nrPolar_params *pp)
{
  free(pp->sc.rr_src);
  pp->sc.rr_src = NULL;
  free(pp->sc.mem);
  pp->sc.mem = NULL;
}

/* kernels with a constant half size sz (4 .. 512) */
__attribute__((always_inline)) static inline void k_f(const int16_t *av, int16_t *ac, const int sz)
{
  if (sz == 4)
    op_f4(av, ac);
  else
    op_fv(av, ac, sz);
}
__attribute__((always_inline)) static inline void k_ga(const int16_t *av, int16_t *ac, const int sz)
{
  if (sz == 4)
    op_ga4(av, ac);
  else
    op_gav(av, ac, sz);
}
__attribute__((always_inline)) static inline void k_gs(const int16_t *av, int16_t *ac, const int8_t *bl, const int sz)
{
  if (sz == 4)
    op_gs4(av, ac, bl);
  else
    op_gsv(av, ac, bl, sz);
}
__attribute__((always_inline)) static inline void k_bx(int8_t *bv, const int sz)
{
  if (sz < 16)
    beta_xnor(bv, bv + sz, sz);
  else
    op_bxv(bv, bv + sz, sz);
}

/* runs the program: direct threading (one indirect branch per operation, the program ends with OP_END) */
static void sc_run(const nr_polar_sc_t *d)
{
  int16_t *const *A = d->alpha;
  int8_t *const beta = d->beta;
  uint8_t *const u = d->u;
  const nr_polar_sc_op_t *op = d->ops;
#define LEVEL_LABELS(L)                                                                                          \
  [OP_F + L - 3] = &&f##L, [OP_GA + L - 3] = &&ga##L, [OP_GS + L - 3] = &&gs##L, [OP_BX + L - 3] = &&bx##L, \
  [OP_BLF + L - 3] = &&blf##L, [OP_BRF + L - 3] = &&brf##L,
#define QUAD_LABEL(P) [OP_QUAD + P] = &&q##P,
  static const void *const lab[64] = {LEVEL_LABELS(3) LEVEL_LABELS(4) LEVEL_LABELS(5) LEVEL_LABELS(6) LEVEL_LABELS(7)
                                      LEVEL_LABELS(8) LEVEL_LABELS(9) LEVEL_LABELS(10) QUAD_LABEL(0) QUAD_LABEL(1)
                                      QUAD_LABEL(2) QUAD_LABEL(3) QUAD_LABEL(4) QUAD_LABEL(5) QUAD_LABEL(6) QUAD_LABEL(7)
                                      QUAD_LABEL(8) QUAD_LABEL(9) QUAD_LABEL(10) QUAD_LABEL(11) QUAD_LABEL(12)
                                      QUAD_LABEL(13) QUAD_LABEL(14)[OP_END] = &&end};
#undef LEVEL_LABELS
#undef QUAD_LABEL
#define NEXT goto *lab[(++op)->code]
  goto *lab[op->code];
#define LEVEL(L)                                                    \
  f##L : k_f(A[L], A[L - 1], 1 << (L - 1));                         \
  NEXT;                                                             \
  ga##L : k_ga(A[L], A[L - 1], 1 << (L - 1));                       \
  NEXT;                                                             \
  gs##L : k_gs(A[L], A[L - 1], beta + op->f, 1 << (L - 1));         \
  NEXT;                                                             \
  bx##L : k_bx(beta + op->f, 1 << (L - 1));                         \
  NEXT;                                                             \
  blf##L : __builtin_memcpy(beta + op->f, beta + op->f + (1 << (L - 1)), 1 << (L - 1)); \
  NEXT;                                                             \
  brf##L : __builtin_memset(beta + op->f + (1 << (L - 1)), -1, 1 << (L - 1)); \
  NEXT;
  LEVEL(3) LEVEL(4) LEVEL(5) LEVEL(6) LEVEL(7) LEVEL(8) LEVEL(9) LEVEL(10)
#undef LEVEL
#define QUAD(P)                                        \
  q##P : sc_quad(A[2], beta + op->f, u + op->f, P); \
  NEXT;
  QUAD(0) QUAD(1) QUAD(2) QUAD(3) QUAD(4) QUAD(5) QUAD(6) QUAD(7) QUAD(8) QUAD(9) QUAD(10) QUAD(11) QUAD(12) QUAD(13) QUAD(14)
#undef QUAD
#undef NEXT
end:
  return;
}

/* rate recovery (TS 38.212 5.4.1, inverse) of E LLRs into the root alpha (N LLRs), as nr_polar_rate_matching_int16() */
static void sc_rate_recovery(const t_nrPolar_params *pp, const int16_t *in, int16_t *out)
{
  const nr_polar_sc_t *d = &pp->sc;
  const int N = pp->N, E = pp->encoderLength, bs = N / 32;
  if (d->rr_src) {
    int16_t x[E + 1];
    __builtin_memcpy(x, in, sizeof(int16_t) * E);
    x[E] = d->rm_mode == 2 ? INT16_MAX : 0;
    const uint16_t *src = d->rr_src;
    for (int j = 0; j < N; j++)
      out[j] = x[src[j]];
    return;
  }
  int16_t s[N] __attribute__((aligned(32)));
  int16_t tmp[pp->i_bil ? E : 1];
  if (pp->i_bil) {
    const uint16_t *ib = pp->i_bil_pattern;
    for (int i = 0; i < E; i++)
      tmp[i] = in[ib[i]];
    in = tmp;
  }
  if (d->rm_mode == 0) { // repetition: s[m] = sum of in[m + tN] (wrapping int16 sum, as OAI)
    __builtin_memcpy(s, in, sizeof(int16_t) * N);
    for (int k = N; k < E; k += N) {
      const int len = E - k < N ? E - k : N;
      const int16_t *x = in + k;
      int i = 0;
      for (; i + 8 <= len; i += 8)
        simde_mm_store_si128((simde__m128i *)(s + i),
                             simde_mm_add_epi16(simde_mm_load_si128((simde__m128i *)(s + i)),
                                                simde_mm_loadu_si128((const simde__m128i *)(x + i))));
      for (; i < len; i++)
        s[i] = (int16_t)((uint16_t)s[i] + (uint16_t)x[i]);
    }
  } else if (d->rm_mode == 1) { // puncturing: the first N - E positions are 0
    __builtin_memset(s, 0, sizeof(int16_t) * (N - E));
    __builtin_memcpy(s + N - E, in, sizeof(int16_t) * E);
  } else { // shortening: the last N - E positions are INT16_MAX
    __builtin_memcpy(s, in, sizeof(int16_t) * E);
    for (int i = E; i < N; i++)
      s[i] = INT16_MAX;
  }
  const uint8_t *P = d->blk_perm;
  switch (bs) {
#define BLK(n)                                                \
  case n:                                                     \
    for (int b = 0; b < 32; b++)                              \
      __builtin_memcpy(out + P[b] * n, s + b * n, n * sizeof(int16_t)); \
    break;
    BLK(1) BLK(2) BLK(4) BLK(8) BLK(16) BLK(32)
#undef BLK
  }
}

void nr_polar_sc_decode(const t_nrPolar_params *pp, const int16_t *input, uint64_t *B)
{
  const nr_polar_sc_t *d = &pp->sc;
  sc_rate_recovery(pp, input, d->alpha[pp->n]);
  sc_run(d);
  /* decoded word: bit t = decision of leaf info_src[t]; 8 decisions (bytes 0/1) packed per multiplication */
  const uint8_t *u = d->u;
  const uint16_t *src = d->info_src;
  uint64_t b[2] = {0, 0};
  for (int t = 0; t < d->n_info; t += 8, src += 8) {
    const uint64_t x = (uint64_t)u[src[0]] | (uint64_t)u[src[1]] << 8 | (uint64_t)u[src[2]] << 16 | (uint64_t)u[src[3]] << 24
                       | (uint64_t)u[src[4]] << 32 | (uint64_t)u[src[5]] << 40 | (uint64_t)u[src[6]] << 48
                       | (uint64_t)u[src[7]] << 56;
    b[t >> 6] |= ((x * 0x0102040810204080ULL) >> 56) << (t & 63);
  }
  B[0] = b[0];
  B[1] = b[1];
}
