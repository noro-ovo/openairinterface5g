/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*
 * Multi-lane carry-less folding CRC (see crc_fold.h).
 *
 * The message, zero-padded at the front to whole 16-byte blocks, is M = sum_k B_k x^(128 (nb - 1 - k)), each block
 * B_k a 128-bit polynomial (byte 0 holds the highest powers, so the bytes are reversed into the lanes). Folding a
 * lane A = A_h x^64 + A_l forward by D bits uses A x^D == A_h (x^(D+64) mod P) + A_l (x^D mod P) (mod P), two 64x64
 * carry-less products of at most 95 bits. W lanes each take every W-th block (fold by 128 W bits per round), are then
 * folded to the end of the message and XORed into one 128-bit X == M (mod P), and the left blocks are folded in one by
 * one. crc(M) = M x^32 mod P = X x^32 mod P: X_h x^96 + X_l x^32 is folded to 96 bits with x^96 mod P, then to 64
 * with x^64 mod P, and the 64-bit U is reduced with Barrett's q = floor(floor(U / x^32) mu / x^32), mu = floor(x^64 / P),
 * crc = (U + q P) mod x^32.
 * The first block is loaded as the first 16 bytes moved towards the end by the padding (a byte shuffle; the kernels
 * need at least 16 bytes), so nothing outside the message is read.
 */

#include "crc_fold.h"
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#include <sys/auxv.h>
#ifndef HWCAP_PMULL
#define HWCAP_PMULL (1 << 4)
#endif
#ifndef HWCAP_SHA3
#define HWCAP_SHA3 (1 << 17)
#endif
#endif

static uint64_t xn_mod(unsigned n, uint32_t poly)
{
  uint32_t r = 1;
  while (n--)
    r = (r << 1) ^ ((r >> 31) ? poly : 0);
  return r;
}

static void kpair(uint64_t k[2], unsigned d, uint32_t poly)
{
  k[0] = xn_mod(d, poly);
  k[1] = xn_mod(d + 64, poly);
}

/* floor(x^64 / P), P = x^32 + poly */
static uint64_t barrett_mu(uint32_t poly)
{
  const unsigned __int128 P = ((unsigned __int128)1 << 32) | poly;
  unsigned __int128 r = (unsigned __int128)1 << 64;
  uint64_t q = 0;
  for (int b = 64; b >= 32; b--)
    if ((r >> b) & 1) {
      q |= 1ULL << (b - 32);
      r ^= P << (b - 32);
    }
  return q;
}

void crc_fold_init(crc_fold_ctx_t *ctx, uint32_t poly, const uint32_t *table)
{
  ctx->poly = poly;
  ctx->table = table;
  ctx->k96 = xn_mod(96, poly);
  ctx->k64 = xn_mod(64, poly);
  ctx->mu = barrett_mu(poly);
  kpair(ctx->k128, 128, poly);
  kpair(ctx->k512, 512, poly);
  kpair(ctx->k2048, 2048, poly);
  for (int j = 0; j < 4; j++)
    kpair(ctx->kc4[j], 128 * (3 - j), poly);
  for (int j = 0; j < 16; j++)
    kpair(ctx->kc16[j], 128 * (15 - j), poly);
}

/* table CRC of short messages */
static inline uint32_t crc_table(const crc_fold_ctx_t *ctx, const uint8_t *x, uint32_t len)
{
  uint32_t crc = 0;
  for (uint32_t i = 0; i < len; i++)
    crc = (crc << 8) ^ ctx->table[x[i] ^ (crc >> 24)];
  return crc;
}

/* byte shuffle indices: the 16 bytes at rev_shift + h reverse a block and move it by h bytes (0xff: zero) */
static const uint8_t rev_shift[32] __attribute__((aligned(16))) = {15,   14,   13,   12,   11,   10,   9,    8,
                                                                    7,    6,    5,    4,    3,    2,    1,    0,
                                                                    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                                                    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

typedef bool (*fold_fn_t)(const crc_fold_ctx_t *ctx, const uint8_t *data, uint32_t len, uint32_t *crc);

/* ------------------------------------------------------------------------------------------------------------- */
#if defined(__x86_64__)

#define TGT128 __attribute__((target("pclmul,ssse3,sse4.1")))
#define TGT512 __attribute__((target("pclmul,ssse3,sse4.1,avx2,avx512f,avx512bw,avx512vl,vpclmulqdq")))

#define FOLD128(a, k) _mm_xor_si128(_mm_clmulepi64_si128(a, k, 0x00), _mm_clmulepi64_si128(a, k, 0x11))

/* X (high qword: highest powers) to the CRC */
TGT128 static inline uint32_t reduce_x86(const crc_fold_ctx_t *ctx, __m128i x)
{
  const __m128i k = _mm_set_epi64x(ctx->k64, ctx->k96), m = _mm_set_epi64x(ctx->poly | (1ULL << 32), ctx->mu);
  /* X_h (x^96 mod P) + X_l x^32: < 2^96 */
  __m128i t = _mm_xor_si128(_mm_clmulepi64_si128(x, k, 0x01), _mm_slli_si128(_mm_move_epi64(x), 4));
  /* bits 64..95 times (x^64 mod P) + bits 0..63 */
  const __m128i u = _mm_xor_si128(_mm_clmulepi64_si128(_mm_srli_si128(t, 8), k, 0x10), t);
  /* Barrett */
  const __m128i q = _mm_srli_epi64(_mm_clmulepi64_si128(_mm_srli_epi64(u, 32), m, 0x00), 32);
  const __m128i r = _mm_xor_si128(u, _mm_clmulepi64_si128(q, m, 0x10));
  return (uint32_t)_mm_cvtsi128_si32(r);
}

TGT128 static bool fold_pclmul(const crc_fold_ctx_t *ctx, const uint8_t *data, uint32_t len, uint32_t *crc)
{
  if (len < 64)
    return false;
  const __m128i rev = _mm_load_si128((const __m128i *)rev_shift);
#define LD(p) _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(p)), rev)
  const uint32_t h = (16 - (len & 15)) & 15, nb = (len + h) / 16;
  const uint8_t *p = data + 16 - h;
  __m128i x0 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)data), _mm_loadu_si128((const __m128i *)(rev_shift + h)));
  __m128i x1 = LD(p), x2 = LD(p + 16), x3 = LD(p + 32);
  p += 48;
  uint32_t left = nb - 4;
  const __m128i k4 = _mm_loadu_si128((const __m128i *)ctx->k512);
  for (; left >= 4; left -= 4, p += 64) {
    x0 = _mm_xor_si128(FOLD128(x0, k4), LD(p));
    x1 = _mm_xor_si128(FOLD128(x1, k4), LD(p + 16));
    x2 = _mm_xor_si128(FOLD128(x2, k4), LD(p + 32));
    x3 = _mm_xor_si128(FOLD128(x3, k4), LD(p + 48));
  }
  x0 = FOLD128(x0, _mm_loadu_si128((const __m128i *)ctx->kc4[0]));
  x1 = FOLD128(x1, _mm_loadu_si128((const __m128i *)ctx->kc4[1]));
  x2 = FOLD128(x2, _mm_loadu_si128((const __m128i *)ctx->kc4[2]));
  __m128i x = _mm_xor_si128(_mm_xor_si128(x0, x1), _mm_xor_si128(x2, x3));
  const __m128i k1 = _mm_loadu_si128((const __m128i *)ctx->k128);
  for (; left; left--, p += 16)
    x = _mm_xor_si128(FOLD128(x, k1), LD(p));
#undef LD
  *crc = reduce_x86(ctx, x);
  return true;
}

#define FOLD512(a, k) _mm512_xor_si512(_mm512_clmulepi64_epi128(a, k, 0x00), _mm512_clmulepi64_epi128(a, k, 0x11))
#define FOLD512X(a, k, b) \
  _mm512_ternarylogic_epi64(_mm512_clmulepi64_epi128(a, k, 0x00), _mm512_clmulepi64_epi128(a, k, 0x11), b, 0x96)

TGT512 static bool fold_vpclmul(const crc_fold_ctx_t *ctx, const uint8_t *data, uint32_t len, uint32_t *crc)
{
  if (len < 256)
    return fold_pclmul(ctx, data, len, crc);
  const __m512i rev = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i *)rev_shift));
#define LD(p) _mm512_shuffle_epi8(_mm512_loadu_si512(p), rev)
  const uint32_t h = (16 - (len & 15)) & 15, nb = (len + h) / 16;
  /* blocks 0..3 from data - h, the h bytes before data masked off (not read) */
  const uint8_t *p = data - h;
  __m512i z0 = _mm512_shuffle_epi8(_mm512_maskz_loadu_epi8(~0ULL << h, p), rev);
  __m512i z1 = LD(p + 64), z2 = LD(p + 128), z3 = LD(p + 192);
  p += 256;
  uint32_t left = nb - 16;
  const __m512i k16 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)ctx->k2048));
  for (; left >= 16; left -= 16, p += 256) {
    z0 = FOLD512X(z0, k16, LD(p));
    z1 = FOLD512X(z1, k16, LD(p + 64));
    z2 = FOLD512X(z2, k16, LD(p + 128));
    z3 = FOLD512X(z3, k16, LD(p + 192));
  }
  /* lane j of z_v is block 4 v + j of the last 16: fold it by 128 (15 - 4 v - j) bits */
  __m512i t = FOLD512(z0, _mm512_load_si512(ctx->kc16[0]));
  t = FOLD512X(z1, _mm512_load_si512(ctx->kc16[4]), t);
  t = FOLD512X(z2, _mm512_load_si512(ctx->kc16[8]), t);
  t = FOLD512X(z3, _mm512_load_si512(ctx->kc16[12]), t);
  /* the rest by 4 blocks (one zmm, folded by 512 bits), then by single blocks */
  if (left >= 4) {
    /* t holds 4 lane values all aligned to the end of the 16 blocks: put their sum in lane 3 and fold by 4 blocks */
    const __m256i t2 = _mm256_xor_si256(_mm512_castsi512_si256(t), _mm512_extracti64x4_epi64(t, 1));
    const __m128i t1 = _mm_xor_si128(_mm256_castsi256_si128(t2), _mm256_extracti128_si256(t2, 1));
    __m512i y = _mm512_maskz_broadcast_i32x4(0xf000, t1);
    const __m512i k4 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)ctx->k512));
    for (; left >= 4; left -= 4, p += 64)
      y = FOLD512X(y, k4, LD(p));
    t = FOLD512(y, _mm512_load_si512(ctx->kc16[12]));
  }
  const __m256i t2 = _mm256_xor_si256(_mm512_castsi512_si256(t), _mm512_extracti64x4_epi64(t, 1));
  __m128i x = _mm_xor_si128(_mm256_castsi256_si128(t2), _mm256_extracti128_si256(t2, 1));
#undef LD
  const __m128i rev1 = _mm512_castsi512_si128(rev);
  const __m128i k1 = _mm_loadu_si128((const __m128i *)ctx->k128);
  for (; left; left--, p += 16)
    x = _mm_xor_si128(FOLD128(x, k1), _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)p), rev1));
  *crc = reduce_x86(ctx, x);
  return true;
}

static fold_fn_t best_kernel(const char **name)
{
  __builtin_cpu_init();
  if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl")
      && __builtin_cpu_supports("vpclmulqdq") && __builtin_cpu_supports("pclmul")) {
    *name = "vpclmul";
    return fold_vpclmul;
  }
  if (__builtin_cpu_supports("pclmul") && __builtin_cpu_supports("sse4.1")) {
    *name = "pclmul";
    return fold_pclmul;
  }
  *name = "off";
  return NULL;
}

static fold_fn_t named_kernel(const char *name)
{
  __builtin_cpu_init();
  if (!strcmp(name, "pclmul") && __builtin_cpu_supports("pclmul") && __builtin_cpu_supports("sse4.1"))
    return fold_pclmul;
  if (!strcmp(name, "vpclmul") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("vpclmulqdq"))
    return fold_vpclmul;
  return NULL;
}

/* ------------------------------------------------------------------------------------------------------------- */
#elif defined(__aarch64__)

#define TGTP __attribute__((target("+aes")))

TGTP static inline uint8x16_t fold_n(uint8x16_t a, poly64x2_t k)
{
  const poly64x2_t a64 = vreinterpretq_p64_u8(a);
  const poly128_t lo = vmull_p64(vgetq_lane_p64(a64, 0), vgetq_lane_p64(k, 0));
  const poly128_t hi = vmull_high_p64(a64, k);
  return veorq_u8(vreinterpretq_u8_p128(lo), vreinterpretq_u8_p128(hi));
}

TGTP static inline uint64_t clmul_lo(uint64_t a, uint64_t b)
{
  return vgetq_lane_u64(vreinterpretq_u64_p128(vmull_p64(a, b)), 0);
}

TGTP static inline uint32_t reduce_a64(const crc_fold_ctx_t *ctx, uint8x16_t x)
{
  const uint64x2_t x64 = vreinterpretq_u64_u8(x);
  const uint64_t xl = vgetq_lane_u64(x64, 0), xh = vgetq_lane_u64(x64, 1);
  /* X_h (x^96 mod P) + X_l x^32: < 2^96 */
  const uint64x2_t t = veorq_u64(vreinterpretq_u64_p128(vmull_p64(xh, ctx->k96)), vcombine_u64(vcreate_u64(xl << 32), vcreate_u64(xl >> 32)));
  /* bits 64..95 times (x^64 mod P) + bits 0..63 */
  const uint64_t u = clmul_lo(vgetq_lane_u64(t, 1), ctx->k64) ^ vgetq_lane_u64(t, 0);
  /* Barrett */
  const uint64_t q = clmul_lo(u >> 32, ctx->mu) >> 32;
  return (uint32_t)(u ^ clmul_lo(q, ctx->poly | (1ULL << 32)));
}

/* main loop: 4 lanes; with eor3 (FEAT_SHA3) the two products and the data in one EOR3 */
#define FOLD4_BODY(EOR3)                                                       \
  for (; left >= 4; left -= 4, p += 64) {                                     \
    const uint8x16x4_t d = vld1q_u8_x4(p);                                    \
    x0 = EOR3(x0, k4, vqtbl1q_u8(d.val[0], rev));                              \
    x1 = EOR3(x1, k4, vqtbl1q_u8(d.val[1], rev));                              \
    x2 = EOR3(x2, k4, vqtbl1q_u8(d.val[2], rev));                              \
    x3 = EOR3(x3, k4, vqtbl1q_u8(d.val[3], rev));                              \
  }

TGTP static inline uint8x16_t fold_eor(uint8x16_t a, poly64x2_t k, uint8x16_t b)
{
  return veorq_u8(fold_n(a, k), b);
}

__attribute__((target("+aes+sha3"))) static inline uint8x16_t fold_eor3(uint8x16_t a, poly64x2_t k, uint8x16_t b)
{
  const poly64x2_t a64 = vreinterpretq_p64_u8(a);
  const poly128_t lo = vmull_p64(vgetq_lane_p64(a64, 0), vgetq_lane_p64(k, 0));
  const poly128_t hi = vmull_high_p64(a64, k);
  return veor3q_u8(vreinterpretq_u8_p128(lo), vreinterpretq_u8_p128(hi), b);
}

#define FOLD_PMULL_FN(NAME, TGT, EOR3)                                                                  \
  TGT static bool NAME(const crc_fold_ctx_t *ctx, const uint8_t *data, uint32_t len, uint32_t *crc)   \
  {                                                                                                     \
    if (len < 16) {                                                                                     \
      *crc = crc_table(ctx, data, len);                                                                 \
      return true;                                                                                      \
    }                                                                                                   \
    const uint8x16_t rev = vld1q_u8(rev_shift);                                                         \
    const uint32_t h = (16 - (len & 15)) & 15, nb = (len + h) / 16;                                     \
    const uint8_t *p = data + 16 - h;                                                                   \
    uint8x16_t x = vqtbl1q_u8(vld1q_u8(data), vld1q_u8(rev_shift + h));                                 \
    uint32_t left = nb - 1;                                                                             \
    if (nb >= 4) {                                                                                      \
      uint8x16_t x0 = x, x1 = vqtbl1q_u8(vld1q_u8(p), rev), x2 = vqtbl1q_u8(vld1q_u8(p + 16), rev);     \
      uint8x16_t x3 = vqtbl1q_u8(vld1q_u8(p + 32), rev);                                                \
      p += 48;                                                                                          \
      left = nb - 4;                                                                                    \
      const poly64x2_t k4 = vreinterpretq_p64_u64(vld1q_u64(ctx->k512));                                \
      FOLD4_BODY(EOR3)                                                                                  \
      x0 = fold_n(x0, vreinterpretq_p64_u64(vld1q_u64(ctx->kc4[0])));                                   \
      x1 = fold_n(x1, vreinterpretq_p64_u64(vld1q_u64(ctx->kc4[1])));                                   \
      x = EOR3(x2, vreinterpretq_p64_u64(vld1q_u64(ctx->kc4[2])), x0);                                  \
      x = veorq_u8(veorq_u8(x, x1), x3);                                                                \
    }                                                                                                   \
    const poly64x2_t k1 = vreinterpretq_p64_u64(vld1q_u64(ctx->k128));                                  \
    for (; left; left--, p += 16)                                                                       \
      x = EOR3(x, k1, vqtbl1q_u8(vld1q_u8(p), rev));                                                    \
    *crc = reduce_a64(ctx, x);                                                                          \
    return true;                                                                                        \
  }

FOLD_PMULL_FN(fold_pmull, TGTP, fold_eor)
FOLD_PMULL_FN(fold_pmull_eor3, __attribute__((target("+aes+sha3"))), fold_eor3)

static fold_fn_t best_kernel(const char **name)
{
  const unsigned long hw = getauxval(AT_HWCAP);
  if ((hw & HWCAP_PMULL) && (hw & HWCAP_SHA3)) {
    *name = "pmull-eor3";
    return fold_pmull_eor3;
  }
  if (hw & HWCAP_PMULL) {
    *name = "pmull";
    return fold_pmull;
  }
  *name = "off";
  return NULL;
}

static fold_fn_t named_kernel(const char *name)
{
  const unsigned long hw = getauxval(AT_HWCAP);
  if (!strcmp(name, "pmull") && (hw & HWCAP_PMULL))
    return fold_pmull;
  if (!strcmp(name, "pmull-eor3") && (hw & HWCAP_PMULL) && (hw & HWCAP_SHA3))
    return fold_pmull_eor3;
  return NULL;
}

#else
static fold_fn_t best_kernel(const char **name)
{
  *name = "off";
  return NULL;
}
static fold_fn_t named_kernel(const char *name)
{
  (void)name;
  return NULL;
}
#endif

/* ------------------------------------------------------------------------------------------------------------- */

static fold_fn_t kernel;
static const char *kernel_name;
static int kernel_set;

static void select_default(void)
{
  kernel = best_kernel(&kernel_name);
  const char *e = getenv("NR_CRC_FOLD");
  if (e && *e)
    crc_fold_set_variant(e);
  __atomic_store_n(&kernel_set, 1, __ATOMIC_RELEASE);
}

bool crc_fold_set_variant(const char *name)
{
  if (!name) {
    kernel = best_kernel(&kernel_name);
    kernel_set = 1;
    return kernel != NULL;
  }
  if (!strcmp(name, "off")) {
    kernel = NULL;
    kernel_name = "off";
    kernel_set = 1;
    return true;
  }
  fold_fn_t k = named_kernel(name);
  if (!k)
    return false;
  kernel = k;
  kernel_name = name;
  kernel_set = 1;
  return true;
}

const char *crc_fold_variant(void)
{
  if (!__atomic_load_n(&kernel_set, __ATOMIC_ACQUIRE))
    select_default();
  return kernel_name;
}

bool crc_fold(const crc_fold_ctx_t *ctx, const uint8_t *data, uint32_t len, uint32_t *crc)
{
  if (!__atomic_load_n(&kernel_set, __ATOMIC_ACQUIRE))
    select_default();
  const fold_fn_t k = kernel;
  return k && k(ctx, data, len, crc);
}
