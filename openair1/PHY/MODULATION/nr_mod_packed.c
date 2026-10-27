/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*
 * Modulation mapping from packed bits (see nr_mod_packed.h).
 *
 * A group is G symbols per layer (G = 16 for NEON, 64 for AVX-512), i.e. G * n_layers consecutive symbols, which
 * start on a byte boundary for every Qm. The input bytes are loaded once and XORed with the scrambling bytes.
 *  NEON, per 16 symbols: the Qm-bit fields are extracted into one byte per symbol (TBL + USHL), layer mapping
 *   permutes these field vectors (UZP, TBL for 3 layers), four TBL lookups give the low and high bytes of I and Q
 *   (64-entry tables for Qm <= 6; for 256QAM the field is first split into its even bits, which select I, and its odd
 *   bits, which select Q, with nibble TBLs) and ST4 interleaves them into c16_t.
 *  AVX-512 (VBMI + GFNI), per 16 symbols of one layer: see run_x_t(); VPERMB gathers each symbol's bytes (this also
 *   does the layer mapping), VPSRLVW + GF2P8AFFINEQB form the I and Q indices, VPERMW looks them up.
 * The tables are derived from OAI's modulation tables and checked against every table entry nr_modulation() and
 * nr_modulation_layer_mapping() use, so both give the same symbols. The last, partial group goes through a
 * zero-padded copy of its input bytes, so nothing past ceil(length / 8) bytes is read and nothing past
 * length / Qm / n_layers symbols per layer is written.
 */

#include "nr_mod_packed.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include "PHY/sse_intrin.h"
#include "PHY/NR_REFSIG/nr_mod_table.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__)
#include <immintrin.h>
#endif

#define AI static inline __attribute__((always_inline))

typedef struct {
  /* [Qm / 2 - 1][I low, I high, Q low, Q high][field value] (Qm = 8: [even bits][odd bits] index, 16 entries) */
  uint8_t t[4][4][64] __attribute__((aligned(64)));
  /* 256QAM NEON: even bits of the low nibble, even bits of the high nibble << 2, same for the odd bits */
  uint8_t nib[4][16] __attribute__((aligned(16)));
  /* [Qm / 2 - 1][index of the even (I) or odd (Q) bits]: the amplitude, same for I and Q (32 words for VPERMW) */
  int16_t t16[4][32] __attribute__((aligned(64)));
  bool ok[4], sym_ok[4];
} mod_tabs_t;

static mod_tabs_t tabs;
static atomic_int tabs_state; /* 0: not built, 1: built */
static pthread_mutex_t tabs_mutex = PTHREAD_MUTEX_INITIALIZER;
static int enabled = -1; /* -1: not read from NR_MOD_PACKED yet */

static inline c16_t sym_of(int qm, unsigned v)
{
  switch (qm) {
    case 2:
      return nr_qpsk_mod_table[v];
    case 4:
      return ((const c16_t *)nr_16qam_mod_table)[v];
    case 6:
      return ((const c16_t *)nr_64qam_mod_table)[2 * v];
    default:
      return ((const c16_t *)nr_256qam_mod_table)[v];
  }
}

static inline bool c16_eq(c16_t a, c16_t b)
{
  return a.r == b.r && a.i == b.i;
}

static inline unsigned even_bits(unsigned v)
{
  return (v & 1) | ((v >> 1) & 2) | ((v >> 2) & 4) | ((v >> 3) & 8);
}

static void put(int q, int e, c16_t s)
{
  tabs.t[q][0][e] = (uint16_t)s.r & 0xff;
  tabs.t[q][1][e] = (uint16_t)s.r >> 8;
  tabs.t[q][2][e] = (uint16_t)s.i & 0xff;
  tabs.t[q][3][e] = (uint16_t)s.i >> 8;
}

static c16_t get(int q, int ei, int eq)
{
  c16_t s;
  s.r = (int16_t)(tabs.t[q][0][ei] | (tabs.t[q][1][ei] << 8));
  s.i = (int16_t)(tabs.t[q][2][eq] | (tabs.t[q][3][eq] << 8));
  return s;
}

/* called with tabs_mutex held */
static void build_tabs(void)
{
  memset(&tabs, 0, sizeof(tabs));
  for (int q = 0; q < 3; q++) {
    const int qm = 2 * q + 2;
    for (unsigned v = 0; v < (1u << qm); v++)
      put(q, v, sym_of(qm, v));
  }
  tabs.ok[0] = tabs.ok[1] = tabs.ok[2] = true;
  /* the tables nr_modulation() reads for whole bytes / 12-bit pairs, and the 3- and 4-layer paths */
  for (unsigned b = 0; b < 256; b++) {
    const c16_t *q2 = (const c16_t *)&nr_qpsk_byte_mod_table[b];
    const c16_t *q4 = (const c16_t *)&nr_16qam_byte_mod_table[b];
    for (int j = 0; j < 4; j++)
      tabs.ok[0] &= c16_eq(q2[j], get(0, (b >> (2 * j)) & 3, (b >> (2 * j)) & 3));
    for (int j = 0; j < 2; j++)
      tabs.ok[1] &= c16_eq(q4[j], get(1, (b >> (4 * j)) & 15, (b >> (4 * j)) & 15));
  }
  for (unsigned v = 0; v < 4096; v++) {
    const c16_t *q6 = (const c16_t *)&nr_64qam_mod_table[v];
    for (int j = 0; j < 2; j++)
      tabs.ok[2] &= c16_eq(q6[j], get(2, (v >> (6 * j)) & 63, (v >> (6 * j)) & 63));
  }
  /* 256QAM: I from the even bits, Q from the odd bits */
  for (unsigned e = 0; e < 16; e++) {
    unsigned ve = 0, vo = 0;
    for (int b = 0; b < 4; b++) {
      ve |= ((e >> b) & 1) << (2 * b);
      vo |= ((e >> b) & 1) << (2 * b + 1);
    }
    const c16_t se = sym_of(8, ve), so = sym_of(8, vo);
    tabs.t[3][0][e] = (uint16_t)se.r & 0xff;
    tabs.t[3][1][e] = (uint16_t)se.r >> 8;
    tabs.t[3][2][e] = (uint16_t)so.i & 0xff;
    tabs.t[3][3][e] = (uint16_t)so.i >> 8;
  }
  tabs.ok[3] = true;
  for (unsigned v = 0; v < 256; v++)
    tabs.ok[3] &= c16_eq(sym_of(8, v), get(3, even_bits(v), even_bits(v >> 1)));
  for (int q = 0; q < 4; q++) {
    const int qm = 2 * q + 2;
    for (unsigned e = 0; e < (1u << (qm / 2)); e++) {
      unsigned v = 0;
      for (int b = 0; b < qm / 2; b++)
        v |= ((e >> b) & 1) << (2 * b);
      tabs.t16[q][e] = sym_of(qm, v).r;
    }
    tabs.sym_ok[q] = tabs.ok[q];
    for (unsigned v = 0; v < (1u << qm); v++) {
      const c16_t s = sym_of(qm, v);
      tabs.sym_ok[q] &= s.r == tabs.t16[q][even_bits(v)] && s.i == tabs.t16[q][even_bits(v >> 1)];
    }
  }
  for (unsigned n = 0; n < 16; n++) {
    tabs.nib[0][n] = even_bits(n);
    tabs.nib[1][n] = even_bits(n) << 2;
    tabs.nib[2][n] = even_bits(n >> 1);
    tabs.nib[3][n] = even_bits(n >> 1) << 2;
  }
}

/* false while nr_generate_modulation_table() has not run (QPSK amplitude 0) */
static bool tabs_ready(void)
{
  if (atomic_load_explicit(&tabs_state, memory_order_acquire))
    return true;
  pthread_mutex_lock(&tabs_mutex);
  if (!atomic_load_explicit(&tabs_state, memory_order_relaxed) && nr_qpsk_mod_table[0].r != 0) {
    build_tabs();
    atomic_store_explicit(&tabs_state, 1, memory_order_release);
  }
  pthread_mutex_unlock(&tabs_mutex);
  return atomic_load_explicit(&tabs_state, memory_order_acquire);
}

/* run ng groups: group g reads in + g * L * G * qm / 8, writes out[l] + g * G */
typedef void (*mod_run_t)(const uint8_t *in, const uint8_t *scr, uint32_t ng, int qm, int L, c16_t *const *out);

typedef struct {
  mod_run_t run;
  int G; /* symbols per layer per group */
  int overread6; /* bytes read past the last group for Qm = 6 */
  bool sym_tab; /* uses t16 (I and Q amplitude per even-bit index) */
} mod_kernel_t;

/* ------------------------------------------------------------------------------------------------------------- */
#if defined(__aarch64__)

typedef struct {
  uint8x16_t c0, c1, c2, c3; /* field extraction constants */
} fx_t;

static const uint8_t q2_idx[16] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
static const int8_t q2_sh[16] = {0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6};
static uint8_t q6_c[4][16]; /* byte, -shift, next byte, 8 - shift */
static const uint8_t l3_idx[3][16] = {{0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45},
                                      {1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46},
                                      {2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47}};

static void neon_consts(void)
{
  for (int k = 0; k < 16; k++) {
    const int o = 6 * k;
    q6_c[0][k] = o >> 3;
    q6_c[1][k] = (uint8_t)(int8_t)(-(o & 7));
    q6_c[2][k] = (o >> 3) + 1;
    q6_c[3][k] = 8 - (o & 7);
  }
}

AI uint8x16_t fields_neon(const uint8_t *p, const uint8_t *s, const int qm, const bool sc, const fx_t *f)
{
  switch (qm) {
    case 2: {
      uint32_t w, z;
      memcpy(&w, p, 4);
      if (sc) {
        memcpy(&z, s, 4);
        w ^= z;
      }
      const uint8x16_t x = vqtbl1q_u8(vreinterpretq_u8_u32(vdupq_n_u32(w)), f->c0);
      return vandq_u8(vshlq_u8(x, vreinterpretq_s8_u8(f->c1)), vdupq_n_u8(3));
    }
    case 4: {
      uint8x8_t x = vld1_u8(p);
      if (sc)
        x = veor_u8(x, vld1_u8(s));
      const uint8x8_t lo = vand_u8(x, vdup_n_u8(15)), hi = vshr_n_u8(x, 4);
      return vcombine_u8(vzip1_u8(lo, hi), vzip2_u8(lo, hi));
    }
    case 6: {
      uint8x16_t x = vld1q_u8(p);
      if (sc)
        x = veorq_u8(x, vld1q_u8(s));
      const uint8x16_t b0 = vshlq_u8(vqtbl1q_u8(x, f->c0), vreinterpretq_s8_u8(f->c1));
      const uint8x16_t b1 = vshlq_u8(vqtbl1q_u8(x, f->c2), vreinterpretq_s8_u8(f->c3));
      return vandq_u8(vorrq_u8(b0, b1), vdupq_n_u8(63));
    }
    default: {
      uint8x16_t x = vld1q_u8(p);
      if (sc)
        x = veorq_u8(x, vld1q_u8(s));
      return x;
    }
  }
}

typedef struct {
  uint8x16x4_t t[4]; /* Qm 6: 64-entry tables; Qm 2/4/8: t[j].val[0] */
  uint8x16_t nib[4];
} lut_t;

AI void lut_store_neon(uint8x16_t v, const int qm, const lut_t *T, c16_t *o)
{
  uint8x16x4_t r;
  if (qm <= 4) {
    for (int j = 0; j < 4; j++)
      r.val[j] = vqtbl1q_u8(T->t[j].val[0], v);
  } else if (qm == 6) {
    for (int j = 0; j < 4; j++)
      r.val[j] = vqtbl4q_u8(T->t[j], v);
  } else {
    const uint8x16_t lo = vandq_u8(v, vdupq_n_u8(15)), hi = vshrq_n_u8(v, 4);
    const uint8x16_t ii = vorrq_u8(vqtbl1q_u8(T->nib[0], lo), vqtbl1q_u8(T->nib[1], hi));
    const uint8x16_t iq = vorrq_u8(vqtbl1q_u8(T->nib[2], lo), vqtbl1q_u8(T->nib[3], hi));
    r.val[0] = vqtbl1q_u8(T->t[0].val[0], ii);
    r.val[1] = vqtbl1q_u8(T->t[1].val[0], ii);
    r.val[2] = vqtbl1q_u8(T->t[2].val[0], iq);
    r.val[3] = vqtbl1q_u8(T->t[3].val[0], iq);
  }
  vst4q_u8((uint8_t *)o, r);
}

AI void run_neon_t(const uint8_t *in, const uint8_t *scr, uint32_t ng, c16_t *const *out, const int qm, const int L, const bool sc)
{
  const int q = qm / 2 - 1;
  const int bb = 2 * qm; /* bytes per block of 16 symbols */
  fx_t f;
  if (qm == 2) {
    f.c0 = vld1q_u8(q2_idx);
    f.c1 = vreinterpretq_u8_s8(vld1q_s8(q2_sh));
  } else if (qm == 6) {
    f.c0 = vld1q_u8(q6_c[0]);
    f.c1 = vld1q_u8(q6_c[1]);
    f.c2 = vld1q_u8(q6_c[2]);
    f.c3 = vld1q_u8(q6_c[3]);
  }
  lut_t T;
  for (int j = 0; j < 4; j++)
    T.t[j] = vld1q_u8_x4(tabs.t[q][j]);
  if (qm == 8)
    for (int j = 0; j < 4; j++)
      T.nib[j] = vld1q_u8(tabs.nib[j]);
  uint8x16_t l3[3];
  if (L == 3)
    for (int j = 0; j < 3; j++)
      l3[j] = vld1q_u8(l3_idx[j]);
  c16_t *o[4] = {0};
  for (int l = 0; l < L; l++)
    o[l] = out[l];
  for (uint32_t g = 0; g < ng; g++, in += L * bb, scr += sc ? L * bb : 0) {
    uint8x16_t v[4], w[4];
    for (int b = 0; b < L; b++)
      v[b] = fields_neon(in + b * bb, scr + b * bb, qm, sc, &f);
    if (L == 1) {
      w[0] = v[0];
    } else if (L == 2) {
      w[0] = vuzp1q_u8(v[0], v[1]);
      w[1] = vuzp2q_u8(v[0], v[1]);
    } else if (L == 3) {
      const uint8x16x3_t t = {{v[0], v[1], v[2]}};
      for (int j = 0; j < 3; j++)
        w[j] = vqtbl3q_u8(t, l3[j]);
    } else {
      const uint8x16_t a = vuzp1q_u8(v[0], v[1]), b = vuzp2q_u8(v[0], v[1]);
      const uint8x16_t c = vuzp1q_u8(v[2], v[3]), d = vuzp2q_u8(v[2], v[3]);
      w[0] = vuzp1q_u8(a, c);
      w[2] = vuzp2q_u8(a, c);
      w[1] = vuzp1q_u8(b, d);
      w[3] = vuzp2q_u8(b, d);
    }
    for (int l = 0; l < L; l++) {
      lut_store_neon(w[l], qm, &T, o[l]);
      o[l] += 16;
    }
  }
}

static void run_neon(const uint8_t *in, const uint8_t *scr, uint32_t ng, int qm, int L, c16_t *const *out)
{
#define C(QM, LL)                                \
  case QM * 8 + LL:                              \
    if (scr)                                     \
      run_neon_t(in, scr, ng, out, QM, LL, true); \
    else                                         \
      run_neon_t(in, scr, ng, out, QM, LL, false); \
    return;
#define CQ(QM) C(QM, 1) C(QM, 2) C(QM, 3) C(QM, 4)
  switch (qm * 8 + L) {
    CQ(2)
    CQ(4)
    CQ(6)
    CQ(8)
  }
#undef CQ
#undef C
}

static const mod_kernel_t *kernel_init(void)
{
  static const mod_kernel_t k = {run_neon, 16, 4, false};
  neon_consts();
  return &k;
}

/* ------------------------------------------------------------------------------------------------------------- */
#elif defined(__x86_64__)

/*
 * AVX-512 VBMI + GFNI: one zmm per 16 symbols of one layer, c16_t per dword. The 2 * Qm * L input bytes of 16 symbols
 * of each layer are loaded once (XOR scrambling); per layer, VPERMB copies the two bytes holding symbol i's field
 * into both words of dword i, VPSRLVW shifts the low word by the field's bit offset and the high word by one more
 * (odd bits to even positions), GF2P8AFFINEQB keeps the even bits below Qm, compressed (the I index in the low word,
 * the Q index in the high word), and VPERMW looks both up in the 16-entry amplitude table (I and Q use the same
 * amplitude per index, checked at table build).
 */
#define TGT __attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi,gfni")))
#define AIX static inline __attribute__((always_inline)) TGT

static uint8_t x_idx[4][4][4][64] __attribute__((aligned(64))); /* [Qm / 2 - 1][L - 1][layer] */
static uint16_t x_sh[4][4][4][32] __attribute__((aligned(64)));
static uint64_t x_gf[4];

static void x86_consts(void)
{
  for (int q = 0; q < 4; q++) {
    const int qm = 2 * q + 2;
    x_gf[q] = 0;
    for (int b = 0; b < qm / 2; b++)
      x_gf[q] |= (uint64_t)(1u << (2 * b)) << (8 * (7 - b));
    for (int L = 1; L <= 4; L++)
      for (int l = 0; l < L; l++)
        for (int i = 0; i < 16; i++) {
          const int o = qm * (L * i + l), b = o >> 3, s = o & 7;
          const int b1 = b + 1 < 64 ? b + 1 : 63; /* b = 63: the field ends in byte 63 */
          x_idx[q][L - 1][l][4 * i] = x_idx[q][L - 1][l][4 * i + 2] = b;
          x_idx[q][L - 1][l][4 * i + 1] = x_idx[q][L - 1][l][4 * i + 3] = b1;
          x_sh[q][L - 1][l][2 * i] = s;
          x_sh[q][L - 1][l][2 * i + 1] = s + 1;
        }
  }
}

/* the cb bytes at p in the low bytes (broadcast loads where cb is a power of two: no mask operation) */
AIX __m512i load_cb(const uint8_t *p, const int cb, __mmask64 km)
{
  switch (cb) {
    case 4: {
      int32_t v;
      memcpy(&v, p, 4);
      return _mm512_set1_epi32(v);
    }
    case 8: {
      int64_t v;
      memcpy(&v, p, 8);
      return _mm512_set1_epi64(v);
    }
    case 16:
      return _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)p));
    case 32:
      return _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)p));
    case 64:
      return _mm512_loadu_si512(p);
    default:
      return _mm512_maskz_loadu_epi8(km, p);
  }
}

AIX void run_x_t(const uint8_t *in, const uint8_t *scr, uint32_t ng, c16_t *const *out, const int qm, const int L, const bool sc)
{
  const int q = qm / 2 - 1;
  const int cb = 2 * qm * L; /* bytes per 16 symbols of every layer */
  const __mmask64 km = cb == 64 ? ~0ULL : (1ULL << cb) - 1;
  const __m512i gf = _mm512_set1_epi64(x_gf[q]), T = _mm512_load_si512(tabs.t16[q]);
  __m512i idx[4], sh[4];
  for (int l = 0; l < L; l++) {
    idx[l] = _mm512_load_si512(x_idx[q][L - 1][l]);
    sh[l] = _mm512_load_si512(x_sh[q][L - 1][l]);
  }
  c16_t *o[4] = {0};
  for (int l = 0; l < L; l++)
    o[l] = out[l];
  for (uint32_t c = 0; c < 4 * ng; c++, in += cb, scr += sc ? cb : 0) {
    __m512i x = load_cb(in, cb, km);
    if (sc)
      x = _mm512_xor_si512(x, load_cb(scr, cb, km));
    for (int l = 0; l < L; l++) {
      __m512i d = _mm512_srlv_epi16(_mm512_permutexvar_epi8(idx[l], x), sh[l]);
      d = _mm512_gf2p8affine_epi64_epi8(d, gf, 0);
      _mm512_storeu_si512(o[l], _mm512_permutexvar_epi16(d, T));
      o[l] += 16;
    }
  }
}

TGT static void run_x(const uint8_t *in, const uint8_t *scr, uint32_t ng, int qm, int L, c16_t *const *out)
{
#define C(QM, LL)                               \
  case QM * 8 + LL:                             \
    if (scr)                                    \
      run_x_t(in, scr, ng, out, QM, LL, true);  \
    else                                        \
      run_x_t(in, scr, ng, out, QM, LL, false); \
    return;
#define CQ(QM) C(QM, 1) C(QM, 2) C(QM, 3) C(QM, 4)
  switch (qm * 8 + L) {
    CQ(2)
    CQ(4)
    CQ(6)
    CQ(8)
  }
#undef CQ
#undef C
}

static const mod_kernel_t *kernel_init(void)
{
  static const mod_kernel_t k = {run_x, 64, 0, true};
  __builtin_cpu_init();
  if (!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("avx512bw") || !__builtin_cpu_supports("avx512vl")
      || !__builtin_cpu_supports("avx512vbmi") || !__builtin_cpu_supports("gfni"))
    return NULL;
  x86_consts();
  return &k;
}

#else
static const mod_kernel_t *kernel_init(void)
{
  return NULL;
}
#endif

/* ------------------------------------------------------------------------------------------------------------- */

static const mod_kernel_t *kernel;
static pthread_once_t kernel_once = PTHREAD_ONCE_INIT;

static void kernel_once_fn(void)
{
  kernel = kernel_init();
  if (enabled < 0) {
    const char *e = getenv("NR_MOD_PACKED");
    enabled = !(e && e[0] == '0');
  }
}

void nr_mod_packed_enable(int on)
{
  pthread_once(&kernel_once, kernel_once_fn);
  enabled = on;
}

bool nr_mod_packed_available(int Qm, int n_layers)
{
  pthread_once(&kernel_once, kernel_once_fn);
  return kernel && enabled && Qm >= 2 && Qm <= 8 && !(Qm & 1) && n_layers >= 1 && n_layers <= 4 && tabs_ready()
         && (kernel->sym_tab ? tabs.sym_ok : tabs.ok)[Qm / 2 - 1];
}

bool nr_mod_packed(const uint32_t *in, const uint32_t *scr, uint32_t length, int Qm, int n_layers, c16_t *const *layers)
{
  if (!nr_mod_packed_available(Qm, n_layers))
    return false;
  const mod_kernel_t *k = kernel;
  const uint32_t nsym = length / Qm;
  if (nsym % n_layers)
    return false;

  const uint8_t *ib = (const uint8_t *)in, *sb = (const uint8_t *)scr;
  const uint32_t nper = nsym / n_layers, G = k->G;
  const uint32_t gb = G * n_layers * Qm / 8, nbytes = (length + 7) / 8;
  uint32_t ng = nper / G;
  if (Qm == 6 && ng && ng * gb + k->overread6 > nbytes)
    ng--;
  if (ng)
    k->run(ib, sb, ng, Qm, n_layers, layers);

  /* rest: at most two groups, through zero-padded copies of their bytes */
  for (uint32_t done = ng * G; done < nper;) {
    uint8_t bi[256 + 64] __attribute__((aligned(64))) = {0}, bs[256 + 64] __attribute__((aligned(64))) = {0};
    c16_t tmp[4][64] __attribute__((aligned(64)));
    c16_t *tp[4] = {tmp[0], tmp[1], tmp[2], tmp[3]};
    const uint32_t off = done / G * gb;
    const uint32_t cnt = nbytes - off < gb ? nbytes - off : gb;
    memcpy(bi, ib + off, cnt);
    if (sb)
      memcpy(bs, sb + off, cnt);
    k->run(bi, sb ? bs : NULL, 1, Qm, n_layers, tp);
    const uint32_t n = nper - done < G ? nper - done : G;
    for (int l = 0; l < n_layers; l++)
      memcpy(layers[l] + done, tmp[l], n * sizeof(c16_t));
    done += n;
  }
  return true;
}
