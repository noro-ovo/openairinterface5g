/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* x86-64 AVX-512 kernels of the packed LDPC transmit chain: encoder rows with 64-byte chunks, masked tail store and
 * vpternlogd 3-way XOR (F/BW/VL); output stage with VBMI interleaving and a GFNI 8x8 bit transpose (VBMI + GFNI). */
#include "nr_ldpc_tx_internal.h"
#if defined(__x86_64__)
#include <immintrin.h>

#define TGT __attribute__((target("avx512f,avx512bw,avx512vl")))
#define AI static inline __attribute__((always_inline)) TGT

/* whole row in N (1..6) zmm accumulators (explicit variables); the first edge (or the first two) initialises
 * them, then two edges per vpternlogd; last chunk stored with a byte mask (loads may read past the row into the
 * NR_LDPC_TX_PAD slack of the work area) */
#define LD(p, k) _mm512_loadu_si512((p) + 64 * (k))
/* keep loads as [reg + disp] (indexed addresses un-laminate the micro-fused load+op on Intel cores) */
#define OPAQUE(p) __asm__("" : "+r"(p))
#define REP6(M) M(0) M(1) M(2) M(3) M(4) M(5)
AI void row512(const uint8_t *base, const uint16_t *off, int ne, const int N, __mmask64 km, uint8_t *d, int z, int dup)
{
#define DECL(k) __m512i a##k = _mm512_setzero_si512();
  REP6(DECL)
#undef DECL
  int e = 0;
  if (ne & 1) {
    const uint8_t *p = base + off[0];
    OPAQUE(p);
#define I1(k) \
  if (N > k)  \
    a##k = LD(p, k);
    REP6(I1)
#undef I1
    e = 1;
  } else if (ne > 0) {
    const uint8_t *p = base + off[0], *q = base + off[1];
    OPAQUE(p);
    OPAQUE(q);
#define I2(k) \
  if (N > k)  \
    a##k = _mm512_xor_si512(LD(p, k), LD(q, k));
    REP6(I2)
#undef I2
    e = 2;
  }
  for (; e < ne; e += 2) {
    const uint8_t *p = base + off[e], *q = base + off[e + 1];
    OPAQUE(p);
    OPAQUE(q);
#define X3(k) \
  if (N > k)  \
    a##k = _mm512_ternarylogic_epi32(a##k, LD(p, k), LD(q, k), 0x96);
    REP6(X3)
#undef X3
  }
  for (int k = 0; k <= dup; k++) {
    uint8_t *q = d + k * z;
#define S1(k)                                \
  if (N - 1 > k)                             \
    _mm512_storeu_si512(q + 64 * (k), a##k); \
  else if (N - 1 == k)                       \
    _mm512_mask_storeu_epi8(q + 64 * (k), km, a##k);
    REP6(S1)
#undef S1
  }
}
#undef LD

static TGT void
rows_avx512(const uint8_t *base, const uint16_t *off, const uint8_t *cnt, int nrows, int z, uint8_t *dst, int dstride, int dup)
{
  const int N = (z + 63) >> 6, rem = z - 64 * (N - 1);
  const __mmask64 km = rem == 64 ? ~0ULL : (1ULL << rem) - 1;
  for (int r = 0; r < nrows; r++) {
    const int ne = cnt[r];
    switch (N) {
      case 1:
        row512(base, off, ne, 1, km, dst, z, dup);
        break;
      case 2:
        row512(base, off, ne, 2, km, dst, z, dup);
        break;
      case 3:
        row512(base, off, ne, 3, km, dst, z, dup);
        break;
      case 4:
        row512(base, off, ne, 4, km, dst, z, dup);
        break;
      case 5:
        row512(base, off, ne, 5, km, dst, z, dup);
        break;
      default:
        row512(base, off, ne, 6, km, dst, z, dup);
        break;
    }
    off += ne;
    dst += dstride;
  }
}

/* select-based exchange: x' = m ? x : y<<s ;  y' = m ? x>>s : y   (vpternlogd 0xCA = A ? B : C) */
#define XSW(x, y, s, m)                                                  \
  do {                                                                   \
    __m512i x_ = x;                                                      \
    x = _mm512_ternarylogic_epi32(m, x_, _mm512_slli_epi16(y, s), 0xCA); \
    y = _mm512_ternarylogic_epi32(m, _mm512_srli_epi16(x_, s), y, 0xCA); \
  } while (0)

/* 8 input registers (64 bytes of each stream) -> Q[q] = output bytes [64q, 64q+64) (q = 0..7) */
AI void tr512(__m512i r[8], __m512i Q[8])
{
  const __m512i m1 = _mm512_set1_epi8(0x55), m2 = _mm512_set1_epi8(0x33), m4 = _mm512_set1_epi8(0x0f);
  XSW(r[0], r[1], 1, m1);
  XSW(r[2], r[3], 1, m1);
  XSW(r[4], r[5], 1, m1);
  XSW(r[6], r[7], 1, m1);
  XSW(r[0], r[2], 2, m2);
  XSW(r[1], r[3], 2, m2);
  XSW(r[4], r[6], 2, m2);
  XSW(r[5], r[7], 2, m2);
  XSW(r[0], r[4], 4, m4);
  XSW(r[1], r[5], 4, m4);
  XSW(r[2], r[6], 4, m4);
  XSW(r[3], r[7], 4, m4);
  /* out[8b+u] = r[7-u][b]: per 128-bit lane, 8-way byte interleave by 3 unpack stages */
  __m512i A0 = _mm512_unpacklo_epi8(r[7], r[6]), A1 = _mm512_unpackhi_epi8(r[7], r[6]);
  __m512i B0 = _mm512_unpacklo_epi8(r[5], r[4]), B1 = _mm512_unpackhi_epi8(r[5], r[4]);
  __m512i C0 = _mm512_unpacklo_epi8(r[3], r[2]), C1 = _mm512_unpackhi_epi8(r[3], r[2]);
  __m512i D0 = _mm512_unpacklo_epi8(r[1], r[0]), D1 = _mm512_unpackhi_epi8(r[1], r[0]);
  __m512i E0 = _mm512_unpacklo_epi16(A0, B0), E1 = _mm512_unpackhi_epi16(A0, B0);
  __m512i E2 = _mm512_unpacklo_epi16(A1, B1), E3 = _mm512_unpackhi_epi16(A1, B1);
  __m512i F0 = _mm512_unpacklo_epi16(C0, D0), F1 = _mm512_unpackhi_epi16(C0, D0);
  __m512i F2 = _mm512_unpacklo_epi16(C1, D1), F3 = _mm512_unpackhi_epi16(C1, D1);
  __m512i G[8];
  G[0] = _mm512_unpacklo_epi32(E0, F0);
  G[1] = _mm512_unpackhi_epi32(E0, F0);
  G[2] = _mm512_unpacklo_epi32(E1, F1);
  G[3] = _mm512_unpackhi_epi32(E1, F1);
  G[4] = _mm512_unpacklo_epi32(E2, F2);
  G[5] = _mm512_unpackhi_epi32(E2, F2);
  G[6] = _mm512_unpacklo_epi32(E3, F3);
  G[7] = _mm512_unpackhi_epi32(E3, F3);
  /* G[k] lane L -> out offset 128L + 16k: 4x4 128-bit lane transpose within G[0..3] and G[4..7] */
  for (int h = 0; h < 2; h++) {
    __m512i *g = G + 4 * h;
    __m512i U0 = _mm512_shuffle_i64x2(g[0], g[1], 0x88), U1 = _mm512_shuffle_i64x2(g[0], g[1], 0xDD);
    __m512i U2 = _mm512_shuffle_i64x2(g[2], g[3], 0x88), U3 = _mm512_shuffle_i64x2(g[2], g[3], 0xDD);
    Q[0 + h] = _mm512_shuffle_i64x2(U0, U2, 0x88); /* block q = 2L + h */
    Q[2 + h] = _mm512_shuffle_i64x2(U1, U3, 0x88);
    Q[4 + h] = _mm512_shuffle_i64x2(U0, U2, 0xDD);
    Q[6 + h] = _mm512_shuffle_i64x2(U1, U3, 0xDD);
  }
}

static TGT void transpose_avx512(const uint8_t *const in[8], int nblk, uint8_t *out)
{
  for (int i = 0; i < nblk; i++) {
    __m512i r[8], Q[8];
    for (int j = 0; j < 8; j++)
      r[j] = _mm512_loadu_si512(in[j] + 64 * i);
    tr512(r, Q);
    for (int q = 0; q < 8; q++)
      _mm512_storeu_si512(out + 512 * i + 64 * q, Q[q]);
  }
}

/* column transposer: z/8 <= 48 bytes per stream and column -> one masked load per stream, masked last store */
static TGT void tcols_avx512(const uint8_t *const in[8], int z, int ncols, uint8_t *work, int S, uint8_t *out, int cw)
{
  const int nb = z >> 3, nq = (nb + 7) >> 3, rb = nb - 8 * (nq - 1);
  const __mmask64 kin = nb == 64 ? ~0ULL : (1ULL << nb) - 1, kout = rb == 8 ? ~0ULL : (1ULL << (8 * rb)) - 1;
  for (int c = 0; c < ncols; c++) {
    __m512i r[8], Q[8];
    for (int j = 0; j < 8; j++)
      r[j] = _mm512_maskz_loadu_epi8(kin, in[j] + c * nb);
    tr512(r, Q);
    uint8_t *w0 = work + c * S, *w1 = w0 + z, *o = out + (c - 2) * z;
    const int wo = c >= 2 && c < cw;
    for (int q = 0; q < nq - 1; q++) {
      _mm512_storeu_si512(w0 + 64 * q, Q[q]);
      _mm512_storeu_si512(w1 + 64 * q, Q[q]);
      if (wo)
        _mm512_storeu_si512(o + 64 * q, Q[q]);
    }
    const int q = nq - 1;
    _mm512_mask_storeu_epi8(w0 + 64 * q, kout, Q[q]);
    _mm512_mask_storeu_epi8(w1 + 64 * q, kout, Q[q]);
    if (wo)
      _mm512_mask_storeu_epi8(o + 64 * q, kout, Q[q]);
  }
}

/* ------------------------------------------------------------------------------------------ output stage */

#define TGTX __attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi,gfni")))

static const uint16_t q6_ab[3][32] __attribute__((aligned(64))) = {
    {0, 32, 0, 1, 33, 0, 2, 34, 0, 3, 35, 0, 4, 36, 0, 5, 37, 0, 6, 38, 0, 7, 39, 0, 8, 40, 0, 9, 41, 0, 10, 42},
    {0, 11, 43, 0, 12, 44, 0, 13, 45, 0, 14, 46, 0, 15, 47, 0, 16, 48, 0, 17, 49, 0, 18, 50, 0, 19, 51, 0, 20, 52, 0, 21},
    {53, 0, 22, 54, 0, 23, 55, 0, 24, 56, 0, 25, 57, 0, 26, 58, 0, 27, 59, 0, 28, 60, 0, 29, 61, 0, 30, 62, 0, 31, 63, 0},
};
static const uint16_t q6_c[3][32] __attribute__((aligned(64))) = {
    {0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0},
    {10, 0, 0, 11, 0, 0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0, 18, 0, 0, 19, 0, 0, 20, 0},
    {0, 21, 0, 0, 22, 0, 0, 23, 0, 0, 24, 0, 0, 25, 0, 0, 26, 0, 0, 27, 0, 0, 28, 0, 0, 29, 0, 0, 30, 0, 0, 31},
};
static const __mmask32 q6_cm[3] = {0x24924924, 0x49249249, 0x92492492};
static const uint8_t p8[2][64] __attribute__((aligned(64))) = {
    {0,  64, 1,  65, 2,  66, 3,  67, 4,  68, 5,  69, 6,  70, 7,  71, 8,  72, 9,  73, 10, 74,
     11, 75, 12, 76, 13, 77, 14, 78, 15, 79, 16, 80, 17, 81, 18, 82, 19, 83, 20, 84, 21, 85,
     22, 86, 23, 87, 24, 88, 25, 89, 26, 90, 27, 91, 28, 92, 29, 93, 30, 94, 31, 95},
    {32, 96,  33, 97,  34, 98,  35, 99,  36, 100, 37, 101, 38, 102, 39, 103, 40, 104, 41, 105, 42, 106,
     43, 107, 44, 108, 45, 109, 46, 110, 47, 111, 48, 112, 49, 113, 50, 114, 51, 115, 52, 116, 53, 117,
     54, 118, 55, 119, 56, 120, 57, 121, 58, 122, 59, 123, 60, 124, 61, 125, 62, 126, 63, 127},
};
static const uint16_t p16[2][32] __attribute__((aligned(64))) = {
    {0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47},
    {16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55,
     24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63},
};
static const uint32_t p32[2][16] __attribute__((aligned(64))) = {
    {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23},
    {8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31},
};
static const uint8_t rev8[64] __attribute__((aligned(64))) = {
    7,  6,  5,  4,  3,  2,  1,  0,  15, 14, 13, 12, 11, 10, 9,  8,  23, 22, 21, 20, 19, 18, 17, 16, 31, 30, 29, 28, 27, 26, 25, 24,
    39, 38, 37, 36, 35, 34, 33, 32, 47, 46, 45, 44, 43, 42, 41, 40, 55, 54, 53, 52, 51, 50, 49, 48, 63, 62, 61, 60, 59, 58, 57, 56};
static const uint8_t gath[64] __attribute__((aligned(64))) = {
    0, 8,  16, 24, 32, 40, 48, 56, 1, 9,  17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59,
    4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63};

#define L5(p) _mm512_loadu_si512((const void *)(p))
#define S5(p, v) _mm512_storeu_si512((void *)(p), v)

/* f[qm t + q] = src[q][t] (TS 38.212 5.4.2.2): 64 t per step with byte/word/dword two-source permutes (Qm = 6: byte
 * pairs, then a 3-way 16-bit interleave as a two-source permute merged with a masked permute of the third source),
 * remaining t by the SSE kernel */
static TGTX void interleave_avx512(int qm, const uint8_t *const src[8], int L, uint8_t *f)
{
  int t = 0;
  const __m512i i8l = L5(p8[0]), i8h = L5(p8[1]);
  switch (qm) {
    case 2:
      for (; t + 64 <= L; t += 64) {
        const __m512i a = L5(src[0] + t), b = L5(src[1] + t);
        S5(f + 2 * t, _mm512_permutex2var_epi8(a, i8l, b));
        S5(f + 2 * t + 64, _mm512_permutex2var_epi8(a, i8h, b));
      }
      break;
    case 4: {
      const __m512i i16l = L5(p16[0]), i16h = L5(p16[1]);
      for (; t + 64 <= L; t += 64) {
        const __m512i a = L5(src[0] + t), b = L5(src[1] + t), c = L5(src[2] + t), d = L5(src[3] + t);
        const __m512i ab0 = _mm512_permutex2var_epi8(a, i8l, b), ab1 = _mm512_permutex2var_epi8(a, i8h, b);
        const __m512i cd0 = _mm512_permutex2var_epi8(c, i8l, d), cd1 = _mm512_permutex2var_epi8(c, i8h, d);
        uint8_t *o = f + 4 * t;
        S5(o, _mm512_permutex2var_epi16(ab0, i16l, cd0));
        S5(o + 64, _mm512_permutex2var_epi16(ab0, i16h, cd0));
        S5(o + 128, _mm512_permutex2var_epi16(ab1, i16l, cd1));
        S5(o + 192, _mm512_permutex2var_epi16(ab1, i16h, cd1));
      }
    } break;
    case 6: {
      const __m512i iab0 = L5(q6_ab[0]), iab1 = L5(q6_ab[1]), iab2 = L5(q6_ab[2]);
      const __m512i ic0 = L5(q6_c[0]), ic1 = L5(q6_c[1]), ic2 = L5(q6_c[2]);
      for (; t + 64 <= L; t += 64) {
        __m512i s[6];
        for (int q = 0; q < 6; q++)
          s[q] = L5(src[q] + t);
        uint8_t *o = f + 6 * t;
        for (int h = 0; h < 2; h++) {
          const __m512i ih = h ? i8h : i8l;
          const __m512i A = _mm512_permutex2var_epi8(s[0], ih, s[1]), B = _mm512_permutex2var_epi8(s[2], ih, s[3]);
          const __m512i C = _mm512_permutex2var_epi8(s[4], ih, s[5]);
          S5(o + 192 * h, _mm512_mask_permutexvar_epi16(_mm512_permutex2var_epi16(A, iab0, B), q6_cm[0], ic0, C));
          S5(o + 192 * h + 64, _mm512_mask_permutexvar_epi16(_mm512_permutex2var_epi16(A, iab1, B), q6_cm[1], ic1, C));
          S5(o + 192 * h + 128, _mm512_mask_permutexvar_epi16(_mm512_permutex2var_epi16(A, iab2, B), q6_cm[2], ic2, C));
        }
      }
    } break;
    case 8: {
      const __m512i i16l = L5(p16[0]), i16h = L5(p16[1]), i32l = L5(p32[0]), i32h = L5(p32[1]);
      for (; t + 64 <= L; t += 64) {
        __m512i s[8], a[8], b[8];
        for (int q = 0; q < 8; q++)
          s[q] = L5(src[q] + t);
        /* a[2p + h] = byte pairs (s_2p, s_2p+1) for t + 32h .. */
        for (int p = 0; p < 4; p++) {
          a[2 * p] = _mm512_permutex2var_epi8(s[2 * p], i8l, s[2 * p + 1]);
          a[2 * p + 1] = _mm512_permutex2var_epi8(s[2 * p], i8h, s[2 * p + 1]);
        }
        /* b[4r + 2h + g] = 4-byte groups (streams 4r..4r+3) for t + 32h + 16g .. */
        for (int r = 0; r < 2; r++)
          for (int h = 0; h < 2; h++) {
            b[4 * r + 2 * h] = _mm512_permutex2var_epi16(a[4 * r + h], i16l, a[4 * r + 2 + h]);
            b[4 * r + 2 * h + 1] = _mm512_permutex2var_epi16(a[4 * r + h], i16h, a[4 * r + 2 + h]);
          }
        uint8_t *o = f + 8 * t;
        for (int k = 0; k < 4; k++) {
          S5(o + 128 * k, _mm512_permutex2var_epi32(b[k], i32l, b[4 + k]));
          S5(o + 128 * k + 64, _mm512_permutex2var_epi32(b[k], i32h, b[4 + k]));
        }
      }
    } break;
    default:
      break;
  }
  nr_ldpc_tx_interleave_sse(qm, src, t, L, f);
}

/* 8 blocks of 64 positions -> 8 words per code block. Per block: bytes reversed within each qword, GFNI 8x8 bit
 * transpose of every qword (byte j = bits of code block j, LSB = first position), bytes gathered code-block-major
 * (qword j = code block j); then an 8x8 qword transpose across the 8 blocks. */
/* Qm = 8 and 4, more than 3 code blocks, 512 positions per block: lane registers r_k hold the bytes of positions
 * 8i + k (i = 0..63), loaded straight from the streams (Qm = 8: stream k; Qm = 4: even / odd bytes of stream k & 3),
 * then the 8x8 bit transpose across the registers leaves code block j's 512 bits, LSB first, in r_j: no interleaved
 * staging copy. */
static TGTX void extract_lanes_avx512(int qm,
                                      const uint8_t *const src[8],
                                      const uint8_t *const src2[8],
                                      uint8_t m2,
                                      int L,
                                      uint64_t *w)
{
  /* two rate-matched lengths: bits of the code blocks in m2 from the second stream set (vpternlogd select) */
  const __m512i mv = _mm512_set1_epi8((char)m2);
#define LDM(p, p2, o) (src2 ? _mm512_ternarylogic_epi32(mv, L5((p2) + (o)), L5((p) + (o)), 0xCA) : L5((p) + (o)))
  const int nb = (qm * L + 511) >> 9;
  const __m512i k1 = _mm512_set1_epi8(0x55), k2 = _mm512_set1_epi8(0x33), k4 = _mm512_set1_epi8(0x0f);
  const __m512i ev = _mm512_set_epi64(0x7e7c7a7876747270ULL,
                                      0x6e6c6a6866646260ULL,
                                      0x5e5c5a5856545250ULL,
                                      0x4e4c4a4846444240ULL,
                                      0x3e3c3a3836343230ULL,
                                      0x2e2c2a2826242220ULL,
                                      0x1e1c1a1816141210ULL,
                                      0x0e0c0a0806040200ULL);
  const __m512i od = _mm512_add_epi8(ev, _mm512_set1_epi8(1));
  for (int b = 0; b < nb; b++) {
    __m512i r[8];
    if (qm == 8) {
      for (int k = 0; k < 8; k++)
        r[k] = LDM(src[k], src2 ? src2[k] : NULL, 64 * b);
    } else {
      for (int k = 0; k < 4; k++) {
        const __m512i lo = LDM(src[k], src2 ? src2[k] : NULL, 128 * b), hi = LDM(src[k], src2 ? src2[k] : NULL, 128 * b + 64);
        r[k] = _mm512_permutex2var_epi8(lo, ev, hi);
        r[k + 4] = _mm512_permutex2var_epi8(lo, od, hi);
      }
    }
    XSW(r[0], r[1], 1, k1);
    XSW(r[2], r[3], 1, k1);
    XSW(r[4], r[5], 1, k1);
    XSW(r[6], r[7], 1, k1);
    XSW(r[0], r[2], 2, k2);
    XSW(r[1], r[3], 2, k2);
    XSW(r[4], r[6], 2, k2);
    XSW(r[5], r[7], 2, k2);
    XSW(r[0], r[4], 4, k4);
    XSW(r[1], r[5], 4, k4);
    XSW(r[2], r[6], 4, k4);
    XSW(r[3], r[7], 4, k4);
    for (int j = 0; j < 8; j++)
      S5(w + j * NR_LDPC_TX_WS + 8 * b, r[j]);
  }
#undef LDM
}

/* the lane-register path takes a second stream set */
static int merge_avx512(int qm, unsigned planes)
{
  return (qm == 8 || qm == 4) && __builtin_popcount(planes) > 3;
}

static TGTX void extract_avx512(int qm,
                                const uint8_t *const src[8],
                                const uint8_t *const src2[8],
                                uint8_t m2,
                                int L,
                                uint8_t *stage,
                                uint64_t *w,
                                unsigned planes)
{
  const int n = qm * L, nw = (n + 63) >> 6;
  if ((qm == 8 || qm == 4) && __builtin_popcount(planes) > 3) {
    extract_lanes_avx512(qm, src, src2, m2, L, w);
    return;
  }
  const uint8_t *f = src[0];
  if (qm > 1) {
    interleave_avx512(qm, src, L, stage);
    f = stage;
  }
  if (__builtin_popcount(planes) <= 3) {
    /* few code blocks: one byte test per code block and 64 positions */
    for (int k = 0; k < nw; k++) {
      const __m512i x = L5(f + 64 * k);
      for (int j = 0; j < 8; j++)
        if (planes >> j & 1)
          w[j * NR_LDPC_TX_WS + k] = _mm512_test_epi8_mask(x, _mm512_set1_epi8((char)(1 << j)));
    }
    return;
  }
  const __m512i rev = L5(rev8), gat = L5(gath);
  /* vgf2p8affineqb(x = selector, A = data): result byte i bit k = parity(A.byte[7-k] & x.byte[i]) */
  const __m512i sel = _mm512_set1_epi64(0x8040201008040201ULL);
  for (int k = 0; k < nw; k += 8) {
    __m512i z[8];
    for (int c = 0; c < 8; c++) {
      const __m512i x = _mm512_permutexvar_epi8(rev, L5(f + 64 * (k + c)));
      z[c] = _mm512_permutexvar_epi8(gat, _mm512_gf2p8affine_epi64_epi8(sel, x, 0));
    }
    __m512i t[8], u[8];
    for (int c = 0; c < 8; c += 2) {
      t[c] = _mm512_unpacklo_epi64(z[c], z[c + 1]);
      t[c + 1] = _mm512_unpackhi_epi64(z[c], z[c + 1]);
    }
    /* t[2p] lane L = (z_2p, z_2p+1) word 2L, t[2p+1] lane L = word 2L+1 */
    for (int o = 0; o < 2; o++) {
      u[4 * o + 0] = _mm512_shuffle_i64x2(t[o], t[2 + o], 0x88);
      u[4 * o + 1] = _mm512_shuffle_i64x2(t[o], t[2 + o], 0xDD);
      u[4 * o + 2] = _mm512_shuffle_i64x2(t[4 + o], t[6 + o], 0x88);
      u[4 * o + 3] = _mm512_shuffle_i64x2(t[4 + o], t[6 + o], 0xDD);
      S5(w + (0 + o) * NR_LDPC_TX_WS + k, _mm512_shuffle_i64x2(u[4 * o], u[4 * o + 2], 0x88));
      S5(w + (4 + o) * NR_LDPC_TX_WS + k, _mm512_shuffle_i64x2(u[4 * o], u[4 * o + 2], 0xDD));
      S5(w + (2 + o) * NR_LDPC_TX_WS + k, _mm512_shuffle_i64x2(u[4 * o + 1], u[4 * o + 3], 0x88));
      S5(w + (6 + o) * NR_LDPC_TX_WS + k, _mm512_shuffle_i64x2(u[4 * o + 1], u[4 * o + 3], 0xDD));
    }
  }
}

static TGT void write_avx512(uint64_t *dst, const uint64_t *w, int n, int sh)
{
  const __m128i cl = _mm_cvtsi32_si128(sh), cr = _mm_cvtsi32_si128(64 - sh);
  int k = 0;
  for (; k + 8 <= n; k += 8)
    S5(dst + k, _mm512_or_si512(_mm512_sll_epi64(L5(w + k), cl), _mm512_srl_epi64(L5(w + k - 1), cr)));
  if (k < n) {
    const __mmask8 m = (1u << (n - k)) - 1;
    const __m512i v = _mm512_or_si512(_mm512_sll_epi64(_mm512_maskz_loadu_epi64(m, w + k), cl),
                                      _mm512_srl_epi64(_mm512_maskz_loadu_epi64(m, w + k - 1), cr));
    _mm512_mask_storeu_epi64(dst + k, m, v);
  }
}
#undef L5
#undef S5

static nr_ldpc_tx_kern_t k_avx512 = {.name = "avx512",
                                     .rows = rows_avx512,
                                     .transpose = transpose_avx512,
                                     .tb = 64,
                                     .tcols = tcols_avx512,
                                     .tcols_zmin = 8,
                                     .extract = extract_avx512,
                                     .merge = merge_avx512,
                                     .write = write_avx512};
const nr_ldpc_tx_kern_t *nr_ldpc_tx_kern_avx512(void)
{
  __builtin_cpu_init();
  if (!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("avx512bw") || !__builtin_cpu_supports("avx512vl"))
    return 0;
  /* output stage without VBMI/GFNI (Skylake-SP/Cascade Lake): the AVX2 kernels */
  if (!__builtin_cpu_supports("avx512vbmi") || !__builtin_cpu_supports("gfni")) {
    k_avx512.extract = nr_ldpc_tx_extract_avx2;
    k_avx512.merge = NULL;
  }
  return &k_avx512;
}
#else
const nr_ldpc_tx_kern_t *nr_ldpc_tx_kern_avx512(void)
{
  return 0;
}
#endif
