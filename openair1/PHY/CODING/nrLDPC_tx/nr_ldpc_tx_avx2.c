/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* x86-64 AVX2 kernels of the packed LDPC transmit chain (function-level target attribute, selected at run time). */
#include "nr_ldpc_tx_internal.h"
#if defined(__x86_64__)
#include <immintrin.h>

#define TGT __attribute__((target("avx2")))
#define AI static inline __attribute__((always_inline)) TGT

/* N contiguous 32-byte chunks at offset o (+ optional overlapping tail chunk at ot) of one row; the first edge
 * initialises the accumulators (explicit variables so that they stay in registers) */
#define LD(p, k) _mm256_loadu_si256((const __m256i *)((p) + 32 * (k)))
/* hide the pointer arithmetic from the compiler so that loads use [reg + disp] (an indexed address would split
 * the micro-fused load+xor into two uops on Intel cores) */
#define OPAQUE(p) __asm__("" : "+r"(p))
#define REP12(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7) M(8) M(9) M(10) M(11)
AI void
grp(const uint8_t *base, const uint16_t *off, int ne, int o, const int N, const int tail, int ot, uint8_t *d, int z, int dup)
{
#define DECL(k) __m256i a##k = _mm256_setzero_si256();
  REP12(DECL)
#undef DECL
  __m256i t = _mm256_setzero_si256();
  if (ne > 0) {
    const uint8_t *p = base + off[0] + o, *pt = base + off[0] + ot;
    OPAQUE(p);
#define I1(k) \
  if (N > k)  \
    a##k = LD(p, k);
    REP12(I1)
#undef I1
    if (tail)
      t = LD(pt, 0);
    for (int e = 1; e < ne; e++) {
      p = base + off[e] + o;
      pt = base + off[e] + ot;
      OPAQUE(p);
#define X1(k) \
  if (N > k)  \
    a##k = _mm256_xor_si256(a##k, LD(p, k));
      REP12(X1)
#undef X1
      if (tail)
        t = _mm256_xor_si256(t, LD(pt, 0));
    }
  }
  for (int k = 0; k <= dup; k++) {
    uint8_t *q = d + k * z + o;
#define S1(k) \
  if (N > k)  \
    _mm256_storeu_si256((__m256i *)(q + 32 * (k)), a##k);
    REP12(S1)
#undef S1
    if (tail)
      _mm256_storeu_si256((__m256i *)(d + k * z + ot), t);
  }
}
#undef LD

/* 16 <= z < 32: one or two overlapping 16-byte chunks */
static TGT void
rows_sse(const uint8_t *base, const uint16_t *off, const uint8_t *cnt, int nrows, int z, uint8_t *dst, int dstride, int dup)
{
  const int o1 = z - 16;
  for (int r = 0; r < nrows; r++) {
    __m128i a = _mm_setzero_si128(), b = _mm_setzero_si128();
    for (int e = 0; e < cnt[r]; e++) {
      a = _mm_xor_si128(a, _mm_loadu_si128((const __m128i *)(base + off[e])));
      b = _mm_xor_si128(b, _mm_loadu_si128((const __m128i *)(base + off[e] + o1)));
    }
    for (int k = 0; k <= dup; k++) {
      _mm_storeu_si128((__m128i *)(dst + k * z), a);
      _mm_storeu_si128((__m128i *)(dst + k * z + o1), b);
    }
    off += cnt[r];
    dst += dstride;
  }
}

#define CASE(n)                                     \
  case n:                                           \
    if (tl)                                         \
      grp(base, off, ne, o, n, 1, ot, dst, z, dup); \
    else                                            \
      grp(base, off, ne, o, n, 0, ot, dst, z, dup); \
    break;

static TGT void
rows_avx2(const uint8_t *base, const uint16_t *off, const uint8_t *cnt, int nrows, int z, uint8_t *dst, int dstride, int dup)
{
  if (z < 32) {
    if (z >= 16)
      rows_sse(base, off, cnt, nrows, z, dst, dstride, dup);
    else
      nr_ldpc_tx_rows_scalar(base, off, cnt, nrows, z, dst, dstride, dup);
    return;
  }
  const int nfull = z >> 5, ot = z - 32, tl = (z & 31) != 0;
  for (int r = 0; r < nrows; r++) {
    const int ne = cnt[r];
    int o = 0, n = nfull;
    while (n > 12) {
      grp(base, off, ne, o, 12, 0, ot, dst, z, dup);
      o += 384;
      n -= 12;
    }
    switch (n) {
      CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) CASE(9) CASE(10) CASE(11) CASE(12)
    }
    off += ne;
    dst += dstride;
  }
}

/* delta swap between registers: bits t (t & s == 0) of y <-> bits t+s of x ... see kern_neon.c */
#define DSW(x, y, s, m)                                                             \
  do {                                                                              \
    __m256i t_ = _mm256_and_si256(_mm256_xor_si256(_mm256_srli_epi16(x, s), y), m); \
    y = _mm256_xor_si256(y, t_);                                                    \
    x = _mm256_xor_si256(x, _mm256_slli_epi16(t_, s));                              \
  } while (0)

/* 8 registers (32 bytes of each stream) -> Q[q] = output bytes [32q, 32q+32) */
AI void tr256(__m256i r[8], __m256i Q[8])
{
  const __m256i m1 = _mm256_set1_epi8(0x55), m2 = _mm256_set1_epi8(0x33), m4 = _mm256_set1_epi8(0x0f);
  DSW(r[0], r[1], 1, m1);
  DSW(r[2], r[3], 1, m1);
  DSW(r[4], r[5], 1, m1);
  DSW(r[6], r[7], 1, m1);
  DSW(r[0], r[2], 2, m2);
  DSW(r[1], r[3], 2, m2);
  DSW(r[4], r[6], 2, m2);
  DSW(r[5], r[7], 2, m2);
  DSW(r[0], r[4], 4, m4);
  DSW(r[1], r[5], 4, m4);
  DSW(r[2], r[6], 4, m4);
  DSW(r[3], r[7], 4, m4);
  /* out[8b+u] = r[7-u][b]; per 128-bit lane: 8-way byte interleave by 3 unpack stages */
  __m256i A0 = _mm256_unpacklo_epi8(r[7], r[6]), A1 = _mm256_unpackhi_epi8(r[7], r[6]);
  __m256i B0 = _mm256_unpacklo_epi8(r[5], r[4]), B1 = _mm256_unpackhi_epi8(r[5], r[4]);
  __m256i C0 = _mm256_unpacklo_epi8(r[3], r[2]), C1 = _mm256_unpackhi_epi8(r[3], r[2]);
  __m256i D0 = _mm256_unpacklo_epi8(r[1], r[0]), D1 = _mm256_unpackhi_epi8(r[1], r[0]);
  __m256i E0 = _mm256_unpacklo_epi16(A0, B0), E1 = _mm256_unpackhi_epi16(A0, B0);
  __m256i E2 = _mm256_unpacklo_epi16(A1, B1), E3 = _mm256_unpackhi_epi16(A1, B1);
  __m256i F0 = _mm256_unpacklo_epi16(C0, D0), F1 = _mm256_unpackhi_epi16(C0, D0);
  __m256i F2 = _mm256_unpacklo_epi16(C1, D1), F3 = _mm256_unpackhi_epi16(C1, D1);
  __m256i G[8];
  G[0] = _mm256_unpacklo_epi32(E0, F0);
  G[1] = _mm256_unpackhi_epi32(E0, F0);
  G[2] = _mm256_unpacklo_epi32(E1, F1);
  G[3] = _mm256_unpackhi_epi32(E1, F1);
  G[4] = _mm256_unpacklo_epi32(E2, F2);
  G[5] = _mm256_unpackhi_epi32(E2, F2);
  G[6] = _mm256_unpacklo_epi32(E3, F3);
  G[7] = _mm256_unpackhi_epi32(E3, F3);
  /* G[k] lane L holds b = 16L + 2k, 2k+1 -> out offset 128L + 16k */
  for (int k = 0; k < 8; k += 2) {
    Q[k / 2] = _mm256_permute2x128_si256(G[k], G[k + 1], 0x20);
    Q[4 + k / 2] = _mm256_permute2x128_si256(G[k], G[k + 1], 0x31);
  }
}

/* SSE version: 16 (or, with half, 8) bytes of each stream -> Q[k] = output bytes [16k, 16k+16) */
#define DSW128(x, y, s, m)                                                 \
  do {                                                                     \
    __m128i t_ = _mm_and_si128(_mm_xor_si128(_mm_srli_epi16(x, s), y), m); \
    y = _mm_xor_si128(y, t_);                                              \
    x = _mm_xor_si128(x, _mm_slli_epi16(t_, s));                           \
  } while (0)
AI void tr128(__m128i r[8], __m128i G[8])
{
  const __m128i m1 = _mm_set1_epi8(0x55), m2 = _mm_set1_epi8(0x33), m4 = _mm_set1_epi8(0x0f);
  DSW128(r[0], r[1], 1, m1);
  DSW128(r[2], r[3], 1, m1);
  DSW128(r[4], r[5], 1, m1);
  DSW128(r[6], r[7], 1, m1);
  DSW128(r[0], r[2], 2, m2);
  DSW128(r[1], r[3], 2, m2);
  DSW128(r[4], r[6], 2, m2);
  DSW128(r[5], r[7], 2, m2);
  DSW128(r[0], r[4], 4, m4);
  DSW128(r[1], r[5], 4, m4);
  DSW128(r[2], r[6], 4, m4);
  DSW128(r[3], r[7], 4, m4);
  __m128i A0 = _mm_unpacklo_epi8(r[7], r[6]), A1 = _mm_unpackhi_epi8(r[7], r[6]);
  __m128i B0 = _mm_unpacklo_epi8(r[5], r[4]), B1 = _mm_unpackhi_epi8(r[5], r[4]);
  __m128i C0 = _mm_unpacklo_epi8(r[3], r[2]), C1 = _mm_unpackhi_epi8(r[3], r[2]);
  __m128i D0 = _mm_unpacklo_epi8(r[1], r[0]), D1 = _mm_unpackhi_epi8(r[1], r[0]);
  __m128i E0 = _mm_unpacklo_epi16(A0, B0), E1 = _mm_unpackhi_epi16(A0, B0);
  __m128i E2 = _mm_unpacklo_epi16(A1, B1), E3 = _mm_unpackhi_epi16(A1, B1);
  __m128i F0 = _mm_unpacklo_epi16(C0, D0), F1 = _mm_unpackhi_epi16(C0, D0);
  __m128i F2 = _mm_unpacklo_epi16(C1, D1), F3 = _mm_unpackhi_epi16(C1, D1);
  G[0] = _mm_unpacklo_epi32(E0, F0);
  G[1] = _mm_unpackhi_epi32(E0, F0);
  G[2] = _mm_unpacklo_epi32(E1, F1);
  G[3] = _mm_unpackhi_epi32(E1, F1);
  G[4] = _mm_unpacklo_epi32(E2, F2);
  G[5] = _mm_unpackhi_epi32(E2, F2);
  G[6] = _mm_unpacklo_epi32(E3, F3);
  G[7] = _mm_unpackhi_epi32(E3, F3);
}

static TGT void transpose_avx2(const uint8_t *const in[8], int nblk, uint8_t *out)
{
  for (int i = 0; i < nblk; i++) {
    __m256i r[8], Q[8];
    for (int j = 0; j < 8; j++)
      r[j] = _mm256_loadu_si256((const __m256i *)(in[j] + 32 * i));
    tr256(r, Q);
    for (int q = 0; q < 8; q++)
      _mm256_storeu_si256((__m256i *)(out + 256 * i + 32 * q), Q[q]);
  }
}

/* column transposer: 32-byte blocks (last one overlapping), SSE 16-byte blocks for 16 <= z/8 < 32 and 8-byte
 * blocks (low half of the SSE transposer) for 8 <= z/8 < 16 */
static TGT void tcols_avx2(const uint8_t *const in[8], int z, int ncols, uint8_t *work, int S, uint8_t *out, int cw)
{
  const int nb = z >> 3;
  for (int c = 0; c < ncols; c++) {
    uint8_t *w0 = work + c * S, *w1 = w0 + z, *o = out + (c - 2) * z;
    const int wo = c >= 2 && c < cw;
    if (nb >= 32) {
      for (int b = 0; b < nb; b += 32) {
        const int bb = (b + 32 <= nb ? b : nb - 32) + c * nb;
        __m256i r[8], Q[8];
        for (int j = 0; j < 8; j++)
          r[j] = _mm256_loadu_si256((const __m256i *)(in[j] + bb));
        tr256(r, Q);
        const int ob = 8 * (bb - c * nb);
        for (int q = 0; q < 8; q++) {
          _mm256_storeu_si256((__m256i *)(w0 + ob + 32 * q), Q[q]);
          _mm256_storeu_si256((__m256i *)(w1 + ob + 32 * q), Q[q]);
          if (wo)
            _mm256_storeu_si256((__m256i *)(o + ob + 32 * q), Q[q]);
        }
      }
    } else {
      const int half = nb < 16, step = half ? 8 : 16, nq = half ? 4 : 8;
      for (int b = 0; b < nb; b += step) {
        const int bb = (b + step <= nb ? b : nb - step) + c * nb;
        __m128i r[8], G[8];
        for (int j = 0; j < 8; j++)
          r[j] = half ? _mm_loadl_epi64((const __m128i *)(in[j] + bb)) : _mm_loadu_si128((const __m128i *)(in[j] + bb));
        tr128(r, G);
        const int ob = 8 * (bb - c * nb);
        for (int q = 0; q < nq; q++) {
          _mm_storeu_si128((__m128i *)(w0 + ob + 16 * q), G[q]);
          _mm_storeu_si128((__m128i *)(w1 + ob + 16 * q), G[q]);
          if (wo)
            _mm_storeu_si128((__m128i *)(o + ob + 16 * q), G[q]);
        }
      }
    }
  }
}

/* ------------------------------------------------------------------------------------------ output stage */

#define LDU(p) _mm_loadu_si128((const __m128i *)(p))
#define STU(p, v) _mm_storeu_si128((__m128i *)(p), v)

/* f[qm t + q] = src[q][t] (TS 38.212 5.4.2.2) for t in [t, L): SSE unpacks, 16 t per step (pshufb 3-way 16-bit
 * interleave of the byte pairs for Qm = 6), scalar tail */
TGT void nr_ldpc_tx_interleave_sse(int qm, const uint8_t *const src[8], int t, int L, uint8_t *f)
{
  switch (qm) {
    case 2:
      for (; t + 16 <= L; t += 16) {
        const __m128i a = LDU(src[0] + t), b = LDU(src[1] + t);
        STU(f + 2 * t, _mm_unpacklo_epi8(a, b));
        STU(f + 2 * t + 16, _mm_unpackhi_epi8(a, b));
      }
      break;
    case 4:
      for (; t + 16 <= L; t += 16) {
        const __m128i a = LDU(src[0] + t), b = LDU(src[1] + t), c = LDU(src[2] + t), d = LDU(src[3] + t);
        const __m128i ab0 = _mm_unpacklo_epi8(a, b), ab1 = _mm_unpackhi_epi8(a, b);
        const __m128i cd0 = _mm_unpacklo_epi8(c, d), cd1 = _mm_unpackhi_epi8(c, d);
        uint8_t *o = f + 4 * t;
        STU(o, _mm_unpacklo_epi16(ab0, cd0));
        STU(o + 16, _mm_unpackhi_epi16(ab0, cd0));
        STU(o + 32, _mm_unpacklo_epi16(ab1, cd1));
        STU(o + 48, _mm_unpackhi_epi16(ab1, cd1));
      }
      break;
    case 6: {
      const __m128i m00 = _mm_setr_epi8(0, 1, -128, -128, -128, -128, 2, 3, -128, -128, -128, -128, 4, 5, -128, -128);
      const __m128i m01 = _mm_setr_epi8(-128, -128, 0, 1, -128, -128, -128, -128, 2, 3, -128, -128, -128, -128, 4, 5);
      const __m128i m02 = _mm_setr_epi8(-128, -128, -128, -128, 0, 1, -128, -128, -128, -128, 2, 3, -128, -128, -128, -128);
      const __m128i m10 = _mm_setr_epi8(-128, -128, 6, 7, -128, -128, -128, -128, 8, 9, -128, -128, -128, -128, 10, 11);
      const __m128i m11 = _mm_setr_epi8(-128, -128, -128, -128, 6, 7, -128, -128, -128, -128, 8, 9, -128, -128, -128, -128);
      const __m128i m12 = _mm_setr_epi8(4, 5, -128, -128, -128, -128, 6, 7, -128, -128, -128, -128, 8, 9, -128, -128);
      const __m128i m20 = _mm_setr_epi8(-128, -128, -128, -128, 12, 13, -128, -128, -128, -128, 14, 15, -128, -128, -128, -128);
      const __m128i m21 = _mm_setr_epi8(10, 11, -128, -128, -128, -128, 12, 13, -128, -128, -128, -128, 14, 15, -128, -128);
      const __m128i m22 = _mm_setr_epi8(-128, -128, 10, 11, -128, -128, -128, -128, 12, 13, -128, -128, -128, -128, 14, 15);
#define S3(o, A, B, C)                                                                                                     \
  do {                                                                                                                     \
    STU(o, _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(A, m00), _mm_shuffle_epi8(B, m01)), _mm_shuffle_epi8(C, m02)));      \
    STU(o + 16, _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(A, m10), _mm_shuffle_epi8(B, m11)), _mm_shuffle_epi8(C, m12))); \
    STU(o + 32, _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(A, m20), _mm_shuffle_epi8(B, m21)), _mm_shuffle_epi8(C, m22))); \
  } while (0)
      for (; t + 16 <= L; t += 16) {
        const __m128i s0 = LDU(src[0] + t), s1 = LDU(src[1] + t), s2 = LDU(src[2] + t);
        const __m128i s3 = LDU(src[3] + t), s4 = LDU(src[4] + t), s5 = LDU(src[5] + t);
        uint8_t *o = f + 6 * t;
        S3(o, _mm_unpacklo_epi8(s0, s1), _mm_unpacklo_epi8(s2, s3), _mm_unpacklo_epi8(s4, s5));
        S3(o + 48, _mm_unpackhi_epi8(s0, s1), _mm_unpackhi_epi8(s2, s3), _mm_unpackhi_epi8(s4, s5));
      }
#undef S3
    } break;
    case 8:
      for (; t + 16 <= L; t += 16) {
        __m128i s[8];
        for (int q = 0; q < 8; q++)
          s[q] = LDU(src[q] + t);
        const __m128i a0 = _mm_unpacklo_epi8(s[0], s[1]), a1 = _mm_unpackhi_epi8(s[0], s[1]);
        const __m128i b0 = _mm_unpacklo_epi8(s[2], s[3]), b1 = _mm_unpackhi_epi8(s[2], s[3]);
        const __m128i c0 = _mm_unpacklo_epi8(s[4], s[5]), c1 = _mm_unpackhi_epi8(s[4], s[5]);
        const __m128i d0 = _mm_unpacklo_epi8(s[6], s[7]), d1 = _mm_unpackhi_epi8(s[6], s[7]);
        const __m128i e0 = _mm_unpacklo_epi16(a0, b0), e1 = _mm_unpackhi_epi16(a0, b0);
        const __m128i e2 = _mm_unpacklo_epi16(a1, b1), e3 = _mm_unpackhi_epi16(a1, b1);
        const __m128i g0 = _mm_unpacklo_epi16(c0, d0), g1 = _mm_unpackhi_epi16(c0, d0);
        const __m128i g2 = _mm_unpacklo_epi16(c1, d1), g3 = _mm_unpackhi_epi16(c1, d1);
        uint8_t *o = f + 8 * t;
        STU(o, _mm_unpacklo_epi32(e0, g0));
        STU(o + 16, _mm_unpackhi_epi32(e0, g0));
        STU(o + 32, _mm_unpacklo_epi32(e1, g1));
        STU(o + 48, _mm_unpackhi_epi32(e1, g1));
        STU(o + 64, _mm_unpacklo_epi32(e2, g2));
        STU(o + 80, _mm_unpackhi_epi32(e2, g2));
        STU(o + 96, _mm_unpacklo_epi32(e3, g3));
        STU(o + 112, _mm_unpackhi_epi32(e3, g3));
      }
      break;
    default:
      break;
  }
  for (; t < L; t++)
    for (int q = 0; q < qm; q++)
      f[qm * t + q] = src[q][t];
}
#undef LDU
#undef STU

/* bit j of 32 bytes -> 32 bits */
#define MM(x, j) ((uint32_t)_mm256_movemask_epi8((j) == 7 ? (x) : _mm256_slli_epi16(x, 7 - (j))))
TGT void nr_ldpc_tx_extract_avx2(int qm, const uint8_t *const src[8], int L, uint8_t *stage, uint64_t *w, unsigned planes)
{
  const int n = qm * L, nw = (n + 63) >> 6;
  const uint8_t *f = src[0];
  if (qm > 1) {
    nr_ldpc_tx_interleave_sse(qm, src, 0, L, stage);
    f = stage;
  }
  for (int k = 0; k < nw; k++) {
    const __m256i x0 = _mm256_loadu_si256((const __m256i *)(f + 64 * k));
    const __m256i x1 = _mm256_loadu_si256((const __m256i *)(f + 64 * k + 32));
#define E1(j)            \
  if (planes >> (j) & 1) \
    w[(j) * NR_LDPC_TX_WS + k] = MM(x0, j) | (uint64_t)MM(x1, j) << 32;
    E1(0) E1(1) E1(2) E1(3) E1(4) E1(5) E1(6) E1(7)
#undef E1
  }
}
#undef MM

TGT void nr_ldpc_tx_write_avx2(uint64_t *dst, const uint64_t *w, int n, int sh)
{
  const __m128i cl = _mm_cvtsi32_si128(sh), cr = _mm_cvtsi32_si128(64 - sh);
  int k = 0;
  for (; k + 4 <= n; k += 4) {
    const __m256i a = _mm256_loadu_si256((const __m256i *)(w + k)), b = _mm256_loadu_si256((const __m256i *)(w + k - 1));
    _mm256_storeu_si256((__m256i *)(dst + k), _mm256_or_si256(_mm256_sll_epi64(a, cl), _mm256_srl_epi64(b, cr)));
  }
  for (; k < n; k++)
    nr_ldpc_tx_st64(dst + k, (w[k] << sh) | (sh ? w[k - 1] >> (64 - sh) : 0));
}

static const nr_ldpc_tx_kern_t k_avx2 = {.name = "avx2",
                                         .rows = rows_avx2,
                                         .transpose = transpose_avx2,
                                         .tb = 32,
                                         .tcols = tcols_avx2,
                                         .tcols_zmin = 64,
                                         .extract = nr_ldpc_tx_extract_avx2,
                                         .write = nr_ldpc_tx_write_avx2};
const nr_ldpc_tx_kern_t *nr_ldpc_tx_kern_avx2(void)
{
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") ? &k_avx2 : 0;
}
#else
const nr_ldpc_tx_kern_t *nr_ldpc_tx_kern_avx2(void)
{
  return 0;
}
#endif
