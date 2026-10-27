/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* aarch64 Advanced SIMD (NEON) kernels of the packed LDPC transmit chain, baseline armv8-a (row kernel also with
 * FEAT_SHA3 EOR3, selected at run time). */
#include "nr_ldpc_tx_internal.h"
#if defined(__aarch64__)
#include <arm_neon.h>
#include <sys/auxv.h>
#ifndef HWCAP_SHA3
#define HWCAP_SHA3 (1 << 17)
#endif

#define AI static inline __attribute__((always_inline))

/* row kernels: plain NEON (EOR) and NEON + FEAT_SHA3 (EOR3, 3-way XOR), same body */
#define FN(x) x##_neon
#define TGT
#define GMAX 24
#include "nr_ldpc_tx_neon_rows.h"
#undef FN
#undef TGT
#undef GMAX
#define NR_LDPC_TX_EOR3 1
#define FN(x) x##_neon3
#define TGT __attribute__((target("+sha3")))
#define GMAX 24
#include "nr_ldpc_tx_neon_rows.h"

/* exchange of bit groups between two registers (one delta-swap stage of the 8x8 bit transpose):
 * x' = m ? x : y<<s,  y' = m ? x>>s : y */
#define XSW(x, y, s, m)                    \
  do {                                     \
    uint8x16_t x_ = x;                     \
    x = vbslq_u8(m, x_, vshlq_n_u8(y, s)); \
    y = vbslq_u8(m, vshrq_n_u8(x_, s), y); \
  } while (0)
/* s = 4 maps exactly onto SLI/SRI */
#define XSW4(x, y)            \
  do {                        \
    uint8x16_t x_ = x;        \
    x = vsliq_n_u8(x_, y, 4); \
    y = vsriq_n_u8(y, x_, 4); \
  } while (0)

/* 16 bytes of each of the 8 streams at byte offset b -> 128 bit-sliced bytes as two st4-ready halves */
static inline __attribute__((always_inline)) void tr16(const uint8_t *const p[8], size_t b, uint16x8x4_t *lo, uint16x8x4_t *hi)
{
  const uint8x16_t m1 = vdupq_n_u8(0x55), m2 = vdupq_n_u8(0x33);
  /* r_j = 16 bytes of stream j; after the transpose r_t byte b bit j = in[j][b] bit t */
  uint8x16_t r0 = vld1q_u8(p[0] + b), r1 = vld1q_u8(p[1] + b), r2 = vld1q_u8(p[2] + b), r3 = vld1q_u8(p[3] + b),
             r4 = vld1q_u8(p[4] + b), r5 = vld1q_u8(p[5] + b), r6 = vld1q_u8(p[6] + b), r7 = vld1q_u8(p[7] + b);
  XSW(r0, r1, 1, m1);
  XSW(r2, r3, 1, m1);
  XSW(r4, r5, 1, m1);
  XSW(r6, r7, 1, m1);
  XSW(r0, r2, 2, m2);
  XSW(r1, r3, 2, m2);
  XSW(r4, r6, 2, m2);
  XSW(r5, r7, 2, m2);
  XSW4(r0, r4);
  XSW4(r1, r5);
  XSW4(r2, r6);
  XSW4(r3, r7);
  /* out[8b+u] = r_(7-u)[b]: zip byte pairs (r7,r6) (r5,r4) (r3,r2) (r1,r0), then 4-way 16-bit interleave (st4) */
  lo->val[0] = vreinterpretq_u16_u8(vzip1q_u8(r7, r6));
  hi->val[0] = vreinterpretq_u16_u8(vzip2q_u8(r7, r6));
  lo->val[1] = vreinterpretq_u16_u8(vzip1q_u8(r5, r4));
  hi->val[1] = vreinterpretq_u16_u8(vzip2q_u8(r5, r4));
  lo->val[2] = vreinterpretq_u16_u8(vzip1q_u8(r3, r2));
  hi->val[2] = vreinterpretq_u16_u8(vzip2q_u8(r3, r2));
  lo->val[3] = vreinterpretq_u16_u8(vzip1q_u8(r1, r0));
  hi->val[3] = vreinterpretq_u16_u8(vzip2q_u8(r1, r0));
}

/* 8-byte version (64-bit registers), for 8 <= z/8 < 16 */
#define XSW8(x, y, s, m)                 \
  do {                                   \
    uint8x8_t x_ = x;                    \
    x = vbsl_u8(m, x_, vshl_n_u8(y, s)); \
    y = vbsl_u8(m, vshr_n_u8(x_, s), y); \
  } while (0)
#define XSW48(x, y)          \
  do {                       \
    uint8x8_t x_ = x;        \
    x = vsli_n_u8(x_, y, 4); \
    y = vsri_n_u8(y, x_, 4); \
  } while (0)
static inline __attribute__((always_inline)) void tr8(const uint8_t *const p[8], size_t b, uint16x4x4_t *lo, uint16x4x4_t *hi)
{
  const uint8x8_t m1 = vdup_n_u8(0x55), m2 = vdup_n_u8(0x33);
  uint8x8_t r0 = vld1_u8(p[0] + b), r1 = vld1_u8(p[1] + b), r2 = vld1_u8(p[2] + b), r3 = vld1_u8(p[3] + b), r4 = vld1_u8(p[4] + b),
            r5 = vld1_u8(p[5] + b), r6 = vld1_u8(p[6] + b), r7 = vld1_u8(p[7] + b);
  XSW8(r0, r1, 1, m1);
  XSW8(r2, r3, 1, m1);
  XSW8(r4, r5, 1, m1);
  XSW8(r6, r7, 1, m1);
  XSW8(r0, r2, 2, m2);
  XSW8(r1, r3, 2, m2);
  XSW8(r4, r6, 2, m2);
  XSW8(r5, r7, 2, m2);
  XSW48(r0, r4);
  XSW48(r1, r5);
  XSW48(r2, r6);
  XSW48(r3, r7);
  lo->val[0] = vreinterpret_u16_u8(vzip1_u8(r7, r6));
  hi->val[0] = vreinterpret_u16_u8(vzip2_u8(r7, r6));
  lo->val[1] = vreinterpret_u16_u8(vzip1_u8(r5, r4));
  hi->val[1] = vreinterpret_u16_u8(vzip2_u8(r5, r4));
  lo->val[2] = vreinterpret_u16_u8(vzip1_u8(r3, r2));
  hi->val[2] = vreinterpret_u16_u8(vzip2_u8(r3, r2));
  lo->val[3] = vreinterpret_u16_u8(vzip1_u8(r1, r0));
  hi->val[3] = vreinterpret_u16_u8(vzip2_u8(r1, r0));
}

static void transpose_neon(const uint8_t *const in[8], int nblk, uint8_t *out)
{
  const uint8_t *const p[8] = {in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7]};
  for (int i = 0; i < nblk; i++, out += 128) {
    uint16x8x4_t lo, hi;
    tr16(p, 16 * (size_t)i, &lo, &hi);
    vst4q_u16((uint16_t *)out, lo);
    vst4q_u16((uint16_t *)(out + 64), hi);
  }
}

/* column transposer: 16-byte blocks (the last one overlapping when z/8 % 16 != 0), 8-byte blocks for z < 128.
 * Each block is stored once with st4 and then copied with ld1/st1 {4 regs} (storing the st4 operands three times
 * makes GCC 13 shuffle registers) */
static void tcols_neon(const uint8_t *const in[8], int z, int ncols, uint8_t *work, int S, uint8_t *out, int cw)
{
  const int nb = z >> 3;
  for (int c = 0; c < ncols; c++) {
    const uint8_t *const p[8] = {in[0] + c * nb,
                                 in[1] + c * nb,
                                 in[2] + c * nb,
                                 in[3] + c * nb,
                                 in[4] + c * nb,
                                 in[5] + c * nb,
                                 in[6] + c * nb,
                                 in[7] + c * nb};
    uint8_t *w0 = work + c * S, *w1 = w0 + z, *o = out + (c - 2) * z;
    const int wo = c >= 2 && c < cw;
    if (nb >= 16) {
      for (int b = 0; b < nb; b += 16) {
        const int bb = b + 16 <= nb ? b : nb - 16;
        uint16x8x4_t lo, hi;
        tr16(p, bb, &lo, &hi);
        uint8_t *q = w0 + 8 * bb;
        vst4q_u16((uint16_t *)q, lo);
        vst4q_u16((uint16_t *)(q + 64), hi);
        uint8x16x4_t x = vld1q_u8_x4(q), y = vld1q_u8_x4(q + 64);
        vst1q_u8_x4(w1 + 8 * bb, x);
        vst1q_u8_x4(w1 + 8 * bb + 64, y);
        if (wo) {
          vst1q_u8_x4(o + 8 * bb, x);
          vst1q_u8_x4(o + 8 * bb + 64, y);
        }
      }
    } else {
      for (int b = 0; b < nb; b += 8) {
        const int bb = b + 8 <= nb ? b : nb - 8;
        uint16x4x4_t lo, hi;
        tr8(p, bb, &lo, &hi);
        uint8_t *q = w0 + 8 * bb;
        vst4_u16((uint16_t *)q, lo);
        vst4_u16((uint16_t *)(q + 32), hi);
        uint8x16x4_t x = vld1q_u8_x4(q);
        vst1q_u8_x4(w1 + 8 * bb, x);
        if (wo)
          vst1q_u8_x4(o + 8 * bb, x);
      }
    }
  }
}

/* Column transposer: writes the work columns and the output directly instead of transposing into a linear buffer +
 * doubling + one large memcpy. Measured per Z under qemu on the cortex-a710 model (no FEAT_MOPS, like the A720) it
 * saves up to 4 % of the instructions when the 16-byte blocks tile a column with little overlap (z/8 % 16 == 0 or
 * >= 10: Z = 128, 208, 224, 240, 256, 352, 384) and costs up to 15 % otherwise (8-byte blocks for Z < 128, large
 * overlaps), so it is used only there. -DNR_LDPC_TX_NEON_TCOLS_ZMIN=1073741824 disables it. */
#ifndef NR_LDPC_TX_NEON_TCOLS_ZMIN
#define NR_LDPC_TX_NEON_TCOLS_ZMIN 128
#endif
static int tcols_use_neon(int z)
{
  const int r = (z >> 3) & 15;
  return r == 0 || r >= 10;
}

/* ------------------------------------------------------------------------------------------ output stage */

/* Extraction, 128 positions per block: lane registers R_r (r = 0..7) hold the bytes of positions 8k + r
 * (k = 0..15) of the interleaved chunk. They are loaded straight from the streams with structure loads (Qm = 8: one
 * stream each; Qm = 4: LD2, even/odd bytes; Qm = 2: LD4; Qm = 1 and the interleaved Qm = 6 chunk: LD4 + UZP).
 * The 8x8 bit transpose across the 8 registers (bit j of R_r -> bit r of R_j) then leaves code block j's 128
 * bits, LSB first, in R_j. */
#define TR8(r0, r1, r2, r3, r4, r5, r6, r7)                          \
  do {                                                               \
    const uint8x16_t m1_ = vdupq_n_u8(0x55), m2_ = vdupq_n_u8(0x33); \
    XSW(r0, r1, 1, m1_);                                             \
    XSW(r2, r3, 1, m1_);                                             \
    XSW(r4, r5, 1, m1_);                                             \
    XSW(r6, r7, 1, m1_);                                             \
    XSW(r0, r2, 2, m2_);                                             \
    XSW(r1, r3, 2, m2_);                                             \
    XSW(r4, r6, 2, m2_);                                             \
    XSW(r5, r7, 2, m2_);                                             \
    XSW4(r0, r4);                                                    \
    XSW4(r1, r5);                                                    \
    XSW4(r2, r6);                                                    \
    XSW4(r3, r7);                                                    \
  } while (0)

#define STW(j, r) vst1q_u8((uint8_t *)(w + (j) * NR_LDPC_TX_WS + 2 * b), r)
#define STALL(r0, r1, r2, r3, r4, r5, r6, r7) \
  do {                                        \
    STW(0, r0);                               \
    STW(1, r1);                               \
    STW(2, r2);                               \
    STW(3, r3);                               \
    STW(4, r4);                               \
    STW(5, r5);                               \
    STW(6, r6);                               \
    STW(7, r7);                               \
  } while (0)

/* 128 contiguous positions (Qm = 1 layout) */
static inline __attribute__((always_inline)) void blk_lin(const uint8_t *f, uint64_t *w, int b)
{
  const uint8x16x4_t a = vld4q_u8(f), c = vld4q_u8(f + 64);
  uint8x16_t r0 = vuzp1q_u8(a.val[0], c.val[0]), r4 = vuzp2q_u8(a.val[0], c.val[0]);
  uint8x16_t r1 = vuzp1q_u8(a.val[1], c.val[1]), r5 = vuzp2q_u8(a.val[1], c.val[1]);
  uint8x16_t r2 = vuzp1q_u8(a.val[2], c.val[2]), r6 = vuzp2q_u8(a.val[2], c.val[2]);
  uint8x16_t r3 = vuzp1q_u8(a.val[3], c.val[3]), r7 = vuzp2q_u8(a.val[3], c.val[3]);
  TR8(r0, r1, r2, r3, r4, r5, r6, r7);
  STALL(r0, r1, r2, r3, r4, r5, r6, r7);
}

/* f[6 t + q] = src[q][t]: byte pairs zipped into 16-bit elements, 3-way 16-bit interleave by st3 */
static void interleave6_neon(const uint8_t *const src[8], int L, uint8_t *f)
{
  int t = 0;
  for (; t + 16 <= L; t += 16) {
    const uint8x16_t s0 = vld1q_u8(src[0] + t), s1 = vld1q_u8(src[1] + t), s2 = vld1q_u8(src[2] + t);
    const uint8x16_t s3 = vld1q_u8(src[3] + t), s4 = vld1q_u8(src[4] + t), s5 = vld1q_u8(src[5] + t);
    uint16x8x3_t lo, hi;
    lo.val[0] = vreinterpretq_u16_u8(vzip1q_u8(s0, s1));
    hi.val[0] = vreinterpretq_u16_u8(vzip2q_u8(s0, s1));
    lo.val[1] = vreinterpretq_u16_u8(vzip1q_u8(s2, s3));
    hi.val[1] = vreinterpretq_u16_u8(vzip2q_u8(s2, s3));
    lo.val[2] = vreinterpretq_u16_u8(vzip1q_u8(s4, s5));
    hi.val[2] = vreinterpretq_u16_u8(vzip2q_u8(s4, s5));
    vst3q_u16((uint16_t *)(f + 6 * t), lo);
    vst3q_u16((uint16_t *)(f + 6 * t + 48), hi);
  }
  for (; t < L; t++)
    for (int q = 0; q < 6; q++)
      f[6 * t + q] = src[q][t];
}

/* natural-order interleaving f[qm t + q] = src[q][t] with structure stores (Qm = 8: byte pairs zipped into 16-bit
 * elements, then st4), scalar tail */
static void interleave_neon(int qm, const uint8_t *const src[8], int L, uint8_t *f)
{
  int t = 0;
  switch (qm) {
    case 2:
      for (; t + 16 <= L; t += 16) {
        const uint8x16x2_t v = {{vld1q_u8(src[0] + t), vld1q_u8(src[1] + t)}};
        vst2q_u8(f + 2 * t, v);
      }
      break;
    case 4:
      for (; t + 16 <= L; t += 16) {
        const uint8x16x4_t v = {{vld1q_u8(src[0] + t), vld1q_u8(src[1] + t), vld1q_u8(src[2] + t), vld1q_u8(src[3] + t)}};
        vst4q_u8(f + 4 * t, v);
      }
      break;
    case 6:
      interleave6_neon(src, L, f);
      return;
    case 8:
      for (; t + 16 <= L; t += 16) {
        uint8x16_t s[8];
        for (int q = 0; q < 8; q++)
          s[q] = vld1q_u8(src[q] + t);
        uint16x8x4_t lo, hi;
        for (int p = 0; p < 4; p++) {
          lo.val[p] = vreinterpretq_u16_u8(vzip1q_u8(s[2 * p], s[2 * p + 1]));
          hi.val[p] = vreinterpretq_u16_u8(vzip2q_u8(s[2 * p], s[2 * p + 1]));
        }
        vst4q_u16((uint16_t *)(f + 8 * t), lo);
        vst4q_u16((uint16_t *)(f + 8 * t + 64), hi);
      }
      break;
    default:
      break;
  }
  for (; t < L; t++)
    for (int q = 0; q < qm; q++)
      f[qm * t + q] = src[q][t];
}

/* bit j (mask bit) of 64 bytes -> 64 bits: CMTST, bit weights, pairwise-add tree */
static inline __attribute__((always_inline)) uint64_t plane64(const uint8_t *f, uint8x16_t bit, uint8x16_t wts)
{
  const uint8x16_t v0 = vandq_u8(vtstq_u8(vld1q_u8(f), bit), wts), v1 = vandq_u8(vtstq_u8(vld1q_u8(f + 16), bit), wts);
  const uint8x16_t v2 = vandq_u8(vtstq_u8(vld1q_u8(f + 32), bit), wts), v3 = vandq_u8(vtstq_u8(vld1q_u8(f + 48), bit), wts);
  const uint8x16_t c = vpaddq_u8(vpaddq_u8(v0, v1), vpaddq_u8(v2, v3));
  return vgetq_lane_u64(vreinterpretq_u64_u8(vpaddq_u8(c, c)), 0);
}

void nr_ldpc_tx_extract_neon(int qm, const uint8_t *const src[8], int L, uint8_t *stage, uint64_t *w, unsigned planes)
{
  if (__builtin_popcount(planes) <= 2) {
    /* few code blocks: natural order, one bit plane at a time */
    static const uint8_t wt[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t wts = vld1q_u8(wt);
    const int nw = (qm * L + 63) >> 6;
    const uint8_t *f = src[0];
    if (qm > 1) {
      interleave_neon(qm, src, L, stage);
      f = stage;
    }
    for (int j = 0; j < 8; j++)
      if (planes >> j & 1) {
        const uint8x16_t bit = vdupq_n_u8((uint8_t)(1 << j));
        for (int k = 0; k < nw; k++)
          w[j * NR_LDPC_TX_WS + k] = plane64(f + 64 * k, bit, wts);
      }
    return;
  }
  const int nb = (qm * L + 127) >> 7;
  switch (qm) {
    case 8: {
      const uint8_t *s0 = src[0], *s1 = src[1], *s2 = src[2], *s3 = src[3];
      const uint8_t *s4 = src[4], *s5 = src[5], *s6 = src[6], *s7 = src[7];
      for (int b = 0; b < nb; b++) {
        uint8x16_t r0 = vld1q_u8(s0 + 16 * b), r1 = vld1q_u8(s1 + 16 * b), r2 = vld1q_u8(s2 + 16 * b);
        uint8x16_t r3 = vld1q_u8(s3 + 16 * b), r4 = vld1q_u8(s4 + 16 * b), r5 = vld1q_u8(s5 + 16 * b);
        uint8x16_t r6 = vld1q_u8(s6 + 16 * b), r7 = vld1q_u8(s7 + 16 * b);
        TR8(r0, r1, r2, r3, r4, r5, r6, r7);
        STALL(r0, r1, r2, r3, r4, r5, r6, r7);
      }
    } break;
    case 4: {
      const uint8_t *s0 = src[0], *s1 = src[1], *s2 = src[2], *s3 = src[3];
      for (int b = 0; b < nb; b++) {
        const uint8x16x2_t a0 = vld2q_u8(s0 + 32 * b), a1 = vld2q_u8(s1 + 32 * b);
        const uint8x16x2_t a2 = vld2q_u8(s2 + 32 * b), a3 = vld2q_u8(s3 + 32 * b);
        uint8x16_t r0 = a0.val[0], r1 = a1.val[0], r2 = a2.val[0], r3 = a3.val[0];
        uint8x16_t r4 = a0.val[1], r5 = a1.val[1], r6 = a2.val[1], r7 = a3.val[1];
        TR8(r0, r1, r2, r3, r4, r5, r6, r7);
        STALL(r0, r1, r2, r3, r4, r5, r6, r7);
      }
    } break;
    case 2: {
      const uint8_t *s0 = src[0], *s1 = src[1];
      for (int b = 0; b < nb; b++) {
        const uint8x16x4_t a = vld4q_u8(s0 + 64 * b), c = vld4q_u8(s1 + 64 * b);
        uint8x16_t r0 = a.val[0], r1 = c.val[0], r2 = a.val[1], r3 = c.val[1];
        uint8x16_t r4 = a.val[2], r5 = c.val[2], r6 = a.val[3], r7 = c.val[3];
        TR8(r0, r1, r2, r3, r4, r5, r6, r7);
        STALL(r0, r1, r2, r3, r4, r5, r6, r7);
      }
    } break;
    case 6:
      interleave6_neon(src, L, stage);
      for (int b = 0; b < nb; b++)
        blk_lin(stage + 128 * b, w, b);
      break;
    default:
      for (int b = 0; b < nb; b++)
        blk_lin(src[0] + 128 * b, w, b);
      break;
  }
}
#undef STW
#undef STALL
#undef TR8

void nr_ldpc_tx_write_neon(uint64_t *dst, const uint64_t *w, int n, int sh)
{
  const int64x2_t cl = vdupq_n_s64(sh), cr = vdupq_n_s64(sh - 64);
  /* (w[k-1], w[k]) pairs by EXT of consecutive loads */
  uint64x2_t prev = vcombine_u64(vdup_n_u64(0), vld1_u64(w - 1));
  int k = 0;
  for (; k + 4 <= n; k += 4) {
    const uint64x2_t c0 = vld1q_u64(w + k), c1 = vld1q_u64(w + k + 2);
    const uint64x2_t e0 = vextq_u64(prev, c0, 1), e1 = vextq_u64(c0, c1, 1);
    vst1q_u64(dst + k, vorrq_u64(vshlq_u64(c0, cl), vshlq_u64(e0, cr)));
    vst1q_u64(dst + k + 2, vorrq_u64(vshlq_u64(c1, cl), vshlq_u64(e1, cr)));
    prev = c1;
  }
  for (; k < n; k++)
    nr_ldpc_tx_st64(dst + k, (w[k] << sh) | (sh ? w[k - 1] >> (64 - sh) : 0));
}

static const nr_ldpc_tx_kern_t k_neon = {.name = "neon",
                                         .rows = rows_neon,
                                         .transpose = transpose_neon,
                                         .tb = 16,
                                         .tcols = tcols_neon,
                                         .tcols_zmin = NR_LDPC_TX_NEON_TCOLS_ZMIN,
                                         .tcols_use = tcols_use_neon,
                                         .extract = nr_ldpc_tx_extract_neon,
                                         .write = nr_ldpc_tx_write_neon};
static const nr_ldpc_tx_kern_t k_neon3 = {.name = "neon-eor3",
                                          .rows = rows_neon3,
                                          .transpose = transpose_neon,
                                          .tb = 16,
                                          .tcols = tcols_neon,
                                          .tcols_zmin = NR_LDPC_TX_NEON_TCOLS_ZMIN,
                                          .tcols_use = tcols_use_neon,
                                          .extract = nr_ldpc_tx_extract_neon,
                                          .write = nr_ldpc_tx_write_neon};
const nr_ldpc_tx_kern_t *nr_ldpc_tx_kern_neon(void)
{
  return &k_neon;
}
const nr_ldpc_tx_kern_t *nr_ldpc_tx_kern_neon3(void)
{
  return (getauxval(AT_HWCAP) & HWCAP_SHA3) ? &k_neon3 : 0;
}
#else
const nr_ldpc_tx_kern_t *nr_ldpc_tx_kern_neon(void)
{
  return 0;
}
const nr_ldpc_tx_kern_t *nr_ldpc_tx_kern_neon3(void)
{
  return 0;
}
#endif
