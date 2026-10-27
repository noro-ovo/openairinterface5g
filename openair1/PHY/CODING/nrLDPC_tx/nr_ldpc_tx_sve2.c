/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* aarch64 SVE2 encoder kernels of the packed LDPC transmit chain, vector-length agnostic, selected at run time
 * (HWCAP2_SVE2). The output stage uses the NEON kernels.
 * Only the functions between push_options/pop_options are compiled for SVE2. */
#include "nr_ldpc_tx_internal.h"
#if defined(__aarch64__) && !defined(NR_LDPC_TX_NO_SVE2)
#include <sys/auxv.h>
#ifndef HWCAP_SVE
#define HWCAP_SVE (1 << 22)
#endif
#ifndef HWCAP2_SVE2
#define HWCAP2_SVE2 (1 << 1)
#endif
#pragma GCC push_options
#pragma GCC target("arch=armv8.5-a+sve2")
#include <arm_sve.h>

#define AI static inline __attribute__((always_inline))

/* N (1..24) consecutive vectors starting at byte o of one row, accumulated in registers; the first edge
 * initialises the accumulators, then two edges per EOR3; stores predicated on i < z (no tail handling needed,
 * loads may read up to one vector past the row: the work area has NR_LDPC_TX_PAD bytes of slack) */
#define REP24(M) \
  M(0)           \
  M(1) M(2) M(3) M(4) M(5) M(6) M(7) M(8) M(9) M(10) M(11) M(12) M(13) M(14) M(15) M(16) M(17) M(18) M(19) M(20) M(21) M(22) M(23)
AI void grp(const uint8_t *base, const uint16_t *off, int ne, int o, const int N, uint8_t *d, int z, int dup)
{
  const svbool_t pt = svptrue_b8();
#define DECL(k) svuint8_t a##k = svdup_n_u8(0);
  REP24(DECL)
#undef DECL
  const int64_t vl8 = 8 * (int64_t)svcntb();
  /* vnum immediates only reach [-8, 7]: one base pointer per 8 vectors, made opaque to the compiler (otherwise
   * GCC folds them back into one addvl per load) */
#define BASES(p)                                        \
  const uint8_t *p##_1 = p + vl8, *p##_2 = p + 2 * vl8; \
  __asm__("" : "+r"(p##_1), "+r"(p##_2));               \
  const uint8_t *const p##_b[3] = {p, p##_1, p##_2};
#define L(p, k) svld1_vnum_u8(pt, p##_b[(k) >> 3], (k) & 7)
  if (ne > 0) {
    int e = 1;
    {
      const uint8_t *p = base + off[0] + o;
      BASES(p)
#define I1(k) \
  if (N > k)  \
    a##k = L(p, k);
      REP24(I1)
#undef I1
    }
    if (!(ne & 1)) {
      const uint8_t *p = base + off[1] + o;
      BASES(p)
#define X2(k) \
  if (N > k)  \
    a##k = sveor_u8_x(pt, a##k, L(p, k));
      REP24(X2)
#undef X2
      e = 2;
    }
    for (; e < ne; e += 2) {
      const uint8_t *p = base + off[e] + o, *q = base + off[e + 1] + o;
      BASES(p)
      BASES(q)
#define X3(k) \
  if (N > k)  \
    a##k = sveor3_u8(a##k, L(p, k), L(q, k));
      REP24(X3)
#undef X3
    }
  }
  /* only the last vector of the row can be partial */
  const svbool_t pl = svwhilelt_b8_s32(o + (N - 1) * (int)svcntb(), z);
  uint8_t *s0 = d + o, *s1 = d + z + o;
  BASES(s0)
  BASES(s1)
#define ST(k)                                                      \
  if (N > k) {                                                     \
    const svbool_t pg = (k == N - 1) ? pl : pt;                    \
    svst1_vnum_u8(pg, (uint8_t *)s0_b[(k) >> 3], (k) & 7, a##k);   \
    if (dup)                                                       \
      svst1_vnum_u8(pg, (uint8_t *)s1_b[(k) >> 3], (k) & 7, a##k); \
  }
  REP24(ST)
#undef BASES
#undef L
#undef ST
}

#define GMAX 24
static void
rows_sve2(const uint8_t *base, const uint16_t *off, const uint8_t *cnt, int nrows, int z, uint8_t *dst, int dstride, int dup)
{
  const int vl = (int)svcntb(), nch = (z + vl - 1) / vl;
  for (int r = 0; r < nrows; r++) {
    const int ne = cnt[r];
    for (int c0 = 0; c0 < nch; c0 += GMAX) {
      const int o = c0 * vl;
#define CASE(n)                            \
  case n:                                  \
    grp(base, off, ne, o, n, dst, z, dup); \
    break;
      switch (nch - c0 >= GMAX ? GMAX : nch - c0) {
        CASE(1)
        CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) CASE(9) CASE(10) CASE(11) CASE(12) CASE(13) CASE(14) CASE(15)
            CASE(16) CASE(17) CASE(18) CASE(19) CASE(20) CASE(21) CASE(22) CASE(23) CASE(24)
      }
#undef CASE
    }
    off += ne;
    dst += dstride;
  }
}

/* transposes the VL bytes at offset b of the 8 streams (only the first nb bytes: predicated loads) and stores the
 * 8*nb output bytes at out + 8*b (predicated st4) */
static inline __attribute__((always_inline)) void tr_sve(const uint8_t *const in[8], int b, int nb, uint8_t *out)
{
  const int vl = (int)svcntb();
  const svbool_t pt = svptrue_b8(), pin = svwhilelt_b8_s32(b, nb);
  const svuint8_t m1 = svdup_n_u8(0x55), m2 = svdup_n_u8(0x33);
  svuint8_t r0 = svld1_u8(pin, in[0] + b), r1 = svld1_u8(pin, in[1] + b);
  svuint8_t r2 = svld1_u8(pin, in[2] + b), r3 = svld1_u8(pin, in[3] + b);
  svuint8_t r4 = svld1_u8(pin, in[4] + b), r5 = svld1_u8(pin, in[5] + b);
  svuint8_t r6 = svld1_u8(pin, in[6] + b), r7 = svld1_u8(pin, in[7] + b);
  svuint8_t x, y;
#define SW(A, B, s, m)                        \
  x = A;                                      \
  y = B;                                      \
  A = svbsl_u8(x, svlsl_n_u8_x(pt, y, s), m); \
  B = svbsl_u8(svlsr_n_u8_x(pt, x, s), y, m);
#define SW4(A, B)          \
  x = A;                   \
  y = B;                   \
  A = svsli_n_u8(x, y, 4); \
  B = svsri_n_u8(y, x, 4);
  SW(r0, r1, 1, m1)
  SW(r2, r3, 1, m1) SW(r4, r5, 1, m1) SW(r6, r7, 1, m1) SW(r0, r2, 2, m2) SW(r1, r3, 2, m2) SW(r4, r6, 2, m2) SW(r5, r7, 2, m2)
      SW4(r0, r4) SW4(r1, r5) SW4(r2, r6) SW4(r3, r7)
#undef SW
#undef SW4
      /* out[8b+u] = r(7-u)[b]: zip byte pairs, then st4 of 16-bit elements (element i of the low halves = byte b+i) */
      svuint16_t l0 = svreinterpret_u16_u8(svzip1_u8(r7, r6)),
                 h0 = svreinterpret_u16_u8(svzip2_u8(r7, r6));
  svuint16_t l1 = svreinterpret_u16_u8(svzip1_u8(r5, r4)), h1 = svreinterpret_u16_u8(svzip2_u8(r5, r4));
  svuint16_t l2 = svreinterpret_u16_u8(svzip1_u8(r3, r2)), h2 = svreinterpret_u16_u8(svzip2_u8(r3, r2));
  svuint16_t l3 = svreinterpret_u16_u8(svzip1_u8(r1, r0)), h3 = svreinterpret_u16_u8(svzip2_u8(r1, r0));
  uint16_t *o16 = (uint16_t *)(out + 8 * b);
  svst4_u16(svwhilelt_b16_s32(b, nb), o16, svcreate4_u16(l0, l1, l2, l3));
  svst4_u16(svwhilelt_b16_s32(b + vl / 2, nb), o16 + 2 * vl, svcreate4_u16(h0, h1, h2, h3));
}

static void transpose_sve2(const uint8_t *const in[8], int nblk, uint8_t *out)
{
  const int vl = (int)svcntb();
  for (int i = 0; i < nblk; i++)
    tr_sve(in, vl * i, vl * (i + 1), out);
}

/* column transposer: predicated blocks of VL bytes into the first work copy, then a predicated vector copy of
 * the z bytes to the doubled copy and the output */
static void tcols_sve2(const uint8_t *const in[8], int z, int ncols, uint8_t *work, int S, uint8_t *out, int cw)
{
  const int nb = z >> 3, vl = (int)svcntb();
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
    for (int b = 0; b < nb; b += vl)
      tr_sve(p, b, nb, w0);
    if (c >= 2 && c < cw) {
      for (int i = 0; i < z; i += vl) {
        const svbool_t pg = svwhilelt_b8_s32(i, z);
        const svuint8_t v = svld1_u8(pg, w0 + i);
        svst1_u8(pg, w1 + i, v);
        svst1_u8(pg, o + i, v);
      }
    } else {
      for (int i = 0; i < z; i += vl) {
        const svbool_t pg = svwhilelt_b8_s32(i, z);
        svst1_u8(pg, w1 + i, svld1_u8(pg, w0 + i));
      }
    }
  }
}

static int sve_vl_bytes(void)
{
  return (int)svcntb();
}
#pragma GCC pop_options

/* column transposer off by default (about 2 % more instructions than the linear-buffer path under qemu at VL 128 and
 * 256, with and without FEAT_MOPS); -DNR_LDPC_TX_SVE_TCOLS_ZMIN=8 enables it */
#ifndef NR_LDPC_TX_SVE_TCOLS_ZMIN
#define NR_LDPC_TX_SVE_TCOLS_ZMIN (1 << 30)
#endif
static nr_ldpc_tx_kern_t k_sve2 = {.name = "sve2",
                                   .rows = rows_sve2,
                                   .transpose = transpose_sve2,
                                   .tcols = tcols_sve2,
                                   .tcols_zmin = NR_LDPC_TX_SVE_TCOLS_ZMIN,
                                   .extract = nr_ldpc_tx_extract_neon,
                                   .write = nr_ldpc_tx_write_neon};
const nr_ldpc_tx_kern_t *nr_ldpc_tx_kern_sve2(void)
{
  /* both bits, as OAI does (qemu -cpu max,sve=off still reports HWCAP2_SVE2) */
  if (!(getauxval(AT_HWCAP) & HWCAP_SVE) || !(getauxval(AT_HWCAP2) & HWCAP2_SVE2))
    return 0;
  int vl = sve_vl_bytes();
  if (vl > 256)
    return 0;
  k_sve2.tb = vl;
  return &k_sve2;
}
#else
const nr_ldpc_tx_kern_t *nr_ldpc_tx_kern_sve2(void)
{
  return 0;
}
#endif
