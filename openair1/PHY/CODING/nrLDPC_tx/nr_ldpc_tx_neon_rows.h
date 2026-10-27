/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* NEON row kernel body, included twice by kern_neon.c (FN/TGT/GMAX/NR_LDPC_TX_EOR3 select the flavour).
 * One pass over the edges of a row computes up to GMAX 16-byte chunks held in registers (explicit variables
 * a0..a23 so that the compiler keeps them in registers).  The accumulators are initialised with the first edge
 * (no zeroing), the remaining edges are XORed one at a time (plain: ld1 {4 regs} + EOR) or two at a time
 * (EOR3 flavour: ldp + EOR3).  A z % 16 tail is one extra chunk overlapping the previous one (no masks). */

/* ldp q,q,[x,#imm]: 32 bytes per load with an immediate offset (GCC 13 does not pair vld1q_u8 by itself) */
#ifndef LDPQ
#define LDPQ(x, y, ptr, off) __asm__("ldp %q0, %q1, [%2, %3]" : "=w"(x), "=w"(y) : "r"(ptr), "i"(off) : "memory")
#endif
#define PAIRS(M) M(0, 1) M(2, 3) M(4, 5) M(6, 7) M(8, 9) M(10, 11) PAIRS_HI(M)
#if GMAX > 12
#define PAIRS_HI(M) M(12, 13) M(14, 15) M(16, 17) M(18, 19) M(20, 21) M(22, 23)
#define G4_HI G4(12, 13, 14, 15) G4(16, 17, 18, 19) G4(20, 21, 22, 23)
#else
#define PAIRS_HI(M)
#define G4_HI
#endif

static inline __attribute__((always_inline)) TGT void FN(
    grp)(const uint8_t *base, const uint16_t *off, int ne, int o, const int N, const int tail, int ot, uint8_t *d, int z, int dup)
{
  const uint8x16_t zero = vdupq_n_u8(0);
  uint8x16_t a0 = zero, a1 = zero, a2 = zero, a3 = zero, a4 = zero, a5 = zero, a6 = zero, a7 = zero, a8 = zero, a9 = zero,
             a10 = zero, a11 = zero, t = zero;
#if GMAX > 12
  uint8x16_t a12 = zero, a13 = zero, a14 = zero, a15 = zero, a16 = zero, a17 = zero, a18 = zero, a19 = zero, a20 = zero, a21 = zero,
             a22 = zero, a23 = zero;
#endif
  if (ne > 0) {
    { /* the first edge initialises the accumulators */
      const uint8_t *p = base + off[0] + o;
#ifdef NR_LDPC_TX_EOR3
/* plain loads here: with the asm ldp GCC adds register moves into the accumulators in this flavour */
#define I2(k0, k1)                 \
  if (N > k0)                      \
    a##k0 = vld1q_u8(p + 16 * k0); \
  if (N > k1)                      \
    a##k1 = vld1q_u8(p + 16 * k1);
#else
#define I2(k0, k1)                  \
  if (N > k1)                       \
    LDPQ(a##k0, a##k1, p, 16 * k0); \
  else if (N > k0)                  \
    a##k0 = vld1q_u8(p + 16 * k0);
#endif
      PAIRS(I2)
#undef I2
      if (tail)
        t = vld1q_u8(base + off[0] + ot);
    }
    int e = 1;
    /* one edge: a ^= x */
#define G4(k0, k1, k2, k3)                            \
  if (N > k3) {                                       \
    uint8x16x4_t x = vld1q_u8_x4(p + 16 * k0);        \
    a##k0 = veorq_u8(a##k0, x.val[0]);                \
    a##k1 = veorq_u8(a##k1, x.val[1]);                \
    a##k2 = veorq_u8(a##k2, x.val[2]);                \
    a##k3 = veorq_u8(a##k3, x.val[3]);                \
  } else if (N > k1) {                                \
    uint8x16_t x0, x1;                                \
    LDPQ(x0, x1, p, 16 * k0);                         \
    a##k0 = veorq_u8(a##k0, x0);                      \
    a##k1 = veorq_u8(a##k1, x1);                      \
    if (N > k2)                                       \
      a##k2 = veorq_u8(a##k2, vld1q_u8(p + 16 * k2)); \
  } else if (N > k0) {                                \
    a##k0 = veorq_u8(a##k0, vld1q_u8(p + 16 * k0));   \
  }
#define ONE_EDGE(E)                                                                                               \
  {                                                                                                               \
    const uint8_t *p = base + off[E] + o;                                                                         \
    G4(0, 1, 2, 3) G4(4, 5, 6, 7) G4(8, 9, 10, 11) G4_HI if (tail) t = veorq_u8(t, vld1q_u8(base + off[E] + ot)); \
  }
#ifdef NR_LDPC_TX_EOR3
    if (!(ne & 1)) { /* make the remaining count even */
      ONE_EDGE(1)
      e = 2;
    }
    /* two edges per step: a = a ^ x ^ y */
#define G2_3(k0, k1)                  \
  if (N > k1) {                       \
    uint8x16_t x0, x1, y0, y1;        \
    LDPQ(x0, x1, p, 16 * k0);         \
    LDPQ(y0, y1, q, 16 * k0);         \
    a##k0 = veor3q_u8(a##k0, x0, y0); \
    a##k1 = veor3q_u8(a##k1, x1, y1); \
  } else if (N > k0)                  \
    a##k0 = veor3q_u8(a##k0, vld1q_u8(p + 16 * k0), vld1q_u8(q + 16 * k0));
    for (; e < ne; e += 2) {
      const uint8_t *p = base + off[e] + o, *q = base + off[e + 1] + o;
      PAIRS(G2_3)
      if (tail)
        t = veor3q_u8(t, vld1q_u8(base + off[e] + ot), vld1q_u8(base + off[e + 1] + ot));
    }
#undef G2_3
#else
    for (; e < ne; e++)
      ONE_EDGE(e)
#endif
#undef ONE_EDGE
  }
  /* plain stores: the compiler pairs them into stp (st1 {4 regs} would force register shuffles) */
#define S2(k0, k1)                \
  if (N > k0)                     \
    vst1q_u8(q + 16 * k0, a##k0); \
  if (N > k1)                     \
    vst1q_u8(q + 16 * k1, a##k1);
  for (int k = 0; k <= dup; k++) {
    uint8_t *q = d + k * z + o;
    PAIRS(S2)
    if (tail)
      vst1q_u8(d + k * z + ot, t);
  }
#undef S2
}
#undef G4
#undef G4_HI
#undef PAIRS
#undef PAIRS_HI

static TGT void FN(
    rows)(const uint8_t *base, const uint16_t *off, const uint8_t *cnt, int nrows, int z, uint8_t *dst, int dstride, int dup)
{
  if (z < 16) {
    nr_ldpc_tx_rows_scalar(base, off, cnt, nrows, z, dst, dstride, dup);
    return;
  }
  const int nfull = z >> 4, ot = z - 16, tl = (z & 15) != 0;
  for (int r = 0; r < nrows; r++) {
    const int ne = cnt[r];
    int o = 0, n = nfull;
    while (n > GMAX) {
      FN(grp)(base, off, ne, o, GMAX, 0, ot, dst, z, dup);
      o += 16 * GMAX;
      n -= GMAX;
    }
#define CASE(n)                                           \
  case n:                                                 \
    if (tl)                                               \
      FN(grp)(base, off, ne, o, (n), 1, ot, dst, z, dup); \
    else                                                  \
      FN(grp)(base, off, ne, o, (n), 0, ot, dst, z, dup); \
    break;
    switch (n) {
      CASE(1)
      CASE(2)
      CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) CASE(9) CASE(10) CASE(11) CASE(12)
#if GMAX > 12
          CASE(13) CASE(14) CASE(15) CASE(16) CASE(17) CASE(18) CASE(19) CASE(20) CASE(21) CASE(22) CASE(23) CASE(24)
#endif
    }
#undef CASE
    off += ne;
    dst += dstride;
  }
}
