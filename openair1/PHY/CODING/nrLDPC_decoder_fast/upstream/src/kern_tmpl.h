/* Layered min-sum kernel template, instantiated once per ISA. One function per row shape (NN multi edges, N1
 * degree-1 edges, KU vectors per step, FIRST iteration or not), generated from one body; the plan stores the row's
 * function pointers. The including file defines:
 *   KSUF            name suffix (token)          KATTR        function attributes (may be empty)
 *   KREGS           vector registers of the ISA (decides which KU each row shape gets)
 *   VT              signed int8 vector type      VW           bytes per vector (may be a run-time expression)
 *   KSETUP(p)       declarations needed by the macros below (predicates, constants)
 *   VLD(ptr) VST(ptr,v)                 unaligned load / store of one vector
 *   VQSUB VQADD                         signed saturating -, +
 *   VABS(a)                             |a| as unsigned byte (|-128| = 128)
 *   VABD(a, b)                          |a - b| as unsigned byte, not saturated (0..255)
 *   VABSD(a)                            VABS where a is dead afterwards (optional, default VABS)
 *   VPIN(v)                             optional register pin (empty asm), default nothing
 *   VP1(x, r, q, ab)                    optional: q = VQSUB(x, r), ab = VABD(x, r) in one step
 *   VMM(m2, m1, ab)                     optional: m2 = max(min(m2, ab), m1), m1 = min(m1, ab) in one step
 *   VMINU VMAXU                         unsigned byte min / max
 *   VXOR VAND VOR VZERO
 *   VLIM                                vector of p->lim (magnitude clamp, <= 127)
 *   VCORR(mc)                           corr() of a clamped magnitude mc <= lim (no rmax clamp needed)
 *   KSG_T, VSGN(sg), VNEGS(c, s)        sign state of the parity accumulator; c negated where sg < 0
 *   VEQP(m1c)                           per-check precomputation for VMSG (NEON/x86: m1c itself; SVE2: -m1c)
 *   VMSG(q, m1c, eqp, c1, c2)           message: (|q| == m1c ? c2 : c1), negated where q < 0
 *   VMSGA(q, ab, m1c, c1, c2)           the same with ab = |x - R| (unsaturated) kept from P1: (ab == m1c ? c2 : c1)
 *   VANYNEG(t)                          non-zero if any lane of t is negative
 *
 * Row processing (checks of a row are independent; superchunk k = KU*VW consecutive checks, k descending):
 *   P1 for each multi edge e:  x = APP_e (unaligned run of the column's read buffer, offset from rdtab),
 *        Q_e = sqsub(x, R_e) kept in a register, |x - R_e| (unsaturated) for the minimum search;
 *        m1 = min, m2 = second minimum (first two edges: one min + one max; then median form, all destructive),
 *        sg = xor of the Q (parity of the signs); degree-1 edge: Q = channel LLR (read only).
 *   state: m1c = min(m1, lim), c1 = +-corr(m1c), c2 = +-corr(min(m2, lim)); degree-1 rows store (m1c, c1, c2).
 *   P2 for each multi edge: R_e = msg(Q_e) stored in place, APP = sqadd(Q_e, R_e) stored aligned into the column's
 *        write buffer (another buffer than the one read: no ordering constraint between P1 reads and P2 writes);
 *        the first min(B, Z) bytes are also stored at +Z (second copy, keeps every later read run contiguous).
 *   The degree-1 edge's APP is not computed here (lazy: syndrome and LLR output recompute it from the state).
 * Exactness of the shortcuts (unsaturated |x - R|, clamp to lim, missing rmax clamp, SVE range test for the
 * equality) is shown in README section 5; every variant is compared with the scalar reference by test_dec.
 */
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wuninitialized"
#define KCAT_(a, b) a##b
#define KCAT(a, b) KCAT_(a, b)
#define KFN(name) KCAT(name, KSUF)

#ifndef VPIN
#define VPIN(v) (void)0
#endif
#ifndef VABSD
#define VABSD VABS
#endif
#ifndef VMM
#define VMM(m2, m1, ab) do { m2 = VMAXU(VMINU(m2, ab), m1); m1 = VMINU(m1, ab); } while (0)
#endif
#ifndef VP1
#define VP1(x, r, q, ab) do { q = VQSUB(x, r); ab = VABD(x, r); } while (0)
#endif
#define KE19(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11) X(12) X(13) X(14) X(15) X(16) X(17) X(18)
#define KU4(X, e) X(e, 0) X(e, 1) X(e, 2) X(e, 3)
#define KU4S(X) X(0) X(1) X(2) X(3)

static KATTR inline __attribute__((always_inline)) void KFN(row_body_)(const ldpc_plan_t *p, const ldpc_row_t *rw, int8_t *ws,
                                                                       const int NN, const int N1, const int KU,
                                                                       const int KAB, const int FIRST)
{
  KSETUP(p);
  const size_t Z = (size_t)p->z, Bq = (size_t)KU * (size_t)(VW);
  const int nsc = rw->nsc, ndbl = rw->ndbl;
  const int32_t *const wo = p->wo + rw->e0;
  const int32_t *const tb = p->rdtab + rw->toff;
  int8_t *const Rrow = ws + rw->roff;
  int8_t *const Srow = ws + rw->soff;
  const VT lim_ = VLIM;
  /* ISAs with few registers keep the Q of large rows in a stack buffer (explicit two-step per superchunk) instead
     of leaving it to the register allocator's spills */
  const int KQM = NN * KU + 4 * KU + 5 > KREGS;
  int8_t qbuf_[KQM ? 19 * 4 * 64 : 1] __attribute__((aligned(64)));
  (void)qbuf_;
#define KDQ(e, u) VT q_##e##_##u, b_##e##_##u;
#define KDQE(e) KU4(KDQ, e)
  KE19(KDQE)
#define KDS(u) VT m1_##u, m2_##u, sg_##u, m1c_##u, c1_##u, c2_##u, eq_##u;
  KU4S(KDS)
  for (int k = nsc - 1; k >= 0; k--) {
    const int32_t *const t_ = tb + (size_t)k * NN;
    int8_t *const R_ = Rrow + (size_t)k * NN * Bq;
    int8_t *const wk = ws + (size_t)k * Bq;
    /* ---- P1 */
#define KP1U(e, u)                                                              \
  if ((u) < KU) {                                                               \
    const int8_t *const a_ = ws + t_[e] + (u) * (VW);                           \
    VT ab_;                                                                     \
    if (FIRST) {                                                                \
      q_##e##_##u = VLD(a_);                                                    \
      ab_ = VABS(q_##e##_##u);                                                  \
    } else {                                                                    \
      const VT x_ = VLD(a_), r_ = VLD(R_ + (e) * Bq + (u) * (VW));              \
      VP1(x_, r_, q_##e##_##u, ab_); /* q = sqsub(x, r), ab = |x - r| (x dead) */ \
    }                                                                           \
    if (KAB) b_##e##_##u = ab_;                                                 \
    if (KQM) VST(qbuf_ + ((e) * KU + (u)) * (VW), q_##e##_##u);                 \
    if ((e) == 0) {                                                             \
      m1_##u = ab_;                                                             \
      sg_##u = q_##e##_##u;                                                     \
    } else if ((e) == 1) {                                                      \
      m2_##u = VMAXU(m1_##u, ab_);                                              \
      m1_##u = VMINU(m1_##u, ab_);                                              \
      sg_##u = VXOR(sg_##u, q_##e##_##u);                                       \
      VPIN(sg_##u); /* materialise q0 ^ q1: later XORs accumulate into it */    \
    } else {                                                                    \
      VMM(m2_##u, m1_##u, ab_); /* m2 = max(min(m2, ab), m1), m1 = min(m1, ab) */ \
      sg_##u = VXOR(sg_##u, q_##e##_##u);                                       \
    }                                                                           \
  }
#define KP1(e) if ((e) < NN) { KU4(KP1U, e) }
    KE19(KP1)
    if (N1) {
      const int8_t *const a1_ = wk + wo[NN];
#define KP1D(u)                                                                 \
  if ((u) < KU) {                                                               \
    const VT q_ = VLD(a1_ + (u) * (VW));                                        \
    sg_##u = VXOR(sg_##u, q_);                                                  \
    const VT ab_ = VABSD(q_); /* q_ is dead: may be destructive */              \
    VMM(m2_##u, m1_##u, ab_);                                                   \
  }
      KU4S(KP1D)
    }
    /* ---- row state */
#define KST(u)                                                                  \
  if ((u) < KU) {                                                               \
    m1c_##u = VMINU(m1_##u, lim_);                                              \
    const KSG_T s_ = VSGN(sg_##u);                                              \
    c1_##u = VNEGS(VCORR(m1c_##u), s_);                                         \
    c2_##u = VNEGS(VCORR(VMINU(m2_##u, lim_)), s_);                             \
    eq_##u = VEQP(m1c_##u);                                                     \
    if (N1) {                                                                   \
      int8_t *const s1_ = Srow + (size_t)k * 3 * Bq + (u) * (VW);               \
      VST(s1_, m1c_##u);                                                        \
      VST(s1_ + Bq, c1_##u);                                                    \
      VST(s1_ + 2 * Bq, c2_##u);                                                \
    }                                                                           \
  }
    KU4S(KST)
    /* ---- P2 */
#define KP2U(e, u)                                                              \
  if ((u) < KU) {                                                               \
    if (KQM) q_##e##_##u = VLD(qbuf_ + ((e) * KU + (u)) * (VW));               \
    const VT r_ = KAB ? VMSGA(q_##e##_##u, b_##e##_##u, m1c_##u, c1_##u, c2_##u)  \
                      : VMSG(q_##e##_##u, m1c_##u, eq_##u, c1_##u, c2_##u);     \
    VST(R_ + (e) * Bq + (u) * (VW), r_);                                        \
    VST(w_ + (u) * (VW), VQADD(q_##e##_##u, r_));                               \
  }
#define KP2(e)                                                                  \
  if ((e) < NN) {                                                               \
    int8_t *const w_ = wk + wo[e];                                              \
    KU4(KP2U, e)                                                                \
  }
    KE19(KP2)
    /* second copy of the first min(B, Z) bytes: copied back from the just written run (a separate step: keeping
       the APP values for a conditional second store would keep them all live and make the compiler spill) */
    if (k < ndbl) {
#define KP2Z(e, u) if ((u) < KU) VST(w_ + Z + (u) * (VW), VLD(w_ + (u) * (VW)));
#define KP2C(e)                                                                 \
  if ((e) < NN) {                                                               \
    int8_t *const w_ = wk + wo[e];                                              \
    KU4(KP2Z, e)                                                                \
  }
      KE19(KP2C)
    }
  }
}

/* row functions for the row shapes of the 38.212 base graphs at OAI's six rates:
   BG1: nn = 2..9 with one degree-1 edge, 19 without; BG2: 2..5 with one, 8 and 10 without.
   KU is limited by the register budget (nn*KU Q vectors + 4*KU state vectors + ~5 must fit in KREGS):
   32 registers: KU = 4 for nn = 2, KU = 2 for nn <= 9; 16 registers (AVX2): KU = 1 only. */
#define KROWFN(nn, n1, u, ab, f)                                                                               \
  static KATTR void KFN(rf_##nn##_##n1##_##u##_##ab##_##f##_)(const ldpc_plan_t *p, const ldpc_row_t *rw,       \
                                                               int8_t *ws)                                     \
  {                                                                                                            \
    KFN(row_body_)(p, rw, ws, nn, n1, u, ab, f);                                                               \
  }
#define KROWFN2(nn, n1, u, ab) KROWFN(nn, n1, u, ab, 0) KROWFN(nn, n1, u, ab, 1)
/* X(nn, n1, U, keep-|x-R|): the first matching entry with the largest U <= umax is used */
#ifndef KSHAPES
#if KREGS >= 32
#define KSHAPES(X)                                                                                             \
  X(2, 1, 4, 0) X(2, 1, 2, 1) X(3, 1, 2, 1) X(4, 1, 2, 1) X(2, 1, 1, 1) X(3, 1, 1, 1) X(4, 1, 1, 1) X(5, 1, 1, 1) X(6, 1, 1, 1) X(7, 1, 1, 1)  \
  X(8, 1, 1, 1) X(9, 1, 1, 1) X(8, 0, 1, 1) X(10, 0, 1, 1) X(19, 0, 1, 0)
#else
#define KSHAPES(X)                                                                                             \
  X(2, 1, 1, 0) X(3, 1, 1, 0) X(4, 1, 1, 0) X(5, 1, 1, 0) X(6, 1, 1, 0) X(7, 1, 1, 0) X(8, 1, 1, 0) X(9, 1, 1, 0)  \
  X(19, 0, 1, 0) X(8, 0, 1, 0) X(10, 0, 1, 0)
#endif
#endif
KSHAPES(KROWFN2)

static ldpc_rowfn_t KFN(row_fn_)(int nn, int n1, int u, int first)
{
#define KSEL(nn_, n1_, u_, ab_) \
  if (nn == nn_ && n1 == n1_ && u == u_) return first ? KFN(rf_##nn_##_##n1_##_##u_##_##ab_##_1_) : KFN(rf_##nn_##_##n1_##_##u_##_##ab_##_0_);
  KSHAPES(KSEL)
#undef KSEL
  return NULL;
}

/* largest supported U <= umax (U = 1 is always provided for the known shapes; 0: unknown shape) */
static int KFN(row_u_)(int nn, int n1, int umax)
{
  for (int u = umax; u >= 1; u >>= 1)
    if (KFN(row_fn_)(nn, n1, u, 0)) return u;
  return 0;
}

/* syndrome of the hard decision, row by row, in superchunks of the plan's B = KU*W bytes (independent of the
   row's own U = UR: no register pressure here; UR | KU fixes the degree-1 state addressing at compile time);
   returns 0 at the first unsatisfied row. The valid-lane mask is applied only where padding lanes exist. */
static KATTR inline __attribute__((always_inline)) int KFN(syn_row_)(const ldpc_plan_t *p, const ldpc_row_t *rw,
                                                                     const int8_t *ws, const int32_t *tab, const int KU,
                                                                     const int UR)
{
  KSETUP(p);
  const size_t Bq = (size_t)KU * (size_t)(VW), Br = (size_t)UR * (size_t)(VW);
  const int nn = rw->nn, nsc = p->zp / (int)Bq, Z = p->z;
  const int8_t *const lm = ws + p->lanemask;
  const int8_t *const a1 = rw->n1 ? ws + p->sb[rw->e0 + nn] : ws;
  VT t_ = VZERO;
  for (int k = 0; k < nsc; k++) {
    const int kb = k * (int)Bq;
#define KX0(u) VT x_##u = VZERO;
    KU4S(KX0)
    const int32_t *const st = tab + rw->yoff + (size_t)k * nn;
    for (int e = 0; e < nn; e++) {
      const int8_t *const a = ws + st[e];
#define KX1(u) if ((u) < KU) x_##u = VXOR(x_##u, VLD(a + (u) * (VW)));
      KU4S(KX1)
    }
    if (rw->n1) { /* degree-1 column: APP = Q + R from the channel and the row state ([kr][3][UR*W] layout) */
      const int8_t *const a = a1 + kb;
#define KX2(u)                                                                  \
  if ((u) < KU) {                                                               \
    const int v_ = k * KU + (u); /* vector index; row superchunk v_ / UR, vector v_ % UR in it */ \
    const int8_t *const s_ = ws + rw->soff + (size_t)(v_ / UR) * 3 * Br + (size_t)(v_ % UR) * (VW); \
    const VT q_ = VLD(a + (u) * (VW)), m1c_ = VLD(s_);                          \
    const VT r_ = VMSG(q_, m1c_, VEQP(m1c_), VLD(s_ + Br), VLD(s_ + 2 * Br));   \
    x_##u = VXOR(x_##u, VQADD(q_, r_));                                         \
  }
      KU4S(KX2)
    }
    if (kb + (int)Bq > Z) { /* superchunk with padding lanes */
#define KX3(u) if ((u) < KU) t_ = VOR(t_, VAND(x_##u, VLD(lm + kb + (u) * (VW))));
      KU4S(KX3)
    } else {
#define KX4(u) if ((u) < KU) t_ = VOR(t_, x_##u);
      KU4S(KX4)
    }
  }
  return !VANYNEG(t_);
}

static KATTR int KFN(syn_)(const ldpc_plan_t *p, const int8_t *ws, int nrows, const int32_t *tab)
{
  const int U = p->U;
  for (int r = 0; r < nrows; r++) {
    const ldpc_row_t *rw = &p->row[r];
    const int ur = rw->n1 ? rw->U : 1;
    int ok;
    if (U == 4) ok = ur == 4 ? KFN(syn_row_)(p, rw, ws, tab, 4, 4) : ur == 2 ? KFN(syn_row_)(p, rw, ws, tab, 4, 2) : KFN(syn_row_)(p, rw, ws, tab, 4, 1);
    else if (U == 2) ok = ur == 2 ? KFN(syn_row_)(p, rw, ws, tab, 2, 2) : KFN(syn_row_)(p, rw, ws, tab, 2, 1);
    else ok = KFN(syn_row_)(p, rw, ws, tab, 1, 1);
    if (!ok) return 0;
  }
  return 1;
}

static void KFN(kern_fill_)(ldpc_kernels_t *k)
{
  k->syndrome = KFN(syn_);
  k->row_u = KFN(row_u_);
  k->row_fn = KFN(row_fn_);
}
