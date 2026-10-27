/* SVE2 instantiation of the kernel template, vector-length agnostic (W = svcntb() at run time; the plan is
 * built for the actual VL). Compiled with baseline flags; only these functions carry the SVE2 target, they are
 * called only after the HWCAP2_SVE2 check in ldpc_dec.c.
 * SVE2-specific forms (fewer instructions than NEON, no MOVPRFX in the edge loops):
 *   |x - R| with the destructive ABD (x is dead after the SQSUB),
 *   equality |q| == m1c as the range test -m1c <= q <= m1c (two chained compares, no ABS),
 *   predicated negation for the signs. */
#include "ldpc_internal.h"
#if defined(__aarch64__)
#if !defined(__ARM_FEATURE_SVE2) /* baseline without SVE2 (e.g. armv8.2-a): enable it for this file only */
#pragma GCC target("arch=armv8.5-a+sve2")
#endif
#include <arm_sve.h>

static inline svint8_t v_corr(svbool_t pg, svint8_t m, int nsh, svuint8_t nshv, svuint8_t vb)
{
  svuint8_t x = svreinterpret_u8_s8(m);
  if (nsh) x = svsub_u8_x(pg, x, svlsr_u8_x(pg, x, nshv));
  return svreinterpret_s8_u8(svqsub_u8(x, vb));
}
static inline svint8_t v_msg(svbool_t pg, svint8_t q, svint8_t m1c, svint8_t nm1c, svint8_t c1, svint8_t c2)
{
  const svbool_t eq = svcmpge_s8(svcmple_s8(pg, q, m1c), q, nm1c);
  const svint8_t s = svsel_s8(eq, c2, c1);
  return svneg_s8_m(s, svcmplt_n_s8(pg, q, 0), s);
}

#define KSUF sve2
#define KATTR
#define KREGS 32
#define VT svint8_t
#define VW svcntb()
#define KSETUP(p)                                          \
  const svbool_t pg_ = svptrue_b8();                       \
  const int nsh_ = (p)->nshift;                            \
  const svuint8_t nshv_ = svdup_n_u8((uint8_t)nsh_);       \
  const svuint8_t vb_ = svdup_n_u8((uint8_t)(p)->beta);    \
  (void)nshv_; (void)vb_
#define VLD(p) svld1_s8(pg_, (const int8_t *)(p))
#define VST(p, v) svst1_s8(pg_, (int8_t *)(p), v)
#define VQSUB svqsub_s8
#define VQADD svqadd_s8
#define VSUBW(a, b) svsub_s8_x(pg_, a, b) /* wrapping subtraction (absorbed messages) */
#define VABS(a) svabs_s8_x(pg_, a)
/* destructive SABD on the (dead) first operand, issued after the SQSUB that also reads it: GCC would otherwise
   schedule the ABD first and add a MOVPRFX */
static inline svint8_t v_abd(svbool_t pg, svint8_t a, svint8_t b)
{
  __asm__("sabd %0.b, %1/m, %0.b, %2.b" : "+w"(a) : "Upl"(pg), "w"(b));
  return a;
}
#define VABD(a, b) v_abd(pg_, a, b)
static inline svint8_t v_absd(svbool_t pg, svint8_t a)
{
  __asm__("abs %0.b, %1/m, %0.b" : "+w"(a) : "Upl"(pg));
  return a;
}
#define VABSD(a) v_absd(pg_, a)
#define VPIN(v) __asm__("" : "+w"(v))
/* fixed instruction sequences (single asm blocks, so the scheduler cannot reorder them into MOV / MOVPRFX copies):
   q = SQSUB x, r (new register), then SABD destructive on x; the min1/min2 update as 3 destructive UMIN/UMAX */
#define VP1(x, r, q, ab)                                                                  \
  do {                                                                                    \
    svint8_t x1_ = (x);                                                                   \
    __asm__("sqsub %0.b, %1.b, %2.b\n\tsabd %1.b, %3/m, %1.b, %2.b"                       \
            : "=&w"(q), "+w"(x1_) : "w"(r), "Upl"(pg_));                                  \
    (ab) = x1_;                                                                           \
  } while (0)
#define VMM(m2, m1, ab)                                                                   \
  __asm__("umin %0.b, %3/m, %0.b, %2.b\n\tumax %0.b, %3/m, %0.b, %1.b\n\tumin %1.b, %3/m, %1.b, %2.b" \
          : "+w"(m2), "+w"(m1) : "w"(ab), "Upl"(pg_))
#define VMINU(a, b) svreinterpret_s8_u8(svmin_u8_x(pg_, svreinterpret_u8_s8(a), svreinterpret_u8_s8(b)))
#define VMAXU(a, b) svreinterpret_s8_u8(svmax_u8_x(pg_, svreinterpret_u8_s8(a), svreinterpret_u8_s8(b)))
#define VXOR(a, b) sveor_s8_x(pg_, a, b)
#define VAND(a, b) svand_s8_x(pg_, a, b)
#define VOR(a, b) svorr_s8_x(pg_, a, b)
#define VZERO svdup_n_s8(0)
#define VLIM svdup_n_s8((int8_t)p->lim)
#define VCORR(m) v_corr(pg_, m, nsh_, nshv_, vb_)
#define KSG_T svbool_t
#define VSGN(sg) svcmplt_n_s8(pg_, sg, 0)
#define VNEGS(c, s) svneg_s8_m(c, s, c)
#define VEQP(m1c) svsub_s8_x(pg_, svdup_n_s8(0), m1c) /* unpredicated SUB: no MOVPRFX */
#define VMSG(q, m1c, eqp, c1, c2) v_msg(pg_, q, m1c, eqp, c1, c2)
static inline svint8_t v_msga(svbool_t pg, svint8_t q, svint8_t ab, svint8_t m1c, svint8_t c1, svint8_t c2)
{
  const svint8_t s = svsel_s8(svcmpeq_s8(pg, ab, m1c), c2, c1);
  return svneg_s8_m(s, svcmplt_n_s8(pg, q, 0), s);
}
#define VMSGA(q, ab, m1c, c1, c2) v_msga(pg_, q, ab, m1c, c1, c2)
#define VANYNEG(t) svptest_any(pg_, svcmplt_n_s8(pg_, t, 0))

#include "kern_tmpl.h"

int ldpc_sve2_bytes(void) { return (int)svcntb(); }

/* hard decisions packed MSB first (bit 7 of out[j] = value 8j): CMPLT into a predicate and STR P (one bit per byte
   lane, LSB first), then RBIT + REV per 64 bits on the integer side; returns the number of values done (multiple of
   64, the caller packs the tail) */
int ldpc_sve2_pack_bits(const int8_t *v, int n, uint8_t *out)
{
  const int W = (int)svcntb();
  if (64 % W) return 0; /* VL > 512 bits: not used */
  const svbool_t pg = svptrue_b8();
  int i = 0;
  for (; i + 64 <= n; i += 64) {
    for (int j = 0; j < 64; j += W) {
      const svbool_t m = svcmplt_n_s8(pg, svld1_s8(pg, v + i + j), 0);
      __asm__ volatile("str %[m], [%[a]]" : : [m] "Upa"(m), [a] "r"(out + ((i + j) >> 3)) : "memory");
    }
    uint64_t x;
    __builtin_memcpy(&x, out + (i >> 3), 8);
    __asm__("rbit %0, %0\n\trev %0, %0" : "+r"(x)); /* LSB-first bits -> MSB first within each byte */
    __builtin_memcpy(out + (i >> 3), &x, 8);
  }
  return i;
}

/* input gain, byte SQRDMULH form of ldpc_gain1(): k = 7 - t >= 0: SQSHL #k then SQRDMULH; k < 0: SQRDMULH then SRSHL
   by k (a rounding right shift); clamp; predicated tail */
void ldpc_sve2_gain_run(int8_t *dst, const int8_t *src, int n, int cmax, int G, int t)
{
  const int k = 7 - t;
  const svint8_t g = svdup_n_s8((int8_t)G), c = svdup_n_s8((int8_t)cmax), nc = svdup_n_s8((int8_t)-cmax);
  const svint8_t kv = svdup_n_s8((int8_t)k);
  const int W = (int)svcntb();
  for (int i = 0; i < n; i += W) {
    const svbool_t pg = svwhilelt_b8_s32(i, n);
    svint8_t v = svld1_s8(pg, src + i);
    if (k > 0) v = svqshl_s8_x(pg, v, kv);
    v = svqrdmulh_s8(v, g);
    if (k < 0) v = svrshl_s8_x(pg, v, kv);
    svst1_s8(pg, dst + i, svmax_s8_x(pg, svmin_s8_x(pg, v, c), nc));
  }
}

int ldpc_kern_sve2(ldpc_kernels_t *k, int *W)
{
  *W = (int)svcntb();
  kern_fill_sve2(k);
  return 1;
}
#else
int ldpc_sve2_bytes(void) { return 0; }
int ldpc_sve2_pack_bits(const int8_t *v, int n, uint8_t *out) { (void)v; (void)n; (void)out; return 0; }
void ldpc_sve2_gain_run(int8_t *dst, const int8_t *src, int n, int cmax, int G, int t) { (void)dst; (void)src; (void)n; (void)cmax; (void)G; (void)t; }
int ldpc_kern_sve2(ldpc_kernels_t *k, int *W) { (void)k; (void)W; return 0; }
#endif
