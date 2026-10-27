/* NEON (ARMv8.0-A baseline) instantiation of the kernel template: 16 checks per vector. kern_neon3.c includes this
   file again with LDPC_NEON_SHA3 and the +sha3 target (GCC then folds pairs of XORs of the sign accumulator and the
   syndrome into EOR3); ldpc_kern_neon() selects it when HWCAP_SHA3 is set. */
#include <stdlib.h>
#include "ldpc_internal.h"
#if defined(__aarch64__)
#include <arm_neon.h>
#if !defined(LDPC_NEON_SHA3) && defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

#define U8(x) vreinterpretq_u8_s8(x)
#define S8(x) vreinterpretq_s8_u8(x)
static inline int8x16_t n_corr(int8x16_t m, int nsh, int8x16_t nshv, uint8x16_t vb)
{
  uint8x16_t x = U8(m);
  if (nsh) x = vsubq_u8(x, vshlq_u8(x, nshv));
  return S8(vqsubq_u8(x, vb));
}
static inline int8x16_t n_negs(int8x16_t c, int8x16_t smask) { return vbslq_s8(U8(smask), vnegq_s8(c), c); }
static inline int8x16_t n_msg(int8x16_t q, int8x16_t m1c, int8x16_t c1, int8x16_t c2)
{
  const int8x16_t s = vbslq_s8(vceqq_s8(vabsq_s8(q), m1c), c2, c1);
  return vbslq_s8(vcltzq_s8(q), vnegq_s8(s), s);
}

#ifdef LDPC_NEON_SHA3
#define KSUF neon3
#else
#define KSUF neon
#endif
#define KATTR
#define KREGS 32
#define VT int8x16_t
#define VW 16
#define KSETUP(p)                                          \
  const int nsh_ = (p)->nshift;                            \
  const int8x16_t nshv_ = vdupq_n_s8((int8_t)-nsh_);       \
  const uint8x16_t vb_ = vdupq_n_u8((uint8_t)(p)->beta);   \
  const int8x16_t vlim_ = vdupq_n_s8((int8_t)(p)->lim);    \
  (void)nshv_; (void)vb_; (void)vlim_
#define VLD(p) vld1q_s8((const int8_t *)(p))
#define VST(p, v) vst1q_s8((int8_t *)(p), v)
#define VQSUB vqsubq_s8
#define VQADD vqaddq_s8
#define VSUBW vsubq_s8 /* wrapping subtraction (absorbed messages) */
#define VABS vabsq_s8
#define VABD(a, b) S8(vreinterpretq_u8_s8(vabdq_s8(a, b)))
#define VMINU(a, b) S8(vminq_u8(U8(a), U8(b)))
#define VMAXU(a, b) S8(vmaxq_u8(U8(a), U8(b)))
#define VXOR veorq_s8
#define VAND vandq_s8
#define VOR vorrq_s8
#define VZERO vdupq_n_s8(0)
#define VLIM vlim_
#define VCORR(m) n_corr(m, nsh_, nshv_, vb_)
#define KSG_T int8x16_t
#define VSGN(sg) vreinterpretq_s8_u8(vcltzq_s8(sg))
#define VNEGS n_negs
#define VEQP(m1c) (m1c)
#define VMSG(q, m1c, eqp, c1, c2) n_msg(q, eqp, c1, c2)
static inline int8x16_t n_msga(int8x16_t q, int8x16_t ab, int8x16_t m1c, int8x16_t c1, int8x16_t c2)
{
  const int8x16_t s = vbslq_s8(vceqq_s8(ab, m1c), c2, c1);
  return vbslq_s8(vcltzq_s8(q), vnegq_s8(s), s);
}
#define VMSGA n_msga
#define VANYNEG(t) (vminvq_s8(t) < 0)

#include "kern_tmpl.h"

#ifdef LDPC_NEON_SHA3
int ldpc_kern_neon3(ldpc_kernels_t *k, int *W)
{
  *W = 16;
  kern_fill_neon3(k);
  return 1;
}
#else
int ldpc_kern_neon3(ldpc_kernels_t *k, int *W);
int ldpc_kern_neon(ldpc_kernels_t *k, int *W)
{
#if defined(__linux__) && defined(HWCAP_SHA3)
  static int sha3 = -1;
  if (sha3 < 0) {
    const char *e = getenv("LDPC_DEC_NOSHA3"); /* measurements: plain ARMv8.0 NEON kernel */
    sha3 = (getauxval(AT_HWCAP) & HWCAP_SHA3) && !(e && atoi(e));
  }
  if (sha3) return ldpc_kern_neon3(k, W);
#endif
  *W = 16;
  kern_fill_neon(k);
  return 1;
}
#endif
#elif !defined(LDPC_NEON_SHA3)
int ldpc_kern_neon(ldpc_kernels_t *k, int *W) { (void)k; (void)W; return 0; }
#endif
