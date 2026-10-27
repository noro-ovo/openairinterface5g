/* Portable C instantiation of the kernel template (8-lane "vectors" of plain int8). Used where no SIMD
 * kernel is compiled in, and as a native check of the template logic itself. */
#include <string.h>
#include "ldpc_internal.h"

typedef struct { int8_t v[8]; } v8_t;
#define SW 8
static inline v8_t s_ld(const int8_t *p) { v8_t r; memcpy(r.v, p, SW); return r; }
static inline void s_st(int8_t *p, v8_t a) { memcpy(p, a.v, SW); }
static inline int8_t s_sat(int x) { return (int8_t)(x > 127 ? 127 : x < -128 ? -128 : x); }
#define S_OP2(name, expr)                                    \
  static inline v8_t name(v8_t a, v8_t b)                    \
  {                                                          \
    v8_t r;                                                  \
    for (int i = 0; i < SW; i++) {                           \
      const int x = a.v[i], y = b.v[i];                      \
      const unsigned ux = (uint8_t)a.v[i], uy = (uint8_t)b.v[i]; \
      (void)x; (void)y; (void)ux; (void)uy;                  \
      r.v[i] = (int8_t)(expr);                               \
    }                                                        \
    return r;                                                \
  }
S_OP2(s_qsub, s_sat(x - y))
S_OP2(s_qadd, s_sat(x + y))
S_OP2(s_wsub, (int8_t)(x - y))
S_OP2(s_minu, ux < uy ? ux : uy)
S_OP2(s_maxu, ux > uy ? ux : uy)
S_OP2(s_xor, x ^ y)
S_OP2(s_and, x & y)
S_OP2(s_or, x | y)
static inline v8_t s_abs(v8_t a) { v8_t r; for (int i = 0; i < SW; i++) r.v[i] = (int8_t)(a.v[i] < 0 ? -a.v[i] : a.v[i]); return r; }
static inline v8_t s_dup(int8_t x) { v8_t r; memset(r.v, x, SW); return r; }
static inline v8_t s_abd(v8_t a, v8_t b) { v8_t r; for (int i = 0; i < SW; i++) { const int d = a.v[i] - b.v[i]; r.v[i] = (int8_t)(uint8_t)(d < 0 ? -d : d); } return r; }
static inline v8_t s_corr(v8_t m, int nsh, int beta) /* m <= lim: no rmax clamp (README section 5) */
{
  v8_t r;
  for (int i = 0; i < SW; i++) {
    unsigned x = (uint8_t)m.v[i];
    if (nsh) x -= x >> nsh;
    r.v[i] = (int8_t)(x > (unsigned)beta ? x - (unsigned)beta : 0);
  }
  return r;
}
static inline v8_t s_negs(v8_t c, v8_t s) { v8_t r; for (int i = 0; i < SW; i++) r.v[i] = (int8_t)(s.v[i] < 0 ? -c.v[i] : c.v[i]); return r; }
static inline v8_t s_msg(v8_t q, v8_t m1c, v8_t c1, v8_t c2)
{
  v8_t r;
  for (int i = 0; i < SW; i++) r.v[i] = ldpc_msg1(q.v[i], (uint8_t)m1c.v[i], c1.v[i], c2.v[i]);
  return r;
}
static inline int s_anyneg(v8_t t) { for (int i = 0; i < SW; i++) if (t.v[i] < 0) return 1; return 0; }

#define KSUF scalar
#define KATTR
#define KREGS 16 /* U = 1 only: keeps this portable fallback small; U > 1 is covered by the SIMD kernels */
#define VT v8_t
#define VW SW
#define KSETUP(p) const int nsh_ = (p)->nshift, beta_ = (p)->beta; (void)nsh_; (void)beta_
#define VLD(p) s_ld((const int8_t *)(p))
#define VST(p, v) s_st((int8_t *)(p), v)
#define VQSUB s_qsub
#define VQADD s_qadd
#define VSUBW s_wsub /* wrapping subtraction (absorbed messages) */
#define VABS s_abs
#define VABD s_abd
#define VMINU s_minu
#define VMAXU s_maxu
#define VXOR s_xor
#define VAND s_and
#define VOR s_or
#define VZERO s_dup(0)
#define VLIM s_dup((int8_t)p->lim)
#define VCORR(m) s_corr(m, nsh_, beta_)
#define KSG_T v8_t
#define VSGN(sg) (sg)
#define VNEGS s_negs
#define VEQP(m1c) (m1c)
#define VMSG(q, m1c, eqp, c1, c2) s_msg(q, m1c, c1, c2)
static inline v8_t s_msga(v8_t q, v8_t ab, v8_t m1c, v8_t c1, v8_t c2)
{
  v8_t r;
  for (int i = 0; i < SW; i++) {
    const int s = (uint8_t)ab.v[i] == (uint8_t)m1c.v[i] ? c2.v[i] : c1.v[i];
    r.v[i] = (int8_t)(q.v[i] < 0 ? -s : s);
  }
  return r;
}
#define VMSGA s_msga
#define VANYNEG s_anyneg

#include "kern_tmpl.h"

int ldpc_kern_scalar(ldpc_kernels_t *k, int *W)
{
  *W = SW;
  kern_fill_scalar(k);
  return 1;
}
