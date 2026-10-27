/* AVX-512 (F+BW) instantiation of the kernel template: 64 checks per vector, k-masks for compares and sign. */
#include "ldpc_internal.h"
#if defined(__x86_64__) || defined(__i386__)
#pragma GCC target("avx512f,avx512bw,avx512vl,avx2")
#include <immintrin.h>

static inline __m512i z_corr(__m512i m, int nsh, __m128i cnt, __m512i shm, __m512i vb)
{
  if (nsh) m = _mm512_sub_epi8(m, _mm512_and_si512(_mm512_srl_epi16(m, cnt), shm));
  return _mm512_subs_epu8(m, vb);
}
static inline __m512i z_msg(__m512i q, __m512i m1c, __m512i c1, __m512i c2)
{
  const __mmask64 eq = _mm512_cmpeq_epi8_mask(_mm512_abs_epi8(q), m1c);
  const __m512i s = _mm512_mask_blend_epi8(eq, c1, c2);
  return _mm512_mask_sub_epi8(s, _mm512_movepi8_mask(q), _mm512_setzero_si512(), s);
}

#define KSUF avx512
#define KATTR
#define KREGS 32
#define VT __m512i
#define VW 64
#define KSETUP(p)                                                             \
  const int nsh_ = (p)->nshift;                                               \
  const __m128i cnt_ = _mm_cvtsi32_si128(nsh_);                               \
  const __m512i shm_ = _mm512_set1_epi8((char)(0xFF >> nsh_));                \
  const __m512i vb_ = _mm512_set1_epi8((char)(p)->beta);                      \
  (void)cnt_; (void)shm_; (void)vb_
#define VLD(p) _mm512_loadu_si512((const void *)(p))
#define VST(p, v) _mm512_storeu_si512((void *)(p), v)
#define VQSUB _mm512_subs_epi8
#define VQADD _mm512_adds_epi8
#define VABS _mm512_abs_epi8
#define VABD(a, b) _mm512_abs_epi8(_mm512_subs_epi8(a, b))
#define VMINU _mm512_min_epu8
#define VMAXU _mm512_max_epu8
#define VXOR _mm512_xor_si512
#define VAND _mm512_and_si512
#define VOR _mm512_or_si512
#define VZERO _mm512_setzero_si512()
#define VLIM _mm512_set1_epi8((char)p->lim)
#define VCORR(m) z_corr(m, nsh_, cnt_, shm_, vb_)
#define KSG_T __mmask64
#define VSGN(sg) _mm512_movepi8_mask(sg)
#define VNEGS(c, s) _mm512_mask_sub_epi8(c, s, _mm512_setzero_si512(), c)
#define VEQP(m1c) (m1c)
#define VMSG(q, m1c, eqp, c1, c2) z_msg(q, m1c, c1, c2)
static inline __m512i z_msga(__m512i q, __m512i ab, __m512i m1c, __m512i c1, __m512i c2)
{
  const __m512i s = _mm512_mask_blend_epi8(_mm512_cmpeq_epi8_mask(ab, m1c), c1, c2);
  return _mm512_mask_sub_epi8(s, _mm512_movepi8_mask(q), _mm512_setzero_si512(), s);
}
#define VMSGA z_msga
#define VANYNEG(t) (_mm512_movepi8_mask(t) != 0)

#include "kern_tmpl.h"

int ldpc_kern_avx512(ldpc_kernels_t *k, int *W)
{
  *W = 64;
  kern_fill_avx512(k);
  return 1;
}
#else
int ldpc_kern_avx512(ldpc_kernels_t *k, int *W) { (void)k; (void)W; return 0; }
#endif
