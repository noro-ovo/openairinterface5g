/* AVX2 instantiation of the kernel template: 32 checks per vector; PSIGNB applies the message sign.
 * 16 vector registers: every row shape uses one vector per step (the compiler spills the Q of the 19-edge rows). */
#include "ldpc_internal.h"
#if defined(__x86_64__) || defined(__i386__)
#pragma GCC target("avx2")
#include <immintrin.h>

static inline __m256i y_corr(__m256i m, int nsh, __m128i cnt, __m256i shm, __m256i vb)
{
  if (nsh) m = _mm256_sub_epi8(m, _mm256_and_si256(_mm256_srl_epi16(m, cnt), shm));
  return _mm256_subs_epu8(m, vb);
}
/* sign_epi8(c, s|1): negate where s < 0, never zero the value */
static inline __m256i y_msg(__m256i q, __m256i m1c, __m256i c1, __m256i c2)
{
  const __m256i eq = _mm256_cmpeq_epi8(_mm256_abs_epi8(q), m1c);
  return _mm256_sign_epi8(_mm256_blendv_epi8(c1, c2, eq), _mm256_or_si256(q, _mm256_set1_epi8(1)));
}

#define KSUF avx2
#define KATTR
#define KREGS 16
#define VT __m256i
#define VW 32
#define KSETUP(p)                                                             \
  const int nsh_ = (p)->nshift;                                               \
  const __m128i cnt_ = _mm_cvtsi32_si128(nsh_);                               \
  const __m256i shm_ = _mm256_set1_epi8((char)(0xFF >> nsh_));                \
  const __m256i vb_ = _mm256_set1_epi8((char)(p)->beta);                      \
  (void)cnt_; (void)shm_; (void)vb_
#define VLD(p) _mm256_loadu_si256((const __m256i *)(const void *)(p))
#define VST(p, v) _mm256_storeu_si256((__m256i *)(void *)(p), v)
#define VQSUB _mm256_subs_epi8
#define VQADD _mm256_adds_epi8
#define VSUBW _mm256_sub_epi8 /* wrapping subtraction (absorbed messages) */
#define VABS _mm256_abs_epi8
#define VABD(a, b) _mm256_abs_epi8(_mm256_subs_epi8(a, b))
#define VMINU _mm256_min_epu8
#define VMAXU _mm256_max_epu8
#define VXOR _mm256_xor_si256
#define VAND _mm256_and_si256
#define VOR _mm256_or_si256
#define VZERO _mm256_setzero_si256()
#define VLIM _mm256_set1_epi8((char)p->lim)
#define VCORR(m) y_corr(m, nsh_, cnt_, shm_, vb_)
#define KSG_T __m256i
#define VSGN(sg) _mm256_or_si256(sg, _mm256_set1_epi8(1))
#define VNEGS(c, s) _mm256_sign_epi8(c, s)
#define VEQP(m1c) (m1c)
#define VMSG(q, m1c, eqp, c1, c2) y_msg(q, m1c, c1, c2)
#define VMSGA(q, ab, m1c, c1, c2) _mm256_sign_epi8(_mm256_blendv_epi8(c1, c2, _mm256_cmpeq_epi8(ab, m1c)), _mm256_or_si256(q, _mm256_set1_epi8(1)))
#define VANYNEG(t) (_mm256_movemask_epi8(t) != 0)

#include "kern_tmpl.h"

int ldpc_kern_avx2(ldpc_kernels_t *k, int *W)
{
  *W = 32;
  kern_fill_avx2(k);
  return 1;
}
#else
int ldpc_kern_avx2(ldpc_kernels_t *k, int *W) { (void)k; (void)W; return 0; }
#endif
