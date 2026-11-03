/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "nr_transport_common_proto.h"
#include "PHY/NR_REFSIG/nr_refsig.h"
#define DEBUG_SCRAMBLING(a)
//#define DEBUG_SCRAMBLING(a) a
const uint32_t *nr_codeword_scrambling_seq(uint32_t size, uint8_t q, uint32_t Nid, uint32_t n_RNTI)
{
  return gold_cache((n_RNTI << 15) + (q << 14) + Nid, (size + 31) / 32);
}

void nr_codeword_scrambling(uint8_t *in,
                            uint32_t size,
                            uint8_t q,
                            uint32_t Nid,
                            uint32_t n_RNTI,
                            uint32_t* out)
{
  const int roundedSz = (size + 31) / 32;
  const uint32_t *seq = nr_codeword_scrambling_seq(size, q, Nid, n_RNTI);
  unsigned int i_32 = 0;
#ifdef __AVX2__
  for (; i_32 < ((roundedSz >> 3) << 3); i_32 += 8) {
    __m256i in_256 = _mm256_loadu_si256((void *)&((uint32_t *)in)[i_32]);
    __m256i seq_256 = _mm256_loadu_si256((void *)&seq[i_32]);
    _mm256_storeu_si256((void *)&out[i_32], _mm256_xor_si256(in_256, seq_256));
  }
#endif
#if defined(__aarch64__)
  for (; i_32 < ((roundedSz >> 2) << 2); i_32 += 4) {
    unsigned int i = i_32 >> 2;
    uint32x4_t in_32x4 = vld1q_u32(&((uint32_t *)in)[i_32]);
    uint32x4_t seq_32x4 = vld1q_u32(&seq[i_32]);
    ((uint32x4_t *)out)[i] = veorq_u32(in_32x4, seq_32x4);
  }
#endif
  for (; i_32 < roundedSz; i_32++) { 
    unsigned int i = i_32;
    out[i] = ((uint32_t *)in)[i] ^ seq[i];
    DEBUG_SCRAMBLING(LOG_D(PHY, "in[%d] %x => %x\n", i, ((uint32_t*)in)[i], out[i]));
  }
}

void nr_codeword_unscrambling(int16_t* llr, uint32_t size, uint8_t q, uint32_t Nid, uint32_t n_RNTI)
{
  const int roundedSz = (size + 31) / 32;
  const uint32_t *seq = gold_cache((n_RNTI << 15) + (q << 14) + Nid, roundedSz);
  nr_codeword_unscrambling_bits(llr, seq, 0, size, llr);
}

/* out[i] = -in[i] if bit first_bit + i of the Gold sequence seq is 1, else in[i] (i < n; out may be in).
 * AVX-512: 32 bits of the sequence go to a mask register and VPSUBW 0 - x under that mask negates the 32 LLRs,
 * 4 x 32 LLRs per step: no +-1 sequence array. -32768 stays -32768, as with the multiply by -1. */
void nr_codeword_unscrambling_bits(const int16_t *in, const uint32_t *seq, uint32_t first_bit, uint32_t n, int16_t *out)
{
  uint32_t i = 0;
#if defined(__AVX512BW__)
  const __m512i zero = _mm512_setzero_si512();
  /* 32 sequence bits starting at bit b; the 64-bit read stays inside the words that hold bits b .. b + 31 when the
     last of them is not the last word: only used while b + 63 < first_bit + n rounded to whole words */
#define SEQ32(b) ((uint32_t)((((uint64_t)seq[((b) >> 5) + 1] << 32) | seq[(b) >> 5]) >> ((b) & 31)))
  const uint32_t last_word = (first_bit + n - 1) >> 5;
  for (; i + 128 <= n && ((first_bit + i + 127) >> 5) + 1 <= last_word; i += 128) {
    const uint32_t b = first_bit + i;
    const __mmask32 k0 = SEQ32(b), k1 = SEQ32(b + 32), k2 = SEQ32(b + 64), k3 = SEQ32(b + 96);
    const __m512i x0 = _mm512_loadu_si512(in + i), x1 = _mm512_loadu_si512(in + i + 32);
    const __m512i x2 = _mm512_loadu_si512(in + i + 64), x3 = _mm512_loadu_si512(in + i + 96);
    _mm512_storeu_si512(out + i, _mm512_mask_sub_epi16(x0, k0, zero, x0));
    _mm512_storeu_si512(out + i + 32, _mm512_mask_sub_epi16(x1, k1, zero, x1));
    _mm512_storeu_si512(out + i + 64, _mm512_mask_sub_epi16(x2, k2, zero, x2));
    _mm512_storeu_si512(out + i + 96, _mm512_mask_sub_epi16(x3, k3, zero, x3));
  }
  for (; i < n; i += 32) {
    const uint32_t b = first_bit + i, w = b >> 5, sh = b & 31, cnt = n - i < 32 ? n - i : 32;
    uint64_t v = seq[w];
    if (w + 1 <= last_word)
      v |= (uint64_t)seq[w + 1] << 32;
    const __mmask32 km = cnt == 32 ? 0xffffffffu : (1u << cnt) - 1, k = (uint32_t)(v >> sh) & km;
    const __m512i x = _mm512_maskz_loadu_epi16(km, in + i);
    _mm512_mask_storeu_epi16(out + i, km, _mm512_mask_sub_epi16(x, k, zero, x));
  }
#undef SEQ32
#endif
  for (; i < n; i++) {
    const uint32_t b = first_bit + i;
    out[i] = (seq[b >> 5] >> (b & 31)) & 1 ? -in[i] : in[i];
  }
}

const uint32_t *nr_codeword_scrambling_gold(uint32_t size, uint8_t q, uint32_t Nid, uint32_t n_RNTI)
{
  return gold_cache((n_RNTI << 15) + (q << 14) + Nid, (size + 31) / 32);
}

void nr_codeword_unscrambling_expand(const uint32_t *seq, int nwords, int16_t *s)
{
  simde__m128i *s128 = (simde__m128i *)s;
  for (int i = 0; i < nwords; i++) {
    uint8_t *s8 = (uint8_t *)(seq + i);
    *s128++ = byte2m128i[s8[0]];
    *s128++ = byte2m128i[s8[1]];
    *s128++ = byte2m128i[s8[2]];
    *s128++ = byte2m128i[s8[3]];
  }
}
