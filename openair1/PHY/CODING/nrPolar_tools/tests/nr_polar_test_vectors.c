/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*
 * Deterministic polar encode / decode sequences over the configurations OAI's callers use (PBCH, PSBCH, DCI A 12..64
 * x AL 1..16, PUCCH format 2 A 12..64 x nPRB 3..16 with K <= E, UCI on PUSCH), for test_nr_polar:
 *  - nr_polar_test_roundtrip(): a clean codeword (LLR +-2) decodes to its payload with a passing CRC;
 *  - nr_polar_test_digest(): FNV-1a hash of every encoder output and every decoder result (return value, payload) for
 *    random payloads and LLR sets including saturated values (-32768/32767), zeros and noise, which pins the exact
 *    behaviour (saturation, ties, partial output bytes) of the original implementation.
 * Built against another implementation of the same API with POLAR_TEST_PREFIX (e.g. old_), it gives that
 * implementation's digest.
 */

#include <stdint.h>
#include <string.h>
#include "PHY/CODING/nrPolar_tools/nr_polar_defs.h"
#include "PHY/CODING/nrPolar_tools/nr_polar_psbch_defs.h"

#ifdef POLAR_TEST_PREFIX
#define PT_CAT2(a, b) a##b
#define PT_CAT(a, b) PT_CAT2(a, b)
#define PT_ENC PT_CAT(POLAR_TEST_PREFIX, polar_encoder_fast)
#define PT_DEC PT_CAT(POLAR_TEST_PREFIX, polar_decoder_int16)
void PT_ENC(uint64_t *, void *, int32_t, uint8_t, int8_t, uint16_t, uint8_t);
uint32_t PT_DEC(const int16_t *, uint64_t *, uint8_t, int8_t, uint16_t, uint8_t);
#else
#define PT_ENC polar_encoder_fast
#define PT_DEC polar_decoder_int16
#endif

typedef struct {
  int8_t type;
  uint16_t A;
  uint16_t agg;
} pt_cfg_t;

static int pt_E(const pt_cfg_t *c)
{
  switch (c->type) {
    case NR_POLAR_PBCH_MESSAGE_TYPE:
      return NR_POLAR_PBCH_E;
    case SL_NR_POLAR_PSBCH_MESSAGE_TYPE:
      return SL_NR_POLAR_PSBCH_E_NORMAL_CP + 2;
    case NR_POLAR_DCI_MESSAGE_TYPE:
      return 108 * c->agg;
    case NR_POLAR_UCI_PUCCH_MESSAGE_TYPE:
      return 16 * c->agg;
    default:
      return c->agg;
  }
}

static int pt_configs(pt_cfg_t *c)
{
  int n = 0;
  c[n++] = (pt_cfg_t){NR_POLAR_PBCH_MESSAGE_TYPE, NR_POLAR_PBCH_PAYLOAD_BITS, 0};
  c[n++] = (pt_cfg_t){SL_NR_POLAR_PSBCH_MESSAGE_TYPE, SL_NR_POLAR_PSBCH_PAYLOAD_BITS, 0};
  for (int A = 12; A <= 64; A++)
    for (int L = 1; L <= 16; L *= 2)
      c[n++] = (pt_cfg_t){NR_POLAR_DCI_MESSAGE_TYPE, A, L};
  for (int A = 12; A <= 64; A++)
    for (int p = 3; p <= 16; p++)
      if (A + (A <= 19 ? 6 : 11) <= 16 * p)
        c[n++] = (pt_cfg_t){NR_POLAR_UCI_PUCCH_MESSAGE_TYPE, A, p};
  static const int pusch_A[] = {12, 19, 20, 33, 64}, pusch_E[] = {96, 128, 160, 200, 240, 255};
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 6; j++)
      c[n++] = (pt_cfg_t){NR_POLAR_UCI_PUSCH_MESSAGE_TYPE, pusch_A[i], pusch_E[j]};
  return n;
}

static uint64_t pt_rnd(uint64_t *s)
{
  *s ^= *s << 13;
  *s ^= *s >> 7;
  *s ^= *s << 17;
  return *s;
}

static void pt_hash(uint64_t *h, const void *p, size_t n)
{
  const uint8_t *b = p;
  for (size_t i = 0; i < n; i++)
    *h = (*h ^ b[i]) * 0x100000001b3ULL;
}

static int16_t pt_sat(int x)
{
  return x > 32767 ? 32767 : x < -32768 ? -32768 : x;
}

/* encodes a random payload of c (CRC mask = RNTI for DCI), returns the codeword in cw (E bits, LSB first) */
static void pt_encode(const pt_cfg_t *c, uint64_t *s, uint64_t a[2], uint32_t *rnti, uint8_t *cw, size_t cwlen)
{
  a[0] = pt_rnd(s);
  a[1] = 0;
  if (c->A < 64)
    a[0] &= (1ULL << c->A) - 1;
  *rnti = c->type == NR_POLAR_DCI_MESSAGE_TYPE ? pt_rnd(s) & 0xffff : 0;
  memset(cw, 0, cwlen);
  uint64_t in[2] = {a[0], a[1]};
  PT_ENC(in, cw, *rnti, c->type == NR_POLAR_DCI_MESSAGE_TYPE, c->type, c->A, c->agg);
}

int nr_polar_test_roundtrip(int *ncfg)
{
  static pt_cfg_t cfg[1200];
  const int n = pt_configs(cfg);
  *ncfg = n;
  uint64_t s = 0x2545F4914F6CDD1DULL;
  int fails = 0;
  for (int i = 0; i < n; i++) {
    const pt_cfg_t *c = &cfg[i];
    const int E = pt_E(c);
    // the coded bit interleaver of polar_encoder_fast() garbles the last partial byte when E is not a multiple of 8
    // (UCI on PUSCH; OAI's UE encodes UCI with PUCCH parameters, E = 16 nPRB)
    if (E & 7)
      continue;
    for (int t = 0; t < 4; t++) {
      uint64_t a[2];
      uint32_t rnti;
      uint8_t cw[256];
      pt_encode(c, &s, a, &rnti, cw, sizeof(cw));
      int16_t llr[2048];
      // small amplitude: G doubles the LLR magnitude per tree level, saturation to -32768 breaks the decoder
      for (int k = 0; k < E; k++)
        llr[k] = (cw[k >> 3] >> (k & 7)) & 1 ? -2 : 2;
      uint64_t out[2] = {0, 0};
      const uint32_t ret = PT_DEC(llr, out, c->type == NR_POLAR_DCI_MESSAGE_TYPE, c->type, c->A, c->agg);
      if (ret != rnti || out[0] != a[0])
        fails++;
    }
  }
  return fails;
}

uint64_t nr_polar_test_digest(int *ncfg)
{
  static pt_cfg_t cfg[1200];
  const int n = pt_configs(cfg);
  *ncfg = n;
  uint64_t s = 0x9E3779B97F4A7C15ULL, h = 0xcbf29ce484222325ULL;
  static const int16_t ex[] = {-32768, 32767, -32767, 0, 1, -1, 16384, -16384};
  for (int i = 0; i < n; i++) {
    const pt_cfg_t *c = &cfg[i];
    const int E = pt_E(c);
    for (int model = 0; model < 4; model++) {
      uint64_t a[2];
      uint32_t rnti;
      uint8_t cw[256];
      pt_encode(c, &s, a, &rnti, cw, sizeof(cw));
      pt_hash(&h, cw, (E + 7) / 8);
      const int scale = 1 << (pt_rnd(&s) % 16); // 1 .. 32768: up to saturation
      int16_t llr[2048];
      for (int k = 0; k < E; k++) {
        const int sgn = (cw[k >> 3] >> (k & 7)) & 1 ? -1 : 1;
        const uint64_t r = pt_rnd(&s);
        switch (model) {
          case 0: // noisy codeword
            llr[k] = pt_sat(sgn * scale + ((int)(r % 65536) - 32768) * scale / 32768);
            break;
          case 1: // uniform
            llr[k] = (int16_t)r;
            break;
          case 2: // extreme and small values
            llr[k] = ex[r & 7];
            break;
          default: // saturated codeword with erasures
            llr[k] = r % 8 == 0 ? 0 : sgn > 0 ? 32767 : (r & 8 ? -32768 : -32767);
            break;
        }
      }
      uint64_t out[2] = {0, 0};
      const uint32_t ret = PT_DEC(llr, out, c->type == NR_POLAR_DCI_MESSAGE_TYPE, c->type, c->A, c->agg);
      pt_hash(&h, &ret, sizeof(ret));
      pt_hash(&h, out, sizeof(out[0]));
    }
  }
  return h;
}
