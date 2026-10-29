/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*! \file PHY/CODING/nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_llr_prescale.c
 * \brief Qm-dependent int16 LLR gain fused into the saturating int16 -> int8 pack (see nrLDPC_llr_prescale.h)
 */

#include "nrLDPC_llr_prescale.h"
#include "PHY/sse_intrin.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

static int set_gain(nr_ldpc_llr_prescale_t *p, double g)
{
  memset(p, 0, sizeof(*p));
  p->q8 = 256;
  if (g == 1.0)
    return 0;
  if (g >= 1.0 / 16 && g < 1.0) {
    long q = lrint(g * 32768.0);
    p->mode = NR_LDPC_LLR_PRESCALE_FRAC;
    p->q15 = q > 32767 ? 32767 : (int16_t)q;
    p->q8 = (uint16_t)lrint(g * 256.0);
    return 0;
  }
  for (int s = 1; s <= 3; s++)
    if (g == (double)(1 << s)) {
      p->mode = NR_LDPC_LLR_PRESCALE_LSHIFT;
      p->lshift = s;
      p->q8 = (uint16_t)(256 << s);
      return 0;
    }
  return -1;
}

static void set_all_x1(nr_ldpc_llr_prescale_rule_t *r)
{
  for (int l = 0; l < 2; l++)
    for (int q = 0; q <= NR_LDPC_LLR_PRESCALE_MAX_QM; q++)
      set_gain(&r->qm[l][q], 1.0);
}

// "g2,g4,g6,g8" into the Qm 2/4/6/8 entries of t; returns the number of characters consumed, -1 on error
static int parse_set(const char *s, nr_ldpc_llr_prescale_t *t)
{
  double g[4];
  int n = 0;
  if (sscanf(s, " %lf ,%lf ,%lf ,%lf %n", &g[0], &g[1], &g[2], &g[3], &n) != 4 || n == 0)
    return -1;
  for (int i = 0; i < 4; i++)
    if (set_gain(&t[2 * (i + 1)], g[i]) != 0)
      return -1;
  return n;
}

int nrLDPC_llr_prescale_parse(const char *str, nr_ldpc_llr_prescale_rule_t *rule)
{
  set_all_x1(rule);
  if (str == NULL)
    return -1;
  while (*str == ' ')
    str++;
  if (!strcasecmp(str, "off") || !strcasecmp(str, "none") || !strcmp(str, "0"))
    return 0;
  if (!strcasecmp(str, "default") || !strcasecmp(str, "on"))
    str = NR_LDPC_LLR_PRESCALE_DEFAULT;
  nr_ldpc_llr_prescale_rule_t r;
  set_all_x1(&r);
  int n = parse_set(str, r.qm[0]);
  if (n < 0)
    return -1;
  str += n;
  if (*str == '/') {
    n = parse_set(str + 1, r.qm[1]);
    if (n < 0)
      return -1;
    str += 1 + n;
  } else {
    memcpy(r.qm[1], r.qm[0], sizeof(r.qm[1]));
  }
  if (*str != 0)
    return -1;
  *rule = r;
  return 0;
}

void nrLDPC_llr_prescale_str(const nr_ldpc_llr_prescale_rule_t *rule, char *buf, int len)
{
  static const char *name[4] = {"QPSK", "16QAM", "64QAM", "256QAM"};
  static const char *lay[2] = {"1 layer:", "; 2+ layers:"};
  int o = 0;
  buf[0] = 0;
  for (int l = 0; l < 2; l++)
    for (int i = 0; i < 4 && o < len; i++)
      o += snprintf(buf + o, len - o, "%s %s x%g", i ? "" : lay[l], name[i], rule->qm[l][2 * (i + 1)].q8 / 256.0);
}

void nrLDPC_llr_pack_int8(const int16_t *z, int8_t *l, int nvec, const nr_ldpc_llr_prescale_t *ps)
{
  const simde__m128i *pv = (const simde__m128i *)z;
  simde__m128i *pl = (simde__m128i *)l;
  const int mode = ps ? ps->mode : NR_LDPC_LLR_PRESCALE_NONE;

  if (mode == NR_LDPC_LLR_PRESCALE_FRAC) {
    // y = sign(x) * round(|x| * gain): mulhrs on |x| rounds half away from zero, so the gain is symmetric in the
    // LLR sign; x = -32768 is first limited to -32767 so that |x| fits in int16. 256-bit vectors (the in-lane packs
    // is reordered by the 64-bit permute), 128-bit tail.
    const simde__m256i m = simde_mm256_set1_epi16(ps->q15);
    const simde__m256i lo = simde_mm256_set1_epi16(-32767);
    int j = 0;
    for (; j + 2 <= nvec; j += 2) {
      const simde__m256i x0 = simde_mm256_max_epi16(simde_mm256_loadu_si256((const simde__m256i *)(z + 16 * j)), lo);
      const simde__m256i x1 = simde_mm256_max_epi16(simde_mm256_loadu_si256((const simde__m256i *)(z + 16 * j + 16)), lo);
      const simde__m256i y0 = simde_mm256_sign_epi16(simde_mm256_mulhrs_epi16(simde_mm256_abs_epi16(x0), m), x0);
      const simde__m256i y1 = simde_mm256_sign_epi16(simde_mm256_mulhrs_epi16(simde_mm256_abs_epi16(x1), m), x1);
      simde_mm256_storeu_si256((simde__m256i *)(l + 16 * j), simde_mm256_permute4x64_epi64(simde_mm256_packs_epi16(y0, y1), 0xd8));
    }
    if (j < nvec) {
      const simde__m128i m1 = simde_mm256_castsi256_si128(m);
      const simde__m128i lo1 = simde_mm256_castsi256_si128(lo);
      const simde__m128i x0 = simde_mm_max_epi16(pv[2 * j], lo1);
      const simde__m128i x1 = simde_mm_max_epi16(pv[2 * j + 1], lo1);
      const simde__m128i y0 = simde_mm_sign_epi16(simde_mm_mulhrs_epi16(simde_mm_abs_epi16(x0), m1), x0);
      const simde__m128i y1 = simde_mm_sign_epi16(simde_mm_mulhrs_epi16(simde_mm_abs_epi16(x1), m1), x1);
      pl[j] = simde_mm_packs_epi16(y0, y1);
    }
  } else if (mode == NR_LDPC_LLR_PRESCALE_GAIN) {
    // y = sign(x) * round(min(|x|, 32767 >> k) * 2^k * q15 / 2^15): the clamp only touches values that saturate the
    // int8 output anyway, the shift is exact, mulhrs rounds half away from zero on |x|. q15 = 0 is not used.
    const int k = ps->lshift;
    const simde__m256i m = simde_mm256_set1_epi16(ps->q15);
    const simde__m256i lo = simde_mm256_set1_epi16(-32767);
    const simde__m256i hi = simde_mm256_set1_epi16(32767 >> k);
    int j = 0;
    for (; j + 2 <= nvec; j += 2) {
      const simde__m256i x0 = simde_mm256_max_epi16(simde_mm256_loadu_si256((const simde__m256i *)(z + 16 * j)), lo);
      const simde__m256i x1 = simde_mm256_max_epi16(simde_mm256_loadu_si256((const simde__m256i *)(z + 16 * j + 16)), lo);
      const simde__m256i a0 = simde_mm256_slli_epi16(simde_mm256_min_epi16(simde_mm256_abs_epi16(x0), hi), k);
      const simde__m256i a1 = simde_mm256_slli_epi16(simde_mm256_min_epi16(simde_mm256_abs_epi16(x1), hi), k);
      const simde__m256i y0 = simde_mm256_sign_epi16(simde_mm256_mulhrs_epi16(a0, m), x0);
      const simde__m256i y1 = simde_mm256_sign_epi16(simde_mm256_mulhrs_epi16(a1, m), x1);
      simde_mm256_storeu_si256((simde__m256i *)(l + 16 * j), simde_mm256_permute4x64_epi64(simde_mm256_packs_epi16(y0, y1), 0xd8));
    }
    if (j < nvec) {
      const simde__m128i m1 = simde_mm256_castsi256_si128(m);
      const simde__m128i lo1 = simde_mm256_castsi256_si128(lo);
      const simde__m128i hi1 = simde_mm256_castsi256_si128(hi);
      const simde__m128i x0 = simde_mm_max_epi16(pv[2 * j], lo1);
      const simde__m128i x1 = simde_mm_max_epi16(pv[2 * j + 1], lo1);
      const simde__m128i a0 = simde_mm_slli_epi16(simde_mm_min_epi16(simde_mm_abs_epi16(x0), hi1), k);
      const simde__m128i a1 = simde_mm_slli_epi16(simde_mm_min_epi16(simde_mm_abs_epi16(x1), hi1), k);
      pl[j] = simde_mm_packs_epi16(simde_mm_sign_epi16(simde_mm_mulhrs_epi16(a0, m1), x0),
                                   simde_mm_sign_epi16(simde_mm_mulhrs_epi16(a1, m1), x1));
    }
  } else if (mode == NR_LDPC_LLR_PRESCALE_LSHIFT) {
    const int s = ps->lshift;
    for (int i = 0, j = 0; j < nvec; i += 2, j++) {
      simde__m128i y0 = pv[i], y1 = pv[i + 1];
      for (int k = 0; k < s; k++) {
        y0 = simde_mm_adds_epi16(y0, y0);
        y1 = simde_mm_adds_epi16(y1, y1);
      }
      pl[j] = simde_mm_packs_epi16(y0, y1);
    }
  } else {
    // legacy conversion, unchanged
    for (int i = 0, j = 0; j < nvec; i += 2, j++) {
      pl[j] = simde_mm_packs_epi16(pv[i], pv[i + 1]);
    }
  }
}

int nrLDPC_llr_agc_parse(const char *str, nr_ldpc_llr_agc_rule_t *rule)
{
  memset(rule, 0, sizeof(*rule));
  if (str == NULL)
    return -1;
  while (*str == ' ')
    str++;
  if (!strcasecmp(str, "off") || !strcasecmp(str, "none") || !strcmp(str, "0"))
    return 0;
  if (!strcasecmp(str, "default") || !strcasecmp(str, "on"))
    str = NR_LDPC_LLR_AGC_DEFAULT;
  float t[8];
  int n1 = 0, n2 = 0;
  if (sscanf(str, " %f , %f , %f , %f %n", &t[0], &t[1], &t[2], &t[3], &n1) != 4 || n1 == 0)
    return -1;
  str += n1;
  if (*str == '/') {
    if (sscanf(str + 1, " %f , %f , %f , %f %n", &t[4], &t[5], &t[6], &t[7], &n2) != 4 || n2 == 0)
      return -1;
    str += 1 + n2;
  } else {
    memcpy(t + 4, t, 4 * sizeof(float));
  }
  if (*str != 0)
    return -1;
  for (int i = 0; i < 8; i++)
    if (!(t[i] >= 0 && t[i] <= 127))
      return -1;
  for (int l = 0; l < 2; l++)
    for (int i = 0; i < 4; i++)
      rule->target[l][2 * (i + 1)] = t[4 * l + i];
  return 0;
}

void nrLDPC_llr_stats(const int16_t *z, int b, int e, int step, nr_ldpc_llr_stats_t *st)
{
  const simde__m256i ones = simde_mm256_set1_epi16(1);
  simde__m256i acc = simde_mm256_setzero_si256(), zc = simde_mm256_setzero_si256();
  int i = b, nv = 0;
  for (; i + 16 <= e; i += 16 * step, nv++) {
    const simde__m256i v = simde_mm256_loadu_si256((const simde__m256i *)(z + i));
    acc = simde_mm256_add_epi32(acc, simde_mm256_madd_epi16(simde_mm256_abs_epi16(v), ones));
    zc = simde_mm256_sub_epi16(zc, simde_mm256_cmpeq_epi16(v, simde_mm256_setzero_si256())); // < 2^15 per lane
  }
  int32_t a[8];
  int16_t c[16];
  simde_mm256_storeu_si256((simde__m256i *)a, acc);
  simde_mm256_storeu_si256((simde__m256i *)c, zc);
  int32_t sabs = 0, nzero = 0;
  for (int k = 0; k < 8; k++)
    sabs += a[k];
  for (int k = 0; k < 16; k++)
    nzero += (uint16_t)c[k];
  int n = 16 * nv;
  if (step == 1) // tail
    for (; i < e; i++, n++) {
      const int v = z[i];
      sabs += v < 0 ? -v : v;
      nzero += v == 0;
    }
  st->sabs += sabs;
  st->nnz += n - nzero;
  st->n += n;
}

nr_ldpc_llr_prescale_t nrLDPC_llr_agc_gain(float target, const nr_ldpc_llr_stats_t *st)
{
  nr_ldpc_llr_prescale_t p = {.mode = NR_LDPC_LLR_PRESCALE_NONE, .q8 = 256};
  if (st->nnz <= 0 || st->sabs <= 0 || target <= 0)
    return p;
  float g = target * (float)st->nnz / (float)st->sabs;
  if (g > 256.0f)
    g = 256.0f;
  int k = 0;
  while (g > (float)(1 << k) && k < 8)
    k++;
  long q = lrintf(g / (float)(1 << k) * 32768.0f);
  if (q >= 32768 && k == 0)
    return p; // gain 1 (within 2^-15)
  p.mode = k ? NR_LDPC_LLR_PRESCALE_GAIN : NR_LDPC_LLR_PRESCALE_FRAC; // gain <= 1: the cheaper fractional path (same result)
  p.lshift = k;
  p.q15 = q > 32767 ? 32767 : q < 1 ? 1 : (int16_t)q;
  const float q8 = g * 256.0f;
  p.q8 = q8 > 65535.0f ? 65535 : q8 < 1.0f ? 1 : (uint16_t)lrintf(q8);
  return p;
}
