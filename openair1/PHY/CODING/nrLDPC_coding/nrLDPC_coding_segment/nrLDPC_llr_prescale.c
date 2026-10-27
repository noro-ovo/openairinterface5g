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
    // LLR sign; x = -32768 is first limited to -32767 so that |x| fits in int16
    const simde__m128i m = simde_mm_set1_epi16(ps->q15);
    const simde__m128i lo = simde_mm_set1_epi16(-32767);
    for (int i = 0, j = 0; j < nvec; i += 2, j++) {
      const simde__m128i x0 = simde_mm_max_epi16(pv[i], lo);
      const simde__m128i x1 = simde_mm_max_epi16(pv[i + 1], lo);
      const simde__m128i y0 = simde_mm_sign_epi16(simde_mm_mulhrs_epi16(simde_mm_abs_epi16(x0), m), x0);
      const simde__m128i y1 = simde_mm_sign_epi16(simde_mm_mulhrs_epi16(simde_mm_abs_epi16(x1), m), x1);
      pl[j] = simde_mm_packs_epi16(y0, y1);
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
