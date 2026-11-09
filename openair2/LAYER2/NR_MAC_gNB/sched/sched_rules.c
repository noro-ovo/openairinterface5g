/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "sched_rules.h"

/* (2048 + 144) kappa 2^-mu Tc, before the shift by mu */
#define SYMBOL_TC_MU0 ((2048 + 144) * RULES_KAPPA)
/* 38.211 5.3.1: 16 kappa more for the first symbol of each half subframe */
#define EXTRA_CP_TC (16 * RULES_KAPPA)

static int64_t floor_div(int64_t a, int64_t b)
{
  return a / b - (a % b != 0 && (a < 0) != (b < 0));
}

/* start of symbol l of a subframe, l in 0..14 x 2^mu (the end of the subframe for the last) */
static int64_t start_in_subframe(int mu, int l)
{
  const int half = 7 << mu;
  return (int64_t)l * (SYMBOL_TC_MU0 >> mu) + EXTRA_CP_TC * ((l > 0) + (l > half));
}

int64_t rules_symbol_start_tc(int mu, int64_t symbol)
{
  const int per_subframe = RULES_SYMBOLS_PER_SLOT << mu;
  const int64_t sf = floor_div(symbol, per_subframe);
  return sf * RULES_TC_PER_MS + start_in_subframe(mu, symbol - sf * per_subframe);
}

int64_t rules_first_symbol_from(int mu, int64_t t_tc)
{
  const int per_subframe = RULES_SYMBOLS_PER_SLOT << mu;
  const int64_t sf = floor_div(t_tc, RULES_TC_PER_MS);
  const int64_t r = t_tc - sf * RULES_TC_PER_MS;
  int l = r / (SYMBOL_TC_MU0 >> mu);
  while (l > 0 && start_in_subframe(mu, l - 1) >= r)
    l--;
  while (start_in_subframe(mu, l) < r)
    l++;
  return sf * per_subframe + l;
}

int rules_n1_x2(int mu, bool cap2, bool dmrs_pos0, bool dmrs_l1_12)
{
  static const int cap1_pos0[] = {8, 10, 17, 20, -1, 80, 160};
  static const int cap1_other[] = {13, 13, 20, 24, -1, 96, 192};
  /* Table 5.3-2 is given for pos0 only, 9 symbols at mu 2 in FR1 */
  static const int cap2_pos0_x2[] = {6, 9, 18};
  if (mu < 0 || mu > 6)
    return -1;
  if (cap2)
    return dmrs_pos0 && mu <= 2 ? cap2_pos0_x2[mu] : -1;
  const int n1 = dmrs_pos0 ? cap1_pos0[mu] : (mu == 0 && dmrs_l1_12 ? 14 : cap1_other[mu]);
  return n1 < 0 ? -1 : 2 * n1;
}

int rules_n2_x2(int mu, bool cap2)
{
  static const int cap1[] = {10, 12, 23, 36, -1, 144, 288};
  static const int cap2_x2[] = {10, 11, 22};
  if (mu < 0 || mu > 6)
    return -1;
  if (cap2)
    return mu <= 2 ? cap2_x2[mu] : -1;
  return cap1[mu] < 0 ? -1 : 2 * cap1[mu];
}

int64_t rules_symbols_x2_tc(int mu, int n_x2)
{
  return (int64_t)n_x2 * SYMBOL_TC_MU0 >> (mu + 1);
}

int rules_d11(bool cap2, bool type_a, int start_symbol, int length, int pdcch_overlap, bool cset3_same_start)
{
  /* the i-th symbol is read as the 0-based symbol index, the larger of the two readings */
  if (type_a) {
    const int last = start_symbol + length - 1;
    return last < 7 ? 7 - last : 0;
  }
  if (length >= 7)
    return 0;
  if (!cap2) {
    if (length >= 4)
      return 7 - length;
    if (length == 3)
      return 3 + (pdcch_overlap < 1 ? pdcch_overlap : 1);
    return 3 + pdcch_overlap;
  }
  if (length == 2 && cset3_same_start)
    return 3;
  return pdcch_overlap;
}

int64_t rules_tproc1_tc(const rules_tproc1_t *in)
{
  /* capability 2 with 30 kHz PDSCH above 136 RBs uses capability 1 */
  const bool cap2 = in->cap2 && !(in->mu_pdsch == 1 && in->n_rb > 136);
  const int d11 = rules_d11(cap2, in->type_a, in->start_symbol, in->length, in->pdcch_overlap, in->cset3_same_start);
  const int mus[] = {in->mu_pdcch, in->mu_pdsch, in->mu_ul};
  int64_t tproc = -1;
  for (int i = 0; i < 3; i++) {
    const int n1_x2 = rules_n1_x2(mus[i], cap2, in->dmrs_pos0, in->dmrs_l1_12);
    if (n1_x2 < 0)
      return -1;
    const int64_t t = rules_symbols_x2_tc(mus[i], n1_x2 + 2 * (d11 + in->d2 + in->d3)) + in->text_tc;
    if (t > tproc)
      tproc = t;
  }
  return tproc;
}

int64_t rules_tproc2_tc(const rules_tproc2_t *in)
{
  const int d21 = in->first_symbol_dmrs_only ? 0 : 1;
  const int mus[] = {in->mu_dl, in->mu_ul};
  int64_t tproc = -1;
  for (int i = 0; i < 2; i++) {
    const int n2_x2 = rules_n2_x2(mus[i], in->cap2);
    if (n2_x2 < 0)
      return -1;
    const int64_t t = rules_symbols_x2_tc(mus[i], n2_x2 + 2 * (d21 + in->d2 + in->d3 + in->d4)) + in->text_tc + in->tswitch_tc;
    if (t > tproc)
      tproc = t;
  }
  return tproc > in->d22_tc ? tproc : in->d22_tc;
}

int64_t rules_earliest_ul_symbol(int mu_ul, int64_t end_tc, int64_t delay_tc, int64_t ta_tc)
{
  /* the UE transmits T_TA ahead of the DL timing: the symbol must start at end + delay + T_TA */
  return rules_first_symbol_from(mu_ul, end_tc + delay_tc + ta_tc);
}

/* N_T,1 of 38.213 8.3/8.4: capability 1 with additional DM-RS, N1,0 = 14 */
static int nt1_x2(int mu)
{
  return mu == 0 ? 28 : rules_n1_x2(mu, false, false, false);
}

int64_t rules_msg3_delay_tc(int mu)
{
  const int n2_x2 = rules_n2_x2(mu, false);
  if (n2_x2 < 0 || nt1_x2(mu) < 0)
    return -1;
  return rules_symbols_x2_tc(mu, nt1_x2(mu) + n2_x2) + RULES_TC_PER_MS / 2;
}

int64_t rules_msg4_harq_delay_tc(int mu)
{
  if (nt1_x2(mu) < 0)
    return -1;
  return rules_symbols_x2_tc(mu, nt1_x2(mu)) + RULES_TC_PER_MS / 2;
}

/* {Qm, R x 2048}, Qm 0 for a reserved index; for the transform precoding tables a Qm of -1 means q
 * and the rate is given for q = 1 */
typedef struct {
  int8_t qm;
  int16_t r_x2048;
} mcs_entry_t;

static const mcs_entry_t mcs_qam64[32] = {
    {2, 240},  {2, 314},  {2, 386},  {2, 502},  {2, 616},  {2, 758},  {2, 898},  {2, 1052}, {2, 1204}, {2, 1358}, {4, 680},
    {4, 756},  {4, 868},  {4, 980},  {4, 1106}, {4, 1232}, {4, 1316}, {6, 876},  {6, 932},  {6, 1034}, {6, 1134}, {6, 1232},
    {6, 1332}, {6, 1438}, {6, 1544}, {6, 1644}, {6, 1746}, {6, 1820}, {6, 1896}, {0, 0},    {0, 0},    {0, 0}};
static const mcs_entry_t mcs_qam256[32] = {
    {2, 240},  {2, 386},  {2, 616},  {2, 898},  {2, 1204}, {4, 756},  {4, 868},  {4, 980},  {4, 1106}, {4, 1232}, {4, 1316},
    {6, 932},  {6, 1034}, {6, 1134}, {6, 1232}, {6, 1332}, {6, 1438}, {6, 1544}, {6, 1644}, {6, 1746}, {8, 1365}, {8, 1422},
    {8, 1508}, {8, 1594}, {8, 1682}, {8, 1770}, {8, 1833}, {8, 1896}, {0, 0},    {0, 0},    {0, 0},    {0, 0}};
static const mcs_entry_t mcs_qam64_lowse[32] = {
    {2, 60},   {2, 80},   {2, 100},  {2, 128},  {2, 156},  {2, 198},  {2, 240},  {2, 314},  {2, 386},  {2, 502},  {2, 616},
    {2, 758},  {2, 898},  {2, 1052}, {2, 1204}, {4, 680},  {4, 756},  {4, 868},  {4, 980},  {4, 1106}, {4, 1232}, {6, 876},
    {6, 932},  {6, 1034}, {6, 1134}, {6, 1232}, {6, 1332}, {6, 1438}, {6, 1544}, {0, 0},    {0, 0},    {0, 0}};
static const mcs_entry_t mcs_qam1024[32] = {
    {2, 240},   {2, 386},   {2, 898},   {4, 756},   {4, 980},   {4, 1232},  {6, 932},  {6, 1034}, {6, 1134}, {6, 1232}, {6, 1332},
    {6, 1438},  {6, 1544},  {6, 1644},  {6, 1746},  {8, 1365},  {8, 1422},  {8, 1508}, {8, 1594}, {8, 1682}, {8, 1770}, {8, 1833},
    {8, 1896},  {10, 1611}, {10, 1706}, {10, 1801}, {10, 1896}, {0, 0},     {0, 0},    {0, 0},    {0, 0},    {0, 0}};
static const mcs_entry_t mcs_tp_qam64[32] = {
    {-1, 480}, {-1, 628}, {2, 386},  {2, 502},  {2, 616},  {2, 758},  {2, 898},  {2, 1052}, {2, 1204}, {2, 1358}, {4, 680},
    {4, 756},  {4, 868},  {4, 980},  {4, 1106}, {4, 1232}, {4, 1316}, {6, 932},  {6, 1034}, {6, 1134}, {6, 1232}, {6, 1332},
    {6, 1438}, {6, 1544}, {6, 1644}, {6, 1746}, {6, 1820}, {6, 1896}, {0, 0},    {0, 0},    {0, 0},    {0, 0}};
static const mcs_entry_t mcs_tp_qam64_lowse[32] = {
    {-1, 120}, {-1, 160}, {-1, 200}, {-1, 256}, {-1, 312}, {-1, 396}, {2, 240},  {2, 314},  {2, 386},  {2, 502},  {2, 616},
    {2, 758},  {2, 898},  {2, 1052}, {2, 1204}, {2, 1358}, {4, 756},  {4, 868},  {4, 980},  {4, 1106}, {4, 1232}, {4, 1316},
    {4, 1398}, {4, 1544}, {6, 1134}, {6, 1232}, {6, 1332}, {6, 1544}, {0, 0},    {0, 0},    {0, 0},    {0, 0}};

bool rules_mcs(rules_mcs_table_t table, int mcs, int q, int *qm, int *r_x2048)
{
  static const mcs_entry_t *tables[] = {mcs_qam64, mcs_qam256, mcs_qam64_lowse, mcs_qam1024, mcs_tp_qam64, mcs_tp_qam64_lowse};
  if (mcs < 0 || mcs > 31 || (int)table < 0 || (int)table > RULES_MCS_TP_QAM64_LOWSE)
    return false;
  const mcs_entry_t e = tables[table][mcs];
  if (e.qm == 0)
    return false;
  if (e.qm < 0) {
    if (q != 1 && q != 2)
      return false;
    *qm = q;
    *r_x2048 = e.r_x2048 / q;
    return true;
  }
  *qm = e.qm;
  *r_x2048 = e.r_x2048;
  return true;
}

int rules_tbs_n_re(int n_symb, int dmrs_re_per_prb, int x_overhead, int n_prb)
{
  const int per_prb = 12 * n_symb - dmrs_re_per_prb - x_overhead;
  return (per_prb < 156 ? per_prb : 156) * n_prb;
}

static int floor_log2(int64_t x)
{
  int n = -1;
  while (x > 0) {
    x >>= 1;
    n++;
  }
  return n;
}

/* Table 5.1.3.2-1 */
static const int16_t tbs_table[93] = {
    24,   32,   40,   48,   56,   64,   72,   80,   88,   96,   104,  112,  120,  128,  136,  144,  152,  160,  168,
    176,  184,  192,  208,  224,  240,  256,  272,  288,  304,  320,  336,  352,  368,  384,  408,  432,  456,  480,
    504,  528,  552,  576,  608,  640,  672,  704,  736,  768,  808,  848,  888,  928,  984,  1032, 1064, 1128, 1160,
    1192, 1224, 1256, 1288, 1320, 1352, 1416, 1480, 1544, 1608, 1672, 1736, 1800, 1864, 1928, 2024, 2088, 2152, 2216,
    2280, 2408, 2472, 2536, 2600, 2664, 2728, 2792, 2856, 2976, 3104, 3240, 3368, 3496, 3624, 3752, 3824};

int rules_tbs(int n_re, int r_x2048, int qm, int layers)
{
  /* N_info = N_RE R Qm v, kept exact as num / 2048 */
  const int64_t num = (int64_t)n_re * r_x2048 * qm * layers;
  if (num <= 0)
    return 0;
  if (num <= 3824LL * 2048) {
    int n = floor_log2(num) - 11 - 6;
    if (n < 3)
      n = 3;
    int64_t ninfo_q = (num >> (n + 11)) << n;
    if (ninfo_q < 24)
      ninfo_q = 24;
    for (int i = 0; i < 93; i++)
      if (tbs_table[i] >= ninfo_q)
        return tbs_table[i];
    return tbs_table[92];
  }
  /* N_info - 24 = (num - 24 x 2048) / 2048 */
  const int64_t rem = num - 24LL * 2048;
  const int n = floor_log2(rem) - 11 - 5;
  /* round((N_info - 24) / 2^n), halves up */
  int64_t ninfo_q = ((rem + (1LL << (n + 10))) >> (n + 11)) << n;
  if (ninfo_q < 3840)
    ninfo_q = 3840;
  int64_t c = 1;
  if (r_x2048 <= 512)
    c = (ninfo_q + 24 + 3815) / 3816;
  else if (ninfo_q > 8424)
    c = (ninfo_q + 24 + 8423) / 8424;
  return 8 * c * ((ninfo_q + 24 + 8 * c - 1) / (8 * c)) - 24;
}

bool rules_ldpc_bg2(int tbs, int r_x2048)
{
  /* R <= 0.67 read as R x 1024 <= 686 (0.67 x 1024 = 686.08), R <= 0.25 as R x 2048 <= 512 */
  return tbs <= 292 || (tbs <= 3824 && r_x2048 <= 1372) || r_x2048 <= 512;
}

int rules_coded_info_bits(int tbs, int r_x2048)
{
  const int b = tbs + (tbs > 3824 ? 24 : 16);
  const int kcb = rules_ldpc_bg2(tbs, r_x2048) ? 3840 : 8448;
  if (b <= kcb)
    return b;
  const int c = (b + kcb - 24 - 1) / (kcb - 24);
  return b + 24 * c;
}

bool rules_rate_ok(int tbs, int r_x2048, int qm, int layers, int n_re_data)
{
  const int64_t g = (int64_t)n_re_data * qm * layers;
  return g > 0 && 20LL * rules_coded_info_bits(tbs, r_x2048) <= 19 * g;
}

int rules_max_mcs(const rules_alloc_t *a, int *tbs)
{
  const int n_re = rules_tbs_n_re(a->n_symb, a->dmrs_re_per_prb, a->x_overhead, a->n_prb);
  for (int mcs = a->max_mcs; mcs >= 0; mcs--) {
    int qm, r;
    if (!rules_mcs(a->table, mcs, a->q, &qm, &r))
      continue;
    const int t = rules_tbs(n_re, r, qm, a->layers);
    if (t > 0 && rules_rate_ok(t, r, qm, a->layers, a->n_re_data)) {
      *tbs = t;
      return mcs;
    }
  }
  return -1;
}

int rules_sliv(int start, int length)
{
  if (start < 0 || length < 1 || length > RULES_SYMBOLS_PER_SLOT - start)
    return -1;
  if (length - 1 <= 7)
    return 14 * (length - 1) + start;
  return 14 * (14 - length + 1) + (14 - 1 - start);
}

bool rules_sliv_decode(int sliv, int *start, int *length)
{
  if (sliv < 0)
    return false;
  /* the two branches of the encoding: L - 1 = SLIV / 14 when it is at most 7 and S + L <= 14 */
  int l = sliv / 14 + 1, s = sliv % 14;
  if (l - 1 > 7 || s + l > 14) {
    l = 14 - sliv / 14 + 1;
    s = 14 - 1 - sliv % 14;
  }
  if (rules_sliv(s, l) != sliv)
    return false;
  *start = s;
  *length = l;
  return true;
}

bool rules_pdsch_sl_valid(bool type_a, bool extended_cp, int dmrs_typea_pos, int start, int length)
{
  const int n = extended_cp ? 12 : 14;
  if (type_a) {
    /* S = 3 only with dmrs-TypeA-Position = 3 */
    const int max_s = dmrs_typea_pos == 3 ? 3 : 2;
    return start >= 0 && start <= max_s && length >= 3 && length <= n && start + length >= 3 && start + length <= n;
  }
  if (extended_cp)
    return start >= 0 && start <= 10 && (length == 2 || length == 4 || length == 6) && start + length <= 12;
  return start >= 0 && start <= 12 && length >= 2 && length <= 13 && start + length <= 14;
}

bool rules_pusch_sl_valid(bool type_a, bool extended_cp, int start, int length)
{
  const int n = extended_cp ? 12 : 14;
  if (type_a)
    return start == 0 && length >= 4 && length <= n;
  return start >= 0 && start <= n - 1 && length >= 1 && length <= n && start + length <= n;
}

int rules_riv(int n_bwp, int rb_start, int n_rb)
{
  if (n_bwp < 1 || rb_start < 0 || n_rb < 1 || n_rb > n_bwp - rb_start)
    return -1;
  if (n_rb - 1 <= n_bwp / 2)
    return n_bwp * (n_rb - 1) + rb_start;
  return n_bwp * (n_bwp - n_rb + 1) + (n_bwp - 1 - rb_start);
}

bool rules_tp_alloc_valid(int n_rb, int layers)
{
  if (n_rb < 1 || layers != 1)
    return false;
  static const int primes[] = {2, 3, 5};
  for (int i = 0; i < 3; i++)
    while (n_rb % primes[i] == 0)
      n_rb /= primes[i];
  return n_rb == 1;
}

int rules_tp_max_rb(int n_rb)
{
  for (int n = n_rb; n > 0; n--)
    if (rules_tp_alloc_valid(n, 1))
      return n;
  return 0;
}

int rules_pdcch_max_candidates(int mu)
{
  static const int max[] = {44, 36, 22, 20};
  return mu >= 0 && mu <= 3 ? max[mu] : -1;
}

int rules_pdcch_max_cces(int mu)
{
  static const int max[] = {56, 56, 48, 32};
  return mu >= 0 && mu <= 3 ? max[mu] : -1;
}
