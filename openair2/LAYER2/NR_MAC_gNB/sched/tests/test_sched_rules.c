/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* Expected values are worked out from the tables and formulas of the specifications, not from the code */

#include <stdio.h>

#include "../sched_rules.h"

static int failures;

#define EXPECT_EQ(value, expected)                                                                     \
  do {                                                                                                 \
    const long long v_ = (value), e_ = (expected);                                                     \
    if (v_ != e_) {                                                                                    \
      printf("FAIL %s:%d %s = %lld, expected %lld\n", __func__, __LINE__, #value, v_, e_);             \
      failures++;                                                                                      \
    }                                                                                                  \
  } while (0)

/* (2048 + 144) x 64 Tc at 15 kHz */
#define SYM0 140288

/* 38.211 5.3.1: 14 x 2^mu symbols per subframe, the first of each half subframe 16 kappa longer */
static void test_symbol_clock(void)
{
  const int mus[] = {0, 1, 2, 3, 5, 6};
  for (int i = 0; i < 6; i++) {
    const int mu = mus[i];
    EXPECT_EQ(rules_symbol_start_tc(mu, 0), 0);
    EXPECT_EQ(rules_symbol_start_tc(mu, 1), (SYM0 >> mu) + 1024);
    EXPECT_EQ(rules_symbol_start_tc(mu, 7 << mu), RULES_TC_PER_MS / 2);
    EXPECT_EQ(rules_symbol_start_tc(mu, 14 << mu), RULES_TC_PER_MS);
    EXPECT_EQ(rules_symbol_start_tc(mu, (14 << mu) * 10240LL), 10240LL * RULES_TC_PER_MS);
    EXPECT_EQ(rules_symbol_start_tc(mu, -1), -(SYM0 >> mu)); /* last symbol of the previous subframe */
    for (int64_t s = -30; s < (28 << mu); s++) {
      const int64_t t = rules_symbol_start_tc(mu, s);
      EXPECT_EQ(rules_first_symbol_from(mu, t), s);
      EXPECT_EQ(rules_first_symbol_from(mu, t + 1), s + 1);
      EXPECT_EQ(rules_first_symbol_from(mu, t - 1), s);
    }
  }
}

/* 38.214 Table 5.3-1, 5.3-2, 6.4-1, 6.4-2 */
static void test_tables(void)
{
  EXPECT_EQ(rules_n1_x2(0, false, true, false), 16);
  EXPECT_EQ(rules_n1_x2(0, false, false, false), 26);
  EXPECT_EQ(rules_n1_x2(0, false, false, true), 28);
  EXPECT_EQ(rules_n1_x2(1, false, false, true), 26);
  EXPECT_EQ(rules_n1_x2(1, false, true, false), 20);
  EXPECT_EQ(rules_n1_x2(3, false, false, false), 48);
  EXPECT_EQ(rules_n1_x2(4, false, false, false), -1);
  EXPECT_EQ(rules_n1_x2(6, false, false, false), 384);
  EXPECT_EQ(rules_n1_x2(1, true, true, false), 9);
  EXPECT_EQ(rules_n1_x2(1, true, false, false), -1);
  EXPECT_EQ(rules_n1_x2(3, true, true, false), -1);
  EXPECT_EQ(rules_n2_x2(1, false), 24);
  EXPECT_EQ(rules_n2_x2(3, false), 72);
  EXPECT_EQ(rules_n2_x2(1, true), 11);
  EXPECT_EQ(rules_n2_x2(5, false), 288);
  EXPECT_EQ(rules_symbols_x2_tc(1, 26), 13 * (SYM0 >> 1));
  EXPECT_EQ(rules_symbols_x2_tc(1, 9), 9 * (SYM0 >> 1) / 2);
}

/* 38.214 5.3: d1,1 */
static void test_d11(void)
{
  EXPECT_EQ(rules_d11(false, true, 1, 13, 0, false), 0);
  EXPECT_EQ(rules_d11(false, true, 1, 5, 0, false), 2); /* last symbol 5 */
  EXPECT_EQ(rules_d11(false, true, 0, 7, 0, false), 1); /* last symbol 6 */
  EXPECT_EQ(rules_d11(false, false, 2, 7, 1, false), 0);
  EXPECT_EQ(rules_d11(false, false, 2, 4, 1, false), 3);
  EXPECT_EQ(rules_d11(false, false, 2, 3, 2, false), 4); /* 3 + min(d, 1) */
  EXPECT_EQ(rules_d11(false, false, 2, 3, 0, false), 3);
  EXPECT_EQ(rules_d11(false, false, 2, 2, 2, false), 5); /* 3 + d */
  EXPECT_EQ(rules_d11(true, false, 2, 5, 1, false), 1);
  EXPECT_EQ(rules_d11(true, false, 0, 2, 2, true), 3);
  EXPECT_EQ(rules_d11(true, false, 0, 2, 2, false), 2);
}

static void test_tproc(void)
{
  /* 30 kHz everywhere, capability 1, additional DM-RS, type A on symbols 1..13: 13 symbols */
  rules_tproc1_t p = {.mu_pdcch = 1, .mu_pdsch = 1, .mu_ul = 1, .type_a = true, .start_symbol = 1, .length = 13};
  EXPECT_EQ(rules_tproc1_tc(&p), 13 * (SYM0 >> 1));
  /* a 15 kHz PDCCH gives the largest Tproc,1: 13 symbols of 15 kHz */
  p.mu_pdcch = 0;
  EXPECT_EQ(rules_tproc1_tc(&p), 13 * SYM0);
  p.mu_pdcch = 1;
  /* mixed slot, type A on 1..5: d1,1 = 2 */
  p.length = 5;
  EXPECT_EQ(rules_tproc1_tc(&p), 15 * (SYM0 >> 1));
  /* capability 2 needs pos0; above 136 RBs at 30 kHz it is capability 1 */
  p.length = 13;
  p.cap2 = true;
  EXPECT_EQ(rules_tproc1_tc(&p), -1);
  p.dmrs_pos0 = true;
  EXPECT_EQ(rules_tproc1_tc(&p), 9 * (SYM0 >> 1) / 2);
  p.n_rb = 137;
  EXPECT_EQ(rules_tproc1_tc(&p), 10 * (SYM0 >> 1));

  /* Tproc,2: N2 = 12 at 30 kHz, d2,1 = 1 unless the first symbol is DM-RS only */
  rules_tproc2_t q = {.mu_dl = 1, .mu_ul = 1};
  EXPECT_EQ(rules_tproc2_tc(&q), 13 * (SYM0 >> 1));
  q.first_symbol_dmrs_only = true;
  EXPECT_EQ(rules_tproc2_tc(&q), 12 * (SYM0 >> 1));
  q.d22_tc = RULES_TC_PER_MS;
  EXPECT_EQ(rules_tproc2_tc(&q), RULES_TC_PER_MS);
  q = (rules_tproc2_t){.mu_dl = 0, .mu_ul = 1};
  EXPECT_EQ(rules_tproc2_tc(&q), 11 * SYM0);
}

/* the audit case DL-55 with min_rxtxtime 1: PDSCH on symbols 1..13 of slot 0, HARQ-ACK one slot later */
static void test_earliest_ul_symbol(void)
{
  const rules_tproc1_t p = {.mu_pdcch = 1, .mu_pdsch = 1, .mu_ul = 1, .type_a = true, .start_symbol = 1, .length = 13};
  const int64_t end = rules_symbol_start_tc(1, 14);
  const int64_t tproc = rules_tproc1_tc(&p);
  /* without timing advance: 983040 + 911872 = 1894912 Tc, symbol 27 starts at 1895936: slot 1 symbol 13 */
  EXPECT_EQ(rules_earliest_ul_symbol(1, end, tproc, 0), 27);
  /* N_TA,offset 25600 Tc of FR1 TDD: slot 2 symbol 0, so a PUCCH at slot 1 symbol 13 is too early */
  EXPECT_EQ(rules_earliest_ul_symbol(1, end, tproc, 25600), 28);
  /* a symbol starting exactly at the limit is legal */
  EXPECT_EQ(rules_earliest_ul_symbol(1, end, rules_symbol_start_tc(1, 27) - end, 0), 27);
}

/* 38.213 8.3 and 8.4, N1,0 = 14 at 15 kHz */
static void test_random_access_delays(void)
{
  EXPECT_EQ(rules_msg3_delay_tc(1), 25 * (SYM0 >> 1) + RULES_TC_PER_MS / 2);
  EXPECT_EQ(rules_msg3_delay_tc(0), 24 * SYM0 + RULES_TC_PER_MS / 2);
  EXPECT_EQ(rules_msg3_delay_tc(3), 60 * (SYM0 >> 3) + RULES_TC_PER_MS / 2);
  EXPECT_EQ(rules_msg4_harq_delay_tc(1), 13 * (SYM0 >> 1) + RULES_TC_PER_MS / 2);
  EXPECT_EQ(rules_msg4_harq_delay_tc(0), 14 * SYM0 + RULES_TC_PER_MS / 2);
  EXPECT_EQ(rules_msg4_harq_delay_tc(4), -1);
}

/* 38.214 5.1.3.2, values worked out by hand */
static void test_tbs(void)
{
  int qm, r;
  /* MCS 27 of Table 5.1.3.1-1 on 106 PRBs, 13 symbols, 12 DM-RS REs: N_RE 15264, N_info 81388.1,
   * n = 11, N'_info 81920, C = 10, TBS = 80 x ceil(81944 / 80) - 24 */
  EXPECT_EQ(rules_mcs(RULES_MCS_QAM64, 27, 2, &qm, &r), 1);
  EXPECT_EQ(rules_tbs(rules_tbs_n_re(13, 12, 0, 106), r, qm, 1), 81976);
  /* N_info = 10146 x 193 / 1024 x 2 = 3824.6 is above 3824: step 4, not step 3 (OAI truncates N_info
   * and gives 3824). N'_info = 3840, R <= 1/4 so C = 2, TBS = 16 x ceil(3864 / 16) - 24 */
  EXPECT_EQ(rules_mcs(RULES_MCS_QAM64, 2, 2, &qm, &r), 1);
  EXPECT_EQ(rules_tbs(rules_tbs_n_re(10, 6, 0, 89), r, qm, 1), 3848);
  /* small TB: 1 PRB, 12 symbols, 12 DM-RS REs, MCS 0: N_info = 132 x 120 / 1024 x 2 = 30.9, n = 3,
   * N'_info = max(24, 8 x 3) = 24 */
  EXPECT_EQ(rules_mcs(RULES_MCS_QAM64, 0, 2, &qm, &r), 1);
  EXPECT_EQ(rules_tbs(rules_tbs_n_re(12, 12, 0, 1), r, qm, 1), 24);
  /* N'_RE is capped at 156 and the xOverhead is removed */
  EXPECT_EQ(rules_tbs_n_re(14, 12, 0, 10), 1560);
  EXPECT_EQ(rules_tbs_n_re(14, 12, 18, 10), 1380);
  /* 682.5 and 916.5 of Table 5.1.3.1-2 are exact; MCS 28 of that table is reserved */
  EXPECT_EQ(rules_mcs(RULES_MCS_QAM256, 20, 2, &qm, &r) && qm == 8 && r == 1365, 1);
  EXPECT_EQ(rules_mcs(RULES_MCS_QAM256, 28, 2, &qm, &r), 0);
  EXPECT_EQ(rules_mcs(RULES_MCS_QAM1024, 23, 2, &qm, &r) && qm == 10 && r == 1611, 1);
  /* transform precoding: q = 1 with pi/2-BPSK, 240 / q */
  EXPECT_EQ(rules_mcs(RULES_MCS_TP_QAM64, 0, 1, &qm, &r) && qm == 1 && r == 480, 1);
  EXPECT_EQ(rules_mcs(RULES_MCS_TP_QAM64, 0, 2, &qm, &r) && qm == 2 && r == 240, 1);
  EXPECT_EQ(rules_mcs(RULES_MCS_TP_QAM64_LOWSE, 5, 2, &qm, &r) && qm == 2 && r == 198, 1);
  EXPECT_EQ(rules_mcs(RULES_MCS_TP_QAM64, 0, 3, &qm, &r), 0);
}

/* 38.212 7.2.2 base graph, 5.2.2 segmentation, 38.214 5.1.3 limit 0.95 */
static void test_rate(void)
{
  EXPECT_EQ(rules_ldpc_bg2(292, 1896), 1);
  EXPECT_EQ(rules_ldpc_bg2(293, 1896), 0);
  EXPECT_EQ(rules_ldpc_bg2(3824, 1372), 1);
  EXPECT_EQ(rules_ldpc_bg2(3824, 1373), 0);
  EXPECT_EQ(rules_ldpc_bg2(100000, 512), 1);
  /* one code block: TB CRC only, 16 bits up to 3824 */
  EXPECT_EQ(rules_coded_info_bits(3824, 1896), 3840);
  /* BG1, B = 8448 + 24: 2 code blocks of 24 CRC bits */
  EXPECT_EQ(rules_coded_info_bits(8448, 1896), 8472 + 48);
  EXPECT_EQ(rules_coded_info_bits(8424, 1896), 8448);
  /* exactly 0.95: 19 x 400 = 20 x 380 */
  EXPECT_EQ(rules_rate_ok(364, 1896, 2, 1, 200), 1);
  EXPECT_EQ(rules_rate_ok(368, 1896, 2, 1, 200), 0);

  /* the DL-38 case of the audit: a short PDSCH in the mixed slot. 1 PRB, 5 symbols, 12 DM-RS REs,
   * 48 data REs at 256QAM (384 bits): MCS 27 gives TBS 352 and (352 + 16) / 384 = 0.958, MCS 26
   * gives TBS 336 and 352 / 384 = 0.917 */
  rules_alloc_t a = {.table = RULES_MCS_QAM256, .max_mcs = 27, .n_symb = 5, .dmrs_re_per_prb = 12, .n_prb = 1, .layers = 1, .n_re_data = 48};
  int tbs = 0;
  EXPECT_EQ(rules_max_mcs(&a, &tbs), 26);
  EXPECT_EQ(tbs, 336);
  /* 2 REs less for data (CSI-RS): 368 bits. MCS 26 and 25 both give TBS 336, 352 / 368 = 0.957;
   * MCS 24: N'_info 312, TBS 320 (next entry of Table 5.1.3.2-1), 336 / 368 = 0.913 */
  a.n_re_data = 46;
  EXPECT_EQ(rules_max_mcs(&a, &tbs), 24);
  EXPECT_EQ(tbs, 320);
  /* a full slot allows MCS 27 */
  a = (rules_alloc_t){.table = RULES_MCS_QAM256, .max_mcs = 27, .n_symb = 13, .dmrs_re_per_prb = 12, .n_prb = 106, .layers = 1, .n_re_data = 144 * 106};
  EXPECT_EQ(rules_max_mcs(&a, &tbs), 27);
  /* no REs: nothing fits */
  a.n_re_data = 0;
  EXPECT_EQ(rules_max_mcs(&a, &tbs), -1);
}

/* 38.214 5.1.2.1 and Tables 5.1.2.1-1, 6.1.2.1-1 */
static void test_sliv(void)
{
  /* L - 1 <= 7: 14 (L - 1) + S; otherwise 14 (14 - L + 1) + (13 - S) */
  EXPECT_EQ(rules_sliv(0, 14), 27);
  EXPECT_EQ(rules_sliv(2, 12), 53); /* 14 x 3 + 11 */
  EXPECT_EQ(rules_sliv(10, 3), 38); /* the msgA PUSCH occasion of RA_2-Step.conf: symbols 10..12 */
  EXPECT_EQ(rules_sliv(0, 8), 98);
  EXPECT_EQ(rules_sliv(1, 14), -1);
  EXPECT_EQ(rules_sliv(13, 1), 13);
  /* every SLIV decodes back, and nothing else decodes */
  int decoded = 0;
  for (int v = 0; v < 128; v++) {
    int s0, l0;
    if (rules_sliv_decode(v, &s0, &l0)) {
      decoded++;
      EXPECT_EQ(rules_sliv(s0, l0), v);
    }
  }
  EXPECT_EQ(decoded, 105); /* 14 + 13 + ... + 1 (S, L) pairs */
  int s0, l0;
  EXPECT_EQ(rules_sliv_decode(38, &s0, &l0) && s0 == 10 && l0 == 3, 1);

  EXPECT_EQ(rules_pdsch_sl_valid(true, false, 2, 1, 13), 1);
  EXPECT_EQ(rules_pdsch_sl_valid(true, false, 2, 3, 11), 0); /* S = 3 needs dmrs-TypeA-Position 3 */
  EXPECT_EQ(rules_pdsch_sl_valid(true, false, 3, 3, 11), 1);
  EXPECT_EQ(rules_pdsch_sl_valid(true, false, 2, 0, 2), 0);
  EXPECT_EQ(rules_pdsch_sl_valid(false, false, 2, 12, 2), 1);
  EXPECT_EQ(rules_pdsch_sl_valid(false, false, 2, 0, 14), 0);
  EXPECT_EQ(rules_pdsch_sl_valid(false, true, 2, 4, 6), 1);
  EXPECT_EQ(rules_pdsch_sl_valid(false, true, 2, 4, 5), 0);
  EXPECT_EQ(rules_pdsch_sl_valid(true, true, 2, 2, 11), 0); /* S + L <= 12 */
  EXPECT_EQ(rules_pusch_sl_valid(true, false, 0, 14), 1);
  EXPECT_EQ(rules_pusch_sl_valid(true, false, 1, 13), 0);
  EXPECT_EQ(rules_pusch_sl_valid(true, false, 0, 3), 0);
  EXPECT_EQ(rules_pusch_sl_valid(false, false, 13, 1), 1);
  EXPECT_EQ(rules_pusch_sl_valid(false, false, 10, 5), 0);
  EXPECT_EQ(rules_pusch_sl_valid(false, true, 11, 1), 1);
  EXPECT_EQ(rules_pusch_sl_valid(false, true, 12, 1), 0);
}

/* 38.214 6.1.2.2.2 RIV, 38.211 6.3.1.4 transform precoding sizes, 38.213 Tables 10.1-2 and 10.1-3 */
static void test_alloc_limits(void)
{
  EXPECT_EQ(rules_riv(106, 0, 1), 0);
  EXPECT_EQ(rules_riv(106, 5, 53), 106 * 52 + 5);
  EXPECT_EQ(rules_riv(106, 0, 106), 106 * 1 + 105);
  EXPECT_EQ(rules_riv(106, 10, 54), 106 * 53 + 10); /* L - 1 = 53 <= floor(106 / 2) */
  EXPECT_EQ(rules_riv(106, 10, 55), 106 * 52 + 95); /* N (N - L + 1) + (N - 1 - RB_start) */
  EXPECT_EQ(rules_riv(106, 100, 7), -1);

  EXPECT_EQ(rules_tp_alloc_valid(75, 1), 1);
  EXPECT_EQ(rules_tp_alloc_valid(7, 1), 0);
  EXPECT_EQ(rules_tp_alloc_valid(14, 1), 0);
  EXPECT_EQ(rules_tp_alloc_valid(1, 1), 1);
  EXPECT_EQ(rules_tp_alloc_valid(12, 2), 0);
  EXPECT_EQ(rules_tp_max_rb(106), 100);
  EXPECT_EQ(rules_tp_max_rb(273), 270);
  EXPECT_EQ(rules_tp_max_rb(7), 6);
  int valid = 0;
  for (int n = 1; n <= 275; n++)
    valid += rules_tp_alloc_valid(n, 1);
  EXPECT_EQ(valid, 53); /* 2^a 3^b 5^c up to 275 */

  EXPECT_EQ(rules_pdcch_max_candidates(1), 36);
  EXPECT_EQ(rules_pdcch_max_cces(1), 56);
  EXPECT_EQ(rules_pdcch_max_candidates(3), 20);
  EXPECT_EQ(rules_pdcch_max_cces(2), 48);
  EXPECT_EQ(rules_pdcch_max_cces(5), -1);
}

int main(void)
{
  test_symbol_clock();
  test_tables();
  test_d11();
  test_tproc();
  test_earliest_ul_symbol();
  test_random_access_delays();
  test_tbs();
  test_rate();
  test_sliv();
  test_alloc_limits();
  if (failures) {
    printf("%d failures\n", failures);
    return 1;
  }
  printf("all sched_rules tests passed\n");
  return 0;
}
