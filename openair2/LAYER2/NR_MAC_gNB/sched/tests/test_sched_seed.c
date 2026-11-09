/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* sched_seed: grid marks from FAPI PDUs; expected RBs computed by hand from 38.211 7.3.2.2 (CORESET,
 * REG bundles, interleaver), 38.211 6.3.2.1 (PUCCH hops) and the FAPI fields */

#include <stdio.h>
#include <string.h>

#include "../sched_seed.h"

static int failures;

#define EXPECT_EQ(value, expected)                                                         \
  do {                                                                                     \
    const long long v_ = (value), e_ = (expected);                                         \
    if (v_ != e_) {                                                                        \
      printf("FAIL %s:%d %s = %lld, expected %lld\n", __func__, __LINE__, #value, v_, e_); \
      failures++;                                                                          \
    }                                                                                      \
  } while (0)

#define SLOT 100

static sched_grid_t *all_flex_grid(void)
{
  static grid_cfg_t cfg;
  cfg = (grid_cfg_t){.n_rb = 106, .horizon_slots = 64, .period_slots = 1};
  for (int i = 0; i < GRID_SYMBOLS; i++)
    cfg.sym[0][i] = GRID_SYM_FLEX;
  sched_grid_t *g = grid_create(&cfg);
  grid_set_current(g, SLOT);
  return g;
}

/* 1 if RB rb of symbol sym is taken in direction ul */
static int used(sched_grid_t *g, int64_t slot, bool ul, int rb, int sym)
{
  const grid_alloc_t a = {.kind = ul ? GRID_PUSCH : GRID_PDSCH, .ul = ul, .owner = 1, .rb_start = rb, .n_rb = 1, .sym_start = sym, .n_sym = 1};
  return grid_check(g, slot, &a) == GRID_ERR_CONFLICT;
}

static seed_cfg_t cfg106(void)
{
  seed_cfg_t c = {.n_rb = 106, .ssb_mu = 1, .prach_first_rb = 43, .prach_n_rb = 12, .prach_dur = 4, .prach_slots = 1};
  c.ssb_start_symbol[0] = 4;
  c.ssb_start_symbol[1] = 8;
  return c;
}

static void test_ssb(void)
{
  sched_grid_t *g = all_flex_grid();
  const seed_cfg_t c = cfg106();
  nfapi_nr_dl_tti_request_body_t dl = {.nPDUs = 1};
  /* 86 RBs of 15 kHz from point A = RB 43 at 30 kHz; k_SSB 2: one more RB, 43..63 */
  dl.dl_tti_pdu_list[0].PDUType = NFAPI_NR_DL_TTI_SSB_PDU_TYPE;
  dl.dl_tti_pdu_list[0].ssb_pdu.ssb_pdu_rel15 = (nfapi_nr_dl_tti_ssb_pdu_rel15_t){.ssbOffsetPointA = 86, .SsbSubcarrierOffset = 2, .SsbBlockIndex = 0};
  EXPECT_EQ(seed_dl_fapi(g, SLOT, &c, &dl, NULL), 1);
  EXPECT_EQ(used(g, SLOT, false, 43, 4), 1);
  EXPECT_EQ(used(g, SLOT, false, 63, 7), 1);
  EXPECT_EQ(used(g, SLOT, false, 64, 4), 0);
  EXPECT_EQ(used(g, SLOT, false, 42, 4), 0);
  EXPECT_EQ(used(g, SLOT, false, 50, 3), 0);
  EXPECT_EQ(used(g, SLOT, false, 50, 8), 0);
  grid_free(g);
}

static void test_pdcch_offset(void)
{
  /* CORESET of PDCCH-Config in a BWP starting at RB 43: groups of 6 counted from 6 x ceil(43 / 6) = 48;
   * bitmap 11 -> RBs 48..59; non-interleaved, 1 symbol: CCE 1 = REG bundle 1 = RBs 54..59 */
  sched_grid_t *g = all_flex_grid();
  const seed_cfg_t c = cfg106();
  nfapi_nr_dl_tti_request_body_t dl = {.nPDUs = 1};
  dl.dl_tti_pdu_list[0].PDUType = NFAPI_NR_DL_TTI_PDCCH_PDU_TYPE;
  nfapi_nr_dl_tti_pdcch_pdu_rel15_t *p = &dl.dl_tti_pdu_list[0].pdcch_pdu.pdcch_pdu_rel15;
  *p = (nfapi_nr_dl_tti_pdcch_pdu_rel15_t){.BWPStart = 43,
                                           .StartSymbolIndex = 0,
                                           .DurationSymbols = 1,
                                           .CceRegMappingType = NFAPI_NR_CCE_REG_MAPPING_NON_INTERLEAVED,
                                           .RegBundleSize = 6,
                                           .CoreSetType = NFAPI_NR_CSET_CONFIG_PDCCH_CONFIG,
                                           .numDlDci = 1};
  p->FreqDomainResource[0] = 0xC0;
  p->dci_pdu[0].CceIndex = 1;
  p->dci_pdu[0].AggregationLevel = 1;
  EXPECT_EQ(seed_dl_fapi(g, SLOT, &c, &dl, NULL), 1);
  EXPECT_EQ(used(g, SLOT, false, 54, 0), 1);
  EXPECT_EQ(used(g, SLOT, false, 59, 0), 1);
  EXPECT_EQ(used(g, SLOT, false, 53, 0), 0);
  EXPECT_EQ(used(g, SLOT, false, 60, 0), 0);
  EXPECT_EQ(used(g, SLOT, false, 54, 1), 0);
  /* the legacy VRB map formula (BWP start + f x 6) books RBs 49..54 instead */
  EXPECT_EQ(used(g, SLOT, false, 49, 0), 0);
  grid_free(g);
}

static void test_pdcch_interleaved(void)
{
  /* CORESET 0 from RB 10, 48 RBs, 2 symbols, bundle 6, R 2, shift 0: N_REG 96, C = 96 / 12 = 8.
   * AL 4 from CCE 0: bundles k 0..3, f = (k%2 x 8 + k/2) % 16 = 0, 8, 1, 9; a bundle is 3 RBs x 2
   * symbols: RBs 0-2, 24-26, 3-5, 27-29 of the CORESET, so 10..15 and 34..39 */
  sched_grid_t *g = all_flex_grid();
  const seed_cfg_t c = cfg106();
  nfapi_nr_ul_dci_request_t ul_dci = {.numPdus = 1};
  nfapi_nr_dl_tti_pdcch_pdu_rel15_t *p = &ul_dci.ul_dci_pdu_list[0].pdcch_pdu.pdcch_pdu_rel15;
  *p = (nfapi_nr_dl_tti_pdcch_pdu_rel15_t){.BWPStart = 10,
                                           .StartSymbolIndex = 0,
                                           .DurationSymbols = 2,
                                           .CceRegMappingType = NFAPI_NR_CCE_REG_MAPPING_INTERLEAVED,
                                           .RegBundleSize = 6,
                                           .InterleaverSize = 2,
                                           .ShiftIndex = 0,
                                           .CoreSetType = NFAPI_NR_CSET_CONFIG_MIB_SIB1,
                                           .numDlDci = 1};
  p->FreqDomainResource[0] = 0xFF;
  p->dci_pdu[0].CceIndex = 0;
  p->dci_pdu[0].AggregationLevel = 4;
  EXPECT_EQ(seed_dl_fapi(g, SLOT, &c, NULL, &ul_dci), 1);
  for (int rb = 10; rb <= 15; rb++)
    EXPECT_EQ(used(g, SLOT, false, rb, 1), 1);
  for (int rb = 34; rb <= 39; rb++)
    EXPECT_EQ(used(g, SLOT, false, rb, 0), 1);
  EXPECT_EQ(used(g, SLOT, false, 16, 0), 0);
  EXPECT_EQ(used(g, SLOT, false, 33, 0), 0);
  EXPECT_EQ(used(g, SLOT, false, 40, 0), 0);
  EXPECT_EQ(used(g, SLOT, false, 10, 2), 0);
  grid_free(g);
}

static void test_pdcch_bundle2(void)
{
  /* CORESET 0 at RB 0, 48 RBs, 1 symbol, bundle 2, R 2, shift 0: N_REG 48, C = 48 / 4 = 12, 24 bundles.
   * AL 1 at CCE 1: bundles k = 3, 4, 5, f = (k%2 x 12 + k/2) % 24 = 13, 2, 14; 2 RBs each: 26-27, 4-5, 28-29 */
  sched_grid_t *g = all_flex_grid();
  const seed_cfg_t c = cfg106();
  nfapi_nr_dl_tti_request_body_t dl = {.nPDUs = 1};
  dl.dl_tti_pdu_list[0].PDUType = NFAPI_NR_DL_TTI_PDCCH_PDU_TYPE;
  nfapi_nr_dl_tti_pdcch_pdu_rel15_t *p = &dl.dl_tti_pdu_list[0].pdcch_pdu.pdcch_pdu_rel15;
  *p = (nfapi_nr_dl_tti_pdcch_pdu_rel15_t){.BWPStart = 0,
                                           .StartSymbolIndex = 0,
                                           .DurationSymbols = 1,
                                           .CceRegMappingType = NFAPI_NR_CCE_REG_MAPPING_INTERLEAVED,
                                           .RegBundleSize = 2,
                                           .InterleaverSize = 2,
                                           .CoreSetType = NFAPI_NR_CSET_CONFIG_MIB_SIB1,
                                           .numDlDci = 1};
  p->FreqDomainResource[0] = 0xFF;
  p->dci_pdu[0].CceIndex = 1;
  p->dci_pdu[0].AggregationLevel = 1;
  EXPECT_EQ(seed_dl_fapi(g, SLOT, &c, &dl, NULL), 1);
  const int expected[] = {4, 5, 26, 27, 28, 29};
  for (int i = 0; i < 6; i++)
    EXPECT_EQ(used(g, SLOT, false, expected[i], 0), 1);
  EXPECT_EQ(used(g, SLOT, false, 2, 0), 0);
  EXPECT_EQ(used(g, SLOT, false, 24, 0), 0);
  EXPECT_EQ(used(g, SLOT, false, 30, 0), 0);
  grid_free(g);
}

static void test_pdsch(void)
{
  sched_grid_t *g = all_flex_grid();
  const seed_cfg_t c = cfg106();
  nfapi_nr_dl_tti_request_body_t dl = {.nPDUs = 2};
  dl.dl_tti_pdu_list[0].PDUType = NFAPI_NR_DL_TTI_PDSCH_PDU_TYPE;
  dl.dl_tti_pdu_list[0].pdsch_pdu.pdsch_pdu_rel15 =
      (nfapi_nr_dl_tti_pdsch_pdu_rel15_t){.BWPStart = 5, .BWPSize = 101, .resourceAlloc = 1, .rbStart = 15, .rbSize = 10, .StartSymbolIndex = 2, .NrOfSymbols = 12};
  /* type 0 bitmap 0x03 (LSB first): RBs 0 and 1 of a BWP starting at RB 50 */
  dl.dl_tti_pdu_list[1].PDUType = NFAPI_NR_DL_TTI_PDSCH_PDU_TYPE;
  dl.dl_tti_pdu_list[1].pdsch_pdu.pdsch_pdu_rel15 =
      (nfapi_nr_dl_tti_pdsch_pdu_rel15_t){.BWPStart = 50, .BWPSize = 56, .resourceAlloc = 0, .StartSymbolIndex = 1, .NrOfSymbols = 1};
  dl.dl_tti_pdu_list[1].pdsch_pdu.pdsch_pdu_rel15.rbBitmap[0] = 0x03;
  EXPECT_EQ(seed_dl_fapi(g, SLOT, &c, &dl, NULL), 2);
  EXPECT_EQ(used(g, SLOT, false, 20, 2), 1);
  EXPECT_EQ(used(g, SLOT, false, 29, 13), 1);
  EXPECT_EQ(used(g, SLOT, false, 30, 13), 0);
  EXPECT_EQ(used(g, SLOT, false, 20, 1), 0);
  EXPECT_EQ(used(g, SLOT, false, 19, 2), 0);
  EXPECT_EQ(used(g, SLOT, false, 50, 1), 1);
  EXPECT_EQ(used(g, SLOT, false, 51, 1), 1);
  EXPECT_EQ(used(g, SLOT, false, 52, 1), 0);
  EXPECT_EQ(used(g, SLOT, false, 56, 1), 0);
  EXPECT_EQ(used(g, SLOT, false, 57, 1), 0);
  grid_free(g);
}

/* CSI-RS: RBs start_rb .. start_rb + nr_of_rbs - 1 (carrier RBs) on the symbols of its row, 38.211
 * table 7.4.1.5.3-1: l0 alone (row 1), l0 and l0 + 1 (row 5), l0, l0 + 1, l1, l1 + 1 (row 13),
 * l0 .. l0 + 3 (row 18); a row out of the table marks nothing */
static void test_csirs(void)
{
  sched_grid_t *g = all_flex_grid();
  const seed_cfg_t c = cfg106();
  nfapi_nr_dl_tti_request_body_t dl = {.nPDUs = 5};
  const nfapi_nr_dl_tti_csi_rs_pdu_rel15_t rows[5] = {
      {.row = 1, .start_rb = 10, .nr_of_rbs = 24, .symb_l0 = 6},
      {.row = 5, .start_rb = 40, .nr_of_rbs = 24, .symb_l0 = 3},
      {.row = 13, .start_rb = 70, .nr_of_rbs = 24, .symb_l0 = 2, .symb_l1 = 9},
      {.row = 18, .start_rb = 0, .nr_of_rbs = 4, .symb_l0 = 8},
      {.row = 0, .start_rb = 100, .nr_of_rbs = 4, .symb_l0 = 0},
  };
  for (int i = 0; i < 5; i++) {
    dl.dl_tti_pdu_list[i].PDUType = NFAPI_NR_DL_TTI_CSI_RS_PDU_TYPE;
    dl.dl_tti_pdu_list[i].csi_rs_pdu.csi_rs_pdu_rel15 = rows[i];
  }
  EXPECT_EQ(seed_dl_fapi(g, SLOT, &c, &dl, NULL), 4);
  /* row 1 */
  EXPECT_EQ(used(g, SLOT, false, 10, 6), 1);
  EXPECT_EQ(used(g, SLOT, false, 33, 6), 1);
  EXPECT_EQ(used(g, SLOT, false, 34, 6), 0);
  EXPECT_EQ(used(g, SLOT, false, 9, 6), 0);
  EXPECT_EQ(used(g, SLOT, false, 20, 5), 0);
  EXPECT_EQ(used(g, SLOT, false, 20, 7), 0);
  /* row 5 */
  EXPECT_EQ(used(g, SLOT, false, 50, 3), 1);
  EXPECT_EQ(used(g, SLOT, false, 50, 4), 1);
  EXPECT_EQ(used(g, SLOT, false, 50, 5), 0);
  EXPECT_EQ(used(g, SLOT, false, 50, 2), 0);
  /* row 13 */
  EXPECT_EQ(used(g, SLOT, false, 80, 2), 1);
  EXPECT_EQ(used(g, SLOT, false, 80, 3), 1);
  EXPECT_EQ(used(g, SLOT, false, 80, 4), 0);
  EXPECT_EQ(used(g, SLOT, false, 80, 8), 0);
  EXPECT_EQ(used(g, SLOT, false, 80, 9), 1);
  EXPECT_EQ(used(g, SLOT, false, 93, 10), 1);
  EXPECT_EQ(used(g, SLOT, false, 94, 10), 0);
  EXPECT_EQ(used(g, SLOT, false, 80, 11), 0);
  /* row 18 */
  EXPECT_EQ(used(g, SLOT, false, 0, 8), 1);
  EXPECT_EQ(used(g, SLOT, false, 3, 11), 1);
  EXPECT_EQ(used(g, SLOT, false, 3, 12), 0);
  EXPECT_EQ(used(g, SLOT, false, 4, 8), 0);
  /* row 0 */
  EXPECT_EQ(used(g, SLOT, false, 100, 0), 0);
  grid_free(g);
}

static void test_ul(void)
{
  sched_grid_t *g = all_flex_grid();
  const seed_cfg_t c = cfg106();
  static nfapi_nr_ul_tti_request_t ul;
  memset(&ul, 0, sizeof(ul));
  ul.n_pdus = 4;
  ul.pdus_list[0].pdu_type = NFAPI_NR_UL_CONFIG_PUSCH_PDU_TYPE;
  ul.pdus_list[0].pusch_pdu = (nfapi_nr_pusch_pdu_t){.bwp_start = 0, .bwp_size = 106, .resource_alloc = 1, .rb_start = 60, .rb_size = 5, .start_symbol_index = 0, .nr_of_symbols = 14};
  /* PUCCH 14 symbols with hopping: first hop floor(14 / 2) = 7 symbols at RB 0, then RB 105 */
  ul.pdus_list[1].pdu_type = NFAPI_NR_UL_CONFIG_PUCCH_PDU_TYPE;
  ul.pdus_list[1].pucch_pdu = (nfapi_nr_pucch_pdu_t){.bwp_start = 0, .prb_start = 0, .prb_size = 1, .start_symbol_index = 0, .nr_of_symbols = 14, .freq_hop_flag = 1, .second_hop_prb = 105};
  /* SRS: the BWP on symbol 13 */
  ul.pdus_list[2].pdu_type = NFAPI_NR_UL_CONFIG_SRS_PDU_TYPE;
  ul.pdus_list[2].srs_pdu = (nfapi_nr_srs_pdu_t){.bwp_start = 0, .bwp_size = 106, .time_start_position = 13, .num_symbols = 0};
  /* PRACH: 3 occasions of 4 symbols from symbol 0 on RBs 43..54 */
  ul.pdus_list[3].pdu_type = NFAPI_NR_UL_CONFIG_PRACH_PDU_TYPE;
  ul.pdus_list[3].prach_pdu = (nfapi_nr_prach_pdu_t){.num_prach_ocas = 3, .prach_start_symbol = 0, .prach_format = 5};
  EXPECT_EQ(seed_ul_fapi(g, SLOT, &c, &ul, NULL, 0), 4);
  EXPECT_EQ(used(g, SLOT, true, 64, 5), 1);
  EXPECT_EQ(used(g, SLOT, true, 65, 5), 0);
  EXPECT_EQ(used(g, SLOT, true, 0, 6), 1);
  EXPECT_EQ(used(g, SLOT, true, 0, 7), 0);
  EXPECT_EQ(used(g, SLOT, true, 105, 7), 1);
  EXPECT_EQ(used(g, SLOT, true, 105, 6), 0);
  EXPECT_EQ(used(g, SLOT, true, 80, 13), 1);
  EXPECT_EQ(used(g, SLOT, true, 80, 12), 0);
  EXPECT_EQ(used(g, SLOT, true, 43, 11), 1);
  EXPECT_EQ(used(g, SLOT, true, 54, 0), 1);
  EXPECT_EQ(used(g, SLOT, true, 55, 0), 0);
  EXPECT_EQ(used(g, SLOT, true, 43, 12), 0);
  grid_free(g);
}

static void test_ul_odd(void)
{
  /* PUCCH of 5 symbols from 9 with hopping: first hop floor(5 / 2) = 2 symbols (9, 10), then 11..13;
   * SRS of 4 symbols (num_symbols 2) from l0 10: symbols 10..13 */
  sched_grid_t *g = all_flex_grid();
  const seed_cfg_t c = cfg106();
  static nfapi_nr_ul_tti_request_t ul;
  memset(&ul, 0, sizeof(ul));
  ul.n_pdus = 2;
  ul.pdus_list[0].pdu_type = NFAPI_NR_UL_CONFIG_PUCCH_PDU_TYPE;
  ul.pdus_list[0].pucch_pdu = (nfapi_nr_pucch_pdu_t){.bwp_start = 0, .prb_start = 0, .prb_size = 1, .start_symbol_index = 9, .nr_of_symbols = 5, .freq_hop_flag = 1, .second_hop_prb = 105};
  ul.pdus_list[1].pdu_type = NFAPI_NR_UL_CONFIG_SRS_PDU_TYPE;
  ul.pdus_list[1].srs_pdu = (nfapi_nr_srs_pdu_t){.bwp_start = 10, .bwp_size = 50, .time_start_position = 10, .num_symbols = 2};
  EXPECT_EQ(seed_ul_fapi(g, SLOT, &c, &ul, NULL, 0), 2);
  EXPECT_EQ(used(g, SLOT, true, 0, 10), 1);
  EXPECT_EQ(used(g, SLOT, true, 0, 11), 0);
  EXPECT_EQ(used(g, SLOT, true, 105, 11), 1);
  EXPECT_EQ(used(g, SLOT, true, 105, 10), 0);
  EXPECT_EQ(used(g, SLOT, true, 30, 13), 1);
  EXPECT_EQ(used(g, SLOT, true, 30, 9), 0);
  EXPECT_EQ(used(g, SLOT, true, 60, 13), 0);
  grid_free(g);
}

static void test_long_prach(void)
{
  /* a long-format PRACH of 2 slots written in the slot before: all symbols of this slot */
  sched_grid_t *g = all_flex_grid();
  seed_cfg_t c = cfg106();
  c.prach_dur = 14;
  c.prach_slots = 2;
  static nfapi_nr_ul_tti_request_t prev;
  memset(&prev, 0, sizeof(prev));
  prev.n_pdus = 1;
  prev.pdus_list[0].pdu_type = NFAPI_NR_UL_CONFIG_PRACH_PDU_TYPE;
  prev.pdus_list[0].prach_pdu = (nfapi_nr_prach_pdu_t){.num_prach_ocas = 1, .prach_format = 0};
  const nfapi_nr_ul_tti_request_t *list[1] = {&prev};
  EXPECT_EQ(seed_ul_fapi(g, SLOT, &c, NULL, list, 1), 1);
  EXPECT_EQ(used(g, SLOT, true, 43, 0), 1);
  EXPECT_EQ(used(g, SLOT, true, 54, 13), 1);
  EXPECT_EQ(used(g, SLOT, true, 55, 13), 0);
  /* one slot only: nothing carried over */
  c.prach_slots = 1;
  EXPECT_EQ(seed_ul_fapi(g, SLOT + 1, &c, NULL, list, 1), 0);
  grid_free(g);
}

int main(void)
{
  test_ssb();
  test_pdcch_offset();
  test_pdcch_interleaved();
  test_pdcch_bundle2();
  test_pdsch();
  test_csirs();
  test_ul();
  test_ul_odd();
  test_long_prach();
  if (failures)
    printf("%d failures\n", failures);
  else
    printf("all sched_seed tests passed\n");
  return failures != 0;
}
