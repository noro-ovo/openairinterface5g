/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* Each rule must fire on a crafted violation and stay silent on the legal
 * variant next to it: a checker that cannot fail proves nothing. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../sched_check.h"

static int failures;

#define EXPECT_COUNT(report, rule, expected)                                                                  \
  do {                                                                                                        \
    if ((report).count[rule] != (uint64_t)(expected)) {                                                       \
      printf("FAIL %s:%d rule %s: count %lu, expected %d\n", __func__, __LINE__, sched_rule_id(rule),         \
             (unsigned long)(report).count[rule], (int)(expected));                                           \
      failures++;                                                                                             \
    }                                                                                                         \
  } while (0)

static sched_check_cfg_t cfg;
/* too large for the stack */
static nfapi_nr_dl_tti_request_t dl;
static nfapi_nr_ul_dci_request_t ul_dci;
static nfapi_nr_ul_tti_request_t ul;
static nfapi_nr_tx_data_request_t tx;

/* DDDSU at 30 kHz, mixed slot 6 DL, 4 flexible, 4 UL */
static void reset(void)
{
  memset(&cfg, 0, sizeof(cfg));
  memset(&dl, 0, sizeof(dl));
  memset(&ul_dci, 0, sizeof(ul_dci));
  memset(&ul, 0, sizeof(ul));
  cfg.slots_per_period = 5;
  cfg.ssb_mu = 1;
  cfg.num_beams = 1;
  for (int i = 0; i < 14; i++) {
    cfg.sym[0][i] = cfg.sym[1][i] = cfg.sym[2][i] = SCHED_SYM_DL;
    cfg.sym[3][i] = i < 6 ? SCHED_SYM_DL : i < 10 ? SCHED_SYM_FLEX : SCHED_SYM_UL;
    cfg.sym[4][i] = SCHED_SYM_UL;
  }
}

static nfapi_nr_dl_tti_pdsch_pdu_rel15_t *add_pdsch(int rnti, int rb_start, int rb_size, int start, int len)
{
  nfapi_nr_dl_tti_request_body_t *b = &dl.dl_tti_request_body;
  nfapi_nr_dl_tti_request_pdu_t *pdu = &b->dl_tti_pdu_list[b->nPDUs++];
  pdu->PDUType = NFAPI_NR_DL_TTI_PDSCH_PDU_TYPE;
  nfapi_nr_dl_tti_pdsch_pdu_rel15_t *p = &pdu->pdsch_pdu.pdsch_pdu_rel15;
  p->rnti = rnti;
  p->resourceAlloc = 1;
  p->rbStart = rb_start;
  p->rbSize = rb_size;
  p->StartSymbolIndex = start;
  p->NrOfSymbols = len;
  p->NrOfCodewords = 1;
  p->nrOfLayers = 1;
  p->qamModOrder[0] = 2;
  p->TBSize[0] = 1;
  p->dlDmrsSymbPos = 1 << (start + 1);
  p->numDmrsCdmGrpsNoData = 1;
  return p;
}

static void add_dci(nfapi_nr_dl_tti_pdcch_pdu_rel15_t *p, int rnti, int cce, int level)
{
  p->StartSymbolIndex = 0;
  p->DurationSymbols = 1;
  p->FreqDomainResource[0] = 0xff;
  nfapi_nr_dl_dci_pdu_t *d = &p->dci_pdu[p->numDlDci++];
  d->RNTI = rnti;
  d->CceIndex = cce;
  d->AggregationLevel = level;
}

static nfapi_nr_dl_tti_pdcch_pdu_rel15_t *new_dl_pdcch(void)
{
  nfapi_nr_dl_tti_request_body_t *b = &dl.dl_tti_request_body;
  nfapi_nr_dl_tti_request_pdu_t *pdu = &b->dl_tti_pdu_list[b->nPDUs++];
  pdu->PDUType = NFAPI_NR_DL_TTI_PDCCH_PDU_TYPE;
  return &pdu->pdcch_pdu.pdcch_pdu_rel15;
}

static nfapi_nr_dl_tti_ssb_pdu_rel15_t *add_ssb(int index, int start_symbol, int offset_point_a)
{
  nfapi_nr_dl_tti_request_body_t *b = &dl.dl_tti_request_body;
  nfapi_nr_dl_tti_request_pdu_t *pdu = &b->dl_tti_pdu_list[b->nPDUs++];
  pdu->PDUType = NFAPI_NR_DL_TTI_SSB_PDU_TYPE;
  nfapi_nr_dl_tti_ssb_pdu_rel15_t *s = &pdu->ssb_pdu.ssb_pdu_rel15;
  s->SsbBlockIndex = index;
  s->ssbOffsetPointA = offset_point_a;
  cfg.ssb_start_symbol[index] = start_symbol;
  return s;
}

static nfapi_nr_pusch_pdu_t *add_pusch(int rnti, int rb_start, int rb_size, int start, int len)
{
  nfapi_nr_ul_tti_request_number_of_pdus_t *pdu = &ul.pdus_list[ul.n_pdus++];
  pdu->pdu_type = NFAPI_NR_UL_CONFIG_PUSCH_PDU_TYPE;
  nfapi_nr_pusch_pdu_t *p = &pdu->pusch_pdu;
  p->rnti = rnti;
  p->rb_start = rb_start;
  p->rb_size = rb_size;
  p->start_symbol_index = start;
  p->nr_of_symbols = len;
  p->nrOfLayers = 1;
  p->transform_precoding = 1;
  return p;
}

static nfapi_nr_pucch_pdu_t *add_pucch(int rnti, int prb, int start, int len)
{
  nfapi_nr_ul_tti_request_number_of_pdus_t *pdu = &ul.pdus_list[ul.n_pdus++];
  pdu->pdu_type = NFAPI_NR_UL_CONFIG_PUCCH_PDU_TYPE;
  nfapi_nr_pucch_pdu_t *p = &pdu->pucch_pdu;
  p->rnti = rnti;
  p->prb_start = prb;
  p->prb_size = 1;
  p->start_symbol_index = start;
  p->nr_of_symbols = len;
  return p;
}

static void test_pdsch_direction(void)
{
  sched_check_report_t r = {0};
  reset();
  dl.Slot = 3;
  add_pdsch(0x1234, 0, 10, 1, 5); // symbols 1..5: all DL
  sched_check_dl_slot(&cfg, &dl, NULL, &r);
  EXPECT_COUNT(r, SCHED_RULE_PDSCH_ON_UL, 0);
  if (r.checked[SCHED_PDU_PDSCH] != 1) {
    printf("FAIL %s: PDSCH not counted as checked\n", __func__);
    failures++;
  }

  memset(&r, 0, sizeof(r));
  reset();
  dl.Slot = 3;
  add_pdsch(0x1234, 0, 10, 1, 9); // symbol 9 flexible: legal for this rule
  sched_check_dl_slot(&cfg, &dl, NULL, &r);
  EXPECT_COUNT(r, SCHED_RULE_PDSCH_ON_UL, 0);

  memset(&r, 0, sizeof(r));
  reset();
  dl.Slot = 3;
  add_pdsch(0x1234, 0, 10, 1, 13); // reaches UL symbols
  sched_check_dl_slot(&cfg, &dl, NULL, &r);
  EXPECT_COUNT(r, SCHED_RULE_PDSCH_ON_UL, 1);

  memset(&r, 0, sizeof(r));
  reset();
  dl.Slot = 8; // same as 3 in the period
  add_pdsch(0x1234, 0, 10, 1, 13);
  cfg.slots_per_period = 0; // FDD: no direction rule
  sched_check_dl_slot(&cfg, &dl, NULL, &r);
  EXPECT_COUNT(r, SCHED_RULE_PDSCH_ON_UL, 0);
}

static void test_ul_direction(void)
{
  sched_check_report_t r = {0};
  reset();
  ul.Slot = 3;
  add_pusch(0x1234, 0, 12, 10, 4);
  add_pucch(0x1234, 100, 12, 2);
  sched_check_ul_slot(&cfg, &ul, &r);
  EXPECT_COUNT(r, SCHED_RULE_PUSCH_ON_DL, 0);
  EXPECT_COUNT(r, SCHED_RULE_PUCCH_ON_DL, 0);

  memset(&r, 0, sizeof(r));
  reset();
  ul.Slot = 3;
  add_pusch(0x1234, 0, 12, 4, 10);
  add_pucch(0x5678, 100, 4, 10); // the Msg4 case of DL-54
  sched_check_ul_slot(&cfg, &ul, &r);
  EXPECT_COUNT(r, SCHED_RULE_PUSCH_ON_DL, 1);
  EXPECT_COUNT(r, SCHED_RULE_PUCCH_ON_DL, 1);
}

static void test_ssb(void)
{
  sched_check_report_t r = {0};
  reset();
  dl.Slot = 0;
  /* symbol 2, offset 40 in 15 kHz PRBs: PRB 20..39 at 30 kHz */
  add_ssb(0, 2, 40);
  add_pdsch(0x1234, 40, 10, 1, 13); // beside
  sched_check_dl_slot(&cfg, &dl, NULL, &r);
  EXPECT_COUNT(r, SCHED_RULE_PDSCH_ON_SSB, 0);

  memset(&r, 0, sizeof(r));
  reset();
  dl.Slot = 0;
  add_ssb(0, 2, 40);
  add_pdsch(0x1234, 30, 10, 6, 8); // same PRBs, after the SSB symbols
  sched_check_dl_slot(&cfg, &dl, NULL, &r);
  EXPECT_COUNT(r, SCHED_RULE_PDSCH_ON_SSB, 0);

  memset(&r, 0, sizeof(r));
  reset();
  dl.Slot = 0;
  add_ssb(0, 2, 40);
  add_pdsch(0xffff, 30, 10, 1, 13); // SIB1 is exempt
  sched_check_dl_slot(&cfg, &dl, NULL, &r);
  EXPECT_COUNT(r, SCHED_RULE_PDSCH_ON_SSB, 0);

  memset(&r, 0, sizeof(r));
  reset();
  dl.Slot = 0;
  cfg.num_beams = 2; // DL-25: the rule holds across beams
  add_ssb(0, 2, 40);
  add_pdsch(0x1234, 30, 10, 1, 13);
  sched_check_dl_slot(&cfg, &dl, NULL, &r);
  EXPECT_COUNT(r, SCHED_RULE_PDSCH_ON_SSB, 1);

  /* a non-zero subcarrier offset makes the SSB span a 21st PRB */
  memset(&r, 0, sizeof(r));
  reset();
  add_ssb(0, 2, 40);
  add_pdsch(0x1234, 40, 10, 1, 13);
  sched_check_dl_slot(&cfg, &dl, NULL, &r);
  EXPECT_COUNT(r, SCHED_RULE_PDSCH_ON_SSB, 0);

  memset(&r, 0, sizeof(r));
  reset();
  add_ssb(0, 2, 40)->SsbSubcarrierOffset = 4;
  add_pdsch(0x1234, 40, 10, 1, 13);
  sched_check_dl_slot(&cfg, &dl, NULL, &r);
  EXPECT_COUNT(r, SCHED_RULE_PDSCH_ON_SSB, 1);

  memset(&r, 0, sizeof(r));
  reset();
  dl.Slot = 4;
  add_ssb(0, 2, 40);
  sched_check_dl_slot(&cfg, &dl, NULL, &r);
  EXPECT_COUNT(r, SCHED_RULE_SSB_ON_UL, 1);
}

static void test_rate(void)
{
  sched_check_report_t r = {0};
  reset();
  /* DL-38: 5 PRB, L=5, one DM-RS symbol, 54 RE/PRB, 64QAM: 1620 physical bits,
   * 0.95 is 1539 bits: 190 bytes + CRC16 = 1536 pass, 191 bytes = 1544 do not */
  nfapi_nr_dl_tti_pdsch_pdu_rel15_t *p = add_pdsch(0x1234, 0, 5, 1, 5);
  p->qamModOrder[0] = 6;
  p->TBSize[0] = 190;
  sched_check_dl_slot(&cfg, &dl, NULL, &r);
  EXPECT_COUNT(r, SCHED_RULE_PDSCH_RATE, 0);

  memset(&r, 0, sizeof(r));
  p->TBSize[0] = 191;
  sched_check_dl_slot(&cfg, &dl, NULL, &r);
  EXPECT_COUNT(r, SCHED_RULE_PDSCH_RATE, 1);
  if (r.max_rate_permille != 1544 * 1000 / 1620 || r.pdsch_symbols[5] != 1) {
    printf("FAIL %s: rate %d histogram L5 %lu\n", __func__, r.max_rate_permille, (unsigned long)r.pdsch_symbols[5]);
    failures++;
  }

  memset(&r, 0, sizeof(r));
  p->rvIndex[0] = 2; // not an initial transmission
  sched_check_dl_slot(&cfg, &dl, NULL, &r);
  EXPECT_COUNT(r, SCHED_RULE_PDSCH_RATE, 0);
}

static void test_transform_precoding(void)
{
  sched_check_report_t r = {0};
  reset();
  ul.Slot = 4;
  nfapi_nr_pusch_pdu_t *p = add_pusch(0x1234, 0, 6, 0, 14);
  p->transform_precoding = 0;
  sched_check_ul_slot(&cfg, &ul, &r);
  EXPECT_COUNT(r, SCHED_RULE_TP_PRB, 0);

  memset(&r, 0, sizeof(r));
  p->rb_size = 7;
  sched_check_ul_slot(&cfg, &ul, &r);
  EXPECT_COUNT(r, SCHED_RULE_TP_PRB, 1);

  memset(&r, 0, sizeof(r));
  p->transform_precoding = 1; // CP-OFDM: any size, any rank
  p->nrOfLayers = 2;
  sched_check_ul_slot(&cfg, &ul, &r);
  EXPECT_COUNT(r, SCHED_RULE_TP_PRB, 0);
  EXPECT_COUNT(r, SCHED_RULE_TP_LAYERS, 0);

  memset(&r, 0, sizeof(r));
  p->transform_precoding = 0;
  p->rb_size = 6;
  sched_check_ul_slot(&cfg, &ul, &r);
  EXPECT_COUNT(r, SCHED_RULE_TP_LAYERS, 1);
}

static void test_overlap(void)
{
  sched_check_report_t r = {0};
  reset();
  add_pdsch(0x1234, 0, 10, 1, 13);
  add_pdsch(0x5678, 10, 10, 1, 13);
  sched_check_dl_slot(&cfg, &dl, NULL, &r);
  EXPECT_COUNT(r, SCHED_RULE_PDSCH_OVERLAP, 0);

  memset(&r, 0, sizeof(r));
  add_pdsch(0x9abc, 9, 2, 1, 13); // touches both
  sched_check_dl_slot(&cfg, &dl, NULL, &r);
  EXPECT_COUNT(r, SCHED_RULE_PDSCH_OVERLAP, 2);

  memset(&r, 0, sizeof(r));
  cfg.num_beams = 2;
  sched_check_dl_slot(&cfg, &dl, NULL, &r);
  EXPECT_COUNT(r, SCHED_RULE_PDSCH_OVERLAP, 0);

  memset(&r, 0, sizeof(r));
  reset();
  ul.Slot = 4;
  add_pucch(0x1234, 0, 12, 2);
  add_pucch(0x5678, 0, 12, 2); // PUCCH sharing is legal
  sched_check_ul_slot(&cfg, &ul, &r);
  EXPECT_COUNT(r, SCHED_RULE_PUSCH_OVERLAP, 0);

  memset(&r, 0, sizeof(r));
  add_pusch(0x9abc, 0, 5, 0, 14);
  sched_check_ul_slot(&cfg, &ul, &r);
  EXPECT_COUNT(r, SCHED_RULE_PUSCH_OVERLAP, 2);

  memset(&r, 0, sizeof(r));
  reset();
  ul.Slot = 4;
  nfapi_nr_pucch_pdu_t *p = add_pucch(0x1234, 0, 10, 4);
  p->freq_hop_flag = 1;
  p->second_hop_prb = 50;
  add_pusch(0x5678, 50, 5, 10, 2); // first hop is elsewhere in these symbols
  sched_check_ul_slot(&cfg, &ul, &r);
  EXPECT_COUNT(r, SCHED_RULE_PUSCH_OVERLAP, 0);

  memset(&r, 0, sizeof(r));
  add_pusch(0x9abc, 50, 5, 12, 2); // second hop
  sched_check_ul_slot(&cfg, &ul, &r);
  EXPECT_COUNT(r, SCHED_RULE_PUSCH_OVERLAP, 1);
}

static void test_dci(void)
{
  sched_check_report_t r = {0};
  reset();
  nfapi_nr_dl_tti_pdcch_pdu_rel15_t *p = new_dl_pdcch();
  add_dci(p, 0x1234, 0, 4);
  add_dci(p, 0x5678, 4, 4);
  sched_check_dl_slot(&cfg, &dl, &ul_dci, &r);
  EXPECT_COUNT(r, SCHED_RULE_CCE_OVERLAP, 0);

  /* an UL DCI on the CCEs of a DL one, in the same CORESET */
  memset(&r, 0, sizeof(r));
  ul_dci.numPdus = 1;
  add_dci(&ul_dci.ul_dci_pdu_list[0].pdcch_pdu.pdcch_pdu_rel15, 0x9abc, 6, 2);
  sched_check_dl_slot(&cfg, &dl, &ul_dci, &r);
  EXPECT_COUNT(r, SCHED_RULE_CCE_OVERLAP, 1);

  /* the same CCE numbers in another CORESET are other CCEs */
  memset(&r, 0, sizeof(r));
  reset();
  add_dci(new_dl_pdcch(), 0x1234, 0, 4);
  p = new_dl_pdcch();
  add_dci(p, 0x5678, 0, 4);
  p->FreqDomainResource[0] = 0x0f;
  sched_check_dl_slot(&cfg, &dl, NULL, &r);
  EXPECT_COUNT(r, SCHED_RULE_CCE_OVERLAP, 0);

  /* a C-RNTI may have one DL and one UL DCI */
  memset(&r, 0, sizeof(r));
  reset();
  p = new_dl_pdcch();
  add_dci(p, 0x1234, 0, 4);
  add_dci(p, 0x1234, 4, 4);
  sched_check_dl_slot(&cfg, &dl, NULL, &r);
  EXPECT_COUNT(r, SCHED_RULE_DCI_PER_RNTI, 0);

  memset(&r, 0, sizeof(r));
  reset();
  p = new_dl_pdcch();
  add_dci(p, 0xfffe, 0, 4);
  add_dci(p, 0xfffe, 4, 4);
  sched_check_dl_slot(&cfg, &dl, NULL, &r);
  EXPECT_COUNT(r, SCHED_RULE_DCI_PER_RNTI, 1);

  memset(&r, 0, sizeof(r));
  reset();
  dl.Slot = 4;
  p = new_dl_pdcch();
  add_dci(p, 0x1234, 0, 4);
  sched_check_dl_slot(&cfg, &dl, NULL, &r);
  EXPECT_COUNT(r, SCHED_RULE_PDCCH_ON_UL, 1);
}

static sched_check_timing_t timing;

/* run the timing checker on the PDUs added so far, then clear them for the next slot */
static void timing_step(int frame, int slot, sched_check_report_t *r)
{
  dl.SFN = ul.SFN = frame;
  dl.Slot = ul.Slot = slot;
  sched_check_timing_slot(&cfg, &timing, &dl, &ul_dci, &ul, &tx, r);
  memset(&tx, 0, sizeof(tx));
  memset(&dl, 0, sizeof(dl));
  memset(&ul_dci, 0, sizeof(ul_dci));
  memset(&ul, 0, sizeof(ul));
}

static void timing_reset(void)
{
  reset();
  memset(&timing, 0, sizeof(timing));
  cfg.slots_per_frame = 20;
  sched_check_set_proc_time(&cfg, 1, false);
  cfg.ra_cr_timer_ms = 8;
  memset(&tx, 0, sizeof(tx));
}

/* MAC PDU of the last PDSCH added */
static uint8_t *add_tx(int len)
{
  nfapi_nr_dl_tti_request_body_t *b = &dl.dl_tti_request_body;
  nfapi_nr_dl_tti_pdsch_pdu_rel15_t *p = &b->dl_tti_pdu_list[b->nPDUs - 1].pdsch_pdu.pdsch_pdu_rel15;
  p->pduIndex = tx.Number_of_PDUs;
  nfapi_nr_pdu_t *pdu = &tx.pdu_list[tx.Number_of_PDUs++];
  pdu->PDU_index = p->pduIndex;
  pdu->num_TLV = 1;
  pdu->TLVs[0].length = len;
  return (uint8_t *)pdu->TLVs[0].value.direct;
}

static void add_prach(int start_symbol)
{
  nfapi_nr_ul_tti_request_number_of_pdus_t *pdu = &ul.pdus_list[ul.n_pdus++];
  pdu->pdu_type = NFAPI_NR_UL_CONFIG_PRACH_PDU_TYPE;
  pdu->prach_pdu.prach_start_symbol = start_symbol;
}

/* RAR for one preamble after a backoff subheader, giving the TC-RNTI */
static void add_rar(int ra_rnti, int tc_rnti)
{
  add_pdsch(ra_rnti, 0, 10, 1, 13);
  uint8_t *b = add_tx(9);
  b[0] = 0x80 | 0x05; /* E=1 T=0 BI */
  b[1] = 0x40 | 0x07; /* E=0 T=1 RAPID 7 */
  b[2 + 5] = tc_rnti >> 8;
  b[2 + 6] = tc_rnti & 0xff;
}

static void add_msg4(int tc_rnti, int start, int len)
{
  add_pdsch(tc_rnti, 0, 10, start, len);
  uint8_t *b = add_tx(7);
  b[0] = 62;
}

static nfapi_nr_dl_tti_pdcch_pdu_rel15_t *new_ul_pdcch(void)
{
  return &ul_dci.ul_dci_pdu_list[ul_dci.numPdus++].pdcch_pdu.pdcch_pdu_rel15;
}

static void test_harq_timing(void)
{
  sched_check_report_t r = {0};
  timing_reset();
  if (cfg.n1 != 13 || cfg.n2 != 12)
    printf("FAIL %s: N1 %d N2 %d, expected 13 and 12\n", __func__, cfg.n1, cfg.n2), failures++;

  /* 38.213 8.3/8.4: N_T,1 takes N1,0 = 14 at 15 kHz */
  sched_check_set_proc_time(&cfg, 0, false);
  if (cfg.nt1 != 14)
    printf("FAIL %s: N_T,1 %d at mu 0, expected 14\n", __func__, cfg.nt1), failures++;
  sched_check_set_proc_time(&cfg, 1, false);

  add_pucch(0x4601, 0, 13, 1); /* SR only: the UE is known from now on */
  add_prach(0);
  timing_step(0, 0, &r);
  /* PDSCH ends at symbol 28, Tproc,1 = 13 symbols: symbol 41 is the earliest, slot 2 symbol 13 */
  add_pdsch(0x4601, 0, 10, 1, 13);
  add_pdsch(0xffff, 20, 10, 1, 13);
  add_pdsch(0x0003, 40, 10, 1, 13); /* RA-RNTI of the PRACH of slot 0: never in UL, no HARQ-ACK */
  timing_step(0, 1, &r);
  add_pucch(0x4601, 0, 13, 1)->bit_len_harq = 1;
  timing_step(0, 2, &r);
  EXPECT_COUNT(r, SCHED_RULE_HARQ_TPROC1, 0);
  if (timing.min_harq_margin != 0 || timing.harq_unmatched != 0)
    printf("FAIL %s: margin %d unmatched %lu\n", __func__, timing.min_harq_margin, (unsigned long)timing.harq_unmatched), failures++;

  /* one symbol earlier is too early */
  add_pdsch(0x4601, 0, 10, 1, 13);
  timing_step(0, 3, &r);
  add_pucch(0x4601, 0, 12, 2)->bit_len_harq = 1;
  timing_step(0, 4, &r);
  EXPECT_COUNT(r, SCHED_RULE_HARQ_TPROC1, 1);

  /* mapping type A ending at symbol 5: d1,1 = 2, so slot 7 symbol 7 is legal and symbol 6 is not */
  add_pdsch(0x4601, 0, 10, 1, 5);
  timing_step(0, 6, &r);
  add_pucch(0x4601, 0, 6, 1)->bit_len_harq = 1;
  timing_step(0, 7, &r);
  EXPECT_COUNT(r, SCHED_RULE_HARQ_TPROC1, 2);

  /* two TBs, two bits; one more bit than PDSCHs is unmatched; time wraps at SFN 1024 */
  add_pdsch(0x4601, 0, 10, 1, 13)->NrOfCodewords = 2;
  timing_step(1023, 19, &r);
  add_pucch(0x4601, 0, 13, 1)->bit_len_harq = 3;
  timing_step(0, 1, &r);
  EXPECT_COUNT(r, SCHED_RULE_HARQ_TPROC1, 2);
  if (timing.harq_unmatched != 1)
    printf("FAIL %s: unmatched %lu, expected 1\n", __func__, (unsigned long)timing.harq_unmatched), failures++;

  /* HARQ-ACK on PUSCH counts too */
  add_pdsch(0x4601, 0, 10, 1, 13);
  timing_step(0, 2, &r);
  nfapi_nr_pusch_pdu_t *pusch = add_pusch(0x4601, 0, 10, 12, 2);
  pusch->pdu_bit_map = PUSCH_PDU_BITMAP_PUSCH_UCI;
  pusch->pusch_uci.harq_ack_bit_length = 1;
  timing_step(0, 3, &r);
  EXPECT_COUNT(r, SCHED_RULE_HARQ_TPROC1, 3);
}

/* a UE configured without random access: its first PDSCH comes before any UL PDU */
static void test_harq_first_pdsch(void)
{
  sched_check_report_t r = {0};
  timing_reset();
  add_pdsch(0x4601, 0, 10, 1, 13);
  timing_step(0, 0, &r);
  add_pucch(0x4601, 0, 13, 1); /* SR only */
  timing_step(0, 1, &r);
  /* the bit answers the PDSCH of slot 0, not the one of this slot */
  add_pdsch(0x4601, 0, 10, 1, 13);
  add_pucch(0x4601, 0, 13, 1)->bit_len_harq = 1;
  timing_step(0, 2, &r);
  add_pucch(0x4601, 0, 13, 1)->bit_len_harq = 1;
  timing_step(0, 4, &r);
  EXPECT_COUNT(r, SCHED_RULE_HARQ_TPROC1, 0);
  if (timing.harq_unmatched != 0 || timing.min_harq_margin != 14)
    printf("FAIL %s: margin %d unmatched %lu\n", __func__, timing.min_harq_margin, (unsigned long)timing.harq_unmatched), failures++;
}

/* UEs first seen in the same slot keep their own entry */
static void test_harq_ues_same_slot(void)
{
  sched_check_report_t r = {0};
  timing_reset();
  add_pdsch(0x4601, 0, 10, 1, 13);
  add_pdsch(0x4602, 10, 10, 1, 13);
  timing_step(0, 0, &r);
  add_pucch(0x4601, 0, 13, 1)->bit_len_harq = 1;
  add_pucch(0x4602, 1, 13, 1)->bit_len_harq = 1;
  timing_step(0, 2, &r);
  EXPECT_COUNT(r, SCHED_RULE_HARQ_TPROC1, 0);
  if (timing.harq_unmatched != 0 || timing.dropped != 0)
    printf("FAIL %s: unmatched %lu dropped %lu\n", __func__, (unsigned long)timing.harq_unmatched, (unsigned long)timing.dropped), failures++;
}

/* the UE transmits N_TA,offset early: the HARQ-ACK and PUSCH placed exactly at Tproc become too early */
static void test_timing_advance(void)
{
  sched_check_report_t r = {0};
  timing_reset();
  cfg.ta_symbols = 1;
  add_pucch(0x4601, 0, 13, 1);
  timing_step(0, 0, &r);
  add_pdsch(0x4601, 0, 10, 1, 13);
  add_dci(new_ul_pdcch(), 0x4601, 0, 4);
  timing_step(0, 1, &r);
  add_pucch(0x4601, 0, 13, 1)->bit_len_harq = 1;
  add_pusch(0x4601, 0, 10, 0, 14);
  timing_step(0, 2, &r);
  EXPECT_COUNT(r, SCHED_RULE_HARQ_TPROC1, 1);
  EXPECT_COUNT(r, SCHED_RULE_PUSCH_TPROC2, 1);
  if (timing.min_harq_margin != -1 || timing.min_pusch_margin != -1)
    printf("FAIL %s: margins %d %d, expected -1\n", __func__, timing.min_harq_margin, timing.min_pusch_margin), failures++;
}

static void test_pusch_timing(void)
{
  sched_check_report_t r = {0};
  timing_reset();
  /* DCI ends at symbol 1, Tproc,2 = 12 + d2,1 = 13: symbol 14 is the earliest */
  add_dci(new_ul_pdcch(), 0x4601, 0, 4);
  timing_step(0, 0, &r);
  add_pusch(0x4601, 0, 10, 0, 14);
  timing_step(0, 1, &r);
  EXPECT_COUNT(r, SCHED_RULE_PUSCH_TPROC2, 0);
  if (timing.min_pusch_margin != 0)
    printf("FAIL %s: margin %d, expected 0\n", __func__, timing.min_pusch_margin), failures++;

  /* K2 = 0 */
  add_dci(new_ul_pdcch(), 0x4601, 0, 4);
  add_pusch(0x4601, 0, 10, 10, 4);
  timing_step(0, 2, &r);
  EXPECT_COUNT(r, SCHED_RULE_PUSCH_TPROC2, 1);

  /* DCI ends at 43; PUSCH at 55 is legal only when its first symbol is DM-RS only (d2,1 = 0) */
  add_dci(new_ul_pdcch(), 0x4601, 0, 4);
  timing_step(0, 3, &r);
  nfapi_nr_pusch_pdu_t *p = add_pusch(0x4601, 0, 10, 13, 1);
  p->ul_dmrs_symb_pos = 1 << 13;
  p->num_dmrs_cdm_grps_no_data = 2;
  timing_step(0, 3, &r);
  EXPECT_COUNT(r, SCHED_RULE_PUSCH_TPROC2, 1);
  add_dci(new_ul_pdcch(), 0x4601, 0, 4);
  timing_step(0, 3, &r);
  p = add_pusch(0x4601, 0, 10, 13, 1);
  p->ul_dmrs_symb_pos = 1 << 13;
  p->num_dmrs_cdm_grps_no_data = 1;
  timing_step(0, 3, &r);
  EXPECT_COUNT(r, SCHED_RULE_PUSCH_TPROC2, 2);

  /* Msg3 has no DCI */
  add_pusch(0x4602, 20, 10, 0, 14);
  timing_step(0, 4, &r);
  EXPECT_COUNT(r, SCHED_RULE_PUSCH_TPROC2, 2);
  if (timing.pusch_unmatched != 1)
    printf("FAIL %s: unmatched %lu, expected 1\n", __func__, (unsigned long)timing.pusch_unmatched), failures++;
}

static void test_random_access(void)
{
  sched_check_report_t r = {0};
  timing_reset();
  /* PRACH in slot 8 symbol 0: RA-RNTI = 1 + 0 + 14 x 8 */
  const int ra_rnti = 1 + 14 * 8;
  add_prach(0);
  timing_step(0, 8, &r);
  /* RAR ends at slot 10 symbol 14 = 154; N_T,1 + N_T,2 + 0.5 ms = 13 + 12 + 14 = 39: Msg3 from 193 */
  add_rar(ra_rnti, 0x4601);
  timing_step(0, 10, &r);
  if (timing.ue[0].rnti != 0x4601 && timing.ue[1].rnti != 0x4601)
    printf("FAIL %s: TC-RNTI not taken from the RAR\n", __func__), failures++;
  add_pusch(0x4601, 0, 10, 11, 3); /* slot 13 symbol 11 = 193 */
  timing_step(0, 13, &r);
  EXPECT_COUNT(r, SCHED_RULE_MSG3_DELAY, 0);

  /* Msg3 ends at 196; timer 8 ms = 224 symbols: Msg4 DCI at 420 is too late, at 406 is fine */
  add_msg4(0x4601, 1, 13);
  add_dci(new_dl_pdcch(), 0x4601, 0, 4);
  timing_step(0, 29, &r);
  EXPECT_COUNT(r, SCHED_RULE_MSG4_TIMER, 0);
  /* Msg4 ends at 420: N_T,1 + 0.5 ms = 27, PUCCH from 447 = slot 31 symbol 13 */
  add_pucch(0x4601, 0, 12, 2)->bit_len_harq = 1;
  timing_step(0, 31, &r);
  EXPECT_COUNT(r, SCHED_RULE_MSG4_HARQ, 1);
  EXPECT_COUNT(r, SCHED_RULE_HARQ_TPROC1, 0);
  /* a PUSCH after Msg4 is not a Msg3: a Msg4 retransmission past the timer is late */
  add_pusch(0x4601, 0, 10, 0, 14);
  timing_step(0, 30, &r);
  add_msg4(0x4601, 1, 13);
  timing_step(0, 31, &r);
  EXPECT_COUNT(r, SCHED_RULE_MSG4_TIMER, 1);
  add_pucch(0x4601, 0, 13, 1)->bit_len_harq = 1;
  timing_step(0, 36, &r);
  /* not Msg4 any more: no Msg4 rule on a data PDSCH (LCID 4 first) */
  add_pdsch(0x4601, 0, 10, 1, 13);
  add_tx(10)[0] = 4;
  timing_step(1, 0, &r);
  add_pucch(0x4601, 0, 12, 2)->bit_len_harq = 1;
  timing_step(1, 2, &r);
  EXPECT_COUNT(r, SCHED_RULE_MSG4_HARQ, 1);
  EXPECT_COUNT(r, SCHED_RULE_MSG4_TIMER, 1);

  /* second access: Msg3 one symbol too early, Msg4 too late */
  add_prach(0);
  timing_step(2, 8, &r);
  add_rar(ra_rnti, 0x4602);
  timing_step(2, 10, &r);
  add_pusch(0x4602, 0, 10, 10, 4);
  timing_step(2, 13, &r);
  EXPECT_COUNT(r, SCHED_RULE_MSG3_DELAY, 1);
  add_msg4(0x4602, 1, 13);
  add_dci(new_dl_pdcch(), 0x4602, 0, 4);
  timing_step(3, 10, &r);
  EXPECT_COUNT(r, SCHED_RULE_MSG4_TIMER, 2);

  /* the timer is about the PDCCH: Msg3 ends 1 symbol before a slot, so the timer ends at symbol 13 of
   * the slot where the PDCCH at symbol 12 is in time and the PDSCH at symbol 13 is not */
  add_prach(0);
  timing_step(4, 8, &r);
  add_rar(ra_rnti, 0x4605);
  timing_step(4, 10, &r);
  add_pusch(0x4605, 0, 10, 11, 2);
  timing_step(4, 13, &r);
  add_msg4(0x4605, 13, 1);
  nfapi_nr_dl_tti_pdcch_pdu_rel15_t *c = new_dl_pdcch();
  add_dci(c, 0x4605, 0, 4);
  c->StartSymbolIndex = 12;
  timing_step(5, 9, &r);
  EXPECT_COUNT(r, SCHED_RULE_MSG4_TIMER, 2);

  /* E = 0 ends the RAR: what follows is padding */
  add_prach(0);
  timing_step(6, 8, &r);
  add_pdsch(ra_rnti, 0, 10, 1, 13);
  uint8_t *b = add_tx(16);
  b[0] = 0x40 | 0x07;
  b[0 + 1 + 6] = 0x06;
  b[0 + 1 + 5] = 0x46;
  b[8] = 0x40 | 0x08;
  b[8 + 1 + 5] = 0x46;
  b[8 + 1 + 6] = 0x04;
  timing_step(6, 10, &r);
  for (int i = 0; i < SCHED_CHECK_MAX_UES; i++)
    if (timing.ue[i].rnti == 0x4604)
      printf("FAIL %s: RAR read past E = 0\n", __func__), failures++;

  /* a PDSCH whose RNTI matches no PRACH is not a RAR */
  add_rar(1 + 14 * 3, 0x4603);
  timing_step(3, 11, &r);
  for (int i = 0; i < SCHED_CHECK_MAX_UES; i++)
    if (timing.ue[i].rnti == 0x4603)
      printf("FAIL %s: RAR accepted without PRACH\n", __func__), failures++;
}

static void step_until(int frame, int first, int last, sched_check_report_t *r)
{
  for (int s = first; s <= last; s++)
    timing_step(frame, s, r);
}

static void test_preamble_response(void)
{
  sched_check_report_t r = {0};
  timing_reset();
  cfg.ra_window_slots = 20;
  const int ra_rnti = 1 + 14 * 8;

  /* RAPID 7 detected at slot 8, answered by the RAR of slot 10; window ends at slot 8 + 1 + 20 */
  add_prach(0);
  timing_step(0, 8, &r);
  sched_check_preamble(&cfg, &timing, 0, 8, 7, 0, 0);
  timing_step(0, 9, &r);
  add_rar(ra_rnti, 0x4601);
  step_until(0, 10, 19, &r);
  step_until(1, 0, 19, &r);
  EXPECT_COUNT(r, SCHED_RULE_RA_UNANSWERED, 0);

  /* RAPID 9: the RAR answers RAPID 7 only; reported once the last slot of the window, 29, is over */
  add_prach(0);
  timing_step(2, 8, &r);
  sched_check_preamble(&cfg, &timing, 2, 8, 9, 0, 0);
  timing_step(2, 9, &r);
  add_rar(ra_rnti, 0x4602);
  step_until(2, 10, 19, &r);
  step_until(3, 0, 9, &r);
  EXPECT_COUNT(r, SCHED_RULE_RA_UNANSWERED, 0);
  timing_step(3, 10, &r);
  EXPECT_COUNT(r, SCHED_RULE_RA_UNANSWERED, 1);

  /* a MsgB on the MSGB-RNTI of the occasion answers it */
  add_prach(0);
  timing_step(4, 8, &r);
  sched_check_preamble(&cfg, &timing, 4, 8, 9, 0, 0);
  timing_step(4, 9, &r);
  add_pdsch(ra_rnti + 14 * 80 * 8 * 2, 0, 10, 1, 13);
  step_until(4, 10, 19, &r);
  step_until(5, 0, 19, &r);
  EXPECT_COUNT(r, SCHED_RULE_RA_UNANSWERED, 1);

  /* a RAR of another occasion (symbol 0, preamble at symbol 2) does not answer; the preamble is
   * reported 4 slots after its PRACH slot, so its window still ends at slot 29 */
  add_prach(0);
  step_until(6, 8, 12, &r);
  sched_check_preamble(&cfg, &timing, 6, 8, 7, 2, 0);
  add_rar(ra_rnti, 0x4603);
  step_until(6, 13, 19, &r);
  step_until(7, 0, 9, &r);
  EXPECT_COUNT(r, SCHED_RULE_RA_UNANSWERED, 1);
  timing_step(7, 10, &r);
  EXPECT_COUNT(r, SCHED_RULE_RA_UNANSWERED, 2);

  /* nor does a RAR of another frequency occasion */
  add_prach(0);
  timing_step(8, 8, &r);
  sched_check_preamble(&cfg, &timing, 8, 8, 7, 0, 1);
  add_rar(ra_rnti, 0x4604);
  step_until(8, 9, 19, &r);
  step_until(9, 0, 10, &r);
  EXPECT_COUNT(r, SCHED_RULE_RA_UNANSWERED, 3);
  if (timing.preambles != 5 || timing.preambles_answered != 2)
    printf("FAIL %s: %lu preambles, %lu answered\n", __func__, timing.preambles, timing.preambles_answered), failures++;
}

/* a PUSCH or PUCCH on the RBs and symbols of a PRACH occasion */
static void test_ul_on_prach(void)
{
  sched_check_report_t r = {0};
  reset();
  cfg.prach_first_prb = 0;
  cfg.prach_n_prb = 12;
  cfg.prach_dur = 4;
  ul.Slot = 9;
  nfapi_nr_ul_tti_request_number_of_pdus_t *pdu = &ul.pdus_list[ul.n_pdus++];
  pdu->pdu_type = NFAPI_NR_UL_CONFIG_PRACH_PDU_TYPE;
  pdu->prach_pdu.prach_start_symbol = 0;
  pdu->prach_pdu.num_prach_ocas = 3; /* symbols 0..11 */
  add_pucch(0x4601, 50, 13, 1); /* other RBs */
  add_pusch(0x4602, 12, 10, 0, 14); /* next to the PRACH RBs */
  add_pucch(0x4603, 5, 12, 1); /* symbol 12, after the occasions */
  sched_check_ul_slot(&cfg, &ul, &r);
  EXPECT_COUNT(r, SCHED_RULE_UL_ON_PRACH, 0);
  add_pucch(0x4604, 11, 4, 2);
  add_pusch(0x4605, 0, 4, 10, 4);
  sched_check_ul_slot(&cfg, &ul, &r);
  EXPECT_COUNT(r, SCHED_RULE_UL_ON_PRACH, 2);
  EXPECT_COUNT(r, SCHED_RULE_PUSCH_OVERLAP, 0);
}

/* the PHY receives at most max_pusch_per_slot PUSCHs in a slot (0: no limit): one finding per extra one */
static void test_pusch_capacity(void)
{
  sched_check_report_t r = {0};
  reset();
  ul.Slot = 9;
  cfg.max_pusch_per_slot = 3;
  for (int i = 0; i < 3; i++)
    add_pusch(0x4700 + i, 10 * i, 10, 0, 14);
  add_pucch(0x4710, 100, 13, 1);
  sched_check_ul_slot(&cfg, &ul, &r);
  EXPECT_COUNT(r, SCHED_RULE_PUSCH_CAPACITY, 0);
  add_pusch(0x4703, 30, 10, 0, 14);
  add_pusch(0x4704, 40, 10, 0, 14);
  memset(&r, 0, sizeof(r));
  sched_check_ul_slot(&cfg, &ul, &r);
  EXPECT_COUNT(r, SCHED_RULE_PUSCH_CAPACITY, 2);
  EXPECT_COUNT(r, SCHED_RULE_PUSCH_OVERLAP, 0);
  cfg.max_pusch_per_slot = 0;
  memset(&r, 0, sizeof(r));
  sched_check_ul_slot(&cfg, &ul, &r);
  EXPECT_COUNT(r, SCHED_RULE_PUSCH_CAPACITY, 0);
}

int main(void)
{
  test_pdsch_direction();
  test_ul_direction();
  test_ssb();
  test_rate();
  test_transform_precoding();
  test_overlap();
  test_dci();
  test_harq_timing();
  test_harq_first_pdsch();
  test_harq_ues_same_slot();
  test_pusch_timing();
  test_random_access();
  test_timing_advance();
  test_preamble_response();
  test_ul_on_prach();
  test_pusch_capacity();
  printf(failures ? "%d failures\n" : "all sched_check tests passed\n", failures);
  return failures != 0;
}
