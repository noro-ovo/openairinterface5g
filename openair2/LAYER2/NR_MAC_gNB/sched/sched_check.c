/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "sched_check.h"

#include <string.h>

/* FAPI values, not taken from the MAC headers on purpose */
#define CHECK_SI_RNTI 0xffff
#define CHECK_P_RNTI 0xfffe
#define CHECK_TP_ENABLED 0
#define CHECK_MAX_RECTS 64
#define CHECK_MAX_DCI 64
/* 38.321 6.1.5 and Table 6.2.1-1 */
#define CHECK_RAR_SIZE 7
#define CHECK_LCID_CONTENTION_RESOLUTION 62
/* 38.321 5.1.3a: MSGB-RNTI is the RA-RNTI of the same occasion plus 14 x 80 x 8 x 2 */
#define CHECK_RA_RNTI_SPAN (14 * 80 * 8 * 2)

static const struct {
  const char *id;
  const char *clause;
} rules[SCHED_RULE_COUNT] = {
    [SCHED_RULE_PDCCH_ON_UL] = {"DL-06", "38.213 11.1"},
    [SCHED_RULE_PDSCH_ON_UL] = {"DL-06", "38.213 11.1"},
    [SCHED_RULE_PUSCH_ON_DL] = {"DL-54", "38.213 11.1"},
    [SCHED_RULE_PUCCH_ON_DL] = {"DL-54", "38.213 11.1"},
    [SCHED_RULE_SSB_ON_UL] = {"CM-47", "38.213 11.1"},
    [SCHED_RULE_PDSCH_ON_SSB] = {"DL-25", "38.214 5.1.4"},
    [SCHED_RULE_PDSCH_RATE] = {"DL-38", "38.214 5.1.3"},
    [SCHED_RULE_PDSCH_OVERLAP] = {"CHK-PDSCH", "-"},
    [SCHED_RULE_PUSCH_OVERLAP] = {"CHK-PUSCH", "-"},
    [SCHED_RULE_TP_PRB] = {"UL-23", "38.211 6.3.1.4"},
    [SCHED_RULE_TP_LAYERS] = {"UL-24", "38.211 6.3.1.4"},
    [SCHED_RULE_CCE_OVERLAP] = {"CHK-CCE", "-"},
    [SCHED_RULE_DCI_PER_RNTI] = {"CM-10", "38.213 10.1"},
    [SCHED_RULE_HARQ_TPROC1] = {"DL-55", "38.214 5.3"},
    [SCHED_RULE_PUSCH_TPROC2] = {"UL-07", "38.214 6.4"},
    [SCHED_RULE_MSG3_DELAY] = {"CM-20", "38.213 8.3"},
    [SCHED_RULE_MSG4_HARQ] = {"CM-32", "38.213 8.4"},
    [SCHED_RULE_MSG4_TIMER] = {"CM-27", "38.321 5.1.5"},
    [SCHED_RULE_RA_UNANSWERED] = {"CM-36", "38.321 5.1.4"},
    [SCHED_RULE_UL_ON_PRACH] = {"CHK-PRACH", "-"},
    [SCHED_RULE_PUSCH_CAPACITY] = {"CHK-PUSCH-N", "-"},
};

/* PRBs in absolute position, by symbols of the slot */
typedef struct {
  int first_prb;
  int last_prb;
  uint16_t symbols;
  int rnti;
  bool pucch;
  bool prach;
} rect_t;

typedef struct {
  uint8_t start_symbol;
  uint8_t duration;
  uint8_t freq[6];
  uint16_t bwp_start;
  int first_cce;
  int num_cce;
  int rnti;
} dci_t;

const char *sched_rule_id(sched_rule_t rule)
{
  return rules[rule].id;
}

const char *sched_rule_clause(sched_rule_t rule)
{
  return rules[rule].clause;
}

static int report_add(sched_check_report_t *report, sched_rule_t rule, int frame, int slot, int rnti)
{
  report->count[rule]++;
  if (report->num_items < SCHED_CHECK_MAX_ITEMS)
    report->item[report->num_items++] = (sched_check_item_t){.rule = rule, .frame = frame, .slot = slot, .rnti = rnti};
  return 1;
}

static uint16_t symbols_mask(int start, int len)
{
  return ((1U << len) - 1) << start;
}

static bool on_direction(const sched_check_cfg_t *cfg, int slot, uint16_t symbols, sched_sym_t forbidden)
{
  if (cfg->slots_per_period == 0)
    return false;
  const uint8_t *sym = cfg->sym[slot % cfg->slots_per_period];
  for (int i = 0; i < NR_SYMBOLS_PER_SLOT; i++)
    if ((symbols >> i) & 1 && sym[i] == forbidden)
      return true;
  return false;
}

static bool rect_overlap(const rect_t *a, const rect_t *b)
{
  return (a->symbols & b->symbols) && a->first_prb <= b->last_prb && b->first_prb <= a->last_prb;
}

static int check_pdsch_rate(const nfapi_nr_dl_tti_pdsch_pdu_rel15_t *p, int frame, int slot, sched_check_report_t *report)
{
  /* DM-RS only: other overhead would only raise the rate, so a finding holds */
  const int n_dmrs = __builtin_popcount(p->dlDmrsSymbPos);
  const int re_dmrs = (p->dmrsConfigType == 0 ? 6 : 4) * p->numDmrsCdmGrpsNoData;
  const int n_re = p->rbSize * (NR_NB_SC_PER_RB * p->NrOfSymbols - n_dmrs * re_dmrs);
  int found = 0;
  for (int cw = 0; cw < p->NrOfCodewords; cw++) {
    /* the NDI is not in the PDSCH PDU: RV 0 stands for an initial transmission */
    if (p->rvIndex[cw] != 0)
      continue;
    const uint64_t tbs_bits = (uint64_t)p->TBSize[cw] * 8;
    const uint64_t info_bits = tbs_bits + (tbs_bits > 3824 ? 24 : 16);
    const uint64_t phy_bits = (uint64_t)n_re * p->qamModOrder[cw] * p->nrOfLayers;
    report->pdsch_symbols[p->NrOfSymbols]++;
    if (phy_bits > 0 && (int)(info_bits * 1000 / phy_bits) > report->max_rate_permille)
      report->max_rate_permille = info_bits * 1000 / phy_bits;
    if (info_bits * 100 > phy_bits * 95)
      found += report_add(report, SCHED_RULE_PDSCH_RATE, frame, slot, p->rnti);
  }
  return found;
}

static int check_ssb(const sched_check_cfg_t *cfg,
                     const nfapi_nr_dl_tti_ssb_pdu_rel15_t *ssb,
                     int frame,
                     int slot,
                     rect_t *rect,
                     sched_check_report_t *report)
{
  const int start = cfg->ssb_start_symbol[ssb->SsbBlockIndex];
  const int symbol = start % NR_SYMBOLS_PER_SLOT;
  const int first_prb = ssb->ssbOffsetPointA >> (cfg->fr2 ? cfg->ssb_mu - 2 : cfg->ssb_mu);
  rect->first_prb = first_prb;
  rect->last_prb = first_prb + 20 - 1 + (ssb->SsbSubcarrierOffset > 0);
  rect->symbols = symbols_mask(symbol, 4);
  if (on_direction(cfg, slot, rect->symbols, SCHED_SYM_UL))
    return report_add(report, SCHED_RULE_SSB_ON_UL, frame, slot, 0);
  return 0;
}

static int add_dci(dci_t *dci, int num_dci, const nfapi_nr_dl_tti_pdcch_pdu_rel15_t *p)
{
  for (int i = 0; i < p->numDlDci && num_dci < CHECK_MAX_DCI; i++) {
    const nfapi_nr_dl_dci_pdu_t *d = &p->dci_pdu[i];
    dci_t *o = &dci[num_dci++];
    *o = (dci_t){.start_symbol = p->StartSymbolIndex,
                 .duration = p->DurationSymbols,
                 .bwp_start = p->BWPStart,
                 .first_cce = d->CceIndex,
                 .num_cce = d->AggregationLevel,
                 .rnti = d->RNTI};
    memcpy(o->freq, p->FreqDomainResource, sizeof(o->freq));
  }
  return num_dci;
}

static bool same_coreset(const dci_t *a, const dci_t *b)
{
  return a->start_symbol == b->start_symbol && a->duration == b->duration && a->bwp_start == b->bwp_start
         && memcmp(a->freq, b->freq, sizeof(a->freq)) == 0;
}

static int check_dcis(const dci_t *dci, int num_dci, int frame, int slot, sched_check_report_t *report)
{
  int found = 0;
  for (int i = 0; i < num_dci; i++) {
    for (int j = i + 1; j < num_dci; j++) {
      const dci_t *a = &dci[i];
      const dci_t *b = &dci[j];
      if (same_coreset(a, b) && a->first_cce < b->first_cce + b->num_cce && b->first_cce < a->first_cce + a->num_cce)
        found += report_add(report, SCHED_RULE_CCE_OVERLAP, frame, slot, a->rnti);
      const bool common = a->rnti == CHECK_SI_RNTI || a->rnti == CHECK_P_RNTI;
      if (common && a->rnti == b->rnti)
        found += report_add(report, SCHED_RULE_DCI_PER_RNTI, frame, slot, a->rnti);
    }
  }
  return found;
}

int sched_check_dl_slot(const sched_check_cfg_t *cfg,
                        const nfapi_nr_dl_tti_request_t *dl,
                        const nfapi_nr_ul_dci_request_t *ul_dci,
                        sched_check_report_t *report)
{
  const int frame = dl->SFN;
  const int slot = dl->Slot;
  const nfapi_nr_dl_tti_request_body_t *body = &dl->dl_tti_request_body;
  rect_t ssb[CHECK_MAX_RECTS];
  rect_t pdsch[CHECK_MAX_RECTS];
  dci_t dci[CHECK_MAX_DCI];
  int num_ssb = 0, num_pdsch = 0, num_dci = 0;
  int found = 0;

  for (int i = 0; i < body->nPDUs; i++) {
    const nfapi_nr_dl_tti_request_pdu_t *pdu = &body->dl_tti_pdu_list[i];
    if (pdu->PDUType == NFAPI_NR_DL_TTI_SSB_PDU_TYPE && num_ssb < CHECK_MAX_RECTS) {
      report->checked[SCHED_PDU_SSB]++;
      found += check_ssb(cfg, &pdu->ssb_pdu.ssb_pdu_rel15, frame, slot, &ssb[num_ssb++], report);
    }
  }

  for (int i = 0; i < body->nPDUs; i++) {
    const nfapi_nr_dl_tti_request_pdu_t *pdu = &body->dl_tti_pdu_list[i];
    if (pdu->PDUType == NFAPI_NR_DL_TTI_PDCCH_PDU_TYPE) {
      const nfapi_nr_dl_tti_pdcch_pdu_rel15_t *p = &pdu->pdcch_pdu.pdcch_pdu_rel15;
      report->checked[SCHED_PDU_PDCCH] += p->numDlDci;
      if (on_direction(cfg, slot, symbols_mask(p->StartSymbolIndex, p->DurationSymbols), SCHED_SYM_UL))
        found += report_add(report, SCHED_RULE_PDCCH_ON_UL, frame, slot, p->numDlDci ? p->dci_pdu[0].RNTI : 0);
      num_dci = add_dci(dci, num_dci, p);
    } else if (pdu->PDUType == NFAPI_NR_DL_TTI_PDSCH_PDU_TYPE) {
      const nfapi_nr_dl_tti_pdsch_pdu_rel15_t *p = &pdu->pdsch_pdu.pdsch_pdu_rel15;
      report->checked[SCHED_PDU_PDSCH]++;
      const uint16_t symbols = symbols_mask(p->StartSymbolIndex, p->NrOfSymbols);
      if (on_direction(cfg, slot, symbols, SCHED_SYM_UL))
        found += report_add(report, SCHED_RULE_PDSCH_ON_UL, frame, slot, p->rnti);
      found += check_pdsch_rate(p, frame, slot, report);
      /* interleaved VRB and type 0 are not used by any scheduler yet */
      if (p->resourceAlloc != 1 || p->VRBtoPRBMapping != 0 || num_pdsch == CHECK_MAX_RECTS)
        continue;
      const rect_t r = {.first_prb = p->BWPStart + p->rbStart,
                        .last_prb = p->BWPStart + p->rbStart + p->rbSize - 1,
                        .symbols = symbols,
                        .rnti = p->rnti};
      /* SIB1 may overlap SSB PRBs: 38.214 5.1.4 limits this rule to other SI */
      for (int s = 0; p->rnti != CHECK_SI_RNTI && s < num_ssb; s++)
        if (rect_overlap(&r, &ssb[s]))
          found += report_add(report, SCHED_RULE_PDSCH_ON_SSB, frame, slot, p->rnti);
      for (int o = 0; cfg->num_beams <= 1 && o < num_pdsch; o++)
        if (rect_overlap(&r, &pdsch[o]))
          found += report_add(report, SCHED_RULE_PDSCH_OVERLAP, frame, slot, p->rnti);
      pdsch[num_pdsch++] = r;
    }
  }

  for (int i = 0; ul_dci != NULL && i < ul_dci->numPdus; i++) {
    const nfapi_nr_dl_tti_pdcch_pdu_rel15_t *p = &ul_dci->ul_dci_pdu_list[i].pdcch_pdu.pdcch_pdu_rel15;
    report->checked[SCHED_PDU_PDCCH] += p->numDlDci;
    if (on_direction(cfg, slot, symbols_mask(p->StartSymbolIndex, p->DurationSymbols), SCHED_SYM_UL))
      found += report_add(report, SCHED_RULE_PDCCH_ON_UL, frame, slot, p->numDlDci ? p->dci_pdu[0].RNTI : 0);
    num_dci = add_dci(dci, num_dci, p);
  }
  return found + check_dcis(dci, num_dci, frame, slot, report);
}

static bool is_2a3b5c(int n)
{
  if (n <= 0)
    return false;
  static const int primes[] = {2, 3, 5};
  for (int i = 0; i < 3; i++)
    while (n % primes[i] == 0)
      n /= primes[i];
  return n == 1;
}

static int pucch_rects(const nfapi_nr_pucch_pdu_t *p, rect_t *r)
{
  const int start = p->bwp_start + p->prb_start;
  if (!p->freq_hop_flag) {
    r[0] = (rect_t){start, start + p->prb_size - 1, symbols_mask(p->start_symbol_index, p->nr_of_symbols), p->rnti, true};
    return 1;
  }
  const int first_hop = p->nr_of_symbols / 2;
  const int hop = p->bwp_start + p->second_hop_prb;
  r[0] = (rect_t){start, start + p->prb_size - 1, symbols_mask(p->start_symbol_index, first_hop), p->rnti, true};
  r[1] = (rect_t){hop,
                  hop + p->prb_size - 1,
                  symbols_mask(p->start_symbol_index + first_hop, p->nr_of_symbols - first_hop),
                  p->rnti,
                  true};
  return 2;
}

int sched_check_ul_slot(const sched_check_cfg_t *cfg, const nfapi_nr_ul_tti_request_t *ul, sched_check_report_t *report)
{
  const int frame = ul->SFN;
  const int slot = ul->Slot;
  rect_t rect[CHECK_MAX_RECTS];
  int num_rect = 0;
  int found = 0;
  int num_pusch = 0;

  for (int i = 0; i < ul->n_pdus; i++) {
    const nfapi_nr_ul_tti_request_number_of_pdus_t *pdu = &ul->pdus_list[i];
    if (pdu->pdu_type == NFAPI_NR_UL_CONFIG_PUSCH_PDU_TYPE) {
      const nfapi_nr_pusch_pdu_t *p = &pdu->pusch_pdu;
      report->checked[SCHED_PDU_PUSCH]++;
      /* more PUSCHs than the PHY receives in a slot: the extra ones are lost */
      if (++num_pusch > cfg->max_pusch_per_slot && cfg->max_pusch_per_slot > 0)
        found += report_add(report, SCHED_RULE_PUSCH_CAPACITY, frame, slot, p->rnti);
      const uint16_t symbols = symbols_mask(p->start_symbol_index, p->nr_of_symbols);
      if (on_direction(cfg, slot, symbols, SCHED_SYM_DL))
        found += report_add(report, SCHED_RULE_PUSCH_ON_DL, frame, slot, p->rnti);
      if (p->transform_precoding == CHECK_TP_ENABLED) {
        if (!is_2a3b5c(p->rb_size))
          found += report_add(report, SCHED_RULE_TP_PRB, frame, slot, p->rnti);
        if (p->nrOfLayers != 1)
          found += report_add(report, SCHED_RULE_TP_LAYERS, frame, slot, p->rnti);
      }
      if (num_rect < CHECK_MAX_RECTS)
        rect[num_rect++] = (rect_t){.first_prb = p->bwp_start + p->rb_start,
                                    .last_prb = p->bwp_start + p->rb_start + p->rb_size - 1,
                                    .symbols = symbols,
                                    .rnti = p->rnti};
    } else if (pdu->pdu_type == NFAPI_NR_UL_CONFIG_PRACH_PDU_TYPE) {
      const nfapi_nr_prach_pdu_t *p = &pdu->prach_pdu;
      const int n_sym = p->num_prach_ocas * cfg->prach_dur;
      if (cfg->prach_n_prb > 0 && n_sym > 0 && num_rect < CHECK_MAX_RECTS)
        rect[num_rect++] = (rect_t){.first_prb = cfg->prach_first_prb,
                                    .last_prb = cfg->prach_first_prb + cfg->prach_n_prb - 1,
                                    .symbols = symbols_mask(p->prach_start_symbol, n_sym > 14 - p->prach_start_symbol ? 14 - p->prach_start_symbol : n_sym),
                                    .prach = true};
    } else if (pdu->pdu_type == NFAPI_NR_UL_CONFIG_PUCCH_PDU_TYPE) {
      const nfapi_nr_pucch_pdu_t *p = &pdu->pucch_pdu;
      report->checked[SCHED_PDU_PUCCH]++;
      if (on_direction(cfg, slot, symbols_mask(p->start_symbol_index, p->nr_of_symbols), SCHED_SYM_DL))
        found += report_add(report, SCHED_RULE_PUCCH_ON_DL, frame, slot, p->rnti);
      if (num_rect + 2 <= CHECK_MAX_RECTS)
        num_rect += pucch_rects(p, &rect[num_rect]);
    }
  }

  /* PUCCH of several UEs legally share PRBs through their cyclic shifts */
  for (int a = 0; cfg->num_beams <= 1 && a < num_rect; a++)
    for (int b = a + 1; b < num_rect; b++) {
      if (rect[a].prach && rect[b].prach)
        continue;
      if (rect[a].prach || rect[b].prach) {
        /* a PUSCH or PUCCH on the RBs and symbols of a PRACH occasion */
        if (rect_overlap(&rect[a], &rect[b]))
          found += report_add(report, SCHED_RULE_UL_ON_PRACH, frame, slot, rect[a].prach ? rect[b].rnti : rect[a].rnti);
        continue;
      }
      if (!(rect[a].pucch && rect[b].pucch) && rect_overlap(&rect[a], &rect[b]))
        found += report_add(report, SCHED_RULE_PUSCH_OVERLAP, frame, slot, rect[a].rnti);
    }
  return found;
}

/* pending entries this old were never answered, as when a UE leaves */
#define CHECK_PENDING_SLOTS 64

void sched_check_set_proc_time(sched_check_cfg_t *cfg, int mu, bool dmrs_pos0)
{
  /* N1,0 is 14 instead of 13 with an additional DM-RS at symbol 12, unknown here */
  static const int n1_pos0[] = {8, 10, 17, 20, 0, 80, 160};
  static const int n1_other[] = {13, 13, 20, 24, 0, 96, 192};
  static const int n2[] = {10, 12, 23, 36, 0, 144, 288};
  cfg->n1 = dmrs_pos0 ? n1_pos0[mu] : n1_other[mu];
  cfg->n2 = n2[mu];
  cfg->mu = mu;
  /* N_T,1 of 38.213 8.3 and 8.4: capability 1 with additional DM-RS, N1,0 = 14 */
  cfg->nt1 = mu == 0 ? 14 : n1_other[mu];
}

static sched_check_ue_t *timing_ue(sched_check_timing_t *t, int rnti, bool create)
{
  /* a free entry (RNTI 0 is never assigned) before the least recently seen UE */
  sched_check_ue_t *oldest = &t->ue[0];
  for (int i = 0; i < SCHED_CHECK_MAX_UES; i++) {
    sched_check_ue_t *ue = &t->ue[i];
    if (ue->rnti == rnti)
      return ue;
    if (oldest->rnti != 0 && (ue->rnti == 0 || ue->last_slot < oldest->last_slot))
      oldest = ue;
  }
  if (!create)
    return NULL;
  t->dropped += oldest->harq_len + oldest->dci_len;
  *oldest = (sched_check_ue_t){.rnti = rnti, .last_slot = t->slot, .rar_end = -1, .msg3_end = -1};
  return oldest;
}

static void expire(sched_check_timing_t *t, int64_t *queue, int *head, int *len, int64_t limit)
{
  while (*len > 0 && queue[*head] < limit) {
    *head = (*head + 1) % SCHED_CHECK_QUEUE;
    (*len)--;
    t->dropped++;
  }
}

static void push(sched_check_timing_t *t, int64_t *queue, int head, int *len, int64_t value)
{
  if (*len == SCHED_CHECK_QUEUE) {
    t->dropped++;
    return;
  }
  queue[(head + *len) % SCHED_CHECK_QUEUE] = value;
  (*len)++;
}

static bool pop(int64_t *queue, int *head, int *len, int64_t *value)
{
  if (*len == 0)
    return false;
  *value = queue[*head];
  *head = (*head + 1) % SCHED_CHECK_QUEUE;
  (*len)--;
  return true;
}

/* d1,1 of 38.214 5.3 for capability 1; the mapping type is not in FAPI, but
 * only type A has its first DM-RS at symbol 2 or 3 after the PDSCH start */
static int pdsch_d11(const nfapi_nr_dl_tti_pdsch_pdu_rel15_t *p, int pdcch_overlap)
{
  const int start = p->StartSymbolIndex;
  const int len = p->NrOfSymbols;
  const int first_dmrs = p->dlDmrsSymbPos ? __builtin_ctz(p->dlDmrsSymbPos) : start;
  const bool type_a = (first_dmrs == 2 || first_dmrs == 3) && start <= first_dmrs;
  if (type_a) {
    const int last = start + len - 1;
    return last < 7 ? 7 - last : 0;
  }
  if (len >= 7)
    return 0;
  if (len >= 4)
    return 7 - len;
  if (len == 3)
    return 3 + (pdcch_overlap > 0);
  return 3 + pdcch_overlap;
}

/* PDCCH of the DL DCI for this RNTI in the slot */
static const nfapi_nr_dl_tti_pdcch_pdu_rel15_t *find_pdcch(const nfapi_nr_dl_tti_request_body_t *body, int rnti)
{
  for (int i = 0; i < body->nPDUs; i++) {
    const nfapi_nr_dl_tti_request_pdu_t *pdu = &body->dl_tti_pdu_list[i];
    if (pdu->PDUType != NFAPI_NR_DL_TTI_PDCCH_PDU_TYPE)
      continue;
    const nfapi_nr_dl_tti_pdcch_pdu_rel15_t *c = &pdu->pdcch_pdu.pdcch_pdu_rel15;
    for (int d = 0; d < c->numDlDci; d++)
      if (c->dci_pdu[d].RNTI == rnti)
        return c;
  }
  return NULL;
}

/* symbols of the scheduling PDCCH that are also PDSCH symbols */
static int pdcch_overlap(const nfapi_nr_dl_tti_pdcch_pdu_rel15_t *c, const nfapi_nr_dl_tti_pdsch_pdu_rel15_t *p)
{
  if (c == NULL)
    return 0;
  return __builtin_popcount(symbols_mask(p->StartSymbolIndex, p->NrOfSymbols) & symbols_mask(c->StartSymbolIndex, c->DurationSymbols));
}

/* 38.321 5.1.3: RA-RNTI = 1 + s_id + 14 t_id + 14 x 80 f_id + 14 x 80 x 8 ul_carrier_id; t_id counts
 * slots of the PRACH numerology, subframes for long preambles, so both are accepted */
static bool is_ra_rnti(const sched_check_cfg_t *cfg, const sched_check_timing_t *t, int rnti)
{
  if (rnti < 1 || rnti > CHECK_RA_RNTI_SPAN)
    return false;
  const int t_id = (rnti - 1) / 14 % 80;
  const int f_id = (rnti - 1) / (14 * 80) % 8;
  for (int i = 0; i < SCHED_CHECK_MAX_PRACH; i++) {
    const sched_check_prach_t *p = &t->prach[i];
    if (p->until > t->slot && (t_id == p->slot || t_id == p->slot >> cfg->mu) && f_id >= p->first_ra)
      return true;
  }
  return false;
}

/* MAC PDU of a PDSCH */
static const uint8_t *tx_payload(const nfapi_nr_tx_data_request_t *tx, int pdu_index, uint32_t *len)
{
  for (int i = 0; tx != NULL && i < tx->Number_of_PDUs; i++) {
    const nfapi_nr_pdu_t *pdu = &tx->pdu_list[i];
    if (pdu->PDU_index != pdu_index || pdu->num_TLV < 1)
      continue;
    const nfapi_nr_tx_data_request_tlv_t *tlv = &pdu->TLVs[0];
    *len = tlv->length;
    return (const uint8_t *)(tlv->tag ? tlv->value.ptr : tlv->value.direct);
  }
  return NULL;
}

/* the pending preambles of the occasion of an RA-RNTI, all RAPIDs if rapid < 0 */
static void preamble_answered(const sched_check_cfg_t *cfg, sched_check_timing_t *t, int ra_rnti, int rapid)
{
  const int s_id = (ra_rnti - 1) % 14;
  const int t_id = (ra_rnti - 1) / 14 % 80;
  const int f_id = (ra_rnti - 1) / (14 * 80) % 8;
  for (int i = 0; i < SCHED_CHECK_MAX_PREAMBLES; i++) {
    sched_check_preamble_t *p = &t->preamble[i];
    if (p->deadline < 0 || p->s_id != s_id || p->f_id != f_id || (t_id != p->slot && t_id != p->slot >> cfg->mu))
      continue;
    if (rapid >= 0 && p->rapid != rapid)
      continue;
    p->deadline = -1;
    t->preambles_answered++;
  }
}

void sched_check_preamble(const sched_check_cfg_t *cfg, sched_check_timing_t *t, int frame, int slot, int rapid, int s_id, int f_id)
{
  if (!t->started)
    return;
  /* the PRACH occasion is behind the last scheduled slot */
  const int frame_slots = 1024 * cfg->slots_per_frame;
  const int behind = (t->last_frame * cfg->slots_per_frame + t->last_slot - (frame * cfg->slots_per_frame + slot) + frame_slots) % frame_slots;
  sched_check_preamble_t *free_entry = NULL;
  for (int i = 0; i < SCHED_CHECK_MAX_PREAMBLES && free_entry == NULL; i++)
    if (t->preamble[i].deadline < 0)
      free_entry = &t->preamble[i];
  if (free_entry == NULL) {
    t->dropped++;
    return;
  }
  /* the window starts at the first PDCCH occasion after the PRACH occasion: one slot of slack */
  *free_entry = (sched_check_preamble_t){.deadline = t->slot - behind + 1 + cfg->ra_window_slots,
                                         .rapid = rapid,
                                         .s_id = s_id,
                                         .f_id = f_id,
                                         .slot = slot};
  t->preambles++;
}

/* 38.321 6.1.5: E/T/R/R/BI or E/T/RAPID subheaders, a 7-byte MAC RAR after each RAPID,
 * Temporary C-RNTI in its last 2 bytes; every RAR starts a random access for that RNTI */
static void rar_received(const sched_check_cfg_t *cfg, sched_check_timing_t *t, int ra_rnti, const uint8_t *b, uint32_t len, int64_t end)
{
  uint32_t i = 0;
  while (i < len) {
    const uint8_t h = b[i++];
    if ((h >> 6) & 1) {
      if (i + CHECK_RAR_SIZE > len)
        return;
      preamble_answered(cfg, t, ra_rnti, h & 0x3f);
      sched_check_ue_t *ue = timing_ue(t, b[i + 5] << 8 | b[i + 6], true);
      ue->last_slot = t->slot;
      ue->rar_end = end;
      ue->in_ra = true;
      ue->msg3_end = -1;
      t->rars++;
      i += CHECK_RAR_SIZE;
    }
    if (!(h >> 7))
      return;
  }
}

static int harq_answered(const sched_check_cfg_t *cfg,
                         sched_check_timing_t *t,
                         sched_check_ue_t *ue,
                         int bits,
                         int64_t first_symbol,
                         int frame,
                         int slot,
                         sched_check_report_t *report)
{
  int found = 0;
  for (int b = 0; b < bits; b++) {
    const int64_t earliest_msg4 = ue->harq_earliest_msg4[ue->harq_head];
    int64_t earliest;
    if (!pop(ue->harq_earliest, &ue->harq_head, &ue->harq_len, &earliest)) {
      t->harq_unmatched += bits - b;
      break;
    }
    const int margin = first_symbol - cfg->ta_symbols - earliest;
    if (margin < t->min_harq_margin)
      t->min_harq_margin = margin;
    if (margin < 0)
      found += report_add(report, SCHED_RULE_HARQ_TPROC1, frame, slot, ue->rnti);
    if (earliest_msg4 >= 0) {
      if (first_symbol - earliest_msg4 < t->min_msg4_harq_margin)
        t->min_msg4_harq_margin = first_symbol - earliest_msg4;
      if (first_symbol < earliest_msg4)
        found += report_add(report, SCHED_RULE_MSG4_HARQ, frame, slot, ue->rnti);
    }
  }
  return found;
}

int sched_check_timing_slot(const sched_check_cfg_t *cfg,
                            sched_check_timing_t *t,
                            const nfapi_nr_dl_tti_request_t *dl,
                            const nfapi_nr_ul_dci_request_t *ul_dci,
                            const nfapi_nr_ul_tti_request_t *ul,
                            const nfapi_nr_tx_data_request_t *tx,
                            sched_check_report_t *report)
{
  const int frame = dl->SFN;
  const int slot = dl->Slot;
  const int frame_slots = 1024 * cfg->slots_per_frame;
  if (!t->started) {
    t->started = true;
    t->min_harq_margin = INT32_MAX;
    t->min_pusch_margin = INT32_MAX;
    t->min_msg3_margin = INT32_MAX;
    t->min_msg4_harq_margin = INT32_MAX;
    t->min_msg4_timer_margin = INT32_MAX;
    for (int i = 0; i < SCHED_CHECK_MAX_PREAMBLES; i++)
      t->preamble[i].deadline = -1;
  } else {
    const int now = frame * cfg->slots_per_frame + slot;
    const int before = t->last_frame * cfg->slots_per_frame + t->last_slot;
    t->slot += (now - before + frame_slots) % frame_slots;
  }
  t->last_frame = frame;
  t->last_slot = slot;
  const int64_t slot_start = t->slot * NR_SYMBOLS_PER_SLOT;
  const int64_t expired = (t->slot - CHECK_PENDING_SLOTS) * NR_SYMBOLS_PER_SLOT;
  for (int i = 0; i < SCHED_CHECK_MAX_UES; i++) {
    sched_check_ue_t *ue = &t->ue[i];
    expire(t, ue->harq_earliest, &ue->harq_head, &ue->harq_len, expired);
    expire(t, ue->dci_end, &ue->dci_head, &ue->dci_len, expired);
  }
  int found = 0;
  for (int i = 0; i < SCHED_CHECK_MAX_PREAMBLES; i++) {
    sched_check_preamble_t *p = &t->preamble[i];
    if (p->deadline >= 0 && p->deadline < t->slot) {
      p->deadline = -1;
      found += report_add(report, SCHED_RULE_RA_UNANSWERED, frame, slot, p->rapid);
    }
  }

  /* UL DCIs first: K2 = 0 puts their PUSCH in this slot */
  for (int i = 0; ul_dci != NULL && i < ul_dci->numPdus; i++) {
    const nfapi_nr_dl_tti_pdcch_pdu_rel15_t *c = &ul_dci->ul_dci_pdu_list[i].pdcch_pdu.pdcch_pdu_rel15;
    for (int d = 0; d < c->numDlDci; d++) {
      sched_check_ue_t *ue = timing_ue(t, c->dci_pdu[d].RNTI, true);
      ue->last_slot = t->slot;
      push(t, ue->dci_end, ue->dci_head, &ue->dci_len, slot_start + c->StartSymbolIndex + c->DurationSymbols);
    }
  }

  const nfapi_nr_dl_tti_request_body_t *body = &dl->dl_tti_request_body;
  for (int i = 0; i < body->nPDUs; i++) {
    const nfapi_nr_dl_tti_request_pdu_t *pdu = &body->dl_tti_pdu_list[i];
    if (pdu->PDUType != NFAPI_NR_DL_TTI_PDSCH_PDU_TYPE)
      continue;
    const nfapi_nr_dl_tti_pdsch_pdu_rel15_t *p = &pdu->pdsch_pdu.pdsch_pdu_rel15;
    const int64_t end = slot_start + p->StartSymbolIndex + p->NrOfSymbols;
    uint32_t len = 0;
    const uint8_t *mac_pdu = tx_payload(tx, p->pduIndex, &len);
    sched_check_ue_t *ue = timing_ue(t, p->rnti, false);
    /* SI, P, RA and MsgB-RNTI PDSCH have no HARQ-ACK; any other RNTI is a UE, possibly configured
     * without random access, whose first PDSCH comes before any UL PDU */
    if (ue == NULL) {
      if (p->rnti == CHECK_SI_RNTI || p->rnti == CHECK_P_RNTI)
        continue;
      if (is_ra_rnti(cfg, t, p->rnti)) {
        if (mac_pdu != NULL)
          rar_received(cfg, t, p->rnti, mac_pdu, len, end);
        continue;
      }
      /* a MsgB answers every preamble of its occasion, as successRAR or fallbackRAR */
      if (p->rnti > CHECK_RA_RNTI_SPAN && is_ra_rnti(cfg, t, p->rnti - CHECK_RA_RNTI_SPAN)) {
        preamble_answered(cfg, t, p->rnti - CHECK_RA_RNTI_SPAN, -1);
        continue;
      }
      ue = timing_ue(t, p->rnti, true);
    }
    ue->last_slot = t->slot;
    const nfapi_nr_dl_tti_pdcch_pdu_rel15_t *pdcch = find_pdcch(body, p->rnti);
    const int64_t earliest = end + cfg->n1 + pdsch_d11(p, pdcch_overlap(pdcch, p));
    int64_t earliest_msg4 = -1;
    if (mac_pdu != NULL && len > 0 && (mac_pdu[0] & 0x3f) == CHECK_LCID_CONTENTION_RESOLUTION && ue->msg3_end >= 0) {
      ue->in_ra = false;
      t->msg4s++;
      earliest_msg4 = end + cfg->nt1 + (7 << cfg->mu);
      /* the UE monitors the PDCCH only while the timer runs */
      const int64_t dci_start = slot_start + (pdcch ? pdcch->StartSymbolIndex : p->StartSymbolIndex);
      const int64_t timer_end = ue->msg3_end + (int64_t)cfg->ra_cr_timer_ms * (NR_SYMBOLS_PER_SLOT << cfg->mu);
      if (timer_end - 1 - dci_start < t->min_msg4_timer_margin)
        t->min_msg4_timer_margin = timer_end - 1 - dci_start;
      if (cfg->ra_cr_timer_ms > 0 && dci_start >= timer_end)
        found += report_add(report, SCHED_RULE_MSG4_TIMER, frame, slot, p->rnti);
    }
    /* one HARQ-ACK bit per TB, no spatial bundling */
    for (int cw = 0; cw < p->NrOfCodewords; cw++) {
      ue->harq_earliest_msg4[(ue->harq_head + ue->harq_len) % SCHED_CHECK_QUEUE] = earliest_msg4;
      push(t, ue->harq_earliest, ue->harq_head, &ue->harq_len, earliest);
    }
  }

  for (int i = 0; ul != NULL && i < ul->n_pdus; i++) {
    const nfapi_nr_ul_tti_request_number_of_pdus_t *pdu = &ul->pdus_list[i];
    if (pdu->pdu_type == NFAPI_NR_UL_CONFIG_PUSCH_PDU_TYPE) {
      const nfapi_nr_pusch_pdu_t *p = &pdu->pusch_pdu;
      sched_check_ue_t *ue = timing_ue(t, p->rnti, true);
      ue->last_slot = t->slot;
      const int64_t first_symbol = slot_start + p->start_symbol_index;
      int64_t dci_end;
      if (pop(ue->dci_end, &ue->dci_head, &ue->dci_len, &dci_end)) {
        const int groups = p->dmrs_config_type == 0 ? 2 : 3;
        const bool dmrs_only = (p->ul_dmrs_symb_pos >> p->start_symbol_index) & 1 && p->num_dmrs_cdm_grps_no_data == groups;
        const int margin = first_symbol - cfg->ta_symbols - (dci_end + cfg->n2 + !dmrs_only);
        if (margin < t->min_pusch_margin)
          t->min_pusch_margin = margin;
        if (margin < 0)
          found += report_add(report, SCHED_RULE_PUSCH_TPROC2, frame, slot, p->rnti);
      } else {
        t->pusch_unmatched++;
        /* Msg3, scheduled by the RAR */
        if (ue->rar_end >= 0) {
          const int margin = first_symbol - ue->rar_end - (cfg->nt1 + cfg->n2 + (7 << cfg->mu));
          t->msg3s++;
          if (margin < t->min_msg3_margin)
            t->min_msg3_margin = margin;
          if (margin < 0)
            found += report_add(report, SCHED_RULE_MSG3_DELAY, frame, slot, p->rnti);
        }
      }
      ue->rar_end = -1;
      if (ue->in_ra)
        ue->msg3_end = first_symbol + p->nr_of_symbols;
      if (p->pdu_bit_map & PUSCH_PDU_BITMAP_PUSCH_UCI)
        found += harq_answered(cfg, t, ue, p->pusch_uci.harq_ack_bit_length, first_symbol, frame, slot, report);
    } else if (pdu->pdu_type == NFAPI_NR_UL_CONFIG_PRACH_PDU_TYPE) {
      /* the RAR window is at most 40 ms */
      t->prach[t->prach_next] = (sched_check_prach_t){.slot = ul->Slot,
                                                      .first_ra = pdu->prach_pdu.num_ra,
                                                      .until = t->slot + 40 * (1 << cfg->mu)};
      t->prach_next = (t->prach_next + 1) % SCHED_CHECK_MAX_PRACH;
    } else if (pdu->pdu_type == NFAPI_NR_UL_CONFIG_PUCCH_PDU_TYPE) {
      const nfapi_nr_pucch_pdu_t *p = &pdu->pucch_pdu;
      sched_check_ue_t *ue = timing_ue(t, p->rnti, true);
      ue->last_slot = t->slot;
      found += harq_answered(cfg, t, ue, p->bit_len_harq, slot_start + p->start_symbol_index, frame, slot, report);
    }
  }
  return found;
}
