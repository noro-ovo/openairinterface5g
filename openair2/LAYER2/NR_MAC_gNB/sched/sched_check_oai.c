/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* Link between the MAC and sched_check.c, which must not depend on the MAC */

#include <stdlib.h>
#include <string.h>

#include "common/utils/LOG/log.h"
#include "openair2/LAYER2/NR_MAC_COMMON/nr_mac_common.h"
#include "openair2/LAYER2/NR_MAC_gNB/mac_proto.h"
#include "openair2/NR_PHY_INTERFACE/NR_IF_Module.h"
#include "sched_check.h"
#include "sched_check_oai.h"

#define SCHED_CHECK_SUMMARY_FRAMES 1000

static void fill_cfg(const nr_cell_sched_t *cell, sched_check_cfg_t *cfg)
{
  const NR_ServingCellConfigCommon_t *scc = cell->common_channels.ServingCellConfigCommon;
  const frame_structure_t *fs = &cell->frame_structure;
  if (fs->frame_type == TDD) {
    cfg->slots_per_period = fs->numb_slots_period;
    for (int s = 0; s < fs->numb_slots_period; s++) {
      const tdd_bitmap_t *bm = &fs->period_cfg.tdd_slot_bitmap[s];
      for (int i = 0; i < NR_SYMBOLS_PER_SLOT; i++) {
        if (bm->slot_type == TDD_NR_DOWNLINK_SLOT)
          cfg->sym[s][i] = SCHED_SYM_DL;
        else if (bm->slot_type == TDD_NR_UPLINK_SLOT)
          cfg->sym[s][i] = SCHED_SYM_UL;
        else if (i < bm->num_dl_symbols)
          cfg->sym[s][i] = SCHED_SYM_DL;
        else if (i >= NR_SYMBOLS_PER_SLOT - bm->num_ul_symbols)
          cfg->sym[s][i] = SCHED_SYM_UL;
        else
          cfg->sym[s][i] = SCHED_SYM_FLEX;
      }
    }
  }
  const long band = *scc->downlinkConfigCommon->frequencyInfoDL->frequencyBandList.list.array[0];
  const NR_SubcarrierSpacing_t scs = *scc->ssbSubcarrierSpacing;
  cfg->ssb_mu = scs;
  cfg->fr2 = scc->ssb_PositionsInBurst->present == 3;
  for (int i = 0; i < get_max_ssbs(scc); i++)
    cfg->ssb_start_symbol[i] = get_ssb_start_symbol(band, scs, i);
  cfg->num_beams = cell->beam_info.beam_allocation ? cell->beam_info.config.beams_per_period : 1;
  cfg->slots_per_frame = fs->numb_slots_frame;
  /* the MAC configures dmrs-AdditionalPosition pos1, and nothing before RRC setup: both use the larger N1 */
  sched_check_set_proc_time(cfg, scc->downlinkConfigCommon->initialDownlinkBWP->genericParameters.subcarrierSpacing, false);
  /* sf8 to sf64 */
  cfg->ra_cr_timer_ms = 8 * (scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->ra_ContentionResolutionTimer + 1);
  /* N_TA,offset only: N_TA depends on the distance to each UE. Default of 38.133 7.1.2 when absent */
  static const int nta_offset_tc[] = {0, 25600, 39936};
  const int nta_offset = scc->n_TimingAdvanceOffset ? nta_offset_tc[*scc->n_TimingAdvanceOffset] : cfg->fr2 ? 13792 : 25600;
  const int symbol_tc = (2048 + 144) * 64 >> cfg->mu;
  /* OAI_SCHED_CHECK_NTA: judge as for a UE with this N_TA (in Tc), on top of N_TA,offset */
  const char *nta = getenv("OAI_SCHED_CHECK_NTA");
  const int n_ta = nta ? atoi(nta) : 0;
  cfg->ta_symbols = (nta_offset + n_ta + symbol_tc - 1) / symbol_tc;
  /* sl1 to sl320, then sl60 and sl160 of Rel-16 */
  static const int window_slots[] = {1, 2, 4, 8, 10, 20, 40, 80, 160, 320};
  static const int window_v1610_slots[] = {60, 160};
  const NR_BWP_UplinkCommon_t *ul_bwp = scc->uplinkConfigCommon->initialUplinkBWP;
  const NR_RACH_ConfigGeneric_t *gen = &ul_bwp->rach_ConfigCommon->choice.setup->rach_ConfigGeneric;
  cfg->ra_window_slots = gen->ext1 && gen->ext1->ra_ResponseWindow_v1610 ? window_v1610_slots[*gen->ext1->ra_ResponseWindow_v1610]
                                                                          : window_slots[gen->ra_ResponseWindow];
  /* PRACH occasions where the PHY receives them: FAPI CONFIG k1 of the first FD occasion, n_RA_RB x FDM RBs */
  const NR_COMMON_channels_t *cc = &cell->common_channels;
  const int mu_pusch = scc->uplinkConfigCommon->frequencyInfoUL->scs_SpecificCarrierList.list.array[0]->subcarrierSpacing;
  cfg->prach_first_prb = cell->config.prach_config.num_prach_fd_occasions_list[0].k1.value;
  cfg->prach_n_prb = get_N_RA_RB(cell->config.prach_config.prach_sub_c_spacing.value, mu_pusch) * cell->config.prach_config.num_prach_fd_occasions.value;
  cfg->prach_dur = (cc->prach_info.format & 0xff) < 4 ? 14 : cc->prach_info.N_dur;
  cfg->max_pusch_per_slot = MAX_UL_PDUS_PER_SLOT;
  if (ul_bwp->ext1 && ul_bwp->ext1->msgA_ConfigCommon_r16) {
    const long *msgb = ul_bwp->ext1->msgA_ConfigCommon_r16->choice.setup->rach_ConfigCommonTwoStepRA_r16.rach_ConfigGenericTwoStepRA_r16
                           .msgB_ResponseWindow_r16;
    if (msgb && window_slots[*msgb] > cfg->ra_window_slots)
      cfg->ra_window_slots = window_slots[*msgb];
  }
}

static int enabled = -1;
static sched_check_cfg_t cfg;
static sched_check_timing_t timing;

/* PUSCH of the last slots, to log next to each preamble the PHY reports: a preamble that no UE
 * sent shows what else was on the air in its PRACH slot */
#define PRACH_TRACE_SLOTS 64
#define PRACH_TRACE_PUSCH 8
typedef struct {
  int frame, slot, n;
  struct {
    uint16_t rnti;
    int16_t first_prb, n_prb;
    uint8_t first_sym, n_sym, mcs;
  } p[PRACH_TRACE_PUSCH];
} prach_trace_t;
static prach_trace_t prach_trace[PRACH_TRACE_SLOTS];
static uint64_t preambles_with_pusch, preambles_without_pusch;

static void trace_pusch(const nfapi_nr_ul_tti_request_t *ul)
{
  prach_trace_t *t = &prach_trace[(ul->SFN * 32 + ul->Slot) % PRACH_TRACE_SLOTS];
  *t = (prach_trace_t){.frame = ul->SFN, .slot = ul->Slot};
  for (int i = 0; i < ul->n_pdus && t->n < PRACH_TRACE_PUSCH; i++) {
    if (ul->pdus_list[i].pdu_type != NFAPI_NR_UL_CONFIG_PUSCH_PDU_TYPE)
      continue;
    const nfapi_nr_pusch_pdu_t *p = &ul->pdus_list[i].pusch_pdu;
    t->p[t->n].rnti = p->rnti;
    t->p[t->n].first_prb = p->bwp_start + p->rb_start;
    t->p[t->n].n_prb = p->rb_size;
    t->p[t->n].first_sym = p->start_symbol_index;
    t->p[t->n].n_sym = p->nr_of_symbols;
    t->p[t->n].mcs = p->mcs_index;
    t->n++;
  }
}

void nr_sched_check_preamble(int frame, int slot, int rapid, int s_id, int f_id)
{
  if (enabled <= 0)
    return;
  sched_check_preamble(&cfg, &timing, frame, slot, rapid, s_id, f_id);
  const prach_trace_t *t = &prach_trace[(frame * 32 + slot) % PRACH_TRACE_SLOTS];
  const bool known = t->frame == frame && t->slot == slot;
  if (known && t->n > 0)
    preambles_with_pusch++;
  else
    preambles_without_pusch++;
  char list[400] = {0};
  for (int i = 0, k = 0; known && i < t->n; i++) {
    /* RBs between this PUSCH and the PRACH RBs, 0 if they touch */
    const int last = t->p[i].first_prb + t->p[i].n_prb - 1, prach_last = cfg.prach_first_prb + cfg.prach_n_prb - 1;
    const int gap = t->p[i].first_prb > prach_last ? t->p[i].first_prb - prach_last - 1 : cfg.prach_first_prb > last ? cfg.prach_first_prb - last - 1 : -1;
    k += snprintf(list + k,
                  sizeof(list) - k,
                  " [%04x RB %d+%d sym %d+%d MCS %d gap %d]",
                  t->p[i].rnti,
                  t->p[i].first_prb,
                  t->p[i].n_prb,
                  t->p[i].first_sym,
                  t->p[i].n_sym,
                  t->p[i].mcs,
                  gap);
  }
  LOG_W(NR_MAC,
        "SCHED_CHECK preamble %d at %d.%d s_id %d: PRACH RB %d+%d, PUSCH in the slot:%s\n",
        rapid,
        frame,
        slot,
        s_id,
        cfg.prach_first_prb,
        cfg.prach_n_prb,
        known ? (t->n ? list : " none") : " (slot not traced)");
}

static sched_check_report_t report;
static uint64_t slots;
/* first transmissions only, SI and paging excluded: MAC bytes in simulated time */
static uint64_t dl_new_bytes, ul_new_bytes;
static int slots_per_second;

static void count_new_data(const NR_Sched_Rsp_t *sched_info)
{
  const nfapi_nr_dl_tti_request_body_t *dl = &sched_info->DL_req.dl_tti_request_body;
  for (int i = 0; i < dl->nPDUs; i++) {
    if (dl->dl_tti_pdu_list[i].PDUType != NFAPI_NR_DL_TTI_PDSCH_PDU_TYPE)
      continue;
    const nfapi_nr_dl_tti_pdsch_pdu_rel15_t *p = &dl->dl_tti_pdu_list[i].pdsch_pdu.pdsch_pdu_rel15;
    if (p->rvIndex[0] == 0 && p->rnti != 0xffff && p->rnti != 0xfffe)
      dl_new_bytes += p->TBSize[0];
  }
  const nfapi_nr_ul_tti_request_t *ul = &sched_info->UL_tti_req;
  for (int i = 0; i < ul->n_pdus; i++)
    if (ul->pdus_list[i].pdu_type == NFAPI_NR_UL_CONFIG_PUSCH_PDU_TYPE && ul->pdus_list[i].pusch_pdu.pusch_data.rv_index == 0)
      ul_new_bytes += ul->pdus_list[i].pusch_pdu.pusch_data.tb_size;
}

void nr_sched_check_slot(const nr_cell_sched_t *cell, const NR_Sched_Rsp_t *sched_info)
{
  static uint64_t reported[SCHED_RULE_COUNT];

  if (enabled < 0) {
    /* OAI_SCHED_CHECK=all logs every finding, any other value the first of each rule */
    const char *env = getenv("OAI_SCHED_CHECK");
    enabled = env == NULL ? 0 : strcmp(env, "all") == 0 ? 2 : 1;
    if (enabled)
      fill_cfg(cell, &cfg);
  }
  if (!enabled)
    return;

  slots_per_second = cell->frame_structure.numb_slots_frame * 100;
  count_new_data(sched_info);
  trace_pusch(&sched_info->UL_tti_req);
  sched_check_dl_slot(&cfg, &sched_info->DL_req, &sched_info->UL_dci_req, &report);
  sched_check_ul_slot(&cfg, &sched_info->UL_tti_req, &report);
  sched_check_timing_slot(&cfg,
                          &timing,
                          &sched_info->DL_req,
                          &sched_info->UL_dci_req,
                          &sched_info->UL_tti_req,
                          &sched_info->TX_req,
                          &report);

  /* the first finding of each rule, with its position */
  for (int i = 0; i < report.num_items; i++) {
    const sched_check_item_t *it = &report.item[i];
    if (reported[it->rule]++ == 0 || enabled == 2)
      LOG_W(NR_MAC,
            "SCHED_CHECK %s %s (%s) at %d.%d RNTI %04x\n",
            reported[it->rule] == 1 ? "first" : "again",
            sched_rule_id(it->rule),
            sched_rule_clause(it->rule),
            it->frame,
            it->slot,
            it->rnti);
  }
  report.num_items = 0;

  if (++slots % (SCHED_CHECK_SUMMARY_FRAMES * cell->frame_structure.numb_slots_frame) == 0)
    nr_sched_check_summary();
}

uint64_t nr_sched_check_dropped(void)
{
  return timing.dropped;
}

uint64_t nr_sched_check_summary(void)
{
  if (enabled <= 0)
    return 0;
  const uint64_t *c = report.checked;
  LOG_A(NR_MAC,
        "SCHED_CHECK %lu slots: checked SSB %lu DCI %lu PDSCH %lu PUSCH %lu PUCCH %lu\n",
        slots,
        c[SCHED_PDU_SSB],
        c[SCHED_PDU_PDCCH],
        c[SCHED_PDU_PDSCH],
        c[SCHED_PDU_PUSCH],
        c[SCHED_PDU_PUCCH]);
  char hist[160] = {0};
  for (int l = 1, n = 0; l <= NR_SYMBOLS_PER_SLOT; l++)
    if (report.pdsch_symbols[l])
      n += snprintf(hist + n, sizeof(hist) - n, " L%d:%lu", l, report.pdsch_symbols[l]);
  LOG_A(NR_MAC, "SCHED_CHECK initial PDSCH by L%s, highest rate %d/1000\n", hist, report.max_rate_permille);
  LOG_A(NR_MAC,
        "SCHED_CHECK timing N1 %d N2 %d: smallest margin HARQ %d PUSCH %d symbols, unmatched HARQ bits %lu PUSCH %lu, dropped %lu\n",
        cfg.n1,
        cfg.n2,
        timing.min_harq_margin,
        timing.min_pusch_margin,
        timing.harq_unmatched,
        timing.pusch_unmatched,
        timing.dropped);
  LOG_A(NR_MAC,
        "SCHED_CHECK random access: RAR %lu Msg3 %lu Msg4 %lu, smallest margin Msg3 %d Msg4 HARQ %d Msg4 timer %d symbols\n",
        timing.rars,
        timing.msg3s,
        timing.msg4s,
        timing.min_msg3_margin,
        timing.min_msg4_harq_margin,
        timing.min_msg4_timer_margin);
  const double sim_s = slots_per_second ? (double)slots / slots_per_second : 0;
  LOG_A(NR_MAC,
        "SCHED_CHECK new data over %.2f s simulated: DL %lu bytes %.2f Mb/s, UL granted %lu bytes %.2f Mb/s\n",
        sim_s,
        dl_new_bytes,
        sim_s > 0 ? dl_new_bytes * 8e-6 / sim_s : 0,
        ul_new_bytes,
        sim_s > 0 ? ul_new_bytes * 8e-6 / sim_s : 0);
  LOG_A(NR_MAC,
        "SCHED_CHECK preambles reported by the PHY: %lu with a PUSCH in their slot, %lu without\n",
        preambles_with_pusch,
        preambles_without_pusch);
  LOG_A(NR_MAC,
        "SCHED_CHECK preambles %lu, answered by RAR or MsgB within %d slots %lu\n",
        timing.preambles,
        cfg.ra_window_slots,
        timing.preambles_answered);
  for (int r = 0; r < SCHED_RULE_COUNT; r++)
    LOG_A(NR_MAC, "SCHED_CHECK %lu slots: rule %d %s (%s): %lu\n", slots, r, sched_rule_id(r), sched_rule_clause(r), report.count[r]);
  uint64_t findings = 0;
  for (int r = 0; r < SCHED_RULE_COUNT; r++)
    findings += report.count[r];
  return findings;
}
