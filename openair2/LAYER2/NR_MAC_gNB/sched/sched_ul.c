/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* 3gpp scheduler, UL (phase 3). In each DL slot, for the reachable UL slots in order, one grid
 * transaction per UE:
 *   TDA of the slot whose first symbol is at least Tproc,2 + T_TA after the end of the PDCCH
 *   (38.214 6.4) and whose S/L is valid (Table 6.1.2.1-1); RBs from the UL grid, which holds what
 *   is already placed in that slot (VRB map and the PDUs written ahead, MsgA PUSCH occasions
 *   included); 2^a 3^b 5^c RBs and one layer with transform precoding (38.211 6.3.1.4); MCS from an
 *   outer loop on the CRCs with a fast start; RBs for the BSR, at least min_grant_prb; CCE.
 * One UL DCI per UE per DCI slot; no fixed number of UEs per slot (CCEs and the FAPI bound decide).
 * Emission is the legacy nr_ul_apply_*_transmission (TBS, DMRS, FAPI, HARQ). */

#include "sched_ul.h"

#include <math.h>
#include <stdlib.h>

#include "common/utils/LOG/log.h"
#include "common/utils/assertions.h"
#include "common/utils/nr/nr_common.h"
#include "openair2/LAYER2/NR_MAC_COMMON/nr_mac_common.h"
#include "openair2/LAYER2/NR_MAC_gNB/mac_proto.h"
#include "openair2/NR_PHY_INTERFACE/NR_IF_Module.h"
#include "sched_grid.h"
#include "sched_rules.h"
#include "sched_seed.h"
#include "sched_seed_oai.h"

#define LA_STEP_UP 0.1
#define LA_STEP_DOWN 1.0
#define LA_FAST_START_STEP 1.0
/* MAC subheaders, BSR and PHR, as the legacy sizing */
#define UL_OVERHEAD_BYTES 10

typedef struct {
  bool init, converged;
  double offset;
  uint64_t rounds0, rounds1;
} ue_la_t;

static struct {
  const nr_cell_sched_t *cell;
  sched_grid_t *grid;
  bool started;
  int64_t abs_slot;
  int last_frame, last_slot;
  int64_t max_nta_tc;
  ue_la_t la[MAX_MOBILES_PER_GNB];
  seed_cfg_t seed;
  bool check; ///< OAI_SCHED_CHECK set: the grid is compared with the legacy VRB map
  /* diagnosis switches (environment, read once): OAI_SCHED_UL_KSPAN, the UL slots ahead of k2_min a DL slot may
   * grant (default up to a frame); OAI_SCHED_UL_MAX_DCI, the UL DCIs per slot (default MAX_DCI_CORESET) */
  int k_span, max_dci;
  uint64_t compared, map_only, seed_only;
} ul;

void sched3gpp_ul_set_max_nta(int64_t n_ta_tc)
{
  ul.max_nta_tc = n_ta_tc;
}

static sched_grid_t *create_grid(const nr_cell_sched_t *cell, int n_rb)
{
  const frame_structure_t *fs = &cell->frame_structure;
  static grid_cfg_t cfg;
  cfg = (grid_cfg_t){.n_rb = n_rb, .horizon_slots = 2 * fs->numb_slots_frame, .period_slots = 1};
  if (fs->frame_type == TDD) {
    cfg.period_slots = fs->numb_slots_period;
    for (int s = 0; s < fs->numb_slots_period; s++) {
      const tdd_bitmap_t *bm = &fs->period_cfg.tdd_slot_bitmap[s];
      for (int i = 0; i < GRID_SYMBOLS; i++) {
        if (bm->slot_type == TDD_NR_DOWNLINK_SLOT)
          cfg.sym[s][i] = GRID_SYM_DL;
        else if (bm->slot_type == TDD_NR_UPLINK_SLOT)
          cfg.sym[s][i] = GRID_SYM_UL;
        else
          cfg.sym[s][i] = i < bm->num_dl_symbols ? GRID_SYM_DL : i >= GRID_SYMBOLS - bm->num_ul_symbols ? GRID_SYM_UL : GRID_SYM_FLEX;
      }
    }
  } else {
    for (int i = 0; i < GRID_SYMBOLS; i++)
      cfg.sym[0][i] = GRID_SYM_UL;
  }
  return grid_create(&cfg);
}

static void advance_slot(const nr_cell_sched_t *cell, int frame, int slot)
{
  const int spf = cell->frame_structure.numb_slots_frame;
  if (!ul.started) {
    ul.abs_slot = (int64_t)frame * spf + slot;
    ul.started = true;
  } else {
    const int span = 1024 * spf;
    ul.abs_slot += ((frame * spf + slot) - (ul.last_frame * spf + ul.last_slot) + span) % span;
  }
  ul.last_frame = frame;
  ul.last_slot = slot;
}

typedef struct {
  sched_grid_t *grid;
  int64_t slot;
} pucch_seed_arg_t;

static void seed_pending_pucch(const nfapi_nr_pucch_pdu_t *pdu, void *arg)
{
  const pucch_seed_arg_t *a = arg;
  seed_ul_pucch(a->grid, a->slot, &ul.seed, pdu);
}

/* what the PHY will receive in the UL slot, as far as it is known now: the PDUs written ahead (PRACH,
 * Msg3, MsgA PUSCH occasions, SRS, PUSCH, and the long-format PRACH of the slots before), the PUCCHs
 * pending in sched_pucch (written only in their own slot) and the prohibited PRBs */
static void seed_ul_pdus(sched_grid_t *grid, gNB_MAC_INST *mac, nr_cell_sched_t *cell, int64_t abs_slot, int frame, int slot)
{
  const int spf = cell->frame_structure.numb_slots_frame;
  const int size = cell->UL_tti_req_ahead_size;
  const nfapi_nr_ul_tti_request_t *req = &cell->UL_tti_req_ahead[ul_buffer_index(frame, slot, spf, size)];
  if (req->SFN != frame || req->Slot != slot)
    req = NULL;
  const nfapi_nr_ul_tti_request_t *prev[4] = {0};
  const int num_prev = min(ul.seed.prach_slots - 1, 4);
  for (int k = 0; k < num_prev; k++) {
    const int abs = (frame * spf + slot - 1 - k + 1024 * spf) % (1024 * spf);
    const nfapi_nr_ul_tti_request_t *r = &cell->UL_tti_req_ahead[ul_buffer_index(abs / spf, abs % spf, spf, size)];
    prev[k] = r->SFN == abs / spf && r->Slot == abs % spf ? r : NULL;
  }
  seed_ul_fapi(grid, abs_slot, &ul.seed, req, prev, num_prev);
  pucch_seed_arg_t arg = {.grid = grid, .slot = abs_slot};
  nr_pending_pucch_pdus(mac, cell, frame, slot, seed_pending_pucch, &arg);
  for (int sym = 0; sym < GRID_SYMBOLS; sym++) {
    uint64_t rbs[GRID_RB_WORDS] = {0};
    for (int rb = 0; rb < ul.seed.n_rb; rb++)
      if (cell->ulprbbl[rb] >> sym & 1)
        rbs[rb / 64] |= 1ULL << (rb % 64);
    grid_mark(grid, abs_slot, true, 0, sym, rbs);
  }
}

/* the grid holds what the PHY will receive in the slot, from the PDUs above only. The legacy VRB map is
 * still written (PUCCH, PRACH, Msg3, SRS, and the PUSCHs granted here) for the legacy allocations that test it.
 * With OAI_SCHED_CHECK it is compared with the grid before any grant of this scheduler: RB x symbols booked
 * by one and not the other. "In the map only" would be a resource this scheduler could grant twice */
static void seed_ul_slot(gNB_MAC_INST *mac, nr_cell_sched_t *cell, int64_t abs_slot, int frame, int slot, int n_rb)
{
  seed_ul_pdus(ul.grid, mac, cell, abs_slot, frame, slot);
  if (!ul.check)
    return;
  const int spf = cell->frame_structure.numb_slots_frame;
  const int index = ul_buffer_index(frame, slot, spf, cell->vrb_map_UL_size);
  const uint16_t *vrb_map = &cell->common_channels.vrb_map_UL[0][index * MAX_BWP_SIZE];
  for (int sym = 0; sym < GRID_SYMBOLS; sym++) {
    uint64_t rbs[GRID_RB_WORDS] = {0};
    for (int rb = 0; rb < n_rb; rb++)
      if (vrb_map[rb] >> sym & 1)
        rbs[rb / 64] |= 1ULL << (rb % 64);
    uint64_t seeded[GRID_RB_WORDS] = {0};
    grid_used(ul.grid, abs_slot, true, sym, seeded);
    for (int w = 0; w < GRID_RB_WORDS; w++) {
      ul.map_only += __builtin_popcountll(rbs[w] & ~seeded[w]);
      ul.seed_only += __builtin_popcountll(seeded[w] & ~rbs[w]);
    }
  }
  if (++ul.compared % 20000 == 0)
    LOG_A(NR_MAC,
          "3gpp UL seed vs legacy VRB map over %lu slots: RB x symbols in the map only %lu, from the PDUs only %lu\n",
          ul.compared,
          ul.map_only,
          ul.seed_only);
}

/* layers of the PUSCH: those of the first transmission for a retransmission, else the rank chosen by
 * mac->ul_ri_tpmi_select (SRS feedback, 1 with DCI 0_0); 1 with transform precoding (38.211 6.3.1.4) */
static int ul_layers(const nr_ul_candidate_t *cand)
{
  if (cand->is_retx)
    return cand->UE->UE_sched_ctrl.ul_harq_processes[cand->retx_harq_pid].sched_pusch.nrOfLayers;
  if (cand->UE->current_UL_BWP.transform_precoding == NR_PUSCH_Config__transformPrecoder_enabled)
    return 1;
  return cand->sched_pusch.nrOfLayers > 0 ? cand->sched_pusch.nrOfLayers : 1;
}

/* PUSCH PDUs already written for frame.slot */
static int puschs_in_slot(const nr_cell_sched_t *cell, int frame, int slot)
{
  const int spf = cell->frame_structure.numb_slots_frame;
  const nfapi_nr_ul_tti_request_t *req = &cell->UL_tti_req_ahead[ul_buffer_index(frame, slot, spf, cell->UL_tti_req_ahead_size)];
  if (req->SFN != frame || req->Slot != slot)
    return 0;
  int n = 0;
  for (int i = 0; i < req->n_pdus; i++)
    n += req->pdus_list[i].pdu_type == NFAPI_NR_UL_CONFIG_PUSCH_PDU_TYPE;
  return n;
}

/* MCS of the PUSCH SNR in the UE's table: get_mcs_from_SINRx10 maps the SNR onto MCS table 0 only (and
 * answers 28 for any other table, a reserved index in table 1); in another table, the highest MCS up to
 * max_mcs whose spectral efficiency Qm x R does not exceed that of the table 0 MCS */
static int snr_mcs(int table, int snrx10, int max_mcs)
{
  const int m0 = get_mcs_from_SINRx10(0, snrx10, 1);
  if (table == 0)
    return m0;
  const uint64_t eff0 = (uint64_t)nr_get_Qm_ul(m0, 0) * nr_get_code_rate_ul(m0, 0);
  int best = 0;
  for (int m = 0; m <= max_mcs; m++)
    if ((uint64_t)nr_get_Qm_ul(m, table) * nr_get_code_rate_ul(m, table) <= eff0)
      best = m;
  return best;
}

/* outer loop on the PUSCH CRCs: base from the PUSCH SNR, fast start until the first retransmission */
static int la_mcs(const nr_ul_candidate_t *cand)
{
  NR_UE_info_t *UE = cand->UE;
  ue_la_t *la = &ul.la[UE->uid % MAX_MOBILES_PER_GNB];
  const NR_mac_dir_stats_t *st = &UE->mac_stats.ul;
  const int max_mcs = cand->max_mcs;
  int base = cand->snrx10 > 0 ? snr_mcs(cand->mcs_table, cand->snrx10, max_mcs) : 0;
  base = base > max_mcs ? max_mcs : base;
  if (!la->init) {
    *la = (ue_la_t){.init = true, .rounds0 = st->rounds[0], .rounds1 = st->rounds[1]};
  } else {
    const double first = st->rounds[0] - la->rounds0, retx = st->rounds[1] - la->rounds1;
    if (retx > 0)
      la->converged = true;
    la->offset += (la->converged ? LA_STEP_UP : LA_FAST_START_STEP) * first - LA_STEP_DOWN * retx;
    la->rounds0 = st->rounds[0];
    la->rounds1 = st->rounds[1];
  }
  la->offset = fmax(-base - 1.0, fmin(max_mcs - base + 1.0, la->offset));
  const int mcs = base + (int)floor(la->offset);
  return mcs < 0 ? 0 : mcs > max_mcs ? max_mcs : mcs;
}

static int64_t ta_tc(const NR_ServingCellConfigCommon_t *scc)
{
  static const int offset[] = {0, 25600, 39936};
  int64_t n_ta_offset;
  if (scc->n_TimingAdvanceOffset)
    n_ta_offset = offset[*scc->n_TimingAdvanceOffset];
  else
    n_ta_offset = get_freq_range_from_arfcn(scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencyPointA) == FR2 ? 13792 : 25600;
  return n_ta_offset + ul.max_nta_tc;
}

static NR_pusch_dmrs_t dmrs_of(const NR_ServingCellConfigCommon_t *scc, const NR_UE_UL_BWP_t *bwp, NR_tda_info_t *tda, int layers)
{
  /* as nr_ul_port_select_default */
  uint8_t cdm_groups;
  if (bwp->transform_precoding == NR_PUSCH_Config__transformPrecoder_enabled)
    cdm_groups = 2;
  else if (bwp->dci_format == NR_UL_DCI_FORMAT_0_0)
    cdm_groups = tda->nrOfSymbols <= 2 ? 1 : 2;
  else
    cdm_groups = layers < 3 ? 1 : 2;
  return get_ul_dmrs_params(scc, bwp, tda, layers, (1 << layers) - 1, cdm_groups);
}

typedef struct {
  int tda;
  NR_tda_info_t info;
  NR_pusch_dmrs_t dmrs;
  int rb_start, n_rb_free;
} ul_choice_t;

/* the TDA of this K2 whose first symbol is legal and with the most free REs */
static bool choose_tda(gNB_MAC_INST *mac,
                       nr_cell_sched_t *cell,
                       nr_ul_candidate_t *cand,
                       int64_t target,
                       int sched_slot,
                       int k2,
                       int64_t earliest_abs,
                       ul_choice_t *best)
{
  const NR_ServingCellConfigCommon_t *scc = cell->common_channels.ServingCellConfigCommon;
  NR_UE_UL_BWP_t *bwp = &cand->UE->current_UL_BWP;
  const NR_tda_info_t *list = NULL;
  const int n = get_num_ul_tda(mac, cell, sched_slot, k2, &list);
  const bool ext_cp = bwp->cyclicprefix != NULL;
  const NR_sched_pusch_t *orig = cand->is_retx ? &cand->UE->UE_sched_ctrl.ul_harq_processes[cand->retx_harq_pid].sched_pusch : NULL;
  int64_t best_res = -1;
  for (int i = 0; i < n; i++) {
    NR_tda_info_t info = list[i];
    if (!info.valid_tda)
      continue;
    const int S = info.startSymbolIndex, L = info.nrOfSymbols;
    if (!rules_pusch_sl_valid(info.mapping_type == typeA, ext_cp, S, L))
      continue;
    /* 38.214 6.4: the PUSCH, DM-RS included, starts no earlier than L2 */
    if (target * RULES_SYMBOLS_PER_SLOT + S < earliest_abs)
      continue;
    const int tda = seq_arr_dist(&cell->ul_tda, seq_arr_front(&cell->ul_tda), &list[i]);
    int start;
    /* test only: OAI_SCHED_TEST_RB_HI bounds the RBs a PUSCH may use (e.g. to put it below the PRACH RBs) */
    static int rb_hi = -2;
    if (rb_hi == -2) {
      const char *e = getenv("OAI_SCHED_TEST_RB_HI");
      rb_hi = e ? atoi(e) : -1;
    }
    const int hi = rb_hi >= 0 && rb_hi < cand->bwp_start + cand->bwp_size ? rb_hi : cand->bwp_start + cand->bwp_size;
    const int run = grid_free_run(ul.grid, target, true, 0, S, L, cand->bwp_start, hi, &start);
    if (run == 0)
      continue;
    int n_rb = run;
    if (orig) {
      /* a retransmission keeps its TB: same TDA and RBs, or a refit that preserves the TBS */
      n_rb = orig->time_domain_allocation == tda ? orig->rbSize : check_ul_retx_feasibility(cand, tda, &info, scc, run);
      if (n_rb == 0 || n_rb > run)
        continue;
    }
    if (bwp->transform_precoding == NR_PUSCH_Config__transformPrecoder_enabled && !rules_tp_alloc_valid(n_rb, 1)) {
      if (orig)
        continue;
      n_rb = rules_tp_max_rb(run);
    }
    NR_pusch_dmrs_t dmrs = orig ? orig->dmrs_info : dmrs_of(scc, bwp, &info, ul_layers(cand));
    if (orig && orig->time_domain_allocation != tda)
      dmrs = get_ul_dmrs_params(scc, bwp, &info, orig->nrOfLayers, orig->dmrs_info.dmrs_ports, orig->dmrs_info.num_dmrs_cdm_grps_no_data);
    const int64_t res = (int64_t)n_rb * (12 * L - dmrs.N_PRB_DMRS * dmrs.num_dmrs_symb);
    if (res > best_res) {
      best_res = res;
      *best = (ul_choice_t){.tda = tda, .info = info, .dmrs = dmrs, .rb_start = start, .n_rb_free = n_rb};
    }
  }
  return best_res >= 0;
}

/* 38.213 7.1.1: the PUSCH power grows with 10 log10(2^mu M_RB) + Delta_TF (Delta_TF with deltaMCS, one layer). The
 * PHR gives the headroom normalized to that of the PUSCH it came with (handle_single_entry_phr: sched_ctrl->ph =
 * PH + 10 log10(2^mu M_RB) + Delta_TF of that PUSCH): an allocation needing more is sent below its power density */
static bool ul_power_ok(const nr_ul_candidate_t *cand, const ul_choice_t *c, int n_rb, uint32_t tbs_bits)
{
  if (cand->pcmax == 0) /* no PHR yet */
    return true;
  const NR_UE_UL_BWP_t *bwp = &cand->UE->current_UL_BWP;
  double need = round(10 * log10((double)(n_rb << bwp->scs)));
  const bool delta_mcs = bwp->pusch_Config && bwp->pusch_Config->pusch_PowerControl
                         && bwp->pusch_Config->pusch_PowerControl->deltaMCS;
  if (delta_mcs && ul_layers(cand) == 1) {
    const int n_re = (NR_NB_SC_PER_RB * c->info.nrOfSymbols - c->dmrs.num_dmrs_symb * c->dmrs.N_PRB_DMRS) * n_rb;
    const double bpre = (double)tbs_bits / n_re;
    need += 10 * log10(pow(2, bpre * 1.25) - 1);
  }
  return need <= cand->ph;
}

/* the fewest RBs carrying the BSR estimate, at least min_grant_prb, legal with transform precoding, within the
 * power headroom (the fewest RBs when even those exceed it: the UE then sends at its maximum power) */
static int size_new_tx(const nr_cell_sched_t *cell, const nr_ul_candidate_t *cand, const ul_choice_t *c, int mcs)
{
  const NR_UE_UL_BWP_t *bwp = &cand->UE->current_UL_BWP;
  const bool tp = bwp->transform_precoding == NR_PUSCH_Config__transformPrecoder_enabled;
  const uint8_t qm = nr_get_Qm_ul(mcs, cand->mcs_table);
  const uint16_t r = nr_get_code_rate_ul(mcs, cand->mcs_table);
  const uint32_t want = (cand->pending_bytes + UL_OVERHEAD_BYTES) * 8;
  const int n_min = cell->min_grant_prb < 1 ? 1 : cell->min_grant_prb;
  /* the RB counts this PUSCH may take, in increasing order */
  int sizes[GRID_MAX_RB], n_sizes = 0;
  const int dmrs_re = c->dmrs.N_PRB_DMRS * c->dmrs.num_dmrs_symb;
  for (int n = 1; n <= c->n_rb_free; n++) {
    if (tp && !rules_tp_alloc_valid(n, 1))
      continue;
    if (n_sizes > 0 && n > n_min
        && !ul_power_ok(cand, c, n, nr_compute_tbs(qm, r, n, c->info.nrOfSymbols, dmrs_re, 0, 0, ul_layers(cand))))
      break; /* the power needed grows with the RBs */
    sizes[n_sizes++] = n;
  }
  if (n_sizes == 0)
    return 0;
  /* the first size at least min_grant_prb whose TBS carries the BSR estimate; the TBS grows with
   * the RBs, so bisect; all the free RBs when none carries it */
  int lo = 0, hi = n_sizes - 1;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    const int n = sizes[mid];
    const bool enough = n >= n_min
                        && nr_compute_tbs(qm, r, n, c->info.nrOfSymbols, c->dmrs.N_PRB_DMRS * c->dmrs.num_dmrs_symb, 0, 0, ul_layers(cand)) >= want;
    if (enough)
      hi = mid;
    else
      lo = mid + 1;
  }
  return sizes[lo];
}

/* UL DCIs per slot up to which new grants are given when the DL has data waiting, and grants of long inactivity
 * always (the legacy UL DCIs per slot) */
#define UL_SHARED_DCI 4

/* a connected UE has DL data in its RLC buffers (as of its last DL buffer update) or a DL HARQ retransmission
 * pending */
static bool dl_data_waiting(gNB_MAC_INST *mac)
{
  FOR_EACH_CONNECTED_UE(&mac->UE_info, UE) {
    const NR_UE_sched_ctrl_t *sc = &UE->UE_sched_ctrl;
    if (sc->num_total_bytes > 0 || sc->retrans_dl_harq.head >= 0)
      return true;
  }
  return false;
}

static int ul_dcis(const nfapi_nr_ul_dci_request_t *req)
{
  int n = 0;
  for (int i = 0; i < req->numPdus; i++)
    n += req->ul_dci_pdu_list[i].pdcch_pdu.pdcch_pdu_rel15.numDlDci;
  return n;
}

static bool schedule_ue(gNB_MAC_INST *mac,
                        nr_cell_sched_t *cell,
                        post_process_pusch_t *pp,
                        nr_ul_candidate_t *cand,
                        int64_t target,
                        int sched_frame,
                        int sched_slot,
                        int k2)
{
  NR_UE_info_t *UE = cand->UE;
  NR_UE_sched_ctrl_t *sc = &UE->UE_sched_ctrl;
  NR_UE_UL_BWP_t *bwp = &UE->current_UL_BWP;
  const NR_ServingCellConfigCommon_t *scc = cell->common_channels.ServingCellConfigCommon;
  const int mu_dl = UE->current_DL_BWP.scs, mu_ul = bwp->scs;

  /* 38.214 6.4: L2 from the end of the scheduling PDCCH, d2,1 = 1 (the larger) */
  const NR_sched_pdcch_t *pdcch = &sc->sched_pdcch;
  const int64_t dci_slot_symbol = ul.abs_slot * RULES_SYMBOLS_PER_SLOT;
  const int64_t pdcch_end = rules_symbol_start_tc(mu_dl, dci_slot_symbol + pdcch->StartSymbolIndex + pdcch->DurationSymbols);
  const rules_tproc2_t tp = {.mu_dl = mu_dl, .mu_ul = mu_ul};
  const int64_t earliest = rules_earliest_ul_symbol(mu_ul, pdcch_end, rules_tproc2_tc(&tp), ta_tc(scc));

  ul_choice_t c;
  if (!choose_tda(mac, cell, cand, target, sched_slot, k2, earliest, &c))
    return false;

  int n_rb = c.n_rb_free, mcs = cand->current_mcs;
  if (!cand->is_retx) {
    mcs = la_mcs(cand);
    n_rb = size_new_tx(cell, cand, &c, mcs);
    if (n_rb == 0)
      return false;
  }

  grid_txn_t txn;
  grid_txn_begin(&txn);
  const grid_alloc_t pusch = {.kind = GRID_PUSCH,
                              .ul = true,
                              .owner = UE->rnti,
                              .rb_start = c.rb_start,
                              .n_rb = n_rb,
                              .sym_start = c.info.startSymbolIndex,
                              .n_sym = c.info.nrOfSymbols};
  if (grid_try_reserve(ul.grid, &txn, target, &pusch) != GRID_OK)
    return false;

  int agg = sc->aggregation_level;
  NR_sched_pdcch_t sched_pdcch = sc->sched_pdcch;
  const int cce = get_cce_index(cell, pp->slot, UE->rnti, &agg, 0, sc->search_space, sc->coreset, &sched_pdcch, sc->pdcch_cl_adjust);
  if (cce < 0) {
    sc->ul_cce_fail++;
    grid_txn_abort(ul.grid, &txn);
    return false;
  }
  grid_txn_commit(&txn);
  fill_pdcch_vrb_map(cell, &sched_pdcch, cce, agg, 0);
  sc->cce_index = cce;
  sc->aggregation_level = agg;
  sc->sched_pdcch = sched_pdcch;

  const int spf = cell->frame_structure.numb_slots_frame;
  uint16_t *rballoc_mask = &cell->common_channels.vrb_map_UL[0][ul_buffer_index(sched_frame, sched_slot, spf, cell->vrb_map_UL_size) * MAX_BWP_SIZE];
  const bwp_info_t bi = get_pusch_bwp_start_size(UE);
  cand->alloc_beam_idx = 0;
  cand->alloc_dci_beam_idx = 0;
  cand->alloc_cce_index = cce;
  cand->alloc_aggregation_level = agg;
  cand->alloc_sched_pdcch = sched_pdcch;
  cand->sched_pusch.rbStart = c.rb_start - bi.bwpStart;
  cand->sched_pusch.rbSize = n_rb;
  cand->sched_pusch.time_domain_allocation = c.tda;
  cand->sched_pusch.tda_info = c.info;
  cand->scheduled = true;
  if (cand->is_retx) {
    nr_ul_apply_retransmission(mac, cell, pp, cand, scc, c.tda, &c.info, rballoc_mask, bi, sched_frame, sched_slot);
  } else {
    cand->sched_pusch.mcs = mcs;
    sc->ul_bler_stats.mcs = mcs;
    /* the TPMI chosen by mac->ul_ri_tpmi_select belongs to its rank: when transform precoding brings the
     * rank down to 1, TPMI 0 */
    const int layers = ul_layers(cand);
    if (layers < cand->sched_pusch.nrOfLayers)
      cand->sched_pusch.tpmi = 0;
    cand->sched_pusch.nrOfLayers = layers;
    cand->sched_pusch.dmrs_info = c.dmrs;
    cand->alloc_slbitmap = SL_to_bitmap(c.info.startSymbolIndex, c.info.nrOfSymbols);
    nr_ul_apply_new_transmission(mac, cell, pp, cand, scc, c.tda, &c.info, rballoc_mask, cand->alloc_slbitmap, bi, sched_frame, sched_slot);
  }
  return true;
}

/* a candidate in the UL grant order */
typedef struct {
  int cls; /* 3 retransmission, 2 SR, 1 data, 0 grant of long inactivity only */
  int sr_cnt; /* SRs since the last grant */
  int age; /* frames since the last UL grant */
  double w; /* proportional fair weight, class 0 */
  int i; /* index in the candidates */
} ul_order_t;

/* retransmissions, then the UEs that sent SRs, most SRs first (as the legacy compare_ul_pf()), then the UEs with
 * data by PF weight, then the grants of long inactivity, which no UE asked for; ties go to the UE waiting for
 * longest. A fixed order (the UE list) gives the slots where the PUSCHs run out to the same UEs, and the
 * inactivity grants of many UEs come before an SR: the others starve until their SRs reach sr-TransMax
 * (TS 38.321 5.4.4) */
static int compare_ul_order(const void *a, const void *b)
{
  const ul_order_t *x = a, *y = b;
  if (x->cls != y->cls)
    return y->cls - x->cls;
  if (x->cls == 2 && x->sr_cnt != y->sr_cnt)
    return y->sr_cnt - x->sr_cnt;
  if (x->cls == 1 && x->w != y->w)
    return (x->w < y->w) - (x->w > y->w);
  return y->age - x->age;
}

void sched3gpp_ul_preprocessor(gNB_MAC_INST *mac, nr_cell_sched_t *cell, post_process_pusch_t *pp)
{
  const NR_ServingCellConfigCommon_t *scc = cell->common_channels.ServingCellConfigCommon;
  const frame_structure_t *fs = &cell->frame_structure;
  const int spf = fs->numb_slots_frame;
  if (cell->beam_info.beam_allocation) {
    /* several beams: phase 7 */
    nr_ulsch_preprocessor(mac, cell, pp);
    return;
  }
  const int n_rb = scc->uplinkConfigCommon->frequencyInfoUL->scs_SpecificCarrierList.list.array[0]->carrierBandwidth;
  if (ul.grid == NULL || ul.cell != cell) {
    grid_free(ul.grid);
    /* OAI_SCHED_MAX_NTA: largest N_TA of the cell (its radius), in Tc, when nothing set it */
    const char *nta = getenv("OAI_SCHED_MAX_NTA");
    ul = (typeof(ul)){.cell = cell, .grid = create_grid(cell, n_rb), .max_nta_tc = ul.max_nta_tc ? ul.max_nta_tc : nta ? atoll(nta) : 0};
    AssertFatal(ul.grid != NULL, "cannot create the UL scheduler grid\n");
    seed_cfg_from_cell(cell, n_rb, &ul.seed);
    ul.check = getenv("OAI_SCHED_CHECK") != NULL;
    const char *ks = getenv("OAI_SCHED_UL_KSPAN"), *md = getenv("OAI_SCHED_UL_MAX_DCI");
    ul.k_span = ks ? atoi(ks) : 2 * spf;
    ul.max_dci = md ? min(atoi(md), MAX_DCI_CORESET) : MAX_DCI_CORESET;
  }
  advance_slot(cell, pp->frame, pp->slot);
  grid_set_current(ul.grid, ul.abs_slot);
  /* the UL DCI is sent in a DL slot */
  if (!is_dl_slot(pp->slot, fs))
    return;

  const int koffset = get_NTN_Koffset(scc);
  const int k2_min = cell->radio_config.minRXTXTIME + koffset;
  bool granted[MAX_MOBILES_PER_GNB] = {0};
  static nr_ul_candidate_t cands[MAX_MOBILES_PER_GNB];
  /* the UL DCIs take the CORESET before the DL scheduler runs: with DL data waiting, the new grants (not the
     retransmissions) stop at UL_SHARED_DCI UL DCIs in the slot, so that the CCEs left carry PDSCH DCIs */
  const int new_dci = dl_data_waiting(mac) ? min(UL_SHARED_DCI, ul.max_dci) : ul.max_dci;
  for (int k = k2_min; k < spf && k < 2 * spf - 1 && k < k2_min + ul.k_span; k++) {
    const int64_t target = ul.abs_slot + k;
    const int sched_slot = (pp->slot + k) % spf;
    const int sched_frame = (pp->frame + (pp->slot + k) / spf) % 1024;
    if (!is_ul_slot(sched_slot, fs))
      continue;
    const int k2 = k - koffset;
    const NR_tda_info_t *list = NULL;
    if (get_num_ul_tda(mac, cell, sched_slot, k2, &list) == 0)
      continue;
    seed_ul_slot(mac, cell, target, sched_frame, sched_slot, n_rb);
    const int n = nr_ul_collect_candidates(mac, cell, &mac->UE_info, cands, MAX_MOBILES_PER_GNB, pp->frame, pp->slot, k2, sched_frame, sched_slot);
    if (n == 0)
      continue;
    /* rank and TPMI from the SRS feedback, as the legacy preprocessor */
    mac->ul_ri_tpmi_select(mac, cands, n);
    /* retransmissions, SRs, proportional fair on the rate of the link-adapted MCS, then long inactivity
     * (compare_ul_order) */
    ul_order_t order[MAX_MOBILES_PER_GNB];
    int m = 0;
    for (int i = 0; i < n; i++) {
      nr_ul_candidate_t *c = &cands[i];
      if (granted[c->UE->uid % MAX_MOBILES_PER_GNB])
        continue;
      ul_order_t *o = &order[m++];
      o->i = i;
      o->cls = c->is_retx ? 3 : c->sr_cnt > 0 ? 2 : c->pending_bytes > 0 ? 1 : 0;
      o->sr_cnt = c->sr_cnt;
      o->age = (pp->frame - c->UE->UE_sched_ctrl.last_ul_frame + 1024) % 1024;
      o->w = o->cls == 1 ? (double)nr_get_Qm_ul(la_mcs(c), c->mcs_table) * nr_get_code_rate_ul(la_mcs(c), c->mcs_table)
                               / fmax(c->avg_throughput, 1.0)
                         : 0;
    }
    qsort(order, m, sizeof(order[0]), compare_ul_order);
    for (int j = 0; j < m; j++) {
      const int dcis = ul_dcis(pp->ul_dci_req);
      if (dcis >= ul.max_dci)
        return;
      /* grants of long inactivity (no data, no SR, the last of the order) stop at UL_SHARED_DCI UL DCIs in the
         slot, so that with many idle UEs they do not take the CCEs of the DL slots carrying UL DCIs */
      if (order[j].cls < 3 && dcis >= (order[j].cls == 0 ? UL_SHARED_DCI : new_dci))
        continue;
      /* the PHY receives at most MAX_UL_PDUS_PER_SLOT PUSCHs in a slot, those already written (Msg3,
       * earlier grants) included: the next ones would be lost */
      if (puschs_in_slot(cell, sched_frame, sched_slot) >= MAX_UL_PDUS_PER_SLOT)
        break;
      nr_ul_candidate_t *c = &cands[order[j].i];
      if (schedule_ue(mac, cell, pp, c, target, sched_frame, sched_slot, k2))
        granted[c->UE->uid % MAX_MOBILES_PER_GNB] = true;
    }
  }
}
