/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* 3gpp scheduler, DL. Per UE, one transaction on the grid:
 *   TDA valid for the slot (38.214 Table 5.1.2.1-1, DL or flexible symbols, after the CORESET),
 *   RBs from the grid, MCS from outer-loop link adaptation capped by the 0.95 limit of this
 *   allocation (38.214 5.1.3), TBS of 38.214 5.1.3.2, CCE, and a HARQ-ACK PUCCH no earlier than the
 *   end of the PDSCH plus Tproc,1 and T_TA (38.214 5.3).
 * Anything that fails undoes the transaction. Common channels are still placed by the legacy code:
 * their VRB map is copied into the grid each slot. Emission (DCI, FAPI, MAC PDU, HARQ) is
 * post_process_dlsch.
 * Phase 2 limits: one beam, one layer, contiguous RBs, no CQI base for link adaptation. */

#include "sched_dl.h"

#include <math.h>
#include <stdlib.h>

#include "common/utils/LOG/log.h"
#include "common/utils/assertions.h"
#include "common/utils/nr/nr_common.h"
#include "openair2/LAYER2/NR_MAC_COMMON/nr_mac_common.h"
#include "openair2/LAYER2/NR_MAC_gNB/mac_proto.h"
#include "sched_grid.h"
#include "sched_seed_oai.h"
#include "sched_rules.h"

/* outer loop: +0.05 MCS per first transmission, -0.5 per retransmission, 10 % BLER at equilibrium,
 * whatever the rate at which the UE is served. Steps of +0.1/-1 swing 2-3 MCS around the target in
 * rfsim AWGN (the NACKs of the HARQ feedback delay add up), -12..-16 % DL against +0.05/-0.5 */
#define LA_STEP_UP 0.05
#define LA_STEP_DOWN 0.5
#define LA_FAST_START_STEP 1.0
/* the loop corrects the CQI both ways: up to this many MCS above the MCS of the reported CQI (the OAI
 * UE reports a CQI 2-4 MCS below the 10 % BLER point in rfsim AWGN; srsRAN bounds its CQI offset to 4 too) */
#define LA_MAX_ABOVE_CQI 4
/* MAC subheaders and CEs, as the legacy sizing */
#define MAC_OVERHEAD_BYTES 12

typedef struct {
  bool init;
  bool converged; ///< a retransmission was seen: the fast start is over
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
  /* longest free run of the slot per (S, L, BWP), valid until the next reservation: most UEs share
   * the same TDA list, so choose_tda reads it instead of scanning the grid again */
  struct {
    int S, L, lo, hi, run, start;
  } free_run[64];
  int free_runs;
  /* a HARQ-ACK PUCCH allocation failed in this slot: the next UEs are checked for a free PUCCH first */
  bool pucch_short;
  bool compare; ///< OAI_SCHED_CHECK set: compare the seed with the legacy VRB map
  uint64_t compared, map_only, seed_only;
} dl;

void sched3gpp_set_max_nta(int64_t n_ta_tc)
{
  dl.max_nta_tc = n_ta_tc;
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
    /* FDD: the DL grid only holds DL */
    for (int i = 0; i < GRID_SYMBOLS; i++)
      cfg.sym[0][i] = GRID_SYM_DL;
  }
  return grid_create(&cfg);
}

static void advance_slot(const nr_cell_sched_t *cell, int frame, int slot)
{
  const int spf = cell->frame_structure.numb_slots_frame;
  if (!dl.started) {
    dl.abs_slot = (int64_t)frame * spf + slot;
    dl.started = true;
  } else {
    const int span = 1024 * spf;
    dl.abs_slot += ((frame * spf + slot) - (dl.last_frame * spf + dl.last_slot) + span) % span;
  }
  dl.last_frame = frame;
  dl.last_slot = slot;
}

/* what is already placed in this slot, from the FAPI PDUs (SSB, SIB, paging, RA, PDCCH of the UL DCIs).
 * With OAI_SCHED_CHECK, the legacy VRB map is compared with it: RB x symbols booked by one and not the other */
static void seed_dl_slot(const nr_cell_sched_t *cell, const post_process_pdsch_t *pp, int n_rb)
{
  seed_dl_fapi(dl.grid, dl.abs_slot, &dl.seed, pp->dl_req, cell->ul_dci_req_slot);
  if (!dl.compare)
    return;
  const uint16_t *vrb_map = cell->common_channels.vrb_map[0];
  for (int sym = 0; sym < GRID_SYMBOLS; sym++) {
    uint64_t grid_rbs[GRID_RB_WORDS], map_rbs[GRID_RB_WORDS] = {0};
    grid_used(dl.grid, dl.abs_slot, false, sym, grid_rbs);
    for (int rb = 0; rb < n_rb; rb++)
      if (vrb_map[rb] >> sym & 1)
        map_rbs[rb / 64] |= 1ULL << (rb % 64);
    for (int w = 0; w < GRID_RB_WORDS; w++) {
      dl.map_only += __builtin_popcountll(map_rbs[w] & ~grid_rbs[w]);
      dl.seed_only += __builtin_popcountll(grid_rbs[w] & ~map_rbs[w]);
    }
  }
  if (++dl.compared % 20000 == 0)
    LOG_A(NR_MAC, "3gpp DL seed vs legacy VRB map over %lu slots: RB x symbols in the map only %lu, from the PDUs only %lu\n", dl.compared, dl.map_only, dl.seed_only);
}

static rules_mcs_table_t rules_table(int mcs_table_idx)
{
  return mcs_table_idx == 1 ? RULES_MCS_QAM256 : mcs_table_idx == 2 ? RULES_MCS_QAM64_LOWSE : RULES_MCS_QAM64;
}

/* outer-loop link adaptation, from the HARQ round counters of the UE. The base is the MCS of the
 * reported CQI (cand->max_mcs then carries it), 0 without CQI; the MCS stays within the table and
 * configured caps and LA_MAX_ABOVE_CQI of the CQI; the offset stays within one MCS of the reachable
 * range so it does not wind up while the MCS is at a bound */
static int la_mcs(const nr_dl_candidate_t *cand)
{
  NR_UE_info_t *UE = cand->UE;
  ue_la_t *la = &dl.la[UE->uid % MAX_MOBILES_PER_GNB];
  const NR_mac_dir_stats_t *st = &UE->mac_stats.dl;
  const int cqi_mcs = cand->max_mcs;
  const int base = cand->cqi > 0 ? cqi_mcs : 0;
  const int cfg_mcs = min(cand->mcs_table == 1 ? 27 : 28, dl.cell->dl_bler.max_mcs);
  const int max_mcs = cand->cqi > 0 ? min(cfg_mcs, cqi_mcs + LA_MAX_ABOVE_CQI) : cqi_mcs;
  if (!la->init) {
    *la = (ue_la_t){.init = true, .rounds0 = st->rounds[0], .rounds1 = st->rounds[1]};
  } else {
    const double first = st->rounds[0] - la->rounds0, retx = st->rounds[1] - la->rounds1;
    /* fast start: one MCS per first transmission until the first retransmission, then the 10 % loop */
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

/* 38.133 7.1.2: N_TA,offset when n-TimingAdvanceOffset is absent, 25600 Tc in FR1, 13792 in FR2 */
static int64_t ta_tc(const NR_ServingCellConfigCommon_t *scc)
{
  static const int offset[] = {0, 25600, 39936};
  int64_t n_ta_offset;
  if (scc->n_TimingAdvanceOffset)
    n_ta_offset = offset[*scc->n_TimingAdvanceOffset];
  else
    n_ta_offset = get_freq_range_from_arfcn(scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencyPointA) == FR2 ? 13792 : 25600;
  return n_ta_offset + dl.max_nta_tc;
}

int sched3gpp_msg4_earliest_pucch_symbol(const nr_cell_sched_t *cell, const NR_UE_info_t *UE, int frame, int slot, int start, int length)
{
  const int mu_dl = UE->current_DL_BWP.scs, mu_ul = UE->current_UL_BWP.scs;
  /* N_T,1 and N_T,2 take the smaller of the SCS configurations */
  const int mu = mu_dl < mu_ul ? mu_dl : mu_ul;
  const int64_t slot_symbol = ((int64_t)frame * cell->frame_structure.numb_slots_frame + slot) * RULES_SYMBOLS_PER_SLOT;
  const int64_t end_tc = rules_symbol_start_tc(mu_dl, slot_symbol + start + length);
  const int64_t earliest = rules_earliest_ul_symbol(mu_ul, end_tc, rules_msg4_harq_delay_tc(mu), ta_tc(cell->common_channels.ServingCellConfigCommon));
  return earliest - rules_first_symbol_from(mu_ul, rules_symbol_start_tc(mu_dl, slot_symbol));
}

typedef struct {
  int tda;
  NR_tda_info_t info;
  NR_pdsch_dmrs_t dmrs;
  int rb_start, n_rb_free; ///< longest free run in the BWP on the TDA symbols, absolute RBs
} tda_choice_t;

static int cached_free_run(int S, int L, int lo, int hi, int *start)
{
  for (int i = 0; i < dl.free_runs; i++)
    if (dl.free_run[i].S == S && dl.free_run[i].L == L && dl.free_run[i].lo == lo && dl.free_run[i].hi == hi) {
      *start = dl.free_run[i].start;
      return dl.free_run[i].run;
    }
  const int run = grid_free_run(dl.grid, dl.abs_slot, false, 0, S, L, lo, hi, start);
  if (dl.free_runs < (int)(sizeof(dl.free_run) / sizeof(dl.free_run[0])))
    dl.free_run[dl.free_runs++] = (typeof(dl.free_run[0])){.S = S, .L = L, .lo = lo, .hi = hi, .run = run, .start = *start};
  return run;
}

/* layers of the PDSCH: those of the first transmission for a retransmission, else the reported rank
 * (csi_ri is 0 for DCI 1_0, as in the legacy RI/PMI selection) */
static int dl_layers(const nr_dl_candidate_t *cand)
{
  if (cand->is_retx)
    return cand->UE->UE_sched_ctrl.harq_processes[cand->retx_harq_pid].sched_pdsch.nrOfLayers;
  return cand->csi_ri + 1;
}

/* the TDA with the most data REs in the free RBs of this slot */
static bool choose_tda(const nr_cell_sched_t *cell, nr_dl_candidate_t *cand, tda_choice_t *best)
{
  NR_UE_info_t *UE = cand->UE;
  NR_UE_sched_ctrl_t *sc = &UE->UE_sched_ctrl;
  NR_UE_DL_BWP_t *bwp = &UE->current_DL_BWP;
  const NR_ServingCellConfigCommon_t *scc = cell->common_channels.ServingCellConfigCommon;
  const int ss_type = sc->search_space->searchSpaceType->present;
  const int coreset_id = sc->coreset->controlResourceSetId;
  const NR_PDSCH_TimeDomainResourceAllocationList_t *list = get_dl_tdalist(bwp, coreset_id, ss_type, TYPE_C_RNTI_);
  if (list == NULL)
    return false;
  const int pdcch_end = sc->sched_pdcch.StartSymbolIndex + sc->sched_pdcch.DurationSymbols;
  const bool ext_cp = bwp->cyclicprefix != NULL;
  int64_t best_res = -1;
  for (int t = 0; t < list->list.count; t++) {
    if (list->list.array[t]->k0 && *list->list.array[t]->k0 != 0)
      continue;
    NR_tda_info_t info = get_dl_tda_info(bwp, ss_type, t, scc->dmrs_TypeA_Position, 1, TYPE_C_RNTI_, coreset_id, false);
    if (!info.valid_tda)
      continue;
    const int S = info.startSymbolIndex, L = info.nrOfSymbols;
    if (!rules_pdsch_sl_valid(info.mapping_type == typeA, ext_cp, 2 + scc->dmrs_TypeA_Position, S, L) || S < pdcch_end)
      continue;
    int start;
    const int run = cached_free_run(S, L, cand->bwp_start, cand->bwp_start + cand->bwp_size, &start);
    if (run == 0)
      continue;
    int need = 0;
    if (cand->is_retx) {
      const NR_sched_pdsch_t *orig = &sc->harq_processes[cand->retx_harq_pid].sched_pdsch;
      const bool same = S == orig->tda_info.startSymbolIndex && L == orig->tda_info.nrOfSymbols;
      need = same ? orig->rbSize : check_dl_retx_feasibility(cand, t, &info, scc, run);
      if (need == 0 || need > run)
        continue;
    }
    const NR_pdsch_dmrs_t dmrs = get_dl_dmrs_params(scc, bwp, &info, dl_layers(cand));
    const int64_t res = (int64_t)run * (12 * L - dmrs.N_PRB_DMRS * dmrs.N_DMRS_SLOT);
    if (res > best_res) {
      best_res = res;
      *best = (tda_choice_t){.tda = t, .info = info, .dmrs = dmrs, .rb_start = start, .n_rb_free = cand->is_retx ? need : run};
    }
  }
  return best_res >= 0;
}

/* RBs and MCS of a new transmission: the fewest RBs carrying the buffer, the highest MCS up to the
 * link adaptation that respects 0.95 on these RBs */
/* highest legal MCS up to la on n RBs and its TBS, -1 if none */
static int mcs_on(const nr_dl_candidate_t *cand, const tda_choice_t *c, int la, int n, int *tbs)
{
  const int dmrs_re = c->dmrs.N_PRB_DMRS * c->dmrs.N_DMRS_SLOT;
  const int L = c->info.nrOfSymbols;
  rules_alloc_t a = {.table = rules_table(cand->mcs_table),
                     .max_mcs = la,
                     .n_symb = L,
                     .dmrs_re_per_prb = dmrs_re,
                     .n_prb = n,
                     .layers = dl_layers(cand),
                     .n_re_data = n * (12 * L - dmrs_re)};
  return rules_max_mcs(&a, tbs);
}

static bool size_new_tx(nr_dl_candidate_t *cand, const tda_choice_t *c, int la, int *n_rb, int *mcs, int *tbs_bits)
{
  const NR_UE_sched_ctrl_t *sc = &cand->UE->UE_sched_ctrl;
  const uint32_t want_bits = (sc->num_total_bytes + MAC_OVERHEAD_BYTES + (sc->ta_apply ? 2 : 0)) * 8;
  int t, m = mcs_on(cand, c, la, c->n_rb_free, &t);
  if (m < 0)
    return false;
  if ((uint32_t)t < want_bits) {
    /* the buffer needs more than the free RBs: all of them */
    *n_rb = c->n_rb_free;
    *mcs = m;
    *tbs_bits = t;
    return true;
  }
  /* the fewest RBs carrying the buffer: the TBS grows with the RBs, so bisect, then step back over
   * the few RBs where the 0.95 limit and the TBS quantization break the monotony */
  int lo = 1, hi = c->n_rb_free;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    int tm;
    if (mcs_on(cand, c, la, mid, &tm) >= 0 && (uint32_t)tm >= want_bits)
      hi = mid;
    else
      lo = mid + 1;
  }
  for (int n = lo - 1; n >= 1 && n >= lo - 4; n--) {
    int tm;
    if (mcs_on(cand, c, la, n, &tm) >= 0 && (uint32_t)tm >= want_bits)
      lo = n;
  }
  *n_rb = lo;
  *mcs = mcs_on(cand, c, la, lo, tbs_bits);
  return *mcs >= 0;
}

/* first DL DCI count of the slot, for the FAPI limit of one PDCCH PDU */
static int dl_dcis(const nfapi_nr_dl_tti_request_body_t *req)
{
  int n = 0;
  for (int i = 0; i < req->nPDUs; i++)
    if (req->dl_tti_pdu_list[i].PDUType == NFAPI_NR_DL_TTI_PDCCH_PDU_TYPE)
      n += req->dl_tti_pdu_list[i].pdcch_pdu.pdcch_pdu_rel15.numDlDci;
  return n;
}

static bool schedule_ue(gNB_MAC_INST *mac, nr_cell_sched_t *cell, post_process_pdsch_t *pp, nr_dl_candidate_t *cand)
{
  NR_UE_info_t *UE = cand->UE;
  NR_UE_sched_ctrl_t *sc = &UE->UE_sched_ctrl;
  NR_UE_DL_BWP_t *bwp = &UE->current_DL_BWP;
  const NR_ServingCellConfigCommon_t *scc = cell->common_channels.ServingCellConfigCommon;
  const int frame = pp->frame, slot = pp->slot;

  /* once a HARQ-ACK PUCCH allocation failed in this slot (resource set exhausted, the usual case with many small UEs),
   * a UE without any PUCCH possible is not tried: its allocation would fail on it, after the TDA, sizing, grid and CCE
   * work. Before that, the test costs more than it saves */
  const int fb_pid = cand->is_retx ? cand->retx_harq_pid : sc->available_dl_harq.head;
  if (dl.pucch_short && !get_FeedbackDisabled(UE->sc_info.downlinkHARQ_FeedbackDisabled_r17, fb_pid)
      && !nr_acknack_possible(cell, UE, frame, slot, 0)) {
    sc->dl_pucch_fail++;
    return false;
  }

  tda_choice_t c;
  if (!choose_tda(cell, cand, &c))
    return false;

  int n_rb, mcs, tbs_bits;
  const NR_sched_pdsch_t *orig = cand->is_retx ? &sc->harq_processes[cand->retx_harq_pid].sched_pdsch : NULL;
  if (orig) {
    n_rb = c.n_rb_free;
    mcs = orig->mcs;
    tbs_bits = orig->tb_size * 8;
    int qm, r;
    if (!rules_mcs(rules_table(cand->mcs_table), mcs, 2, &qm, &r))
      return false;
    const int dmrs_re = c.dmrs.N_PRB_DMRS * c.dmrs.N_DMRS_SLOT;
    if (!rules_rate_ok(tbs_bits, r, qm, orig->nrOfLayers, n_rb * (12 * c.info.nrOfSymbols - dmrs_re)))
      return false;
  } else {
    const int la = la_mcs(cand);
    if (!size_new_tx(cand, &c, la, &n_rb, &mcs, &tbs_bits))
      return false;
  }

  grid_txn_t txn;
  grid_txn_begin(&txn);
  const grid_alloc_t pdsch = {.kind = GRID_PDSCH,
                              .owner = UE->rnti,
                              .rb_start = c.rb_start,
                              .n_rb = n_rb,
                              .sym_start = c.info.startSymbolIndex,
                              .n_sym = c.info.nrOfSymbols};
  if (grid_try_reserve(dl.grid, &txn, dl.abs_slot, &pdsch) != GRID_OK)
    return false;

  int agg = sc->aggregation_level;
  NR_sched_pdcch_t pdcch = sc->sched_pdcch;
  const int cce = get_cce_index(cell, slot, UE->rnti, &agg, 0, sc->search_space, sc->coreset, &pdcch, sc->pdcch_cl_adjust);
  if (cce < 0) {
    sc->dl_cce_fail++;
    grid_txn_abort(dl.grid, &txn);
    return false;
  }

  const int harq_pid = orig ? cand->retx_harq_pid : sc->available_dl_harq.head;
  int pucch = -1;
  if (!get_FeedbackDisabled(UE->sc_info.downlinkHARQ_FeedbackDisabled_r17, harq_pid)) {
    /* 38.214 5.3: the PUCCH starts after the end of the PDSCH plus Tproc,1, advanced by T_TA */
    const int mu = bwp->scs;
    const int S = c.info.startSymbolIndex, L = c.info.nrOfSymbols;
    const int pdcch_start = pdcch.StartSymbolIndex, pdcch_len = pdcch.DurationSymbols;
    int overlap = 0;
    for (int s = pdcch_start; s < pdcch_start + pdcch_len; s++)
      overlap += s >= S && s < S + L;
    const rules_tproc1_t tp = {.mu_pdcch = mu,
                               .mu_pdsch = mu,
                               .mu_ul = UE->current_UL_BWP.scs,
                               /* the column with additional DM-RS, and N1,0 = 14: the larger values */
                               .dmrs_l1_12 = true,
                               .type_a = c.info.mapping_type == typeA,
                               .start_symbol = S,
                               .length = L,
                               .pdcch_overlap = overlap,
                               .n_rb = n_rb};
    const int64_t slot_symbol = dl.abs_slot * RULES_SYMBOLS_PER_SLOT;
    const int64_t end_tc = rules_symbol_start_tc(mu, slot_symbol + S + L);
    const int64_t earliest = rules_earliest_ul_symbol(UE->current_UL_BWP.scs, end_tc, rules_tproc1_tc(&tp), ta_tc(scc));
    /* in UL symbols counted from the start of this slot */
    const int64_t ul_base = rules_first_symbol_from(UE->current_UL_BWP.scs, rules_symbol_start_tc(mu, slot_symbol));
    const int earliest_rel = earliest - ul_base;
    const int r_pucch = nr_get_pucch_resource(sc->coreset, UE->current_UL_BWP.pucch_Config, cce);
    pucch = nr_acknack_scheduling_from(cell, UE, frame, slot, UE->UE_beam_index, r_pucch, 0, earliest_rel);
    if (pucch < 0) {
      dl.pucch_short = true;
      sc->dl_pucch_fail++;
      grid_txn_abort(dl.grid, &txn);
      return false;
    }
  }
  grid_txn_commit(&txn);

  /* the legacy maps still serve the common channels and the UL scheduler */
  fill_pdcch_vrb_map(cell, &pdcch, cce, agg, 0);
  const uint16_t symbols = SL_to_bitmap(c.info.startSymbolIndex, c.info.nrOfSymbols);
  for (int rb = 0; rb < n_rb; rb++)
    cell->common_channels.vrb_map[0][c.rb_start + rb] |= symbols;
  sc->cce_index = cce;
  sc->aggregation_level = agg;
  sc->sched_pdcch = pdcch;

  NR_sched_pdsch_t s = {0};
  if (orig) {
    s = *orig;
    s.dl_harq_pid = cand->retx_harq_pid;
  } else {
    s.mcs = mcs;
    /* what the UE statistics print */
    sc->dl_bler_stats.mcs = mcs;
    s.Qm = nr_get_Qm_dl(mcs, bwp->mcsTableIdx);
    s.R = nr_get_code_rate_dl(mcs, bwp->mcsTableIdx);
    s.tb_size = tbs_bits / 8;
    s.nrOfLayers = dl_layers(cand);
    s.pm_index = cand->csi_pm_index;
    s.dl_harq_pid = harq_pid;
    /* the UE applies the new CellGroup once it has the whole RRCReconfiguration: switch on the ACK of the PDSCH
       carrying the last SRB1 bytes, not of a first RLC segment (as the legacy nr_dl_schedule) */
    const int srb1_bytes = sc->rlc_status[DL_SCH_LCID_DCCH].bytes_in_buffer;
    if (UE->reconfigCellGroup && srb1_bytes > 0 && (srb1_bytes > 10 || UE->reconfig_partly_sent)) {
      if (s.tb_size >= srb1_bytes + MAC_OVERHEAD_BYTES + (sc->ta_apply ? 2 : 0))
        s.action = nr_dl_reconfig_action();
      else
        UE->reconfig_partly_sent = true;
    }
    const nr_pdsch_AntennaPorts_t *p = &cell->radio_config.pdsch_AntennaPorts;
    s.ant_port_idx.numSpatialStreamIndices = p->XP * p->N1 * p->N2;
    for (int i = 0; i < s.ant_port_idx.numSpatialStreamIndices; i++)
      s.ant_port_idx.spatialStreamIndices[i] = cell->radio_config.spatial_stream_index[i];
  }
  s.time_domain_allocation = c.tda;
  s.tda_info = c.info;
  s.dmrs_parms = c.dmrs;
  s.rbStart = c.rb_start - cand->bwp_start;
  s.rbSize = n_rb;
  s.pucch_allocation = pucch;
  s.alloc_type = PDSCH_TYPE1;
  s.bwp_info = (bwp_info_t){.bwpStart = cand->bwp_start, .bwpSize = cand->bwp_size};
  cand->sched_pdsch = s;
  cand->scheduled = true;
  cand->alloc_cce_index = cce;
  cand->alloc_aggregation_level = agg;
  cand->alloc_sched_pdcch = pdcch;
  post_process_dlsch(mac, cell, pp, UE, &s, cand);
  return true;
}

static int compare_weight(const void *a, const void *b)
{
  const double wa = *(const double *)a, wb = *(const double *)b;
  return (wa < wb) - (wa > wb);
}

void sched3gpp_dl_preprocessor(gNB_MAC_INST *mac, nr_cell_sched_t *cell, post_process_pdsch_t *pp)
{
  /* SIB1 is still placed by schedule_nr_sib1: make it choose its TDA for the slot (no UL symbols) */
  cell->sib1_check_slot = true;
  cell->sched3gpp_ra_rules = true;
  /* the Msg4 rule needs the cell N_TA before any UE is connected */
  if (dl.max_nta_tc == 0 && getenv("OAI_SCHED_MAX_NTA"))
    dl.max_nta_tc = atoll(getenv("OAI_SCHED_MAX_NTA"));
  if (get_num_connected_ue(&mac->UE_info) == 0)
    return;
  const NR_ServingCellConfigCommon_t *scc = cell->common_channels.ServingCellConfigCommon;
  const int n_rb = scc->downlinkConfigCommon->frequencyInfoDL->scs_SpecificCarrierList.list.array[0]->carrierBandwidth;
  if (cell->beam_info.beam_allocation) {
    /* several beams: phase 7 */
    nr_dlsch_preprocessor(mac, cell, pp);
    return;
  }
  if (dl.grid == NULL || dl.cell != cell) {
    grid_free(dl.grid);
    /* OAI_SCHED_MAX_NTA: largest N_TA of the cell (its radius), in Tc, when nothing set it */
    const char *nta = getenv("OAI_SCHED_MAX_NTA");
    dl = (typeof(dl)){.cell = cell, .grid = create_grid(cell, n_rb), .max_nta_tc = dl.max_nta_tc ? dl.max_nta_tc : nta ? atoll(nta) : 0};
    AssertFatal(dl.grid != NULL, "cannot create the scheduler grid\n");
    seed_cfg_from_cell(cell, n_rb, &dl.seed);
    dl.compare = getenv("OAI_SCHED_CHECK") != NULL;
  }
  advance_slot(cell, pp->frame, pp->slot);
  grid_set_current(dl.grid, dl.abs_slot);
  seed_dl_slot(cell, pp, n_rb);
  dl.free_runs = 0;
  dl.pucch_short = false;

  static nr_dl_candidate_t cands[MAX_MOBILES_PER_GNB];
  const int n = nr_dl_collect_candidates(cell, &mac->UE_info, cands, MAX_MOBILES_PER_GNB, pp->frame, pp->slot);
  if (n == 0)
    return;
  /* retransmissions first, then proportional fair on the rate of the link-adapted MCS */
  struct {
    double w;
    int i;
  } order[MAX_MOBILES_PER_GNB];
  for (int i = 0; i < n; i++) {
    nr_dl_candidate_t *c = &cands[i];
    double w;
    if (c->is_retx) {
      w = 1e30;
    } else {
      int qm, r;
      const int m = la_mcs(c);
      rules_mcs(rules_table(c->mcs_table), m, 2, &qm, &r);
      w = (double)qm * r / fmax(c->avg_throughput, 1.0);
    }
    order[i].w = w;
    order[i].i = i;
  }
  qsort(order, n, sizeof(order[0]), compare_weight);
  for (int k = 0; k < n; k++) {
    if (dl_dcis(pp->dl_req) >= MAX_DCI_CORESET)
      break;
    if (schedule_ue(mac, cell, pp, &cands[order[k].i]))
      dl.free_runs = 0; /* the grid changed */
  }
}
