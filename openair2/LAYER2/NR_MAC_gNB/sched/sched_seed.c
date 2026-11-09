/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* Seed the scheduler grid from the FAPI PDUs already written for a slot (see sched_seed.h) */

#include "sched_seed.h"

#include <string.h>

#include "common/utils/nr/nr_common.h"

static void set_rb(uint64_t rbs[GRID_RB_WORDS], int rb, int n_rb)
{
  if (rb >= 0 && rb < n_rb && rb < GRID_MAX_RB)
    rbs[rb / 64] |= 1ULL << (rb % 64);
}

static void set_run(uint64_t rbs[GRID_RB_WORDS], int first, int count, int n_rb)
{
  for (int rb = first; rb < first + count; rb++)
    set_rb(rbs, rb, n_rb);
}

/* FAPI resource allocation type 0: bit i of the bitmap (LSB first in each byte) is RB i of the BWP */
static void set_bitmap(uint64_t rbs[GRID_RB_WORDS], const uint8_t bitmap[36], int bwp_start, int bwp_size, int n_rb)
{
  for (int i = 0; i < bwp_size && i < 36 * 8; i++)
    if (bitmap[i / 8] >> (i % 8) & 1)
      set_rb(rbs, bwp_start + i, n_rb);
}

static void mark(sched_grid_t *grid, int64_t slot, bool ul, int sym_start, int n_sym, const uint64_t rbs[GRID_RB_WORDS])
{
  for (int s = sym_start; s < sym_start + n_sym && s < GRID_SYMBOLS; s++)
    if (s >= 0)
      grid_mark(grid, slot, ul, 0, s, rbs);
}

void seed_pdcch_rbs(const nfapi_nr_dl_tti_pdcch_pdu_rel15_t *p, int dci, uint64_t rbs[GRID_RB_WORDS])
{
  /* as the PHY (nr_fill_reg_list, nr_generate_dci): the CORESET starts at the first RB of its bitmap,
   * counted from 6 x ceil(BWP start / 6) for a CORESET configured by PDCCH-Config (38.211 7.3.2.2) */
  int n_rb_cset, cset_start;
  get_coreset_rballoc(p->FreqDomainResource, &n_rb_cset, &cset_start);
  const int dur = p->DurationSymbols;
  if (n_rb_cset == 0 || dur == 0)
    return;
  const int offset = p->CoreSetType == NFAPI_NR_CSET_CONFIG_PDCCH_CONFIG ? (p->BWPStart + 5) / 6 * 6 - p->BWPStart : 0;
  const int first = p->BWPStart + cset_start + offset;
  const int bsize = p->RegBundleSize, R = p->InterleaverSize;
  const int N_regs = n_rb_cset * dur;
  const int C = p->CceRegMappingType == NFAPI_NR_CCE_REG_MAPPING_INTERLEAVED && bsize > 0 && R > 0 ? N_regs / (bsize * R) : 0;
  const int L = p->dci_pdu[dci].AggregationLevel;
  for (int cce = p->dci_pdu[dci].CceIndex; cce < p->dci_pdu[dci].CceIndex + L; cce++)
    for (int b = 0; bsize > 0 && b < 6 / bsize; b++) {
      const int f = cce_to_reg_interleaving(p->CceRegMappingType == NFAPI_NR_CCE_REG_MAPPING_INTERLEAVED ? R : 0,
                                            6 * cce / bsize + b,
                                            p->ShiftIndex,
                                            C,
                                            bsize,
                                            N_regs);
      for (int r = 0; r < bsize / dur; r++)
        set_rb(rbs, first + f * bsize / dur + r, GRID_MAX_RB);
    }
}

static int seed_pdcch(sched_grid_t *grid, int64_t slot, const seed_cfg_t *cfg, const nfapi_nr_dl_tti_pdcch_pdu_rel15_t *p)
{
  for (int d = 0; d < p->numDlDci; d++) {
    uint64_t rbs[GRID_RB_WORDS] = {0};
    seed_pdcch_rbs(p, d, rbs);
    for (int w = 0; w < GRID_RB_WORDS; w++) /* carrier bound */
      for (int b = 0; b < 64; b++)
        if (w * 64 + b >= cfg->n_rb)
          rbs[w] &= ~(1ULL << b);
    mark(grid, slot, false, p->StartSymbolIndex, p->DurationSymbols, rbs);
  }
  return p->numDlDci;
}

int seed_dl_fapi(sched_grid_t *grid,
                 int64_t slot,
                 const seed_cfg_t *cfg,
                 const nfapi_nr_dl_tti_request_body_t *dl,
                 const nfapi_nr_ul_dci_request_t *ul_dci)
{
  int n = 0;
  for (int i = 0; dl != NULL && i < dl->nPDUs; i++) {
    const nfapi_nr_dl_tti_request_pdu_t *pdu = &dl->dl_tti_pdu_list[i];
    uint64_t rbs[GRID_RB_WORDS] = {0};
    if (pdu->PDUType == NFAPI_NR_DL_TTI_SSB_PDU_TYPE) {
      /* 20 RBs of the SSB, one more when k_SSB > 0; ssbOffsetPointA counts 15 kHz RBs (60 kHz in FR2) */
      const nfapi_nr_dl_tti_ssb_pdu_rel15_t *s = &pdu->ssb_pdu.ssb_pdu_rel15;
      const int first = s->ssbOffsetPointA >> (cfg->fr2 ? cfg->ssb_mu - 2 : cfg->ssb_mu);
      set_run(rbs, first, 20 + (s->SsbSubcarrierOffset > 0), cfg->n_rb);
      mark(grid, slot, false, cfg->ssb_start_symbol[s->SsbBlockIndex % 64] % GRID_SYMBOLS, 4, rbs);
      n++;
    } else if (pdu->PDUType == NFAPI_NR_DL_TTI_PDCCH_PDU_TYPE) {
      n += seed_pdcch(grid, slot, cfg, &pdu->pdcch_pdu.pdcch_pdu_rel15);
    } else if (pdu->PDUType == NFAPI_NR_DL_TTI_PDSCH_PDU_TYPE) {
      const nfapi_nr_dl_tti_pdsch_pdu_rel15_t *p = &pdu->pdsch_pdu.pdsch_pdu_rel15;
      if (p->resourceAlloc == 0)
        set_bitmap(rbs, p->rbBitmap, p->BWPStart, p->BWPSize, cfg->n_rb);
      else
        set_run(rbs, p->BWPStart + p->rbStart, p->rbSize, cfg->n_rb);
      mark(grid, slot, false, p->StartSymbolIndex, p->NrOfSymbols, rbs);
      n++;
    } else if (pdu->PDUType == NFAPI_NR_DL_TTI_CSI_RS_PDU_TYPE) {
      /* the OAI gNB PHY maps the PDSCH over the CSI-RS REs while the UE rate-matches around them
       * (38.214 5.1.4.1, csiRsForRateMatching): a PDSCH must not share their symbols. Whole RB range on
       * the symbols of the row, 38.211 table 7.4.1.5.3-1, as the UE (overlap_csi_symbol) */
      static const int num_l0[18] = {1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 4, 2, 2, 4};
      const nfapi_nr_dl_tti_csi_rs_pdu_rel15_t *c = &pdu->csi_rs_pdu.csi_rs_pdu_rel15;
      if (c->row >= 1 && c->row <= 18) {
        set_run(rbs, c->start_rb, c->nr_of_rbs, cfg->n_rb);
        mark(grid, slot, false, c->symb_l0, num_l0[c->row - 1], rbs);
        if (c->row == 13 || c->row == 14 || c->row == 16 || c->row == 17)
          mark(grid, slot, false, c->symb_l1, 2, rbs);
        n++;
      }
    }
  }
  for (int i = 0; ul_dci != NULL && i < ul_dci->numPdus; i++)
    n += seed_pdcch(grid, slot, cfg, &ul_dci->ul_dci_pdu_list[i].pdcch_pdu.pdcch_pdu_rel15);
  return n;
}

void seed_ul_pucch(sched_grid_t *grid, int64_t slot, const seed_cfg_t *cfg, const nfapi_nr_pucch_pdu_t *p)
{
  uint64_t rbs[GRID_RB_WORDS] = {0};
  set_run(rbs, p->bwp_start + p->prb_start, p->prb_size, cfg->n_rb);
  if (!p->freq_hop_flag) {
    mark(grid, slot, true, p->start_symbol_index, p->nr_of_symbols, rbs);
    return;
  }
  /* first hop floor(N/2) symbols (38.211 6.3.2.1) */
  const int first = p->nr_of_symbols / 2;
  mark(grid, slot, true, p->start_symbol_index, first, rbs);
  uint64_t hop[GRID_RB_WORDS] = {0};
  set_run(hop, p->bwp_start + p->second_hop_prb, p->prb_size, cfg->n_rb);
  mark(grid, slot, true, p->start_symbol_index + first, p->nr_of_symbols - first, hop);
}

int seed_ul_fapi(sched_grid_t *grid,
                 int64_t slot,
                 const seed_cfg_t *cfg,
                 const nfapi_nr_ul_tti_request_t *ul,
                 const nfapi_nr_ul_tti_request_t *const *prev,
                 int num_prev)
{
  int n = 0;
  /* a long-format PRACH written k slots before still takes all symbols of this slot while k < prach_slots */
  for (int k = 0; k < num_prev && k + 1 < cfg->prach_slots; k++)
    for (int i = 0; prev[k] != NULL && i < prev[k]->n_pdus; i++)
      if (prev[k]->pdus_list[i].pdu_type == NFAPI_NR_UL_CONFIG_PRACH_PDU_TYPE) {
        uint64_t rbs[GRID_RB_WORDS] = {0};
        set_run(rbs, cfg->prach_first_rb, cfg->prach_n_rb, cfg->n_rb);
        mark(grid, slot, true, 0, GRID_SYMBOLS, rbs);
        n++;
      }
  for (int i = 0; ul != NULL && i < ul->n_pdus; i++) {
    const nfapi_nr_ul_tti_request_number_of_pdus_t *pdu = &ul->pdus_list[i];
    uint64_t rbs[GRID_RB_WORDS] = {0};
    switch (pdu->pdu_type) {
      case NFAPI_NR_UL_CONFIG_PUSCH_PDU_TYPE: {
        const nfapi_nr_pusch_pdu_t *p = &pdu->pusch_pdu;
        if (p->resource_alloc == 0)
          set_bitmap(rbs, p->rb_bitmap, p->bwp_start, p->bwp_size, cfg->n_rb);
        else
          set_run(rbs, p->bwp_start + p->rb_start, p->rb_size, cfg->n_rb);
        mark(grid, slot, true, p->start_symbol_index, p->nr_of_symbols, rbs);
        n++;
        break;
      }
      case NFAPI_NR_UL_CONFIG_PUCCH_PDU_TYPE:
        seed_ul_pucch(grid, slot, cfg, &pdu->pucch_pdu);
        n++;
        break;
      case NFAPI_NR_UL_CONFIG_SRS_PDU_TYPE: {
        /* the whole BWP on the SRS symbols, as the legacy booking: the RBs of one occasion depend on
         * C_SRS, B_SRS, n_RRC and hopping (38.211 6.4.1.4.3), not modelled yet */
        const nfapi_nr_srs_pdu_t *p = &pdu->srs_pdu;
        set_run(rbs, p->bwp_start, p->bwp_size, cfg->n_rb);
        mark(grid, slot, true, p->time_start_position, 1 << p->num_symbols, rbs);
        n++;
        break;
      }
      case NFAPI_NR_UL_CONFIG_PRACH_PDU_TYPE: {
        const nfapi_nr_prach_pdu_t *p = &pdu->prach_pdu;
        set_run(rbs, cfg->prach_first_rb, cfg->prach_n_rb, cfg->n_rb);
        if (cfg->prach_dur >= GRID_SYMBOLS)
          mark(grid, slot, true, 0, GRID_SYMBOLS, rbs);
        else
          mark(grid, slot, true, p->prach_start_symbol, p->num_prach_ocas * cfg->prach_dur, rbs);
        n++;
        break;
      }
      default:
        break;
    }
  }
  return n;
}
