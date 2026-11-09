/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "sched_seed_oai.h"

#include "common/utils/nr/nr_common.h"
#include "openair2/LAYER2/NR_MAC_COMMON/nr_mac_common.h"
#include "openair2/LAYER2/NR_MAC_gNB/mac_proto.h"

void seed_cfg_from_cell(const nr_cell_sched_t *cell, int n_rb, seed_cfg_t *cfg)
{
  const NR_ServingCellConfigCommon_t *scc = cell->common_channels.ServingCellConfigCommon;
  *cfg = (seed_cfg_t){.n_rb = n_rb};
  const long band = *scc->downlinkConfigCommon->frequencyInfoDL->frequencyBandList.list.array[0];
  const NR_SubcarrierSpacing_t scs = *scc->ssbSubcarrierSpacing;
  cfg->ssb_mu = scs;
  cfg->fr2 = scc->ssb_PositionsInBurst->present == 3;
  for (int i = 0; i < get_max_ssbs(scc) && i < 64; i++)
    cfg->ssb_start_symbol[i] = get_ssb_start_symbol(band, scs, i);
  /* where the PHY receives the PRACH: FAPI CONFIG k1 of the first FD occasion, n_RA_RB x FDM RBs */
  const int mu_pusch = scc->uplinkConfigCommon->frequencyInfoUL->scs_SpecificCarrierList.list.array[0]->subcarrierSpacing;
  cfg->prach_first_rb = cell->config.prach_config.num_prach_fd_occasions_list[0].k1.value;
  cfg->prach_n_rb = get_N_RA_RB(cell->config.prach_config.prach_sub_c_spacing.value, mu_pusch)
                    * cell->config.prach_config.num_prach_fd_occasions.value;
  const NR_COMMON_channels_t *cc = &cell->common_channels;
  const bool long_format = (cc->prach_info.format & 0xff) < 4;
  cfg->prach_dur = long_format ? GRID_SYMBOLS : cc->prach_info.N_dur;
  cfg->prach_slots = long_format ? cc->prach_len : 1;
}
