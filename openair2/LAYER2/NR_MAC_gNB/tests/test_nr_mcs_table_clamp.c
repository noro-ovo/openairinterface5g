/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* When the UE capabilities arrive, the CellGroupConfig update can switch the UE to the 256QAM MCS table (max MCS 27)
 * while its link adaptation sits at MCS 28 of the 64QAM table. Between two BLER updates the MCS selection keeps the
 * current MCS: it must still respect the new limit, else MCS 28 (reserved in the 256QAM table) gives no Qm/R and no
 * TBS ("Error in compute TBS with a NULL input") and the UE stops being scheduled correctly. */

#include <stdlib.h>

#include "common/utils/assertions.h"
#include "LAYER2/NR_MAC_gNB/mac_proto.h"
#include "LAYER2/NR_MAC_gNB/nr_mac_gNB.h"
#include "LAYER2/NR_MAC_gNB/gNB_scheduler_dlsch_default_policies.h"
#include "LAYER2/NR_MAC_gNB/gNB_scheduler_ulsch_default_policies.h"

// returns the MCS selected for a new transmission between two BLER updates, current MCS cur, limit max_mcs; *stored is
// the MCS kept in the UE's BLER state
int test_mcs_table_clamp(int dl, int cur, int max_mcs, int *stored)
{
  NR_UE_info_t *ue = calloc(1, sizeof(*ue));
  nr_cell_sched_t *cell = calloc(1, sizeof(*cell));
  AssertFatal(ue && cell, "out of memory\n");
  NR_bler_options_t bo = {.harq_round_max = 4, .min_mcs = 0, .max_mcs = 28, .lower = 0.05, .upper = 0.15};
  int mcs;
  if (dl) {
    cell->dl_bler = bo;
    nr_dl_candidate_t c = {.UE = ue, .is_retx = false, .bler_updated = false, .current_mcs = cur, .max_mcs = max_mcs, .mcs_table = 1};
    nr_dl_mcs_select_default(cell, &c, 1);
    mcs = c.sched_pdsch.mcs;
    *stored = ue->UE_sched_ctrl.dl_bler_stats.mcs;
  } else {
    cell->ul_bler = bo;
    nr_ul_candidate_t c = {.UE = ue, .is_retx = false, .bler_updated = false, .current_mcs = cur, .max_mcs = max_mcs, .mcs_table = 1};
    nr_ul_mcs_select_default(cell, &c, 1);
    mcs = c.sched_pusch.mcs;
    *stored = ue->UE_sched_ctrl.ul_bler_stats.mcs;
  }
  free(cell);
  free(ue);
  return mcs;
}
