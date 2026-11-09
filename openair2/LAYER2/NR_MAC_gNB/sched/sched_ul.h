/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* 3gpp scheduler, UL (phase 3): decisions from sched_rules on sched_grid, emission by the existing
 * nr_ul_apply_*_transmission */

#ifndef SCHED_UL_H_
#define SCHED_UL_H_

#include "openair2/LAYER2/NR_MAC_gNB/nr_mac_gNB.h"

/// pre_processor_ul of the 3gpp scheduler
void sched3gpp_ul_preprocessor(gNB_MAC_INST *mac, nr_cell_sched_t *cell, post_process_pusch_t *pp_pusch);

/// timing advance assumed for every UE on top of N_TA,offset, in Tc (cell radius)
void sched3gpp_ul_set_max_nta(int64_t n_ta_tc);

#endif /* SCHED_UL_H_ */
