/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* 3gpp scheduler, DL (phase 2): decisions from sched_rules on sched_grid, emission by the
 * existing FAPI/MAC PDU code (post_process_dlsch) */

#ifndef SCHED_DL_H_
#define SCHED_DL_H_

#include "openair2/LAYER2/NR_MAC_gNB/nr_mac_gNB.h"

/// pre_processor_dl of the 3gpp scheduler
void sched3gpp_dl_preprocessor(gNB_MAC_INST *mac, nr_cell_sched_t *cell, post_process_pdsch_t *pp_pdsch);

/// timing advance the scheduler assumes for every UE on top of N_TA,offset, in Tc (cell radius)
void sched3gpp_set_max_nta(int64_t n_ta_tc);

/// 38.213 8.4: first symbol, counted from symbol 0 of the PDSCH slot, of the PUCCH that may carry the
/// HARQ-ACK of a Msg4 (or MsgB) PDSCH ending at symbol start + length: N_T,1 + 0.5 ms + T_TA after it
int sched3gpp_msg4_earliest_pucch_symbol(const nr_cell_sched_t *cell, const NR_UE_info_t *UE, int frame, int slot, int start, int length);

#endif /* SCHED_DL_H_ */
