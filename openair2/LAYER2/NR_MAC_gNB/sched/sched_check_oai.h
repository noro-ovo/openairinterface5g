/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef SCHED_CHECK_OAI_H_
#define SCHED_CHECK_OAI_H_

#include "openair2/LAYER2/NR_MAC_gNB/nr_mac_gNB.h"

/// run the conformance checker on the output of a slot, if OAI_SCHED_CHECK is set
void nr_sched_check_slot(const nr_cell_sched_t *cell, const NR_Sched_Rsp_t *sched_info);
/// a preamble reported by the PHY in a RACH indication, before the MAC filters it
void nr_sched_check_preamble(int frame, int slot, int rapid, int s_id, int f_id);
/// log the counters, return the number of findings (0 if the checker is off)
uint64_t nr_sched_check_summary(void);
/// PDUs the checker could not follow (queue or UE table full): the findings are then incomplete
uint64_t nr_sched_check_dropped(void);

#endif /* SCHED_CHECK_OAI_H_ */
