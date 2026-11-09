/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* sched_seed configuration from the OAI cell */

#ifndef SCHED_SEED_OAI_H_
#define SCHED_SEED_OAI_H_

#include "openair2/LAYER2/NR_MAC_gNB/nr_mac_gNB.h"
#include "sched_seed.h"

void seed_cfg_from_cell(const nr_cell_sched_t *cell, int n_rb, seed_cfg_t *cfg);

#endif /* SCHED_SEED_OAI_H_ */
