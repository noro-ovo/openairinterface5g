/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* What other code already put in a slot, read from the FAPI PDUs written for it (what the PHY will
 * transmit or receive) and marked in the scheduler grid: SSB, PDCCH REGs, PDSCH, CSI-RS, PUSCH, PUCCH,
 * SRS, PRACH. It replaces the copy of the legacy VRB maps, so the grid no longer depends on how the
 * legacy code books its resources. */

#ifndef SCHED_SEED_H_
#define SCHED_SEED_H_

#include <stdbool.h>
#include <stdint.h>
#include "nfapi_nr_interface_scf.h"
#include "sched_grid.h"

typedef struct {
  int n_rb; ///< carrier RBs
  /* SSB */
  int ssb_mu; ///< SSB subcarrier spacing
  bool fr2;
  int ssb_start_symbol[64]; ///< first symbol of each SSB index, counted from the start of the half frame
  /* PRACH: where the PHY receives it (FAPI CONFIG k1 of the first FD occasion) */
  int prach_first_rb, prach_n_rb; ///< n_RA_RB x FDM
  int prach_dur; ///< symbols of one occasion, 14 for the long formats (whole slot)
  int prach_slots; ///< slots of a long-format PRACH (it is written once, in its first slot), 1 otherwise
} seed_cfg_t;

/// RBs x symbols of the SSB, PDCCH (REGs of each DCI), PDSCH and CSI-RS PDUs of a DL slot, and of the UL DCIs
/// (may be NULL); returns the number of PDUs marked
int seed_dl_fapi(sched_grid_t *grid,
                 int64_t slot,
                 const seed_cfg_t *cfg,
                 const nfapi_nr_dl_tti_request_body_t *dl,
                 const nfapi_nr_ul_dci_request_t *ul_dci);

/// RBs x symbols of the PUSCH, PUCCH, SRS and PRACH PDUs of an UL slot, and of the long-format PRACH
/// PDUs written in the slots before it that still cover it (prev[k] is the request of slot - 1 - k, NULL
/// when unknown); returns the number of PDUs marked
int seed_ul_fapi(sched_grid_t *grid,
                 int64_t slot,
                 const seed_cfg_t *cfg,
                 const nfapi_nr_ul_tti_request_t *ul,
                 const nfapi_nr_ul_tti_request_t *const *prev,
                 int num_prev);

/// RBs x symbols of one PUCCH PDU, both hops
void seed_ul_pucch(sched_grid_t *grid, int64_t slot, const seed_cfg_t *cfg, const nfapi_nr_pucch_pdu_t *p);

/// carrier RBs of the REGs of one DCI, in rbs (bit per RB); the PHY mapping of 38.211 7.3.2.2
void seed_pdcch_rbs(const nfapi_nr_dl_tti_pdcch_pdu_rel15_t *p, int dci, uint64_t rbs[GRID_RB_WORDS]);

#endif /* SCHED_SEED_H_ */
