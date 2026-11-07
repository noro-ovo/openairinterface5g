/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* The UE may skip decoding an initial transmission above an effective channel code rate of 0.95 (38.214 5.1.3). The DL
 * scheduler sizes a new transmission with nr_compute_tbs() at the MCS of the link adaptation; nr_dl_limit_code_rate()
 * then lowers that MCS on the same RBs until the rate is met. */

#include <stdio.h>
#include <stdlib.h>

#include "common/utils/assertions.h"
#include "LAYER2/NR_MAC_gNB/mac_proto.h"
#include "LAYER2/NR_MAC_gNB/nr_mac_gNB.h"

// *above: allocations whose TBS at the scheduled MCS is above 0.95, *checked: all allocations; returns the number of
// allocations still above 0.95 (or without a TB) after nr_dl_limit_code_rate()
int test_dl_code_rate(int *above, int *checked)
{
  const int max_mcs[2] = {28, 27}; // MCS table 1 (64QAM) and 2 (256QAM), without the reserved indexes
  // DM-RS REs per PRB and DM-RS symbols: type 1 with 1 or 2 CDM groups without data, 1 to 4 DM-RS symbols
  const struct {
    int n_prb_dmrs, n_dmrs_slot;
  } dmrs[] = {{6, 1}, {12, 1}, {12, 2}, {12, 3}, {12, 4}};
  int bad = 0;
  *above = *checked = 0;
  for (int table = 0; table < 2; table++) {
    for (int mcs = 0; mcs <= max_mcs[table]; mcs++) {
      for (int layers = 1; layers <= 4; layers++) {
        for (int nb_symb = 2; nb_symb <= 14; nb_symb++) {
          for (int d = 0; d < sizeofArray(dmrs); d++) {
            if (dmrs[d].n_dmrs_slot >= nb_symb)
              continue;
            for (int rb = 5; rb <= 273; rb++) { // the DL scheduler allocates at least 5 RBs (min_rbSize, nr_find_nb_rb())
              NR_sched_pdsch_t p = {.rbSize = rb,
                                    .mcs = mcs,
                                    .Qm = nr_get_Qm_dl(mcs, table),
                                    .R = nr_get_code_rate_dl(mcs, table),
                                    .nrOfLayers = layers,
                                    .tda_info = {.nrOfSymbols = nb_symb},
                                    .dmrs_parms = {.N_PRB_DMRS = dmrs[d].n_prb_dmrs, .N_DMRS_SLOT = dmrs[d].n_dmrs_slot}};
              const int nb_dmrs_prb = dmrs[d].n_prb_dmrs * dmrs[d].n_dmrs_slot;
              p.tb_size = nr_compute_tbs(p.Qm, p.R, rb, nb_symb, nb_dmrs_prb, 0, 0, layers) >> 3;
              (*checked)++;
              if (!nr_dl_code_rate_ok(p.tb_size, p.R, p.Qm, layers, rb, nb_symb, nb_dmrs_prb))
                (*above)++;
              // function under test
              nr_dl_limit_code_rate(&p, table);
              if (p.tb_size == 0 || !nr_dl_code_rate_ok(p.tb_size, p.R, p.Qm, layers, rb, nb_symb, nb_dmrs_prb)) {
                if (bad < 5)
                  printf("table %d MCS %d, %d layers, %d symbols, DM-RS %d RE, %d RBs: MCS %d TBS %u still above 0.95\n",
                         table + 1, mcs, layers, nb_symb, nb_dmrs_prb, rb, p.mcs, p.tb_size);
                bad++;
              }
            }
          }
        }
      }
    }
  }
  printf("%d allocations, %d above 0.95 at the scheduled MCS, %d after nr_dl_limit_code_rate()\n", *checked, *above, bad);
  return bad;
}
