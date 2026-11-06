/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* With transform precoding, a PUSCH spans 2^a 3^b 5^c RBs (38.211 6.3.1.4). A retransmission moved to another TDA
 * keeps its TBS with a refitted RB count (check_ul_retx_feasibility()): that count must also be a valid transform
 * precoding size, or the gNB and the UE have no low-PAPR DMRS sequence for it. */

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include "common/ran_context.h"
#include "common/utils/assertions.h"
#include "LAYER2/NR_MAC_gNB/mac_proto.h"
#include "LAYER2/NR_MAC_gNB/nr_mac_gNB.h"

// from nr_unitary_defs.h, which defines globals and is already included by test_nr_other_sib_ssb_index.c
void prepare_scc(NR_ServingCellConfigCommon_t *scc);
void fill_scc_sim(NR_ServingCellConfigCommon_t *scc, uint64_t *ssb_bitmap, int N_RB_DL, int N_RB_UL, int mu_dl, int mu_ul);
void fix_scc(NR_ServingCellConfigCommon_t *scc, uint64_t ssbmap);

// number of refitted RB counts that are not a transform precoding size; *feasible: retx accepted on the new TDA
int test_ul_retx_tp(int *feasible)
{
  NR_ServingCellConfigCommon_t *scc = calloc_or_fail(1, sizeof(*scc));
  prepare_scc(scc);
  uint64_t sim_bitmap;
  fill_scc_sim(scc, &sim_bitmap, 106, 106, 1, 1);
  fix_scc(scc, 0x1);

  NR_UE_info_t *UE = calloc_or_fail(1, sizeof(*UE));
  UE->rnti = 0x1234;
  UE->current_UL_BWP.transform_precoding = NR_PUSCH_Config__transformPrecoder_enabled;
  const int bwp_size = 106, mcs_table = 3, pid = 0; // mcs_table 3: transform precoding 64QAM (38.214 6.1.4.1-1)
  NR_sched_pusch_t *ret = &UE->UE_sched_ctrl.ul_harq_processes[pid].sched_pusch;
  const nr_ul_candidate_t cand = {.UE = UE, .rnti = UE->rnti, .is_retx = true, .retx_harq_pid = pid};

  int bad = 0;
  *feasible = 0;
  for (int old_sym = 4; old_sym <= 14; old_sym++) {
    for (int new_sym = 4; new_sym <= 14; new_sym++) {
      if (new_sym == old_sym)
        continue;
      const NR_tda_info_t old_tda = {.mapping_type = typeB, .startSymbolIndex = 14 - old_sym, .nrOfSymbols = old_sym, .valid_tda = true};
      const NR_tda_info_t new_tda = {.mapping_type = typeB, .startSymbolIndex = 14 - new_sym, .nrOfSymbols = new_sym, .valid_tda = true};
      for (int mcs = 0; mcs <= 27; mcs++) {
        for (int rb = 1; rb <= bwp_size; rb++) {
          if (check_sc_fdma_rbsize(NR_PUSCH_Config__transformPrecoder_enabled, rb) != rb)
            continue; // the first transmission has a transform precoding size
          *ret = (NR_sched_pusch_t){.rbSize = rb, .mcs = mcs, .Qm = nr_get_Qm_ul(mcs, mcs_table), .R = nr_get_code_rate_ul(mcs, mcs_table),
                                    .nrOfLayers = 1, .time_domain_allocation = 0, .tda_info = old_tda};
          ret->dmrs_info = get_ul_dmrs_params(scc, &UE->current_UL_BWP, &old_tda, 1, 0, 0);
          ret->tb_size = nr_compute_tbs(ret->Qm, ret->R, rb, old_sym, ret->dmrs_info.N_PRB_DMRS * ret->dmrs_info.num_dmrs_symb, 0, 0, 1) >> 3;
          // function under test
          const uint16_t new_rb = check_ul_retx_feasibility(&cand, 1, &new_tda, scc, bwp_size);
          if (new_rb == 0)
            continue;
          (*feasible)++;
          if (check_sc_fdma_rbsize(NR_PUSCH_Config__transformPrecoder_enabled, new_rb) != new_rb) {
            if (bad < 5)
              printf("retx mcs %d %d RBs %d symbols -> %d symbols: %d RBs, not a transform precoding size\n", mcs, rb, old_sym, new_sym, new_rb);
            bad++;
          }
        }
      }
    }
  }
  printf("%d retransmissions accepted on another TDA, %d with an RB count that is not a transform precoding size\n", *feasible, bad);
  free(UE);
  return bad;
}
