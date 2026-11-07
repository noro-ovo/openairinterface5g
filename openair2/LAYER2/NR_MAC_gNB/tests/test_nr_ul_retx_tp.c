/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* A retransmission keeps the TBS of the first transmission (38.214 6.1.4.2). When the scheduler moves it to another
 * TDA, check_ul_retx_feasibility() refits the RB count; with transform precoding a PUSCH spans 2^a 3^b 5^c RBs
 * (38.211 6.3.1.4), so apply_ul_retransmission() rounds that count down to such a size and computes the TBS again. The
 * TBS sent must still be the TBS of the first transmission. */

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

// number of retransmissions accepted on another TDA that would be sent with another TBS; *feasible: accepted ones
int test_ul_retx_tp(int *feasible)
{
  NR_ServingCellConfigCommon_t *scc = calloc_or_fail(1, sizeof(*scc));
  prepare_scc(scc);
  uint64_t sim_bitmap;
  fill_scc_sim(scc, &sim_bitmap, 106, 106, 1, 1);
  fix_scc(scc, 0x1);

  NR_UE_info_t *UE = calloc_or_fail(1, sizeof(*UE));
  UE->rnti = 0x1234;
  const long tp = NR_PUSCH_Config__transformPrecoder_enabled;
  UE->current_UL_BWP.transform_precoding = tp;
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
      const NR_pusch_dmrs_t new_dmrs = get_ul_dmrs_params(scc, &UE->current_UL_BWP, &new_tda, 1, 0, 0);
      for (int mcs = 0; mcs <= 27; mcs++) {
        for (int rb = 1; rb <= bwp_size; rb++) {
          if (check_sc_fdma_rbsize(tp, rb) != rb)
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
          // RB count and TBS as apply_ul_retransmission() sends them
          const uint16_t sent_rb = check_sc_fdma_rbsize(tp, new_rb);
          const uint32_t sent_tbs =
              nr_compute_tbs(ret->Qm, ret->R, sent_rb, new_sym, new_dmrs.N_PRB_DMRS * new_dmrs.num_dmrs_symb, 0, 0, 1) >> 3;
          if (sent_tbs != ret->tb_size) {
            if (bad < 5)
              printf("retx mcs %d %d RBs %d -> %d symbols: refit %d RBs, sent %d RBs with TBS %u instead of %u\n",
                     mcs, rb, old_sym, new_sym, new_rb, sent_rb, sent_tbs, ret->tb_size);
            bad++;
          }
        }
      }
    }
  }
  printf("%d retransmissions accepted on another TDA, %d sent with another TBS\n", *feasible, bad);
  free(UE);
  return bad;
}
