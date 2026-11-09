/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* SearchSpace nrofCandidates (38.331) is ENUMERATED {n0, n1, n2, n3, n4, n5, n6, n8}: the last value is 8 candidates.
 * find_aggregation_candidates() gives the number of PDCCH candidates M of 38.213 10.1 for an aggregation level. */

#include <stdio.h>
#include "openair2/LAYER2/NR_MAC_COMMON/nr_mac_common.h"
#include "executables/softmodem-common.h"

softmodem_params_t *get_softmodem_params(void)
{
  return NULL;
}

int main(void)
{
  const int candidates[8] = {0, 1, 2, 3, 4, 5, 6, 8};
  int bad = 0;
  for (int al = 0; al < 5; al++) {
    const int L = 1 << al;
    for (long e = 0; e < 8; e++) {
      struct NR_SearchSpace__nrofCandidates nrof = {0};
      long *field[5] = {&nrof.aggregationLevel1,
                        &nrof.aggregationLevel2,
                        &nrof.aggregationLevel4,
                        &nrof.aggregationLevel8,
                        &nrof.aggregationLevel16};
      *field[al] = e;
      const NR_SearchSpace_t ss = {.nrofCandidates = &nrof};
      int level = 0, m = -1;
      find_aggregation_candidates(&level, &m, &ss, L);
      if (m != candidates[e] || (m > 0 && level != L)) {
        printf("L %d, nrofCandidates %ld: %d candidates at L %d, expected %d\n", L, e, m, level, candidates[e]);
        bad++;
      }
    }
  }
  printf("%d of 40 aggregation level / nrofCandidates values wrong\n", bad);
  return bad != 0;
}
