/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* Long BSR Buffer Size index i of 38.321 Table 6.1.3.1-2 means NR_LONG_BSR_TABLE[i - 1] < BS <= NR_LONG_BSR_TABLE[i],
 * index 254 means BS > 81338368 and index 255 is reserved. nr_estimate_ul_buffer_long_bsr() gives the bytes the
 * scheduler assumes the UE has in its buffer. */

#include <stdio.h>
#include <stdint.h>

#include "LAYER2/NR_MAC_gNB/mac_proto.h"
#include "LAYER2/NR_MAC_COMMON/nr_mac_common.h"

static int estimate(const uint8_t *ce, int ce_len)
{
  return nr_estimate_ul_buffer_long_bsr((const NR_BSR_LONG *)ce, ce_len);
}

// returns the number of Long BSRs whose estimate is below the smallest buffer size their indexes mean
int test_ul_long_bsr(int *checked)
{
  int bad = 0;
  *checked = 0;
  // one LCG, every Buffer Size index
  for (int lcg = 0; lcg < 8; lcg++) {
    for (int idx = 1; idx <= 254; idx++) {
      const uint8_t ce[2] = {1 << lcg, idx};
      const int64_t min_bs = (int64_t)get_long_bsr_value(idx - 1) + 1;
      const int est = estimate(ce, sizeof(ce));
      (*checked)++;
      if (est < min_bs) {
        if (bad < 5)
          printf("LCG %d index %d: estimate %d bytes, the UE has more than %ld bytes\n", lcg, idx, est, min_bs - 1);
        bad++;
      }
    }
  }
  // the eight LCGs with the same index
  for (int idx = 1; idx <= 254; idx++) {
    uint8_t ce[9] = {0xff};
    for (int i = 1; i < 9; i++)
      ce[i] = idx;
    const int64_t min_bs = 8 * ((int64_t)get_long_bsr_value(idx - 1) + 1);
    const int est = estimate(ce, sizeof(ce));
    (*checked)++;
    if (est < min_bs) {
      if (bad < 5)
        printf("8 LCGs index %d: estimate %d bytes, the UE has more than %ld bytes\n", idx, est, min_bs - 1);
      bad++;
    }
  }
  printf("%d Long BSRs, %d estimated below the reported buffer size\n", *checked, bad);
  return bad;
}
