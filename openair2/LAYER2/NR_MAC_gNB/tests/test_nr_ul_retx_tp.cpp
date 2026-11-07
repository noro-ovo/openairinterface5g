/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "gtest/gtest.h"

// returns the number of retransmissions on another TDA that would be sent with another TBS
extern "C" int test_ul_retx_tp(int *feasible);

TEST(UlRetxTransformPrecoding, RetxKeepsTbs)
{
  int feasible = 0;
  EXPECT_EQ(test_ul_retx_tp(&feasible), 0);
  EXPECT_GT(feasible, 0);
}
