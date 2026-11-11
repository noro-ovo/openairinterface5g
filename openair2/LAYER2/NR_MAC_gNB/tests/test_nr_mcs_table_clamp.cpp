/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "gtest/gtest.h"

extern "C" int test_mcs_table_clamp(int dl, int cur, int max_mcs, int *stored);

// MCS 28 chosen with the 64QAM table, then the UE moves to the 256QAM table (max 27) before the next BLER update
TEST(McsTableClamp, UlNewTxRespectsLowerLimit)
{
  int stored = -1;
  EXPECT_EQ(test_mcs_table_clamp(0, 28, 27, &stored), 27);
  EXPECT_EQ(stored, 27);
}

TEST(McsTableClamp, DlNewTxRespectsLowerLimit)
{
  int stored = -1;
  EXPECT_EQ(test_mcs_table_clamp(1, 28, 27, &stored), 27);
  EXPECT_EQ(stored, 27);
}

// an MCS within the limit is kept as before
TEST(McsTableClamp, MCSWithinLimitKept)
{
  int stored = -1;
  EXPECT_EQ(test_mcs_table_clamp(0, 20, 27, &stored), 20);
  EXPECT_EQ(test_mcs_table_clamp(1, 20, 27, &stored), 20);
}
