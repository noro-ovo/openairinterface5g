/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "gtest/gtest.h"

// returns the number of Long BSRs whose estimate is below the smallest buffer size their indexes mean
extern "C" int test_ul_long_bsr(int *checked);

TEST(UlLongBsr, EstimateCoversReportedSize)
{
  int checked = 0;
  EXPECT_EQ(test_ul_long_bsr(&checked), 0);
  EXPECT_GT(checked, 0);
}
