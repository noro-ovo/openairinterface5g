/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "gtest/gtest.h"

// returns the number of allocations still above an effective code rate of 0.95 after nr_dl_limit_code_rate()
extern "C" int test_dl_code_rate(int *above, int *checked);

TEST(DlCodeRate, InitialTransmissionAtMost095)
{
  int above = 0, checked = 0;
  EXPECT_EQ(test_dl_code_rate(&above, &checked), 0);
  EXPECT_GT(checked, 0);
}
