/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "gtest/gtest.h"

// returns the number of gap slots in which the UE can be scheduled, for every gapOffset of an MGRP of 160 ms
extern "C" int test_measgap(int mu, int *checked);

class MeasGap : public ::testing::TestWithParam<int> {};

TEST_P(MeasGap, NoSchedulingInGap)
{
  int checked = 0;
  EXPECT_EQ(test_measgap(GetParam(), &checked), 0);
  EXPECT_GT(checked, 0);
}

INSTANTIATE_TEST_SUITE_P(Numerology, MeasGap, ::testing::Values(0, 1));
