/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

// Polar encoder / decoder of PBCH, PSBCH, DCI and UCI (polar_encoder_fast(), polar_decoder_int16()): round trip, and
// bit-exactness with the original implementation (OAI develop 9313cbea) through a digest of deterministic encode /
// decode sequences (nr_polar_test_vectors.c).

#include "gtest/gtest.h"
#include <cstdint>

extern "C" {
#include "common/utils/LOG/log.h"
uint64_t nr_polar_test_digest(int *ncfg);
int nr_polar_test_roundtrip(int *ncfg);
void crcTableInit(void);
}

TEST(nr_polar, roundtrip)
{
  int ncfg = 0;
  EXPECT_EQ(nr_polar_test_roundtrip(&ncfg), 0);
  EXPECT_EQ(ncfg, 1001);
}

TEST(nr_polar, bit_exact_with_original_implementation)
{
  int ncfg = 0;
  // digest of the original implementation (tree decoder, nr_polar_uxG() encoder) over the same sequences
  EXPECT_EQ(nr_polar_test_digest(&ncfg), 0x098bd2cebdfbd2c9ULL);
  EXPECT_EQ(ncfg, 1001);
}

int main(int argc, char **argv)
{
  logInit();
  crcTableInit();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
