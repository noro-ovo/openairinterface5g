/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

// Qm-dependent int16 -> int8 LLR conversion of the LDPC segment decoder (nrLDPC_llr_prescale.c) against a scalar
// reference: legacy pack when off, symmetric rounding, saturation, filler/zero preservation, rule parsing.

#include "gtest/gtest.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

extern "C" {
#include "openair1/PHY/CODING/nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_llr_prescale.h"
}

namespace {

constexpr int NVEC = 4096; // 65536 values: every int16 once

std::vector<int16_t, std::allocator<int16_t>> all_int16()
{
  std::vector<int16_t> v(NVEC * 16);
  for (int i = 0; i < NVEC * 16; i++)
    v[i] = (int16_t)(i - 32768);
  return v;
}

int8_t sat8(long v)
{
  return v > 127 ? 127 : v < -128 ? -128 : (int8_t)v;
}

// scalar reference of the documented conversion
int8_t ref(int16_t x, const nr_ldpc_llr_prescale_t &p)
{
  switch (p.mode) {
    case NR_LDPC_LLR_PRESCALE_FRAC: {
      const long xc = x < -32767 ? -32767 : x;
      const long a = std::labs(xc);
      const long y = (a * p.q15 + (1 << 14)) >> 15;
      return sat8(xc < 0 ? -y : y);
    }
    case NR_LDPC_LLR_PRESCALE_LSHIFT: {
      long y = (long)x << p.lshift;
      y = y > 32767 ? 32767 : y < -32768 ? -32768 : y;
      return sat8(y);
    }
    default:
      return sat8(x);
  }
}

void run(const std::vector<int16_t> &in, std::vector<int8_t> &out, const nr_ldpc_llr_prescale_t *p)
{
  alignas(16) static int16_t z[NVEC * 16];
  alignas(16) static int8_t l[NVEC * 16];
  memcpy(z, in.data(), sizeof(z));
  nrLDPC_llr_pack_int8(z, l, NVEC, p);
  out.assign(l, l + NVEC * 16);
}

TEST(nrLDPC_llr_prescale, parse)
{
  nr_ldpc_llr_prescale_rule_t r;
  ASSERT_EQ(nrLDPC_llr_prescale_parse("default", &r), 0);
  nr_ldpc_llr_prescale_rule_t d;
  ASSERT_EQ(nrLDPC_llr_prescale_parse(NR_LDPC_LLR_PRESCALE_DEFAULT, &d), 0);
  EXPECT_EQ(memcmp(&r, &d, sizeof(r)), 0);
  ASSERT_EQ(nrLDPC_llr_prescale_parse("1,0.5,0.5,1", &r), 0);
  EXPECT_EQ(r.qm[0][2].mode, NR_LDPC_LLR_PRESCALE_NONE);
  EXPECT_EQ(r.qm[0][4].mode, NR_LDPC_LLR_PRESCALE_FRAC);
  EXPECT_EQ(r.qm[0][4].q15, 16384);
  EXPECT_EQ(r.qm[0][6].q15, 16384);
  EXPECT_EQ(r.qm[0][8].mode, NR_LDPC_LLR_PRESCALE_NONE);
  EXPECT_EQ(r.qm[0][4].q8, 128);
  EXPECT_EQ(r.qm[0][8].q8, 256);
  for (const char *off : {"off", "none", "0", "1,1,1,1"}) {
    ASSERT_EQ(nrLDPC_llr_prescale_parse(off, &r), 0) << off;
    for (int l = 0; l < 2; l++)
      for (int q = 0; q <= NR_LDPC_LLR_PRESCALE_MAX_QM; q++) {
        EXPECT_EQ(r.qm[l][q].mode, NR_LDPC_LLR_PRESCALE_NONE) << off;
        EXPECT_EQ(r.qm[l][q].q8, 256) << off;
      }
  }
  ASSERT_EQ(nrLDPC_llr_prescale_parse("2,0.75,0.25,4", &r), 0);
  EXPECT_EQ(r.qm[0][2].mode, NR_LDPC_LLR_PRESCALE_LSHIFT);
  EXPECT_EQ(r.qm[0][2].lshift, 1);
  EXPECT_EQ(r.qm[0][4].q15, 24576);
  EXPECT_EQ(r.qm[0][6].q15, 8192);
  EXPECT_EQ(r.qm[0][8].lshift, 2);
  EXPECT_EQ(r.qm[0][8].q8, 1024);
  // Qm outside {2,4,6,8} (pi/2-BPSK, unknown) is always x1
  EXPECT_EQ(r.qm[0][0].mode, NR_LDPC_LLR_PRESCALE_NONE);
  EXPECT_EQ(r.qm[0][1].mode, NR_LDPC_LLR_PRESCALE_NONE);
  // optional second set for 2 or more layers; a single set applies to any number of layers
  ASSERT_EQ(nrLDPC_llr_prescale_parse("1,0.5,0.5,1/1, 0.25 ,0.5,0.5", &r), 0);
  EXPECT_EQ(r.qm[0][4].q15, 16384);
  EXPECT_EQ(r.qm[1][4].q15, 8192);
  EXPECT_EQ(r.qm[0][8].mode, NR_LDPC_LLR_PRESCALE_NONE);
  EXPECT_EQ(r.qm[1][8].q15, 16384);
  EXPECT_EQ(nrLDPC_llr_prescale_get(&r, 4, 1), &r.qm[0][4]);
  EXPECT_EQ(nrLDPC_llr_prescale_get(&r, 4, 2), &r.qm[1][4]);
  EXPECT_EQ(nrLDPC_llr_prescale_get(&r, 4, 4), &r.qm[1][4]);
  EXPECT_EQ(nrLDPC_llr_prescale_get(&r, 4, 0), &r.qm[0][4]);
  EXPECT_EQ(nrLDPC_llr_prescale_get(&r, 10, 1), nullptr);
  ASSERT_EQ(nrLDPC_llr_prescale_parse("2,0.75,0.25,4", &r), 0);
  for (int q = 0; q <= NR_LDPC_LLR_PRESCALE_MAX_QM; q++)
    EXPECT_EQ(memcmp(&r.qm[0][q], &r.qm[1][q], sizeof(r.qm[0][q])), 0) << q;
  for (const char *bad : {"1,0.5,0.5,1/", "1,0.5,0.5,1/1,1,1", "1,0.5,0.5,1/1,1,1,1/1,1,1,1", "/1,1,1,1", "", "1,0.5,0.5", "1,0.5,0.5,1,1", "3,1,1,1", "1,0.01,1,1", "1,-0.5,1,1", "16,1,1,1", "x", "1,0.5,0.5,1x"}) {
    EXPECT_EQ(nrLDPC_llr_prescale_parse(bad, &r), -1) << bad;
    EXPECT_EQ(r.qm[0][4].mode, NR_LDPC_LLR_PRESCALE_NONE) << bad;
  }
}

TEST(nrLDPC_llr_prescale, off_is_legacy_pack)
{
  const auto in = all_int16();
  std::vector<int8_t> a, b;
  nr_ldpc_llr_prescale_t none{};
  run(in, a, nullptr);
  run(in, b, &none);
  for (int i = 0; i < NVEC * 16; i++) {
    ASSERT_EQ(a[i], sat8(in[i])) << i;
    ASSERT_EQ(b[i], a[i]) << i;
  }
}

TEST(nrLDPC_llr_prescale, exhaustive_vs_reference)
{
  const auto in = all_int16();
  for (const char *rule : {"default", "1,0.5,0.5,1", "0.0625,0.25,0.75,0.999", "2,4,8,0.3"}) {
    nr_ldpc_llr_prescale_rule_t r;
    ASSERT_EQ(nrLDPC_llr_prescale_parse(rule, &r), 0) << rule;
    for (int ly = 0; ly < 2; ly++)
      for (int q = 2; q <= 8; q += 2) {
        std::vector<int8_t> out;
        run(in, out, &r.qm[ly][q]);
        for (int i = 0; i < NVEC * 16; i++)
          ASSERT_EQ(out[i], ref(in[i], r.qm[ly][q])) << rule << " layers " << ly << " Qm " << q << " x " << in[i];
      }
  }
}

TEST(nrLDPC_llr_prescale, odd_and_short_lengths)
{
  // the segment decoder converts ((Kc * Z) >> 4) + 1 vectors: odd and even counts, down to 1
  const auto in = all_int16();
  nr_ldpc_llr_prescale_rule_t r;
  ASSERT_EQ(nrLDPC_llr_prescale_parse("2,0.25,0.5,0.75/0.25,0.0625,0.5,1", &r), 0);
  for (int nv : {1, 2, 3, 5, 1633, 4095}) {
    for (int ly = 0; ly < 2; ly++)
      for (int q = 2; q <= 8; q += 2) {
        alignas(16) static int16_t z[NVEC * 16];
        alignas(16) static int8_t l[NVEC * 16 + 32];
        memcpy(z, in.data() + 16 * (NVEC - nv), sizeof(int16_t) * 16 * nv); // the large-magnitude end of the range
        memset(l, 0x55, sizeof(l));
        nrLDPC_llr_pack_int8(z, l, nv, &r.qm[ly][q]);
        for (int i = 0; i < 16 * nv; i++)
          ASSERT_EQ(l[i], ref(z[i], r.qm[ly][q])) << "nvec " << nv << " Qm " << q << " i " << i;
        ASSERT_EQ(l[16 * nv], 0x55) << "write past nvec*16, nvec " << nv;
      }
  }
}

TEST(nrLDPC_llr_prescale, half_is_symmetric_rounding)
{
  nr_ldpc_llr_prescale_rule_t r;
  ASSERT_EQ(nrLDPC_llr_prescale_parse("1,0.5,0.5,1", &r), 0);
  const auto in = all_int16();
  std::vector<int8_t> out;
  run(in, out, &r.qm[0][4]);
  for (int i = 0; i < NVEC * 16; i++) {
    const int x = in[i];
    // round half away from zero of x/2, then saturate
    const long want = x >= 0 ? (x + 1) / 2 : -((-x + 1) / 2);
    ASSERT_EQ(out[i], sat8(x == -32768 ? -16384 : want)) << x;
    const int j = 65536 - i; // index of -x
    if (x > -32768 && j < 65536) {
      // symmetric up to the asymmetric int8 rail of the saturating pack (-128 vs +127)
      ASSERT_EQ(out[i] == -128 ? -127 : out[i], -(out[j] == -128 ? -127 : out[j])) << x;
    }
  }
}

TEST(nrLDPC_llr_prescale, filler_and_zero_preserved)
{
  // filler LLRs are set with memset(.., 127, ..) = int16 0x7f7f; punctured / untransmitted LLRs are 0
  for (const char *rule : {"0.0625,0.0625,0.0625,0.0625", "1,0.5,0.5,1", "8,8,8,8"}) {
    nr_ldpc_llr_prescale_rule_t r;
    ASSERT_EQ(nrLDPC_llr_prescale_parse(rule, &r), 0);
    for (int q = 2; q <= 8; q += 2) {
      alignas(16) int16_t z[32];
      alignas(16) int8_t l[32];
      for (int i = 0; i < 16; i++) {
        z[i] = 0x7f7f; // = memset(.., 127, ..) on int16
        z[16 + i] = 0;
      }
      nrLDPC_llr_pack_int8(z, l, 2, &r.qm[0][q]);
      for (int i = 0; i < 16; i++) {
        EXPECT_EQ(l[i], 127) << rule;
        EXPECT_EQ(l[16 + i], 0) << rule;
      }
    }
  }
}

} // namespace

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

// ---------------------------------------------------------------------------------------------- per-code-block AGC

namespace {
int8_t ref_gain(int16_t x, const nr_ldpc_llr_prescale_t &p)
{
  // y = sign(x) * round(min(|x|, 32767 >> k) * 2^k * q15 / 2^15) (round half away from zero), saturated to int8
  long a = x == -32768 ? 32767 : std::labs((long)x);
  const long c = 32767 >> p.lshift;
  if (a > c)
    a = c;
  const long y = ((a << p.lshift) * p.q15 + (1 << 14)) >> 15;
  return sat8(x < 0 ? -y : y);
}
} // namespace

TEST(nrLDPC_llr_agc, parse)
{
  nr_ldpc_llr_agc_rule_t r, d;
  ASSERT_EQ(nrLDPC_llr_agc_parse("default", &r), 0);
  ASSERT_EQ(nrLDPC_llr_agc_parse(NR_LDPC_LLR_AGC_DEFAULT, &d), 0);
  EXPECT_EQ(memcmp(&r, &d, sizeof(r)), 0);
  EXPECT_EQ(r.target[0][4], 31.0f);
  EXPECT_EQ(r.target[1][4], 40.0f);
  ASSERT_EQ(nrLDPC_llr_agc_parse("off", &r), 0);
  for (int l = 0; l < 2; l++)
    for (int q = 0; q <= NR_LDPC_LLR_PRESCALE_MAX_QM; q++)
      EXPECT_EQ(r.target[l][q], 0.0f);
  ASSERT_EQ(nrLDPC_llr_agc_parse("10,20,0,40", &r), 0);
  EXPECT_EQ(r.target[1][2], 10.0f);
  EXPECT_EQ(r.target[0][6], 0.0f);
  for (const char *bad : {"", "1,2,3", "1,2,3,4/", "1,2,3,4/5,6,7", "1,2,3,200", "-1,2,3,4", "x"})
    EXPECT_EQ(nrLDPC_llr_agc_parse(bad, &r), -1) << bad;
}

TEST(nrLDPC_llr_agc, gain_selection)
{
  nr_ldpc_llr_stats_t st = {.sabs = 1000, .nnz = 10, .n = 20}; // mean |LLR| of the non-zero values: 100
  nr_ldpc_llr_prescale_t p = nrLDPC_llr_agc_gain(25.0f, &st);
  EXPECT_EQ(p.mode, NR_LDPC_LLR_PRESCALE_FRAC);
  EXPECT_EQ(p.q15, 8192);
  EXPECT_EQ(p.q8, 64);
  st.sabs = 10; // mean 1 -> gain 25 = 2^5 * 0.78125
  p = nrLDPC_llr_agc_gain(25.0f, &st);
  EXPECT_EQ(p.mode, NR_LDPC_LLR_PRESCALE_GAIN);
  EXPECT_EQ(p.lshift, 5);
  EXPECT_EQ(p.q15, 25600);
  st.sabs = 1; // gain capped at 256
  p = nrLDPC_llr_agc_gain(100.0f, &st);
  EXPECT_EQ(p.lshift, 8);
  st.sabs = 0; // no non-zero LLR: x1
  EXPECT_EQ(nrLDPC_llr_agc_gain(25.0f, &st).mode, NR_LDPC_LLR_PRESCALE_NONE);
  st = {.sabs = 3100, .nnz = 100, .n = 100}; // mean = target: x1, legacy pack
  EXPECT_EQ(nrLDPC_llr_agc_gain(31.0f, &st).mode, NR_LDPC_LLR_PRESCALE_NONE);
}

TEST(nrLDPC_llr_agc, exhaustive_gain_pack_vs_reference)
{
  const auto in = all_int16();
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> e(-15.0f, 8.0f);
  for (int t = 0; t < 200; t++) {
    nr_ldpc_llr_stats_t st = {.sabs = 1000, .nnz = 1000, .n = 1000};
    const nr_ldpc_llr_prescale_t p = nrLDPC_llr_agc_gain(std::pow(2.0f, e(rng)), &st);
    if (p.mode == NR_LDPC_LLR_PRESCALE_NONE)
      continue;
    std::vector<int8_t> out;
    run(in, out, &p);
    for (int i = 0; i < NVEC * 16; i++)
      ASSERT_EQ(out[i], ref_gain(in[i], p)) << "k " << (int)p.lshift << " q15 " << p.q15 << " x " << in[i];
  }
}

TEST(nrLDPC_llr_agc, statistics_vs_scalar)
{
  std::mt19937 rng(11);
  std::vector<int16_t> z(68 * 384 + 32);
  for (auto &v : z)
    v = (rng() % 5 == 0) ? 0 : (int16_t)((int)(rng() % 65536) - 32768);
  for (int b : {0, 1, 7, 16, 33})
    for (int e : {b, b + 1, b + 15, b + 16, b + 17, 1000, 68 * 384}) {
      nr_ldpc_llr_stats_t st = {0};
      nrLDPC_llr_stats(z.data(), b, e, 1, &st);
      long sa = 0, nz = 0;
      for (int i = b; i < e; i++) {
        sa += std::labs((long)z[i]);
        nz += z[i] != 0;
      }
      // the int32 sum is exact as long as it fits (true for a code block of int16 LLRs below |x| ~ 2^15 / 1.0)
      if (sa < (1L << 31)) {
        EXPECT_EQ(st.sabs, sa) << b << " " << e;
      }
      EXPECT_EQ(st.nnz, nz) << b << " " << e;
      EXPECT_EQ(st.n, e - b) << b << " " << e;
    }
  // sub-sampling: one 16-value vector out of step
  nr_ldpc_llr_stats_t st = {0};
  nrLDPC_llr_stats(z.data(), 0, 64 * 16, 4, &st);
  EXPECT_EQ(st.n, 16 * 16);
}
