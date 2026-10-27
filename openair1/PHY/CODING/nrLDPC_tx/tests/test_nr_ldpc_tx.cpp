/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

// Packed LDPC transmit chain against a reference built from OAI's existing functions: the original encoder
// (ldpc_encoder.c, one code block, one byte per bit), nr_rate_matching_ldpc() and nr_interleaving_ldpc(), packed
// LSB first at the code block offsets. Every kernel set available on the CPU is checked, with groups of 1..8 code
// blocks, two E values per group, LBRM, all redundancy versions, Qm 1..8, filler bits and misaligned outputs.

#include "gtest/gtest.h"
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

extern "C" {
#include "openair1/PHY/CODING/nrLDPC_defs.h"
#include "openair1/PHY/CODING/nrLDPC_coding/nrLDPC_coding_segment/nr_rate_matching.h"
#include "openair1/PHY/CODING/nrLDPC_tx/nr_ldpc_tx.h"
#include "common/utils/LOG/log.h"
int LDPCencoder(unsigned char **inputArray, unsigned char *outputArray, encoder_implemparams_t *impp);
}

namespace {

const uint16_t Zs[] = {2, 3, 5, 7, 8, 13, 15, 26, 36, 52, 64, 104, 112, 176, 208, 240, 288, 320, 352, 384};
const char *variants[] = {"scalar", "neon", "neon-eor3", "sve2", "avx2", "avx512"};

struct tx_case {
  nr_ldpc_tx_params_t p;
  uint32_t Tbslbrm, C;
  uint8_t rv;
  int n;
  uint32_t E[8], Eoff[8];
  int mis;
};

// d_0 .. d_{N-1} of one code block, one byte per bit, from the original encoder (fillers are 0 in the input)
std::vector<uint8_t> reference_codeword(const tx_case &t, const uint8_t *c)
{
  const nr_ldpc_tx_params_t &p = t.p;
  std::vector<uint8_t> in(8448 / 8 + 64), d(68 * 384);
  memcpy(in.data(), c, in.size());
  unsigned char *ip[1] = {in.data()};
  encoder_implemparams_t impp = {};
  impp.n_segments = 1;
  impp.K = p.K;
  impp.Kb = p.BG == 1 ? 22 : 10;
  impp.Zc = p.Z;
  impp.BG = p.BG;
  LDPCencoder(ip, d.data(), &impp);
  d.resize((p.BG == 1 ? 66 : 50) * p.Z);
  return d;
}

void reference_output(const tx_case &t, const std::vector<std::vector<uint8_t>> &c, uint8_t *out)
{
  const nr_ldpc_tx_params_t &p = t.p;
  for (int j = 0; j < t.n; j++) {
    std::vector<uint8_t> d = reference_codeword(t, c[j].data());
    std::vector<uint8_t> e(t.E[j] + 64), f(t.E[j] + 64);
    ASSERT_EQ(nr_rate_matching_ldpc(t.Tbslbrm, p.BG, p.Z, d.data(), e.data(), t.C, p.F, p.K - p.F - 2 * p.Z, t.rv, t.E[j]), 0);
    if (p.Qm > 1)
      nr_interleaving_ldpc(t.E[j], p.Qm, e.data(), f.data());
    else
      f = e;
    for (uint32_t i = 0; i < t.E[j]; i++)
      if (f[i])
        out[(t.Eoff[j] + i) >> 3] |= 1 << ((t.Eoff[j] + i) & 7);
  }
}

} // namespace

TEST(nr_ldpc_tx, against_reference)
{
  std::mt19937 rng(7);
  const uint8_t qms[] = {1, 2, 4, 6, 8};
  int n_run = 0;
  while (n_run < 600) {
    tx_case t = {};
    nr_ldpc_tx_params_t &p = t.p;
    p.BG = 1 + rng() % 2;
    p.Z = Zs[rng() % (sizeof(Zs) / sizeof(Zs[0]))];
    p.Qm = qms[rng() % 5];
    p.K = (p.BG == 1 ? 22 : 10) * p.Z;
    const uint32_t N = (p.BG == 1 ? 66 : 50) * p.Z;
    const uint32_t Kp = 2 * p.Z + 1 + rng() % (p.K - 2 * p.Z);
    p.F = p.K - Kp;
    t.rv = rng() % 4;
    t.C = 1 + rng() % 20;
    t.Tbslbrm = (rng() % 2) ? 0 : (2 * t.C * (p.K - 2 * p.Z + 1 + rng() % (N - p.K + 2 * p.Z)) + 2) / 3;
    const nr_ldpc_geometry_t geo = nr_ldpc_soft_buffer_geometry(t.Tbslbrm, p.BG, p.Z, t.C, t.rv);
    p.Ncb = geo.Ncb;
    p.k0 = geo.k0;
    if (Kp - 2 * p.Z + p.F > p.Ncb)
      continue;
    t.n = 1 + rng() % 8;
    // E as nr_get_E() gives it: two values Nl Qm apart, the larger ones last
    const uint32_t nl = 1 + rng() % 4, e1 = p.Qm * nl * (1 + rng() % (3 * N / (p.Qm * nl) + 1));
    const int first2 = rng() % (t.n + 1);
    uint32_t off = rng() % 200;
    for (int j = 0; j < t.n; j++) {
      t.E[j] = e1 + (j >= first2 ? p.Qm * nl : 0);
      t.Eoff[j] = off;
      off += t.E[j];
    }
    t.mis = rng() % 8;
    std::vector<std::vector<uint8_t>> c(t.n, std::vector<uint8_t>(8448 / 8 + 64));
    for (auto &v : c) {
      for (auto &b : v)
        b = rng() & 0xff;
      for (uint32_t k = Kp; k < v.size() * 8; k++)
        v[k >> 3] &= ~(0x80 >> (k & 7));
    }
    const size_t nbytes = off / 8 + 64;
    std::vector<uint8_t> ref(nbytes + 8, 0), out(nbytes + 8);
    reference_output(t, c, ref.data());
    if (::testing::Test::HasFatalFailure())
      return;
    const uint8_t *cp[8];
    for (int j = 0; j < t.n; j++)
      cp[j] = c[j].data();
    for (const char *v : variants) {
      if (!nr_ldpc_tx_variant_available(v))
        continue;
      ASSERT_EQ(nr_ldpc_tx_set_variant(v), 0);
      std::fill(out.begin(), out.end(), 0);
      ASSERT_EQ(nr_ldpc_tx_encode(&p, t.n, cp, t.E, t.Eoff, out.data() + t.mis), 0);
      ASSERT_EQ(memcmp(ref.data(), out.data() + t.mis, nbytes), 0)
          << v << ": BG " << int(p.BG) << " Z " << p.Z << " K' " << Kp << " Qm " << int(p.Qm) << " rv " << int(t.rv) << " Ncb "
          << p.Ncb << " n " << t.n << " E " << t.E[0] << "/" << t.E[t.n - 1];
      for (int k = 0; k < t.mis; k++)
        ASSERT_EQ(out[k], 0) << "write before the output";
    }
    nr_ldpc_tx_set_variant(NULL);
    n_run++;
  }
}

TEST(nr_ldpc_tx, sliced_encoder_against_reference)
{
  std::mt19937 rng(11);
  for (int bg = 1; bg <= 2; bg++)
    for (uint16_t z : Zs) {
      tx_case t = {};
      t.p.BG = bg;
      t.p.Z = z;
      t.p.K = (bg == 1 ? 22 : 10) * z;
      const int N = (bg == 1 ? 66 : 50) * z, Kp = t.p.K - rng() % (t.p.K - 2 * z);
      std::vector<std::vector<uint8_t>> c(8, std::vector<uint8_t>(8448 / 8 + 64));
      const uint8_t *cp[8];
      for (int j = 0; j < 8; j++) {
        for (auto &b : c[j])
          b = rng() & 0xff;
        for (uint32_t k = Kp; k < c[j].size() * 8; k++)
          c[j][k >> 3] &= ~(0x80 >> (k & 7));
        cp[j] = c[j].data();
      }
      std::vector<uint8_t> out(68 * 384 + 512);
      for (const char *v : variants) {
        if (!nr_ldpc_tx_variant_available(v))
          continue;
        ASSERT_EQ(nr_ldpc_tx_set_variant(v), 0);
        ASSERT_EQ(nr_ldpc_tx_encode_sliced(bg, z, Kp, 8, cp, out.data(), N), N);
        for (int j = 0; j < 8; j++) {
          const std::vector<uint8_t> d = reference_codeword(t, cp[j]);
          for (int k = 0; k < N; k++)
            ASSERT_EQ((out[k] >> j) & 1, d[k])
                << v << ": BG " << bg << " Z " << z << " K' " << Kp << " block " << j << " bit " << k;
        }
      }
      nr_ldpc_tx_set_variant(NULL);
    }
}

int main(int argc, char **argv)
{
  logInit();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
