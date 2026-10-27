/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

// Modulation / layer mapping from packed bits (nr_mod_packed.c, nr_modulation.c), the fused PDSCH scrambling, the
// folding CRCs (crc_fold.c via crc_byte.c) and the 64-bit Gold generator (refsig.c), against bit-by-bit references
// built from OAI's own tables and gold_generic().

#include "gtest/gtest.h"
#include <cstdint>
#include <cstring>
#include <vector>
#include <simde/x86/avx512.h>

extern "C" {
#include "common/utils/LOG/log.h"
#include "openair1/PHY/TOOLS/tools_defs.h"
#include "openair1/PHY/NR_REFSIG/nr_mod_table.h"
#include "openair1/PHY/MODULATION/nr_mod_packed.h"
#include "openair1/PHY/CODING/crc_fold.h"
#include "openair1/PHY/gold.h"
void nr_generate_modulation_table(void);
void nr_modulation(const uint32_t *in, uint32_t length, uint16_t mod_order, int16_t *out);
// VLA parameter hidden from C++: c16_t tx_layers[][layerSz]
bool nr_modulation_layer_mapping(const uint32_t *in, uint32_t length, uint16_t mod_order, uint8_t n_layers, int layerSz,
                                 c16_t (*tx_layers)[]);
void nr_codeword_scrambling(uint8_t *in, uint32_t size, uint8_t q, uint32_t Nid, uint32_t n_RNTI, uint32_t *out);
const uint32_t *nr_codeword_scrambling_seq(uint32_t size, uint8_t q, uint32_t Nid, uint32_t n_RNTI);
uint32_t *gold_cache(uint32_t key, int length);
void crcTableInit(void);
uint32_t crc24a(unsigned char *inptr, int bitlen);
uint32_t crc24b(unsigned char *inptr, int bitlen);
uint32_t crc16(unsigned char *inptr, int bitlen);
uint32_t crcbit(unsigned char *inputptr, int octetlen, uint32_t poly);
}

namespace {

uint64_t rng = 0x2545F4914F6CDD1DULL;
uint64_t rnd()
{
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return rng;
}

c16_t ref_sym(int qm, unsigned v)
{
  switch (qm) {
    case 2:
      return nr_qpsk_mod_table[v];
    case 4:
      return ((const c16_t *)nr_16qam_mod_table)[v];
    case 6:
      return ((const c16_t *)nr_64qam_mod_table)[2 * v];
    default:
      return ((const c16_t *)nr_256qam_mod_table)[v];
  }
}

// symbol k of the (optionally scrambled) bits to layer k % L, position k / L
void ref_mod(const uint8_t *in, const uint32_t *scr, uint32_t nsym, int qm, int L, c16_t *out, int layerSz)
{
  for (uint32_t k = 0; k < nsym; k++) {
    unsigned v = 0;
    for (int j = 0; j < qm; j++) {
      const uint32_t b = k * qm + j;
      unsigned bit = (in[b / 8] >> (b % 8)) & 1;
      if (scr)
        bit ^= (scr[b / 32] >> (b % 32)) & 1;
      v |= bit << j;
    }
    out[(k % L) * layerSz + k / L] = ref_sym(qm, v);
  }
}

// 64-byte aligned symbol buffer (OAI's layer arrays are aligned; nr_layer_mapping() stores aligned zmm)
struct Syms {
  c16_t *p;
  explicit Syms(size_t n) : p((c16_t *)aligned_alloc(64, ((n * sizeof(c16_t)) + 63) & ~(size_t)63)) {}
  ~Syms() { free(p); }
  c16_t &operator[](size_t i) { return p[i]; }
  c16_t *data() { return p; }
};

bool same(const c16_t *a, const c16_t *b, size_t n)
{
  return !memcmp(a, b, n * sizeof(c16_t));
}

class ModPacked : public ::testing::Test {
 protected:
  static void SetUpTestSuite()
  {
    nr_generate_modulation_table();
    crcTableInit();
  }
};

TEST_F(ModPacked, ModulationAndLayerMapping)
{
  const uint32_t canary = 0x5a5a5a5a;
  for (int qm = 2; qm <= 8; qm += 2)
    for (int L = 1; L <= 4; L++)
      for (int t = 0; t < 60; t++) {
        // every residue of the group sizes, then sizes up to a 273 PRB layer
        const uint32_t nper = t < 50 ? 1 + t * 3 : 1 + rnd() % 40000;
        const uint32_t nsym = nper * L, length = nsym * qm;
        // the scalar 64QAM path leaves the last symbol unwritten for nsym % 4 == 3 (one layer; PDSCH/PUSCH sizes are
        // even): only checked where nr_mod_packed() runs
        const bool scalar_gap = qm == 6 && nsym % 4 == 3 && !nr_mod_packed_available(qm, 1);
        if (scalar_gap && L == 1)
          continue;
        std::vector<uint8_t> in((length + 7) / 8 + 64);
        for (auto &b : in)
          b = rnd();
        const int layerSz = (nper + 64 + 63) & ~63;
        Syms ref(L * layerSz), out(L * layerSz);
        ref_mod(in.data(), nullptr, nsym, qm, L, ref.data(), layerSz);
        std::fill((uint32_t *)out.data(), (uint32_t *)(out.data() + L * layerSz), canary);
        ASSERT_TRUE(nr_modulation_layer_mapping((const uint32_t *)in.data(), length, qm, L, layerSz, (c16_t (*)[])out.data()));
        for (int l = 0; l < L; l++) {
          ASSERT_TRUE(same(&ref[l * layerSz], &out[l * layerSz], nper)) << "qm " << qm << " L " << L << " nper " << nper;
          if (nr_mod_packed_available(qm, L))
            ASSERT_EQ(((uint32_t *)out.data())[l * layerSz + nper], canary);
        }
        // nr_modulation: all symbols on one layer
        if (scalar_gap)
          continue;
        Syms ref1(nsym), out1(nsym + 64);
        ref_mod(in.data(), nullptr, nsym, qm, 1, ref1.data(), 0);
        nr_modulation((const uint32_t *)in.data(), length, qm, (int16_t *)out1.data());
        ASSERT_TRUE(same(ref1.data(), out1.data(), nsym)) << "nr_modulation qm " << qm << " nsym " << nsym;
      }
}

TEST_F(ModPacked, FusedScrambling)
{
  for (int qm = 2; qm <= 8; qm += 2)
    for (int L = 1; L <= 4; L++) {
      if (!nr_mod_packed_available(qm, L))
        GTEST_SKIP() << "no modulation kernel on this CPU";
      for (int t = 0; t < 20; t++) {
        const uint32_t nper = 1 + rnd() % 20000, nsym = nper * L, length = nsym * qm;
        const uint32_t Nid = rnd() % 1024, rnti = rnd() % 65536;
        std::vector<uint8_t> in((length + 7) / 8 + 64);
        for (auto &b : in)
          b = rnd();
        // OAI's scrambler, then the reference modulation of the scrambled bits
        std::vector<uint32_t> scrambled((length >> 5) + 4);
        nr_codeword_scrambling(in.data(), length, 0, Nid, rnti, scrambled.data());
        const int layerSz = (nper + 63) & ~63;
        Syms ref(L * layerSz), out(L * layerSz);
        ref_mod((const uint8_t *)scrambled.data(), nullptr, nsym, qm, L, ref.data(), layerSz);
        c16_t *lp[4];
        for (int l = 0; l < L; l++)
          lp[l] = &out[l * layerSz];
        const uint32_t *seq = nr_codeword_scrambling_seq(length, 0, Nid, rnti);
        ASSERT_TRUE(nr_mod_packed((const uint32_t *)in.data(), seq, length, qm, L, lp));
        for (int l = 0; l < L; l++)
          ASSERT_TRUE(same(&ref[l * layerSz], lp[l], nper)) << "qm " << qm << " L " << L << " nper " << nper;
      }
    }
}

TEST_F(ModPacked, FoldingCrc)
{
  const uint32_t polys[3] = {0x864cfb00, 0x80006300, 0x10210000};
  uint32_t (*const f[3])(unsigned char *, int) = {crc24a, crc24b, crc16};
  uint32_t tab[3][256];
  for (int w = 0; w < 3; w++)
    for (int c = 0; c < 256; c++) {
      unsigned char b = c;
      tab[w][c] = crcbit(&b, 1, polys[w]);
    }
  std::vector<uint8_t> buf(70000);
  const char *variants[] = {"pclmul", "vpclmul", "pmull", "pmull-eor3", nullptr};
  for (const char *v : variants) {
    if (!crc_fold_set_variant(v))
      continue;
    for (int t = 0; t < 600; t++) {
      const int bytes = t < 500 ? 1 + t : 1 + rnd() % (buf.size() - 32);
      const int bitlen = 8 * bytes - (t % 3 ? 0 : rnd() % 8), off = rnd() % 16;
      for (int i = 0; i < bytes + 1; i++)
        buf[off + i] = rnd();
      for (int w = 0; w < 3; w++) {
        uint32_t crc = 0;
        const uint8_t *p = &buf[off];
        for (int i = 0; i < bitlen / 8; i++)
          crc = (crc << 8) ^ tab[w][p[i] ^ (crc >> 24)];
        for (int b = 0; b < bitlen % 8; b++)
          crc = ((crc >> 31) ^ ((p[bitlen / 8] >> (7 - b)) & 1)) ? (crc << 1) ^ polys[w] : crc << 1;
        ASSERT_EQ(f[w](&buf[off], bitlen), crc) << "kernel " << crc_fold_variant() << " crc " << w << " bits " << bitlen;
      }
    }
  }
  crc_fold_set_variant(nullptr);
}

TEST_F(ModPacked, GoldSequence)
{
  for (int t = 0; t < 200; t++) {
    const uint32_t key = rnd();
    const int len = 1 + rnd() % (t % 10 ? 300 : 20000);
    std::vector<uint32_t> ref(len);
    unsigned int x1 = 0, x2 = key;
    ref[0] = gold_generic(&x1, &x2, 1);
    for (int n = 1; n < len; n++)
      ref[n] = gold_generic(&x1, &x2, 0);
    ASSERT_EQ(memcmp(ref.data(), gold_cache(key, len), len * 4), 0) << "key " << key << " len " << len;
  }
}

} // namespace

int main(int argc, char **argv)
{
  logInit();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
