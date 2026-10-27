/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* Portable kernels of the packed LDPC transmit chain: reference variant, and fallbacks of the SIMD variants. */
#include <string.h>
#include "nr_ldpc_tx_internal.h"

void nr_ldpc_tx_rows_scalar(const uint8_t *base,
                            const uint16_t *off,
                            const uint8_t *cnt,
                            int nrows,
                            int z,
                            uint8_t *dst,
                            int dstride,
                            int dup)
{
  for (int r = 0; r < nrows; r++, off += cnt[r - 1], dst += dstride) {
    const int ne = cnt[r];
    if (z >= 8) {
      /* 64-bit words, last word overlapping the previous one when z % 8 != 0 */
      for (int i = 0; i < z; i += 8) {
        const int o = i + 8 <= z ? i : z - 8;
        uint64_t acc = 0;
        for (int e = 0; e < ne; e++)
          acc ^= nr_ldpc_tx_ld64(base + off[e] + o);
        nr_ldpc_tx_st64(dst + o, acc);
        if (dup)
          nr_ldpc_tx_st64(dst + z + o, acc);
      }
    } else {
      for (int i = 0; i < z; i++) {
        uint8_t acc = 0;
        for (int e = 0; e < ne; e++)
          acc ^= base[off[e] + i];
        dst[i] = acc;
        if (dup)
          dst[z + i] = acc;
      }
    }
  }
}

void nr_ldpc_tx_transpose_scalar(const uint8_t *const in[8], int nblk, uint8_t *out)
{
  /* 8-byte blocks */
  for (int b = 0; b < nblk * 8; b++) {
    uint64_t w = 0; /* byte j = in[j][b] */
    for (int j = 0; j < 8; j++)
      w |= (uint64_t)in[j][b] << (8 * j);
    for (int u = 0; u < 8; u++) {
      /* out byte bit j = bit (7-u) of in[j][b] */
      uint64_t t = (w >> (7 - u)) & 0x0101010101010101ULL;
      out[8 * b + u] = (uint8_t)((t * 0x0102040810204080ULL) >> 56);
    }
  }
}

/* f[qm t + q] = src[q][t] (TS 38.212 5.4.2.2) */
void nr_ldpc_tx_interleave_scalar(int qm, const uint8_t *const src[8], int L, uint8_t *f)
{
  for (int q = 0; q < qm; q++) {
    const uint8_t *s = src[q];
    uint8_t *o = f + q;
    for (int t = 0; t < L; t++)
      o[qm * t] = s[t];
  }
}

/* 8x8 bit matrix transpose: bit 8u + j <-> bit 8j + u (byte u bit j -> byte j bit u) */
static inline uint64_t tr8x8(uint64_t x)
{
  uint64_t t;
  t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAULL;
  x ^= t ^ (t << 7);
  t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCULL;
  x ^= t ^ (t << 14);
  t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0ULL;
  x ^= t ^ (t << 28);
  return x;
}

void nr_ldpc_tx_extract_scalar(int qm,
                               const uint8_t *const src[8],
                               const uint8_t *const src2[8],
                               uint8_t m2,
                               int L,
                               uint8_t *stage,
                               uint64_t *w,
                               unsigned planes)
{
  const int n = qm * L, nw = (n + 63) >> 6;
  const uint8_t *f = src[0];
  if (qm > 1) {
    nr_ldpc_tx_interleave_scalar(qm, src, L, stage);
    f = stage;
  }
  for (int k = 0; k < nw; k++) {
    uint64_t acc[8] = {0};
    for (int b = 0; b < 8; b++) {
      /* positions 64k + 8b .. +7: byte j of the transpose = code block j */
      const uint64_t x = tr8x8(nr_ldpc_tx_ld64(f + 64 * k + 8 * b));
      for (int j = 0; j < 8; j++)
        acc[j] |= ((x >> (8 * j)) & 0xff) << (8 * b);
    }
    for (int j = 0; j < 8; j++)
      w[j * NR_LDPC_TX_WS + k] = acc[j];
  }
}

void nr_ldpc_tx_write_scalar(uint64_t *dst, const uint64_t *w, int n, int sh)
{
  if (sh == 0) {
    memcpy(dst, w, 8 * (size_t)n);
    return;
  }
  for (int k = 0; k < n; k++)
    nr_ldpc_tx_st64(dst + k, (w[k] << sh) | (w[k - 1] >> (64 - sh)));
}

static const nr_ldpc_tx_kern_t k_scalar = {.name = "scalar",
                                           .rows = nr_ldpc_tx_rows_scalar,
                                           .transpose = nr_ldpc_tx_transpose_scalar,
                                           .tb = 8,
                                           .extract = nr_ldpc_tx_extract_scalar,
                                           .write = nr_ldpc_tx_write_scalar};
const nr_ldpc_tx_kern_t *nr_ldpc_tx_kern_scalar(void)
{
  return &k_scalar;
}
