/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* Packed LDPC transmit chain: LDPC encoding (TS 38.212 5.3.2), rate matching (5.4.2.1) and bit interleaving
 * (5.4.2.2) of up to 8 code blocks per call, from the packed code block bits produced by nr_segmentation() straight
 * to the packed transport-block output, in one pass and inside the calling thread.
 *
 * The code blocks of a group are encoded together in bit-sliced bytes (bit j of a byte = code block j, every
 * circulant shift is an unaligned byte load); only the part of the codeword that the circular-buffer window reads
 * is computed. The window is read stream by stream and each 64-bit run of output bits of every code block is
 * extracted and written at its bit offset in the transport block. Kernels are selected at run time (scalar, AVX2,
 * AVX-512, NEON, NEON+SHA3, SVE2). */
#ifndef NR_LDPC_TX_H
#define NR_LDPC_TX_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Parameters shared by the code blocks of a call (all code blocks of a transport block share them) */
typedef struct {
  uint8_t BG; /* base graph 1 or 2 */
  uint8_t Qm; /* modulation order 1, 2, 4, 6 or 8 */
  uint16_t Z; /* lifting size */
  uint32_t K; /* 22 Z (BG1) or 10 Z (BG2) */
  uint32_t F; /* filler bits, the last F of the K systematic bits */
  uint32_t Ncb; /* circular buffer length (5.4.2.1, LBRM applied), <= N = 66 Z / 50 Z */
  uint32_t k0; /* start of the redundancy version in the circular buffer */
} nr_ldpc_tx_params_t;

/* Builds the per-(BG, Z) plans and selects the kernels (otherwise done on first use, thread safe). */
int nr_ldpc_tx_init(void);

/*
 * Encodes, rate matches and interleaves n (1..8) code blocks.
 *   c[j]    : the K - F information bits of code block j, packed MSB first (bit k = c[j][k >> 3] bit 7 - (k & 7));
 *             ceil((K - F) / 8) bytes are read, filler bits are not read (encoded as 0).
 *   E[j]    : number of rate-matched bits of code block j (a multiple of Qm).
 *   Eoff[j] : bit offset of code block j in out.
 *   out     : transport-block output, packed LSB first (bit b = out[b >> 3] bit b & 7), zeroed by the caller.
 *             Bits [Eoff[j], Eoff[j] + E[j]) are set; 64-bit words (relative to out aligned down to 8 bytes)
 *             entirely inside that range are stored, the two partial edge words are atomically ORed, so calls for
 *             disjoint ranges of the same buffer may run concurrently.
 * Returns 0, or < 0 for invalid parameters (nothing is written).
 */
int nr_ldpc_tx_encode(const nr_ldpc_tx_params_t *p,
                      int n,
                      const uint8_t *const *c,
                      const uint32_t *E,
                      const uint32_t *Eoff,
                      uint8_t *out);

/*
 * LDPC encoder alone, OAI LDPCencoder() layout: up to 8 code blocks sharing BG, Z and K' (= K - F).
 *   out[k] bit j = d_k of code block j (codeword without the 2Z punctured bits), bits n..7 zero, k < nout
 *   (nout <= N; parity rows that nout does not reach are not computed). Filler positions are 0.
 * Returns nout, or < 0 for invalid parameters.
 */
int nr_ldpc_tx_encode_sliced(int bg, int z, int kprime, int n, const uint8_t *const *c, uint8_t *out, int nout);

/* Kernel set selection, for tests and benchmarks (not thread safe): "scalar", "neon", "neon-eor3", "sve2", "avx2",
 * "avx512" or NULL/"auto". The environment variable NR_LDPC_TX_VARIANT does the same at initialisation.
 * Returns 0, or -1 if the variant is not available on this CPU. */
int nr_ldpc_tx_set_variant(const char *name);
const char *nr_ldpc_tx_variant(void);
/* 1 if the named variant is compiled in and supported by this CPU */
int nr_ldpc_tx_variant_available(const char *name);

#ifdef __cplusplus
}
#endif
#endif
