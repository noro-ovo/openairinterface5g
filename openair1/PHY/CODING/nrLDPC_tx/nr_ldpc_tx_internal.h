/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* Internal interfaces of the packed LDPC transmit chain: shared by the driver (nr_ldpc_tx.c) and the per-ISA
 * kernels. */
#ifndef NR_LDPC_TX_INTERNAL_H
#define NR_LDPC_TX_INTERNAL_H
#include <stdint.h>
#include <stddef.h>

#define NR_LDPC_TX_ZMAX 384
#define NR_LDPC_TX_MAXCOLS 26 /* kb + 4 core parity columns (BG1) */
#define NR_LDPC_TX_PAD 512 /* slack after the encoder work area: chunked loads may read past the last region */

/* Output stage chunk: CH positions of the interleaved sequence f per pass step. CH is a multiple of 512 (whole
 * AVX-512 extraction blocks) and of every Qm (1, 2, 4, 6, 8), so a chunk maps to CH/Qm bytes of each stream; it is
 * large enough to amortise the per-chunk work (stream spans, edge words) and small enough for L1 (about 24 KB of
 * buffers per pass). */
#define NR_LDPC_TX_CH 6144
#define NR_LDPC_TX_CHW (NR_LDPC_TX_CH / 64)
/* stride of the per-code-block word arrays: [0] = last word of the previous chunk, [1..CHW] = chunk words, slack */
#define NR_LDPC_TX_WS (NR_LDPC_TX_CHW + 8)
/* a stream pointer given to an extraction kernel may be read up to this many bytes past its L bytes */
#define NR_LDPC_TX_OVERREAD 512
/* scratch given to the extraction kernels (interleaved chunk + padding) */
#define NR_LDPC_TX_STAGE (NR_LDPC_TX_CH + 1024)

/*
 * Encoder row kernel: for r in [0, nrows):  dst[r*dstride + i] = XOR_{e in row r} base[off[e] + i],  i in [0, z)
 * where the edges of row r are the next cnt[r] entries of off[] (an edge offset is column_start + P_ij in the
 * work area, every column being stored twice: [s | s]).
 * If dup, the result is also written at dst[r*dstride + z + i] (doubled copy for later circulant loads).
 * Kernels write exactly z (resp. 2z) bytes per row and may read up to NR_LDPC_TX_PAD bytes past base[off+z].
 */
typedef void (*nr_ldpc_tx_rows_fn)(const uint8_t *base,
                                   const uint16_t *off,
                                   const uint8_t *cnt,
                                   int nrows,
                                   int z,
                                   uint8_t *dst,
                                   int dstride,
                                   int dup);

/* Bit transposer: in[j] (j = 0..7) are packed MSB-first streams; converts nblk blocks of tb bytes per stream
 * into nblk*tb*8 bit-sliced bytes:  out[8*b + u] bit j = in[j][b] bit (7-u). */
typedef void (*nr_ldpc_tx_transpose_fn)(const uint8_t *const in[8], int nblk, uint8_t *out);

/* Column transposer (optional, z % 8 == 0 and z >= tcols_zmin): for c in [0, ncols), transposes the z/8 bytes
 * in[j][c*z/8 ..] of each stream into the z bit-sliced bytes of systematic column c and writes them directly to
 * work + c*S, work + c*S + z (doubled copy) and, for 2 <= c < cw, to out + (c-2)*z (codeword output). */
typedef void (*nr_ldpc_tx_tcols_fn)(const uint8_t *const in[8], int z, int ncols, uint8_t *work, int S, uint8_t *out, int cw);

/*
 * Output stage, one chunk of one rate-matching pass (TS 38.212 5.4.2.1 bit selection, 5.4.2.2 bit interleaving).
 *   src[q], q < qm: L bytes of the bit-sliced rate-matched stream q, i.e. e_{q E/Qm + t0} .. e_{q E/Qm + t0 + L-1}
 *   (readable up to NR_LDPC_TX_OVERREAD bytes further). The chunk is f_{i0} .. f_{i0 + qm L - 1} with
 *   f_{i0 + qm t + q} = src[q][t], qm L <= NR_LDPC_TX_CH.
 *   Writes w[j * NR_LDPC_TX_WS + k] for every code block j = 0..7 and k = 0 .. ceil(qm L / 64) - 1: bits 64k .. 64k+63
 *   of the chunk of code block j (bit j of the bytes), packed LSB first. Bits past qm L are unspecified, and the
 *   kernel may write up to 7 more words per code block. stage: NR_LDPC_TX_STAGE bytes of scratch, 64-byte aligned.
 *   Only the code blocks of the bit mask planes are needed (kernels may skip the others: small groups).
 */
typedef void (*nr_ldpc_tx_extract_fn)(int qm, const uint8_t *const src[8], int L, uint8_t *stage, uint64_t *w, unsigned planes);

/* Packed writer: dst[k] = (w[k] << sh) | (w[k-1] >> (64 - sh)) for k = 0 .. n-1 (w[-1] is read; sh in 0..63,
 * w[k-1] >> 64 meaning 0). dst need not be aligned. */
typedef void (*nr_ldpc_tx_write_fn)(uint64_t *dst, const uint64_t *w, int n, int sh);

typedef struct {
  const char *name;
  /* encoder, bit-sliced bytes (8 code blocks per byte) */
  nr_ldpc_tx_rows_fn rows;
  nr_ldpc_tx_transpose_fn transpose;
  int tb; /* transpose block, bytes per stream (<= 256) */
  nr_ldpc_tx_tcols_fn tcols;
  int tcols_zmin;
  int (*tcols_use)(int z); /* optional finer choice (z % 8 == 0 and z >= tcols_zmin already hold) */
  /* output stage */
  nr_ldpc_tx_extract_fn extract;
  nr_ldpc_tx_write_fn write;
} nr_ldpc_tx_kern_t;

/* portable kernels (reference variant, and fallbacks of the SIMD variants) */
void nr_ldpc_tx_rows_scalar(const uint8_t *base,
                            const uint16_t *off,
                            const uint8_t *cnt,
                            int nrows,
                            int z,
                            uint8_t *dst,
                            int dstride,
                            int dup);
void nr_ldpc_tx_transpose_scalar(const uint8_t *const in[8], int nblk, uint8_t *out);
void nr_ldpc_tx_interleave_scalar(int qm, const uint8_t *const src[8], int L, uint8_t *f);
void nr_ldpc_tx_extract_scalar(int qm, const uint8_t *const src[8], int L, uint8_t *stage, uint64_t *w, unsigned planes);
void nr_ldpc_tx_write_scalar(uint64_t *dst, const uint64_t *w, int n, int sh);
#if defined(__x86_64__)
/* AVX2 output-stage kernels, also used by the AVX-512 variant (interleave_sse: positions t .. L-1 only) */
void nr_ldpc_tx_interleave_sse(int qm, const uint8_t *const src[8], int t, int L, uint8_t *f);
void nr_ldpc_tx_extract_avx2(int qm, const uint8_t *const src[8], int L, uint8_t *stage, uint64_t *w, unsigned planes);
void nr_ldpc_tx_write_avx2(uint64_t *dst, const uint64_t *w, int n, int sh);
#endif
#if defined(__aarch64__)
/* NEON output-stage kernels, also used by the NEON+SHA3 and SVE2 variants */
void nr_ldpc_tx_extract_neon(int qm, const uint8_t *const src[8], int L, uint8_t *stage, uint64_t *w, unsigned planes);
void nr_ldpc_tx_write_neon(uint64_t *dst, const uint64_t *w, int n, int sh);
#endif

/* per-ISA kernel sets: NULL if not compiled in or not supported by the running CPU */
const nr_ldpc_tx_kern_t *nr_ldpc_tx_kern_scalar(void);
const nr_ldpc_tx_kern_t *nr_ldpc_tx_kern_neon(void);
const nr_ldpc_tx_kern_t *nr_ldpc_tx_kern_neon3(void);
const nr_ldpc_tx_kern_t *nr_ldpc_tx_kern_sve2(void);
const nr_ldpc_tx_kern_t *nr_ldpc_tx_kern_avx2(void);
const nr_ldpc_tx_kern_t *nr_ldpc_tx_kern_avx512(void);

static inline uint64_t nr_ldpc_tx_ld64(const void *p)
{
  uint64_t v;
  __builtin_memcpy(&v, p, 8);
  return v;
}
static inline void nr_ldpc_tx_st64(void *p, uint64_t v)
{
  __builtin_memcpy(p, &v, 8);
}

#endif
