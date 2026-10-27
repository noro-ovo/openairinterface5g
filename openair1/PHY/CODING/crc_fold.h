/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef CRC_FOLD_H
#define CRC_FOLD_H

#include <stdbool.h>
#include <stdint.h>

/*
 * CRC of whole bytes by carry-less multiplication folding over several independent 128-bit lanes (x86 PCLMULQDQ or
 * AVX-512 VPCLMULQDQ, aarch64 PMULL; kernel chosen at run time), for the left-aligned CRCs of crc_byte.c:
 *   crc = M(x) * x^32 mod (x^32 + poly), M(x) the message with the MSB of byte 0 as the highest power, the result
 *   left-aligned in 32 bits (crc24a/crc24b: bits 31..8, crc16: bits 31..16).
 * The message is folded down to a 128-bit remainder-equivalent, which is reduced to the CRC with two more folds
 * and a Barrett reduction (messages shorter than the kernel minimum: the byte table of crc_byte.c,
 * table[c] = crcbit(&c, 1, poly)). The result equals the table and crc32_calc_pclmulqdq() results.
 */
typedef struct {
  uint32_t poly;
  const uint32_t *table;
  uint64_t k96, k64, mu; /* x^96 mod P, x^64 mod P, floor(x^64 / P): final reduction (Barrett) */
  uint64_t k128[2]; /* {x^128 mod P, x^(128 + 64) mod P}: fold one lane by 128 bits */
  uint64_t k512[2]; /* by 512 bits (4 lanes) */
  uint64_t k2048[2]; /* by 2048 bits (16 lanes, AVX-512) */
  uint64_t kc4[4][2]; /* lane j of 4 to the end: by 128 * (3 - j) bits */
  uint64_t kc16[16][2] __attribute__((aligned(64))); /* lane j of 16: by 128 * (15 - j) bits */
} crc_fold_ctx_t;

void crc_fold_init(crc_fold_ctx_t *ctx, uint32_t poly, const uint32_t *table);

/* crc of len bytes; false (nothing computed) when no kernel is available or len < the kernel's minimum */
bool crc_fold(const crc_fold_ctx_t *ctx, const uint8_t *data, uint32_t len, uint32_t *crc);

/* kernel selection for tests and benchmarks: "off", "pclmul", "vpclmul", "pmull" (best available when NULL);
 * returns false when the kernel is not available on this CPU. Not thread safe. */
bool crc_fold_set_variant(const char *name);
const char *crc_fold_variant(void);

#endif
