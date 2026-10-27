/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef NR_MOD_PACKED_H
#define NR_MOD_PACKED_H

#include <stdbool.h>
#include <stdint.h>
#include "common/platform_types.h"

/*
 * Modulation mapping (TS 38.211 5.1.3-5.1.6, QPSK to 256QAM) straight from packed bits, with optional scrambling
 * (5.3.1 / 6.3.1.1: the packed scrambling sequence is XORed into the bits as they are read) and layer mapping
 * (7.3.1.3 / 6.3.1.3, one codeword: symbol i to layer i mod n_layers, position i / n_layers).
 *   in     : length bits packed LSB first (bit b = in[b / 32] bit b % 32); only ceil(length / 8) bytes are read
 *   scr    : NULL or the scrambling sequence in the same layout (gold_cache())
 *   layers : n_layers (1 to 4) output arrays, length / Qm / n_layers symbols each
 * The symbol values are taken from OAI's modulation tables (nr_generate_modulation_table()), so the output equals
 * nr_modulation() (+ nr_layer_mapping()) on the scrambled bits. Returns false, writing nothing, when the
 * combination is not handled (no NEON / AVX-512 VBMI kernel, tables not generated, other Qm or layer count): the
 * caller then uses the scalar path.
 */
bool nr_mod_packed(const uint32_t *in, const uint32_t *scr, uint32_t length, int Qm, int n_layers, c16_t *const *layers);

/* true when nr_mod_packed() handles Qm and n_layers (it can then only fail for length / Qm not a multiple of
 * n_layers) */
bool nr_mod_packed_available(int Qm, int n_layers);

/* Kernel selection for tests and benchmarks: 0 = disabled (always false), 1 = default. Not thread safe. */
void nr_mod_packed_enable(int on);

#endif
