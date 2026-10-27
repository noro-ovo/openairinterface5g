/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* LDPCencoder() of the LDPC shared-library interface (nrLDPC_extern.h) on top of the bit-sliced encoder of the packed
 * transmit chain: same output as ldpc_encoder_optim8segmulti.c (code blocks first_seg .. min(first_seg + 8,
 * n_segments) - 1, output[k] bit j = codeword bit k without the 2Z punctured bits of code block first_seg + j). Used by
 * ldpctest; the transport-block path (nrLDPC_coding_encoder) calls nr_ldpc_tx_encode() directly.
 * impp->F is honoured: the last F of the K systematic bits are filler bits, encoded as 0 whatever the input holds. */
#include <stdlib.h>
#include <string.h>
#include "openair1/PHY/CODING/nrLDPC_defs.h"
#include "openair1/PHY/CODING/nrLDPC_extern.h"
#include "nr_ldpc_tx.h"

int32_t LDPCencoder(uint8_t **input, uint8_t *output, encoder_implemparams_t *impp)
{
  const int Z = impp->Zc, K = impp->K, BG = impp->BG;
  if (BG != 1 && BG != 2)
    return -1;
  const int kb = BG == 1 ? 22 : 10, nrows = BG == 1 ? 46 : 42, rate = BG == 1 ? 3 : 5;
  const unsigned first = impp->first_seg;
  const unsigned end = impp->n_segments > first + 8 ? first + 8 : impp->n_segments;
  const int n = end > first ? (int)(end - first) : 0;
  // output length as ldpc_encoder_optim8segmulti.c computes it
  const int no_punctured_columns = ((nrows - 2) * Z + K - K * rate) / Z;
  const int removed_bit = (nrows - no_punctured_columns - 2) * Z + K - K * rate;
  const int npar = (nrows - no_punctured_columns) * Z - removed_bit;
  if (K <= 2 * Z || K > kb * Z || npar < 0)
    return -1;
  const int kprime = (impp->F > 0 && (int)impp->F < K) ? K - (int)impp->F : K;
  const uint8_t *const *in = (const uint8_t *const *)(input + first);
  if (K == kb * Z) // always the case with nr_segmentation()
    return nr_ldpc_tx_encode_sliced(BG, Z, kprime, n, in, output, K - 2 * Z + npar) < 0 ? -1 : 0;
  // K < kb Z: the parity follows the K - 2Z systematic bytes directly
  uint8_t *tmp = malloc(68 * 384);
  if (!tmp)
    return -1;
  const int nout = kb * Z - 2 * Z + npar, N = (BG == 1 ? 66 : 50) * Z;
  const int ret = nr_ldpc_tx_encode_sliced(BG, Z, kprime, n, in, tmp, nout <= N ? nout : N);
  if (ret >= 0) {
    memcpy(output, tmp, K - 2 * Z);
    memcpy(output + K - 2 * Z, tmp + kb * Z - 2 * Z, npar);
  }
  free(tmp);
  return ret < 0 ? -1 : 0;
}

uint32_t **LDPCencoder32(uint8_t **input, encoder_implemparams_t *impp)
{
  (void)input;
  (void)impp;
  abort();
}
