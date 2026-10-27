/* OAI-compatible decoder entry points: same signatures and conventions as OAI's nrLDPC_decoder
 * (openair1/PHY/CODING/nrLDPC_decoder, commit 9313cbea), so libldpc can be replaced by this implementation.
 *
 * Built standalone, this header mirrors OAI's parameter structures (layout-identical to nrLDPC_types.h and
 * coding_defs.h). Inside OAI, compile ldpc_oai_entry.c with -DLDPC_DEC_OAI_HEADERS and OAI's include paths so the
 * original definitions are used instead.
 */
#ifndef LDPC_DEC_OAI_H
#define LDPC_DEC_OAI_H
#ifdef LDPC_DEC_OAI_HEADERS
#include "openair1/PHY/CODING/nrLDPC_decoder/nrLDPC_types.h"
#include "openair1/PHY/CODING/coding_defs.h"
#else
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
typedef struct {
  pthread_mutex_t mutex_failure;
  bool failed;
} decode_abort_t;
typedef enum nrLDPC_outMode {
  nrLDPC_outMode_BIT,     /* packed bits, MSB first */
  nrLDPC_outMode_BITINT8, /* one bit per byte */
  nrLDPC_outMode_LLRINT8  /* one LLR per byte */
} e_nrLDPC_outMode;
typedef struct nrLDPC_dec_params {
  uint8_t BG;
  uint16_t Z;
  uint8_t R;
  uint8_t numMaxIter;
  int Kprime;
  e_nrLDPC_outMode outMode;
  int crc_type;
  int (*check_crc)(uint8_t *decoded_bytes, uint32_t n, uint8_t crc_type);
  int n_segments;
  int LastTrial;
} t_nrLDPC_dec_params;
typedef struct nrLDPC_time_stats t_nrLDPC_time_stats; /* not used */
#endif

int32_t LDPCinit(void);
int32_t LDPCshutdown(void);
/* Decodes one code block. Returns (iterations - 1) on success, numMaxIter on failure (and sets the abort
   flag, as OAI does). numMaxIter is the number of layered iterations (OAI: flooding iterations + 1). */
/* optional: modulation order of the next code blocks decoded on this thread (2, 4, 6, 8; 0 unknown), for the AGC
   target; ignored when built with -DLDPC_DEC_OAI_HAS_QM (then p_decParams->Qm is used) */
void LDPCdecoder_set_qm(int qm);
int32_t LDPCdecoder(t_nrLDPC_dec_params *p_decParams, int8_t *p_llr, uint8_t *p_out, t_nrLDPC_time_stats *time_stats,
                    decode_abort_t *ab);
#endif
