/* OAI-signature entry points (see include/ldpc_dec_oai.h). */
#include <string.h>
#include "ldpc_dec.h"
#include "ldpc_dec_oai.h"

#include <stdlib.h>

int32_t LDPCinit(void) { return 0; }

/* modulation order hint (AGC target): OAI's t_nrLDPC_dec_params has no Qm field at 9313cbea. Until it has one
   (build with -DLDPC_DEC_OAI_HAS_QM to read p_decParams->Qm), the segment layer can call LDPCdecoder_set_qm() on the
   decoding thread before LDPCdecoder(); LDPC_DEC_QM=<qm> sets a process-wide default (experiments); else unknown. */
static __thread int t_qm = -1;
void LDPCdecoder_set_qm(int qm) { t_qm = qm; }
#ifndef LDPC_DEC_OAI_HAS_QM
static int env_qm(void)
{
  static int q = -2;
  if (q == -2) {
    const char *e = getenv("LDPC_DEC_QM");
    q = e ? atoi(e) : 0;
  }
  return q;
}
#endif
int32_t LDPCshutdown(void) { return 0; }

static int oai_abort_poll(void *ctx)
{
  decode_abort_t *ab = ctx;
  pthread_mutex_lock(&ab->mutex_failure);
  const int f = ab->failed;
  pthread_mutex_unlock(&ab->mutex_failure);
  return f;
}

int32_t LDPCdecoder(t_nrLDPC_dec_params *p_decParams, int8_t *p_llr, uint8_t *p_out, t_nrLDPC_time_stats *time_stats,
                    decode_abort_t *ab)
{
  (void)time_stats;
  const int maxit = p_decParams->numMaxIter;
  ldpc_dec_params_t p;
  memset(&p, 0, sizeof p);
  p.bg = p_decParams->BG;
  p.z = p_decParams->Z;
  p.rcode = p_decParams->R;
  p.max_iter = (uint8_t)(maxit < 1 ? 1 : maxit);
  p.out_mode = p_decParams->outMode == nrLDPC_outMode_BITINT8 ? LDPC_OUT_BYTES
             : p_decParams->outMode == nrLDPC_outMode_LLRINT8 ? LDPC_OUT_LLR
                                                              : LDPC_OUT_BITS;
  p.impl = LDPC_IMPL_AUTO;
  p.check_crc = p_decParams->check_crc;
  p.kprime = p_decParams->Kprime;
  p.crc_type = (uint8_t)p_decParams->crc_type;
#ifdef LDPC_DEC_OAI_HAS_QM
  p.qm = (uint8_t)p_decParams->Qm;
#else
  p.qm = (uint8_t)(t_qm >= 0 ? t_qm : env_qm());
#endif
  if (ab) {
    p.abort_poll = oai_abort_poll;
    p.abort_ctx = ab;
  }
  ldpc_dec_stats_t st;
  const int ok = ldpc_decode(&p, p_llr, p_out, &st);
  if (ok == 1 && st.iterations - 1 < maxit) return st.iterations - 1;
  if (ab && ok >= 0) {
    pthread_mutex_lock(&ab->mutex_failure);
    ab->failed = true;
    pthread_mutex_unlock(&ab->mutex_failure);
  }
  return maxit;
}
