/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*! \file PHY/CODING/nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_coding_segment_encoder_packed.c
 * \brief Transport-block LDPC encoding with the packed transmit chain (PHY/CODING/nrLDPC_tx)
 *
 * One thread-pool task per group of up to 8 code blocks of a transport block. Each task encodes its code blocks,
 * rate matches (TS 38.212 5.4.2.1) and bit interleaves (5.4.2.2) them and writes the packed bits straight to the
 * transport-block output at the code blocks' offsets: no per-group byte buffers and no serial bit-plane extraction
 * after the join. Same interface and output as nrLDPC_coding_segment_encoder.c.
 */

#include "nr_rate_matching.h"
#include "PHY/CODING/nrLDPC_coding/nrLDPC_coding_interface.h"
#include "PHY/CODING/nrLDPC_tx/nr_ldpc_tx.h"
#include "common/utils/LOG/log.h"
#include "common/utils/threadPool/task_ans.h"

typedef struct {
  nrLDPC_TB_encoding_parameters_t *tb;
  uint32_t first; // first code block of the group
  uint32_t n; // code blocks in the group (1..8)
  uint32_t Eoff; // bit offset of the first code block in the transport-block output
  task_ans_t *ans;
} ldpc_tx_task_t;

static void ldpc_tx_task(void *arg)
{
  const ldpc_tx_task_t *t = arg;
  nrLDPC_TB_encoding_parameters_t *tb = t->tb;
  // 38.212 5.4.2.1: Ncb (with LBRM) and k0, computed as for the legacy chain (C truncated to 8 bits there too)
  const nr_ldpc_geometry_t geo = nr_ldpc_soft_buffer_geometry(tb->tbslbrm, tb->BG, tb->Z, tb->C, tb->rv_index);
  const nr_ldpc_tx_params_t p = {.BG = tb->BG, .Qm = tb->Qm, .Z = tb->Z, .K = tb->K, .F = tb->F, .Ncb = geo.Ncb, .k0 = geo.k0};
  const uint8_t *c[8];
  uint32_t E[8], Eoff[8];
  uint32_t off = t->Eoff;
  for (int j = 0; j < t->n; j++) {
    const nrLDPC_segment_encoding_parameters_t *seg = &tb->segments[t->first + j];
    c[j] = seg->c;
    E[j] = seg->E;
    Eoff[j] = off;
    off += seg->E;
  }
  start_meas(&tb->segments[t->first].ts_ldpc_encode);
  if (nr_ldpc_tx_encode(&p, t->n, c, E, Eoff, tb->output) < 0)
    LOG_E(NR_PHY,
          "LDPC encoding failed, code segments %d...%d/%d (A %d, G %u, Kr %d, F %d, Zc %d, Ncb %u, rv %d, Qm %d), "
          "sending zeroed block\n",
          t->first,
          t->first + t->n,
          tb->C,
          tb->A,
          tb->G,
          tb->K,
          tb->F,
          tb->Z,
          geo.Ncb,
          tb->rv_index,
          tb->Qm);
  stop_meas(&tb->segments[t->first].ts_ldpc_encode);
  completed_task_ans(t->ans);
}

int nrLDPC_coding_encoder(nrLDPC_slot_encoding_parameters_t *slot)
{
  int n_tasks = 0;
  for (int i = 0; i < slot->nb_TBs; i++)
    n_tasks += (slot->TBs[i].C + 7) / 8;
  if (n_tasks == 0)
    return 0;

  ldpc_tx_task_t tasks[n_tasks];
  task_ans_t ans;
  init_task_ans(&ans, n_tasks);
  int k = 0;
  for (int i = 0; i < slot->nb_TBs; i++) {
    nrLDPC_TB_encoding_parameters_t *tb = &slot->TBs[i];
    uint32_t Eoff = 0;
    for (uint32_t first = 0; first < tb->C; first += 8) {
      ldpc_tx_task_t *t = &tasks[k++];
      *t = (ldpc_tx_task_t){.tb = tb, .first = first, .n = tb->C - first < 8 ? tb->C - first : 8, .Eoff = Eoff, .ans = &ans};
      for (uint32_t r = first; r < first + t->n; r++)
        Eoff += tb->segments[r].E;
      pushTpool(slot->threadPool, (task_t){.func = ldpc_tx_task, .args = t});
    }
  }
  join_task_ans(&ans);
  return 0;
}
