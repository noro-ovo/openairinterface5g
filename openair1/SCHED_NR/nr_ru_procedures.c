/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*!
 * \brief Implementation of RU procedures
 */

#include "PHY/defs_gNB.h"
#include "common/platform_types.h"
#include "nr/nr_common.h"
#include "sched_nr.h"
#include "PHY/MODULATION/phy_ofdm_mod.h"
#include "PHY/MODULATION/nr_modulation.h"
#include "openair1/PHY/defs_nr_common.h"
#include "common/utils/LOG/log.h"
#include "common/utils/system.h"

#include "T.h"
#include <math.h>
#include <pthread.h>
#include <stdlib.h>

#include "assertions.h"

#include <time.h>

// RU OFDM Modulator gNodeB
// OFDM modulation core routine, generates a first_symbol to first_symbol+num_symbols on a particular slot and TX antenna port
// from txdataF (one slot): natural order if nbins == 0, else the fftshifted layout of nbins subcarriers
void nr_feptx0(RU_t *ru, int tti_tx, int first_symbol, int num_symbols, int aa, const c16_t *txdataF, int nbins)
{
  NR_DL_FRAME_PARMS *fp = ru->nr_frame_parms;

  unsigned int slot_offset,slot_offsetF;
  int slot = tti_tx;

  if (aa == 0 && first_symbol == 0)
    start_meas(&ru->ofdm_mod_stats);
  slot_offset = get_samples_slot_timestamp(fp, slot);
  slot_offsetF = first_symbol * fp->ofdm_symbol_size;

  int abs_first_symbol = slot * fp->symbols_per_slot;

  for (int idx_sym = abs_first_symbol; idx_sym < abs_first_symbol + first_symbol; idx_sym++)
    slot_offset += (idx_sym % (0x7 << fp->numerology_index)) ? fp->nb_prefix_samples : fp->nb_prefix_samples0;

  slot_offset += fp->ofdm_symbol_size * first_symbol;

  LOG_D(PHY,
        "SFN/SF:RU:TX:%d/%d aa %d Generating slot %d (first_symbol %d num_symbols %d) slot_offset %d, slot_offsetF %d\n",
        ru->proc.frame_tx,
        ru->proc.tti_tx,
        aa,
        slot,
        first_symbol,
        num_symbols,
        slot_offset,
        slot_offsetF);
  
  if (fp->Ncp == 1) {
    PHY_ofdm_mod_shifted((const int *)&txdataF[slot_offsetF],
                         (int*)&ru->common.txdata[aa][slot_offset],
                         fp->ofdm_symbol_size,
                         num_symbols,
                         fp->nb_prefix_samples,
                         CYCLIC_PREFIX,
                         nbins);
  } else {
    if (fp->numerology_index != 0) {
      
      if (!(slot%(fp->slots_per_subframe/2))&&(first_symbol==0)) { // case where first symbol in slot has longer prefix
        PHY_ofdm_mod_shifted((const int *)&txdataF[slot_offsetF],
                             (int*)&ru->common.txdata[aa][slot_offset],
                             fp->ofdm_symbol_size,
                             1,
                             fp->nb_prefix_samples0,
                             CYCLIC_PREFIX,
                             nbins);

        PHY_ofdm_mod_shifted((const int *)&txdataF[slot_offsetF+fp->ofdm_symbol_size],
                             (int*)&ru->common.txdata[aa][slot_offset+fp->nb_prefix_samples0+fp->ofdm_symbol_size],
                             fp->ofdm_symbol_size,
                             num_symbols-1,
                             fp->nb_prefix_samples,
                             CYCLIC_PREFIX,
                             nbins);
      }
      else { // all symbols in slot have shorter prefix
        PHY_ofdm_mod_shifted((const int *)&txdataF[slot_offsetF],
                             (int*)&ru->common.txdata[aa][slot_offset],
                             fp->ofdm_symbol_size,
                             num_symbols,
                             fp->nb_prefix_samples,
                             CYCLIC_PREFIX,
                             nbins);
      }
    } // numerology_index!=0
    else { //numerology_index == 0
      for (int idx_sym = abs_first_symbol; idx_sym < abs_first_symbol+num_symbols; idx_sym++) {
        if (idx_sym % 0x7) {
          PHY_ofdm_mod_shifted((const int *)&txdataF[slot_offsetF],
                               (int*)&ru->common.txdata[aa][slot_offset],
                               fp->ofdm_symbol_size,
                               1,
                               fp->nb_prefix_samples,
                               CYCLIC_PREFIX,
                               nbins);
          slot_offset += fp->nb_prefix_samples+fp->ofdm_symbol_size;
          slot_offsetF += fp->ofdm_symbol_size;
        }
        else {
          PHY_ofdm_mod_shifted((const int *)&txdataF[slot_offsetF],
                               (int*)&ru->common.txdata[aa][slot_offset],
                               fp->ofdm_symbol_size,
                               1,
                               fp->nb_prefix_samples0,
                               CYCLIC_PREFIX,
                               nbins);
          slot_offset += fp->nb_prefix_samples0+fp->ofdm_symbol_size;
          slot_offsetF += fp->ofdm_symbol_size;
        }
      } // for(idx_symbol..
    } //  numerology 0
  }

  if (aa == 0 && first_symbol == 0)
    stop_meas(&ru->ofdm_mod_stats);
}

// RU FEP TX OFDM modulation, single-thread
void nr_feptx_ofdm(RU_t *ru,int frame_tx,int tti_tx)
{
  nfapi_nr_config_request_scf_t *cfg = &ru->gNB_list[0]->gNB_config;
  NR_DL_FRAME_PARMS *fp = ru->nr_frame_parms;

  unsigned int aa=0;
  int slot_sizeF = fp->ofdm_symbol_size * fp->symbols_per_slot;
  int slot = tti_tx;
  int *txdata = &ru->common.txdata[aa][get_samples_slot_timestamp(fp, slot)];

  if (nr_slot_select(cfg,frame_tx,slot) == NR_UPLINK_SLOT)
    return;

  nr_feptx0(ru, slot, 0, fp->symbols_per_slot, aa, (c16_t *)ru->common.txdataF_BF[aa], 0);

  LOG_D(PHY,
        "feptx_ofdm (TXPATH): frame %d, slot %d: txp (time %p) %d dB, txp (freq) %d dB\n",
        frame_tx,
        slot,
        txdata,
        dB_fixed(signal_energy((int32_t *)txdata, get_samples_per_slot(slot, fp))),
        dB_fixed(signal_energy_nodc((c16_t *)ru->common.txdataF_BF[aa], 2 * slot_sizeF)));
}

void nr_feptx_prec(RU_t *ru, int frame_tx, int slot_tx)
{
  PHY_VARS_gNB **gNB_list = ru->gNB_list;
  AssertFatal(ru->num_gNB == 1, "Cannot handle more than 1 gNB\n");
  PHY_VARS_gNB *gNB = gNB_list[0];
  nfapi_nr_config_request_scf_t *cfg = &ru->gNB_list[0]->gNB_config;
  NR_DL_FRAME_PARMS *fp = ru->nr_frame_parms;
  start_meas(&ru->precoding_stats);

  if (nr_slot_select(cfg,frame_tx,slot_tx) == NR_UPLINK_SLOT)
    return;

  // If there is no digital beamforming we just need to copy the data to RU
  if (ru->config.dbt_config.num_dig_beams == 0 || ru->gNB_list[0]->common_vars.analog_bf) {
    for (int i = 0; i < fp->nb_antennas_tx; ++i) {
      memcpy(ru->common.txdataF_BF[i], gNB->common_vars.txdataF[i], fp->samples_per_slot_wCP * sizeof(int32_t));
    }
  }  else {
    AssertFatal(false, "This needs to be fixed by using appropriate beams from config\n");
  }
  stop_meas(&ru->precoding_stats);
}

// core routine for FEP TX, called from threads in RU TX thread-pool 
void nr_feptx(void *arg)
{
  feptx_cmd_t *feptx = (feptx_cmd_t *)arg;

  RU_t *ru = feptx->ru;
  int slot = feptx->slot;
  int aa = feptx->aid;
  int startSymbol = feptx->startSymbol;
  int numSymbols = feptx->numSymbols;

  // Without digital beamforming the gNB frequency-domain data (fftshifted layout) is modulated directly
  AssertFatal(ru->config.dbt_config.num_dig_beams == 0 || ru->gNB_list[0]->common_vars.analog_bf,
              "This needs to be fixed by using appropriate beams from config\n");

  ////////////FEPTX////////////
  const NR_DL_FRAME_PARMS *fp = &ru->gNB_list[0]->frame_parms;
  nr_feptx0(ru, slot, startSymbol, numSymbols, aa, ru->gNB_list[0]->common_vars.txdataF[aa], fp->N_RB_DL * NR_NB_SC_PER_RB);

  // Task completed in //
  completed_task_ans(feptx->ans);
}

// RU FEP TX using thread-pool
void nr_feptx_tp(RU_t *ru, int frame_tx, int slot)
{
  nfapi_nr_config_request_scf_t *cfg = &ru->gNB_list[0]->gNB_config;
  const NR_DL_FRAME_PARMS *fp = ru->nr_frame_parms;

  if (nr_slot_select(cfg, frame_tx, slot) == NR_UPLINK_SLOT)
    return;
  if (ru->rfdevice.trx_write_fd_func)
    return; // the device does the OFDM modulation (tx_rf_symbols hands it the frequency-domain data)
  start_meas(&ru->ofdm_total_stats);

  int nt = fp->nb_antennas_tx;
  size_t const sz = nt + (ru->half_slot_parallelization > 0) * nt;
  feptx_cmd_t arr[sz];
  task_ans_t ans;
  init_task_ans(&ans, sz);

  int nbfeptx = 0;
  for (int aid = 0; aid < nt; aid++) {
    feptx_cmd_t *feptx_cmd = &arr[nbfeptx];
    feptx_cmd->ans = &ans;
    feptx_cmd->aid = aid;
    feptx_cmd->ru = ru;
    feptx_cmd->slot = slot;
    feptx_cmd->startSymbol = 0;
    feptx_cmd->numSymbols =
        (ru->half_slot_parallelization > 0) ? ru->nr_frame_parms->symbols_per_slot >> 1 : ru->nr_frame_parms->symbols_per_slot;

    task_t t = {.func = nr_feptx, .args = feptx_cmd};
    pushTpool(ru->threadPool, t);
    nbfeptx++;
    if (ru->half_slot_parallelization > 0) {
      feptx_cmd_t *feptx_cmd = &arr[nbfeptx];
      feptx_cmd->ans = &ans;
      feptx_cmd->aid = aid;
      feptx_cmd->ru = ru;
      feptx_cmd->slot = slot;
      feptx_cmd->startSymbol = ru->nr_frame_parms->symbols_per_slot >> 1;
      feptx_cmd->numSymbols = ru->nr_frame_parms->symbols_per_slot >> 1;

      task_t t = {.func = nr_feptx, .args = feptx_cmd};
      pushTpool(ru->threadPool, t);
      nbfeptx++;
    }
  }
  join_task_ans(&ans);

  stop_meas(&ru->ofdm_total_stats);
}

// Test only (environment OAI_FD_CHECK=N): every N-th symbol taken from the device is also demodulated by OAI and by an
// exact double-precision DFT of the same window; the errors against the exact DFT are logged every 200 checks.
static void fd_check_fft(double *re, double *im, int n)
{
  for (int i = 1, j = 0; i < n; i++) {
    int b = n >> 1;
    for (; j & b; b >>= 1)
      j ^= b;
    j ^= b;
    if (i < j) {
      double t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    const double a = -2 * M_PI / len;
    for (int i = 0; i < n; i += len)
      for (int k = 0; k < len / 2; k++) {
        const double wr = cos(a * k), wi = sin(a * k);
        const int p = i + k, q = p + len / 2;
        const double xr = re[q] * wr - im[q] * wi, xi = re[q] * wi + im[q] * wr;
        re[q] = re[p] - xr; im[q] = im[p] - xi;
        re[p] += xr; im[p] += xi;
      }
  }
}

static void fd_check_symbol(const NR_DL_FRAME_PARMS *fp, const c16_t *rxdata, const c16_t *dev_bins, int l, int slot, int sample_offset)
{
  static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
  static double e_dev, e_oai, e_do, p_ref, worst_dev = -1000;
  static int checks, dev_sat;
  const int N = fp->ofdm_symbol_size, nbins = fp->N_RB_UL * NR_NB_SC_PER_RB;
  if (N != 4096)
    return;
  c16_t oai[4096] __attribute__((aligned(32)));
  nr_symbol_fep_ul_shifted(fp, rxdata, oai, l, slot, sample_offset);
  // the window nr_symbol_fep_ul_shifted uses
  const int prefix = get_samples_symbol_duration(fp, slot, l, 1) - N;
  int64_t off = get_samples_slot_timestamp(fp, slot) + get_samples_symbol_timestamp(fp, slot, l) + prefix
                - fp->nb_prefix_samples / fp->ofdm_offset_divisor - sample_offset;
  if (off < 0)
    off += fp->samples_per_frame;
  double *re = malloc(2 * N * sizeof(double)), *im = re + N; // test only
  for (int n = 0; n < N; n++) {
    const c16_t x = rxdata[(off + n) % fp->samples_per_frame];
    re[n] = x.r;
    im[n] = x.i;
  }
  fd_check_fft(re, im, N);
  double ed = 0, eo = 0, edo = 0, pr = 0;
  int sat = 0;
  for (int j = 0; j < nbins; j++) {
    const int k = (N - nbins / 2 + j) % N;
    const double rr = re[k] / 64, ri = im[k] / 64;
    const double dr = dev_bins[j].r - rr, di = dev_bins[j].i - ri, orr = oai[j].r - rr, oi = oai[j].i - ri;
    ed += dr * dr + di * di;
    eo += orr * orr + oi * oi;
    edo += (dev_bins[j].r - oai[j].r) * (dev_bins[j].r - oai[j].r) + (dev_bins[j].i - oai[j].i) * (dev_bins[j].i - oai[j].i);
    pr += rr * rr + ri * ri;
    sat += abs(dev_bins[j].r) >= 32767 || abs(dev_bins[j].i) >= 32767;
  }
  free(re);
  pthread_mutex_lock(&mtx);
  e_dev += ed;
  e_oai += eo;
  e_do += edo;
  p_ref += pr;
  dev_sat += sat;
  if (pr > 0 && 10 * log10(ed / pr) > worst_dev)
    worst_dev = 10 * log10(ed / pr);
  if (++checks == 200) {
    LOG_I(PHY,
          "fd check: %d symbols, bin rms %.1f, vs exact DFT/64: device %.1f dB (worst symbol %.1f), OAI %.1f dB; device vs OAI %.1f dB; saturated device bins %d\n",
          checks, sqrt(p_ref / (checks * (double)nbins)), 10 * log10(e_dev / p_ref), worst_dev, 10 * log10(e_oai / p_ref),
          10 * log10(e_do / p_ref), dev_sat);
    e_dev = e_oai = e_do = p_ref = 0;
    worst_dev = -1000;
    checks = dev_sat = 0;
  }
  pthread_mutex_unlock(&mtx);
}

// core RX FEP routine, called by threads in RU thread-pool
void nr_fep(void *arg)
{
  feprx_cmd_t *feprx_cmd = (feprx_cmd_t *)arg;
  int slot = feprx_cmd->slot;
  int startSymbol = feprx_cmd->startSymbol;
  int endSymbol = feprx_cmd->endSymbol;

  const NR_DL_FRAME_PARMS *fp = feprx_cmd->fp;
  openair0_device_t *fd_dev = feprx_cmd->fd_dev;
  const int nbins = fp->N_RB_UL * NR_NB_SC_PER_RB;
  for (int l = startSymbol; l <= endSymbol; l++) {
    if (fd_dev) {
      // the device demodulated the symbol: same FFT window as nr_symbol_fep_ul_shifted (end of the cyclic prefix,
      // advanced by nb_prefix_samples / ofdm_offset_divisor and by the timing advance offset)
      const int prefix = get_samples_symbol_duration(fp, slot, l, 1) - fp->ofdm_symbol_size;
      const openair0_timestamp_t win_ts = feprx_cmd->slot_ts + get_samples_symbol_timestamp(fp, slot, l) + prefix
                                          - fp->nb_prefix_samples / fp->ofdm_offset_divisor - feprx_cmd->sample_offet;
      if (fd_dev->trx_get_rxdataF_func(fd_dev, win_ts, feprx_cmd->aid, &feprx_cmd->rxdataF[l * fp->ofdm_symbol_size], nbins)
          == 0) {
        static int check_every = -1, check_n;
        if (check_every < 0) {
          const char *e = getenv("OAI_FD_CHECK");
          check_every = e ? atoi(e) : 0;
        }
        if (check_every > 0 && __atomic_add_fetch(&check_n, 1, __ATOMIC_RELAXED) % check_every == 0)
          fd_check_symbol(fp, feprx_cmd->rxdata, &feprx_cmd->rxdataF[l * fp->ofdm_symbol_size], l, slot, feprx_cmd->sample_offet);
        continue;
      }
    }
    // OFDM Demod + FFT-shift
    nr_symbol_fep_ul_shifted(fp,
                             feprx_cmd->rxdata,
                             &feprx_cmd->rxdataF[l * fp->ofdm_symbol_size],
                             l,
                             slot,
                             feprx_cmd->sample_offet);
  }

  completed_task_ans(feprx_cmd->ans);
}

// RU RX FEP using thread-pool
void nr_fep_tp(RU_t *ru, int slot)
{
  int nbfeprx = 0;
  start_meas(&ru->ofdm_demod_stats);

  NR_DL_FRAME_PARMS *fp = ru->nr_frame_parms;
  int nt = fp->nb_antennas_rx;
  int tasks_per_slot = (ru->half_slot_parallelization > 0) ? 2 : 1;
  size_t const sz = nt * tasks_per_slot;
  feprx_cmd_t arr[sz];
  task_ans_t ans;
  init_task_ans(&ans, sz);
  int rxdataF_offset = (slot % RU_RX_SLOT_DEPTH) * fp->symbols_per_slot * fp->ofdm_symbol_size;

  int symbols_per_task = fp->symbols_per_slot / tasks_per_slot;

  for (int task_idx = 0; task_idx < tasks_per_slot; task_idx++) {
    int start_symbol = task_idx * symbols_per_task;
    int end_symbol = (task_idx + 1) * symbols_per_task - 1;
    if (task_idx == tasks_per_slot - 1)
      end_symbol = fp->symbols_per_slot - 1;

    for (int aid = 0; aid < nt; aid++) {
      feprx_cmd_t *feprx_cmd = &arr[nbfeprx];
      feprx_cmd->ans = &ans;
      feprx_cmd->fp = fp;
      feprx_cmd->slot = ru->proc.tti_rx;
      feprx_cmd->startSymbol = start_symbol;
      feprx_cmd->endSymbol = end_symbol;
      feprx_cmd->rxdata = (const c16_t *)ru->common.rxdata[aid];
      feprx_cmd->rxdataF = (c16_t *)&ru->common.rxdataF[aid][rxdataF_offset];
      feprx_cmd->sample_offet = ru->N_TA_offset;
      feprx_cmd->aid = aid;
      feprx_cmd->fd_dev = ru->rfdevice.trx_get_rxdataF_func ? &ru->rfdevice : NULL;
      feprx_cmd->slot_ts = ru->ts_offset + ru->proc.timestamp_rx;

      task_t t = {.func = nr_fep, .args = feprx_cmd};
      pushTpool(ru->threadPool, t);
      nbfeprx++;
    }
  }
  join_task_ans(&ans);

  stop_meas(&ru->ofdm_demod_stats);
}
