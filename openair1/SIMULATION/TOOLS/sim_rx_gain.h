/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* Experiment option for the NR simulators: received-amplitude sweep at fixed SNR.
 *
 * Env OAI_SIM_RX_GAIN_DB=<g>: the simulated receive samples (signal after the channel + noise, i.e. both scaled by the
 * same factor, so the SNR is unchanged) are multiplied by 10^(g/20) and then quantised like an int16 ADC: rounded to
 * the nearest integer (OAI_SIM_RX_QUANT=trunc keeps the simulators' legacy truncation toward zero) and saturated to
 * [-32768, 32767] (the legacy float->int16 casts wrap instead).  Unset: the simulators are unchanged (bit-exact).
 * At exit, one line "SIM_RXGAIN: ..." reports the gain, the number of quantised components, the fraction clipped and
 * the RMS level of the complex samples (dBFS re 32768). */

#ifndef SIM_RX_GAIN_H
#define SIM_RX_GAIN_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int init, active, round;
  double g, g_db;
  uint64_t n, nclip, nzero;
  double e, peak;
} sim_rx_gain_t;

static sim_rx_gain_t sim_rxg;

static void sim_rx_gain_report(void)
{
  const double rms = sim_rxg.n ? sqrt(2.0 * sim_rxg.e / sim_rxg.n) : 0.0; // complex RMS (2 components per sample)
  printf("SIM_RXGAIN: g_db %.2f quant %s components %llu clipped %.4e zero %.4e rms %.2f (%.2f dBFS) peak %.0f\n",
         sim_rxg.g_db,
         sim_rxg.round ? "round" : "trunc",
         (unsigned long long)sim_rxg.n,
         sim_rxg.n ? (double)sim_rxg.nclip / sim_rxg.n : 0.0,
         sim_rxg.n ? (double)sim_rxg.nzero / sim_rxg.n : 0.0,
         rms,
         rms > 0 ? 20.0 * log10(rms / 32768.0) : -999.0,
         sim_rxg.peak);
  fflush(stdout);
}

static inline const sim_rx_gain_t *sim_rx_gain_get(void)
{
  if (!sim_rxg.init) {
    sim_rxg.init = 1;
    sim_rxg.g = 1.0;
    const char *s = getenv("OAI_SIM_RX_GAIN_DB");
    if (s && *s) {
      sim_rxg.active = 1;
      sim_rxg.g_db = atof(s);
      sim_rxg.g = pow(10.0, sim_rxg.g_db / 20.0);
      const char *q = getenv("OAI_SIM_RX_QUANT");
      sim_rxg.round = !(q && strcmp(q, "trunc") == 0);
      atexit(sim_rx_gain_report);
    }
  }
  return &sim_rxg;
}

/* v: the sample component after the gain (float, before quantisation) */
static inline int16_t sim_rx_adc(double v)
{
  const double a = sim_rxg.round ? nearbyint(v) : trunc(v);
  sim_rxg.n++;
  int16_t r;
  if (a > 32767.0) {
    sim_rxg.nclip++;
    r = 32767;
  } else if (a < -32768.0) {
    sim_rxg.nclip++;
    r = -32768;
  } else {
    r = (int16_t)a;
  }
  sim_rxg.e += (double)r * r;
  if (r == 0)
    sim_rxg.nzero++;
  if (fabs((double)r) > sim_rxg.peak)
    sim_rxg.peak = fabs((double)r);
  return r;
}

#endif
