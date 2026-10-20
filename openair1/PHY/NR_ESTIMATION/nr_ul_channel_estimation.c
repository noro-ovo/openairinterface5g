/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "nr_common.h"
#include "platform_types.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <limits.h>

#include "nr_ul_estimation.h"
#include "PHY/sse_intrin.h"
#include "PHY/NR_REFSIG/nr_refsig.h"
#include "PHY/NR_REFSIG/dmrs_nr.h"
#include "PHY/NR_REFSIG/ptrs_nr.h"
#include "PHY/NR_TRANSPORT/nr_transport_proto.h"
#include "PHY/NR_UE_ESTIMATION/filt16a_32.h"
#include "PHY/NR_TRANSPORT/nr_sch_dmrs.h"
#include "PHY/NR_REFSIG/ul_ref_seq_nr.h"
#include "executables/softmodem-common.h"
#include "nr_phy_common.h"
#include "openair1/PHY/TOOLS/phy_scope_interface.h"
#include "T.h"

//#define DEBUG_CH
//#define DEBUG_PUSCH
//#define SRS_DEBUG

#define NO_INTERP 1
#define dBc(x, y) (dB_fixed(((int32_t)(x)) * (x) + ((int32_t)(y)) * (y)))

typedef struct puschAntennaProc_s {
  unsigned char Ns;
  int nl;
  unsigned short p;
  unsigned char symbol;
  unsigned short bwp_start_subcarrier;
  int aarx;
  uint16_t ant_port_start;
  int numAntennas;
  const nfapi_nr_pusch_pdu_t *pusch_pdu;
  int *max_ch;
  c16_t *pilot;
  int *nest_count;
  uint64_t *noise_amp2;
  delay_t *delay;
  int chest_freq;
  void *ul_ch_estimates; // c16_t [layer * nb_rx + rx][ch_est_size]
  int ch_est_size;
  int ch_row_offset; // row of this DMRS symbol in each estimate
  NR_DL_FRAME_PARMS *frame_parms;
  c16_t **rxdataF;
  task_ans_t *ans;
  scopeData_t *scope;
  c16_t *pusch_ch_est_dmrs_pos_slot_mem;
  int dmrs_symbol_start_idx;
} puschAntennaProc_t;

__attribute__((always_inline)) inline c16_t c32x16cumulVectVectWithSteps(c16_t *in1,
                                                                         int *offset1,
                                                                         const int step1,
                                                                         c16_t *in2,
                                                                         int *offset2,
                                                                         const int step2,
                                                                         const int modulo2,
                                                                         const int N)
{
  int localOffset1 = *offset1;
  int localOffset2 = *offset2;
  c32_t cumul = {0};
  for (int i = 0; i < N; i++) {
    cumul = c32x16maddShift(in1[localOffset1], in2[localOffset2], cumul, 15);
    localOffset1 += step1;
    localOffset2 = (localOffset2 + step2) % modulo2;
  }
  *offset1 = localOffset1;
  *offset2 = localOffset2;
  return c16x32div(cumul, N);
}

/* DMRS type 1 LS estimate of nr_pusch_antenna_processing(), same results as its per-pilot loop: for pilot pair n
 * (REs k0 + delta + 4n and + 2), ch = sum over the pair of (pilot * rx) >> 16 per component (int32), max_ch over the
 * int32 |re|, |im|, and ls[4n..4n+3] = ch truncated to 16 bits. AVX2: 4 pairs per iteration. */
static void pusch_type1_ls_fast(const c16_t *pil, const c16_t *rx, int npairs, c16_t *ls, int *max_ch)
{
  int mx = *max_ch, n = 0;
  const simde__m256i neg_im = simde_mm256_set1_epi32(0xffff0001);
  const simde__m256i even = simde_mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
  simde__m256i vmax = simde_mm256_setzero_si256();
  for (; n + 4 <= npairs; n += 4) {
    const simde__m256i x0 = simde_mm256_permutevar8x32_epi32(simde_mm256_loadu_si256((simde__m256i *)&rx[4 * n]), even);
    const simde__m256i x1 = simde_mm256_permutevar8x32_epi32(simde_mm256_loadu_si256((simde__m256i *)&rx[4 * n + 8]), even);
    const simde__m256i x = simde_mm256_permute2x128_si256(x0, x1, 0x20); // REs 0, 2, 4, .., 14: pairs n..n+3, 2 each
    const simde__m256i p = simde_mm256_loadu_si256((simde__m256i *)&pil[2 * n]);
    const simde__m256i psw = simde_mm256_shufflehi_epi16(simde_mm256_shufflelo_epi16(p, 0xb1), 0xb1);
    const simde__m256i re = simde_mm256_srai_epi32(simde_mm256_madd_epi16(x, simde_mm256_sign_epi16(p, neg_im)), 16);
    const simde__m256i im = simde_mm256_srai_epi32(simde_mm256_madd_epi16(x, psw), 16);
    // per 128-bit lane: re(n) re(n+1) im(n) im(n+1) | re(n+2) re(n+3) im(n+2) im(n+3)
    const simde__m256i h = simde_mm256_hadd_epi32(re, im);
    vmax = simde_mm256_max_epi32(vmax, simde_mm256_abs_epi32(h));
    int32_t v[8];
    simde_mm256_storeu_si256((simde__m256i *)v, h);
    const int32_t rr[4] = {v[0], v[1], v[4], v[5]}, ii[4] = {v[2], v[3], v[6], v[7]};
    for (int j = 0; j < 4; j++) {
      const c16_t c = {.r = (int16_t)rr[j], .i = (int16_t)ii[j]};
      c16_t *d = &ls[4 * (n + j)];
      d[0] = d[1] = d[2] = d[3] = c;
    }
  }
  int32_t m8[8];
  simde_mm256_storeu_si256((simde__m256i *)m8, vmax);
  for (int j = 0; j < 8; j++)
    mx = max(mx, m8[j]);
  for (; n < npairs; n++) {
    c32_t ch = {0};
    ch = c32x16maddShift(pil[2 * n], rx[4 * n], ch, 16);
    ch = c32x16maddShift(pil[2 * n + 1], rx[4 * n + 2], ch, 16);
    mx = max(mx, max(abs(ch.r), abs(ch.i)));
    const c16_t c = {.r = (int16_t)ch.r, .i = (int16_t)ch.i};
    ls[4 * n] = ls[4 * n + 1] = ls[4 * n + 2] = ls[4 * n + 3] = c;
  }
  *max_ch = mx;
}

/* DMRS type 1 frequency interpolation, delay revert and noise of nr_pusch_antenna_processing(), same results as the
 * per-pilot filter loop. Pilot pc adds round(w * a / 2^14) (a = delayed LS of the pilot, w = filter tap) to a window of
 * 16 REs whose taps are constant per group of 4 REs, so the interpolated channel is one value per group of 4 REs.
 * Middle pilots (3 .. npil-2, tap 2048) cover groups (pc-3)/2 .. +3, so group g gets the middle pilots 2g-3 .. 2g+4:
 * a difference of prefix sums. Edge pilots: pc 0 (4096 on groups 0, 1), pc 1 and 2 (4096 on group 0, 2048 on 1, 2),
 * the last one (4096 on group 3 nb_rb - 2, 8192 on 3 nb_rb - 1). With |a| <= 32768 the terms of any RE add up to at
 * most 32768 in magnitude, so the saturating adds of the filter loop equal an int32 sum saturated once. Groups
 * 0 .. 3 nb_rb (the last one only holds the tail of the last middle windows beyond the 12 nb_rb REs, as written by the
 * filter loop). */
static void pusch_type1_interp_fast(const c16_t *ls,
                                    const c16_t *delay_tab,
                                    const c16_t *inv_delay_tab,
                                    int nb_rb,
                                    c16_t *ul_ch,
                                    uint64_t *noise_amp2,
                                    int *nest_count)
{
  const int npil = 6 * nb_rb, ngrp = 3 * nb_rb + 1, nre = 12 * nb_rb;
  c16_t a[npil];
  for (int pc = 0; pc < npil; pc++)
    a[pc] = c16mulShift(ls[2 * pc], delay_tab[2 * pc], 8);
  // prefix sums of the middle terms (a + 4) >> 3 = (2048 a + 2^13) >> 14
  int32_t pr[npil + 1], pi[npil + 1];
  pr[0] = pi[0] = 0;
  for (int pc = 0; pc < npil; pc++) {
    const int mid = pc >= 3 && pc <= npil - 2;
    pr[pc + 1] = pr[pc] + (mid ? (a[pc].r + 4) >> 3 : 0);
    pi[pc + 1] = pi[pc] + (mid ? (a[pc].i + 4) >> 3 : 0);
  }
  int32_t sr[ngrp], si[ngrp];
  for (int g = 0; g < ngrp; g++) {
    const int lo = max(2 * g - 3, 0), hi = min(2 * g + 5, npil);
    sr[g] = hi > lo ? pr[hi] - pr[lo] : 0;
    si[g] = hi > lo ? pi[hi] - pi[lo] : 0;
  }
#define T4096(x) (((x) + 2) >> 2)
#define T2048(x) (((x) + 4) >> 3)
#define T8192(x) (((x) + 1) >> 1)
  sr[0] += T4096(a[0].r) + T4096(a[1].r) + T4096(a[2].r);
  si[0] += T4096(a[0].i) + T4096(a[1].i) + T4096(a[2].i);
  sr[1] += T4096(a[0].r) + T2048(a[1].r) + T2048(a[2].r);
  si[1] += T4096(a[0].i) + T2048(a[1].i) + T2048(a[2].i);
  sr[2] += T2048(a[1].r) + T2048(a[2].r);
  si[2] += T2048(a[1].i) + T2048(a[2].i);
  sr[3 * nb_rb - 2] += T4096(a[npil - 1].r);
  si[3 * nb_rb - 2] += T4096(a[npil - 1].i);
  sr[3 * nb_rb - 1] += T8192(a[npil - 1].r);
  si[3 * nb_rb - 1] += T8192(a[npil - 1].i);
#undef T4096
#undef T2048
#undef T8192
  for (int g = 0; g < ngrp; g++) {
    const c16_t v = {.r = (int16_t)(sr[g] > 32767 ? 32767 : sr[g] < -32768 ? -32768 : sr[g]),
                     .i = (int16_t)(si[g] > 32767 ? 32767 : si[g] < -32768 ? -32768 : si[g])};
    ul_ch[4 * g] = ul_ch[4 * g + 1] = ul_ch[4 * g + 2] = ul_ch[4 * g + 3] = v;
  }
  // revert the delay (c16mulShift(x, d, 8): int32 products >> 8, truncated to int16) and the noise |ls - ch|^2
  int k = 0;
  const simde__m256i neg_im = simde_mm256_set1_epi32(0xffff0001); // (re, im) -> (re, -im) as 16-bit multipliers
  const simde__m256i lo16 = simde_mm256_set1_epi32(0xffff);
  simde__m256i acc = simde_mm256_setzero_si256(); // 4 x uint64
  for (; k + 8 <= nre; k += 8) {
    const simde__m256i x = simde_mm256_loadu_si256((simde__m256i *)&ul_ch[k]);
    const simde__m256i d = simde_mm256_loadu_si256((simde__m256i *)&inv_delay_tab[k]);
    // re = x.r d.r - x.i d.i: madd(x, (d.r, -d.i)); im = x.r d.i + x.i d.r: madd(x, (d.i, d.r)); |d| <= 256
    const simde__m256i re = simde_mm256_srai_epi32(simde_mm256_madd_epi16(x, simde_mm256_sign_epi16(d, neg_im)), 8);
    const simde__m256i dsw = simde_mm256_shufflehi_epi16(simde_mm256_shufflelo_epi16(d, 0xb1), 0xb1);
    const simde__m256i im = simde_mm256_srai_epi32(simde_mm256_madd_epi16(x, dsw), 8);
    const simde__m256i ch = simde_mm256_or_si256(simde_mm256_and_si256(re, lo16), simde_mm256_slli_epi32(im, 16));
    simde_mm256_storeu_si256((simde__m256i *)&ul_ch[k], ch);
    const simde__m256i diff = simde_mm256_sub_epi16(simde_mm256_loadu_si256((simde__m256i *)&ls[k]), ch);
    const simde__m256i e2 = simde_mm256_madd_epi16(diff, diff); // r^2 + i^2 per RE, as uint32
    acc = simde_mm256_add_epi64(acc, simde_mm256_and_si256(e2, simde_mm256_set1_epi64x(0xffffffff)));
    acc = simde_mm256_add_epi64(acc, simde_mm256_srli_epi64(e2, 32));
  }
  uint64_t q[4];
  simde_mm256_storeu_si256((simde__m256i *)q, acc);
  uint64_t noise = q[0] + q[1] + q[2] + q[3];
  for (; k < nre; k++) {
    ul_ch[k] = c16mulShift(ul_ch[k], inv_delay_tab[k], 8);
    noise += c16amp2(c16sub(ls[k], ul_ch[k]));
  }
  *noise_amp2 += noise;
  *nest_count += 2 * npil;
}

static int pusch_chest_mode(void)
{
  // 0: fast interpolation (default), 1: per-pilot filter loop (OAI_CHEST_LEGACY=1), 2: both, compared (OAI_CHEST_CHECK=1)
  static int mode = -1;
  if (mode < 0)
    mode = getenv("OAI_CHEST_CHECK") ? 2 : getenv("OAI_CHEST_LEGACY") ? 1 : 0;
  return mode;
}

/* chest_freq values for the frequency-domain channel estimation */
enum { CHEST_FREQ_BOX_INTERP = 0, CHEST_FREQ_PRB_AVERAGE = 1, CHEST_FREQ_LMMSE_INTERP = 2 };

/* Frequency-domain LMMSE interpolation of DMRS type 1 (comb 2) estimates.
 * Pilot values p[m] at RE sp m + off2 / 2 (before delay compensation) are combined over the W_K nearest
 * pilots: h[k] = sum_i w_o[i] p[m0+i], with o = k - sp m0 the offset of RE k in its window; the
 * weights depend only on o (sp W_K filters per design). Pilot grids (sp, off2):
 * - DMRS type 1, single layer: per-pilot LS, (2, 0);
 * - DMRS type 1, multi-layer: pair averages of each CDM pilot pair (the frequency OCC despreading), (4, 2);
 * - DMRS type 2: pair averages of the two adjacent pilots, (6, 2 delta + 1), delta = 0, 2 or 4.
 * Channel correlation model:
 * exponential power delay profile starting at delay 0, R(df) = 1 / (1 + j 2 pi df tau).
 * Per call, tau and the noise-to-signal ratio are estimated from pilot correlations c(L):
 * tau from |c(4)|/|c(1)| (noise-free lags), then doubled (robust: over-estimating the delay
 * spread costs little, under-estimating it biases the estimate); s2 = (c(0) - S) / S with
 * S = |c(1)| sqrt(1 + (2 pi df1 tau)^2). Designs are quantised (tau grid x 1 dB SNR) and built at init. */
#define W_K 16
#define W_NTAU 13
#define W_NSNR 61 /* -10 .. 50 dB */
static const double w_tau_grid[W_NTAU] =
    {10e-9, 20e-9, 30e-9, 50e-9, 75e-9, 100e-9, 150e-9, 200e-9, 300e-9, 450e-9, 700e-9, 1000e-9, 1500e-9};
typedef struct {
  float r[W_K], i[W_K];
} wiener_row_t; /* taps of the filter for one output offset o */
/* The weights only depend on the distance between output RE and pilots, so the three DMRS type 2 grids share the
 * delta = 0 design: row o of grid delta is row o - delta of grid 0. That design is extended by W_PRE2 rows before
 * offset 0. */
#define W_NGRID 3
#define W_PRE2 4
static const int w_grids[W_NGRID][3] = {{2, 0, 0}, {4, 2, 0}, {6, 1, W_PRE2}}; /* sp, off2, rows before offset 0 */
static wiener_row_t *w_cache[W_NGRID][W_NTAU][W_NSNR]; /* [pilot grid][tau][SNR], offsets -pre .. sp W_K - 1 */
/* design index of the pilot grid (sp, off2), and the shift in RE of that grid relative to the design */
static int w_grid_index(int sp, int off2, int *shift)
{
  *shift = 0;
  if (sp == 2 && off2 == 0)
    return 0;
  if (sp == 4 && off2 == 2)
    return 1;
  if (sp == 6 && (off2 == 1 || off2 == 5 || off2 == 9)) {
    *shift = off2 / 2;
    return 2;
  }
  AssertFatal(false, "UL LMMSE chest: unsupported pilot grid %d/%d\n", sp, off2);
  return -1;
}

/* solve A x = b for complex A (n x n, row-major), Gaussian elimination with partial pivoting */
static void w_solve(int n, double ar[n][n], double ai[n][n], double br[n], double bi[n])
{
  for (int c = 0; c < n; c++) {
    int p = c;
    for (int r = c + 1; r < n; r++)
      if (ar[r][c] * ar[r][c] + ai[r][c] * ai[r][c] > ar[p][c] * ar[p][c] + ai[p][c] * ai[p][c])
        p = r;
    for (int j = 0; j < n; j++) {
      double t = ar[c][j];
      ar[c][j] = ar[p][j];
      ar[p][j] = t;
      t = ai[c][j];
      ai[c][j] = ai[p][j];
      ai[p][j] = t;
    }
    double t = br[c];
    br[c] = br[p];
    br[p] = t;
    t = bi[c];
    bi[c] = bi[p];
    bi[p] = t;
    const double d = ar[c][c] * ar[c][c] + ai[c][c] * ai[c][c];
    for (int r = c + 1; r < n; r++) {
      /* f = A[r][c] / A[c][c] */
      const double fr = (ar[r][c] * ar[c][c] + ai[r][c] * ai[c][c]) / d;
      const double fi = (ai[r][c] * ar[c][c] - ar[r][c] * ai[c][c]) / d;
      for (int j = c; j < n; j++) {
        ar[r][j] -= fr * ar[c][j] - fi * ai[c][j];
        ai[r][j] -= fr * ai[c][j] + fi * ar[c][j];
      }
      br[r] -= fr * br[c] - fi * bi[c];
      bi[r] -= fr * bi[c] + fi * br[c];
    }
  }
  for (int c = n - 1; c >= 0; c--) {
    double sr = br[c], si = bi[c];
    for (int j = c + 1; j < n; j++) {
      sr -= ar[c][j] * br[j] - ai[c][j] * bi[j];
      si -= ar[c][j] * bi[j] + ai[c][j] * br[j];
    }
    const double d = ar[c][c] * ar[c][c] + ai[c][c] * ai[c][c];
    br[c] = (sr * ar[c][c] + si * ai[c][c]) / d;
    bi[c] = (si * ar[c][c] - sr * ai[c][c]) / d;
  }
}

/* R(df) = 1 / (1 + j a), a = 2 pi df tau */
static inline void w_corr(double df, double tau, double *re, double *im)
{
  const double a = 2 * M_PI * df * tau;
  *re = 1.0 / (1.0 + a * a);
  *im = -a / (1.0 + a * a);
}

/* One design: the filters for offsets -pre .. sp W_K - 1 of the pilot grid (sp, off2), delay spread w_tau_grid[ti] and
 * SNR bin si; offset o is stored in f[pre + o]. */
static void w_design(wiener_row_t *f, int sp, int off2, int pre, int ti, int si, double scs)
{
  const double tau = w_tau_grid[ti], s2 = pow(10.0, -(si - 10) / 10.0);
  for (int o = -pre; o < sp * W_K; o++) {
    /* h_o = a (R + s2 I)^-1 p, a_i = R(f_o - f_i): solve (R + s2 I)^T x = a^T, h_o = x . p */
    double ar[W_K][W_K], ai[W_K][W_K], br[W_K], bi[W_K];
    for (int r = 0; r < W_K; r++) {
      for (int c = 0; c < W_K; c++) {
        double re, im;
        w_corr((sp * c - sp * r) * scs, tau, &re, &im); /* (R^T)[r][c] = R(f_c - f_r) */
        ar[r][c] = re + (r == c ? s2 : 0.0);
        ai[r][c] = im;
      }
      w_corr((o - sp * r - off2 / 2.0) * scs, tau, &br[r], &bi[r]);
    }
    w_solve(W_K, ar, ai, br, bi);
    for (int r = 0; r < W_K; r++) {
      f[pre + o].r[r] = br[r];
      f[pre + o].i[r] = bi[r];
    }
  }
}

static double w_cache_scs; /* subcarrier spacing of the designs, 0 until nr_ul_chest_lmmse_init() */

void nr_ul_chest_lmmse_init(int chest_freq, int subcarrier_spacing)
{
  if (chest_freq != CHEST_FREQ_LMMSE_INTERP)
    return;
  if (w_cache_scs != 0) {
    AssertFatal(w_cache_scs == subcarrier_spacing, "UL LMMSE chest: one subcarrier spacing per process\n");
    return;
  }
  for (int g = 0; g < W_NGRID; g++) {
    const int sp = w_grids[g][0], off2 = w_grids[g][1], pre = w_grids[g][2];
    for (int ti = 0; ti < W_NTAU; ti++)
      for (int si = 0; si < W_NSNR; si++) {
        wiener_row_t *f = malloc((pre + sp * W_K) * sizeof(*f));
        AssertFatal(f, "UL LMMSE chest: out of memory\n");
        w_design(f, sp, off2, pre, ti, si, subcarrier_spacing);
        w_cache[g][ti][si] = f;
      }
  }
  w_cache_scs = subcarrier_spacing;
}

/* All designs are built by nr_ul_chest_lmmse_init(): no allocation or lock in the slot processing. Returns the filter of
 * offset 0, valid for offsets 0 .. sp W_K - 1. */
static const wiener_row_t *w_get_filters(int sp, int off2, int ti, int si, double scs)
{
  AssertFatal(w_cache_scs == scs, "UL LMMSE chest: filters not built for SCS %.0f (nr_ul_chest_lmmse_init)\n", scs);
  int shift;
  const int g = w_grid_index(sp, off2, &shift);
  return w_cache[g][ti][si] + w_grids[g][2] - shift;
}

typedef struct {
  double t0; /* first-path delay, s */
  double t_mean; /* mean delay arg c(1) / (2 pi df1), s */
  double S; /* signal power per pilot, int16^2 units */
  int snr_db; /* estimated SNR per pilot, clamped to [-10, 50] dB */
} ul_chest_wiener_est_t;

/* p: npil pilot values (pilot m at RE sp m + off2 / 2, see the pilot grids above); writes sp * npil RE to out and
 * the channel parameters to est */
static void ul_chest_wiener_interp(const c16_t *p,
                                   const int npil,
                                   const int sp,
                                   const int off2,
                                   const double scs,
                                   c16_t *out,
                                   ul_chest_wiener_est_t *est)
{
  AssertFatal(npil >= W_K, "UL LMMSE chest: need at least %d pilots\n", W_K);
  /* exact integer sums: at most 2 * 3276 products of 2^30 */
  int64_t s0 = 0, s1r = 0, s1i = 0, s4r = 0, s4i = 0;
  for (int m = 0; m < npil; m++) {
    s0 += (int64_t)p[m].r * p[m].r + (int64_t)p[m].i * p[m].i;
    if (m + 1 < npil) {
      s1r += (int64_t)p[m].r * p[m + 1].r + (int64_t)p[m].i * p[m + 1].i;
      s1i += (int64_t)p[m].i * p[m + 1].r - (int64_t)p[m].r * p[m + 1].i;
    }
    if (m + 4 < npil) {
      s4r += (int64_t)p[m].r * p[m + 4].r + (int64_t)p[m].i * p[m + 4].i;
      s4i += (int64_t)p[m].i * p[m + 4].r - (int64_t)p[m].r * p[m + 4].i;
    }
  }
  const double c0 = (double)s0 / npil, c1r = s1r, c1i = s1i, c4r = s4r, c4i = s4i;
  const double a1 = sqrt(c1r * c1r + c1i * c1i) / (npil - 1), a4 = sqrt(c4r * c4r + c4i * c4i) / (npil - 4);
  const double d1 = sp * scs, d4 = 4 * sp * scs;
  double rho = a1 > 0 ? a4 / a1 : 1.0;
  rho = min(max(rho, 1e-3), 0.9999);
  const double x = rho * rho;
  const double den = x * (2 * M_PI * d4) * (2 * M_PI * d4) - (2 * M_PI * d1) * (2 * M_PI * d1);
  double tau = den > 0 ? sqrt(max((1 - x) / den, 0.0)) : 1e-6;
  tau = min(max(tau, 10e-9), 1e-6);
  const double S = a1 * sqrt(1 + (2 * M_PI * d1 * tau) * (2 * M_PI * d1 * tau));
  const double s2 = S > 0 ? max(c0 - S, 1e-5 * S) / S : 1.0;
  /* robust design: twice the estimated delay spread, nearest grid point in log scale */
  const double td = 2 * tau;
  int ti = 0;
  for (int t = 1; t < W_NTAU; t++)
    if (fabs(log(w_tau_grid[t] / td)) < fabs(log(w_tau_grid[ti] / td)))
      ti = t;
  int si = (int)lround(-10 * log10(s2)) + 10;
  si = min(max(si, 0), W_NSNR - 1);
  const wiener_row_t *f = w_get_filters(sp, off2, ti, si, scs);
  /* first-path delay: arg c(1) = 2 pi d1 t0 + atan(2 pi d1 tau) for the exponential model; remove
   * the corresponding phase ramp before filtering and restore it afterwards (timing offset) */
  const double t0 = (atan2(c1i, c1r) - atan(2 * M_PI * d1 * tau)) / (2 * M_PI * d1);
  *est = (ul_chest_wiener_est_t){.t0 = t0, .t_mean = atan2(c1i, c1r) / (2 * M_PI * d1), .S = S, .snr_db = si - 10};
  const double dphi = 2 * M_PI * scs * t0; /* phase step per RE */
  /* derotate the pilots, p * exp(+j 2 pi f_m t0) with f_m = (sp m + off2 / 2) scs: 8 pilots at a time, the phasor
   * of pilot m = 8 b + l is base_b * step_l, with base advanced by a double-precision recursion */
  float pr[npil], pim[npil];
  {
    float sr[8], si[8];
    for (int l = 0; l < 8; l++) {
      sr[l] = cos(sp * dphi * l);
      si[l] = sin(sp * dphi * l);
    }
    const simde__m256 vsr = simde_mm256_loadu_ps(sr), vsi = simde_mm256_loadu_ps(si);
    const simde__m256i deint = simde_mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    const double wr = cos(8 * sp * dphi), wi = sin(8 * sp * dphi);
    double br = cos(off2 * dphi / 2), bi = sin(off2 * dphi / 2);
    int m = 0;
    for (; m + 8 <= npil; m += 8) {
      /* phasor for the 8 pilots */
      const simde__m256 vbr = simde_mm256_set1_ps(br), vbi = simde_mm256_set1_ps(bi);
      const simde__m256 cr = simde_mm256_fmsub_ps(vbr, vsr, simde_mm256_mul_ps(vbi, vsi));
      const simde__m256 ci = simde_mm256_fmadd_ps(vbr, vsi, simde_mm256_mul_ps(vbi, vsr));
      /* 8 c16 pilots -> float, de-interleaved */
      const simde__m256i v = simde_mm256_loadu_si256((const simde__m256i *)(p + m));
      const simde__m256 lo =
          simde_mm256_permutevar8x32_ps(simde_mm256_cvtepi32_ps(simde_mm256_cvtepi16_epi32(simde_mm256_castsi256_si128(v))), deint);
      const simde__m256 hi =
          simde_mm256_permutevar8x32_ps(simde_mm256_cvtepi32_ps(simde_mm256_cvtepi16_epi32(simde_mm256_extracti128_si256(v, 1))),
                                        deint);
      const simde__m256 xr = simde_mm256_permute2f128_ps(lo, hi, 0x20), xi = simde_mm256_permute2f128_ps(lo, hi, 0x31);
      simde_mm256_storeu_ps(pr + m, simde_mm256_fmsub_ps(xr, cr, simde_mm256_mul_ps(xi, ci)));
      simde_mm256_storeu_ps(pim + m, simde_mm256_fmadd_ps(xr, ci, simde_mm256_mul_ps(xi, cr)));
      const double t = br * wr - bi * wi;
      bi = br * wi + bi * wr;
      br = t;
    }
    for (; m < npil; m++) {
      const double cr = br * sr[m % 8] - bi * si[m % 8], ci = br * si[m % 8] + bi * sr[m % 8];
      pr[m] = p[m].r * cr - p[m].i * ci;
      pim[m] = p[m].r * ci + p[m].i * cr;
    }
  }
  const int nre = sp * npil;
  float yr[nre], yi[nre];
  /* interior RE: k = sp j + r (offset o = sp W_K / 2 + r) with window m0 = j - W_K/2, for j in
   * [W_K/2, npil - W_K/2]; consecutive j share the filter and slide the window by one pilot, so 8 outputs
   * are computed at once with one broadcast weight per tap */
  const int j0 = W_K / 2, j1 = npil - W_K / 2; /* inclusive */
  int j = j0;
  for (; j + 7 <= j1; j += 8) {
    for (int r = 0; r < sp; r++) {
      const wiener_row_t *w = &f[sp * W_K / 2 + r];
      simde__m256 er = simde_mm256_setzero_ps(), ei = er;
      for (int i = 0; i < W_K; i++) {
        const simde__m256 vr = simde_mm256_loadu_ps(pr + j - W_K / 2 + i), vi = simde_mm256_loadu_ps(pim + j - W_K / 2 + i);
        const simde__m256 fr = simde_mm256_set1_ps(w->r[i]), fi = simde_mm256_set1_ps(w->i[i]);
        er = simde_mm256_fnmadd_ps(fi, vi, simde_mm256_fmadd_ps(fr, vr, er));
        ei = simde_mm256_fmadd_ps(fi, vr, simde_mm256_fmadd_ps(fr, vi, ei));
      }
      float t_r[8], t_i[8];
      simde_mm256_storeu_ps(t_r, er);
      simde_mm256_storeu_ps(t_i, ei);
      for (int l = 0; l < 8; l++) {
        yr[sp * (j + l) + r] = t_r[l];
        yi[sp * (j + l) + r] = t_i[l];
      }
    }
  }
  /* remaining interior RE and the band edges: scalar */
  for (int k = 0; k < nre; k++) {
    if (k / sp >= j0 && k / sp < j)
      continue; /* done above */
    const int m0 = min(max(k / sp - W_K / 2, 0), npil - W_K);
    const wiener_row_t *w = &f[k - sp * m0];
    float accr = 0, acci = 0;
    for (int i = 0; i < W_K; i++) {
      accr += w->r[i] * pr[m0 + i] - w->i[i] * pim[m0 + i];
      acci += w->r[i] * pim[m0 + i] + w->i[i] * pr[m0 + i];
    }
    yr[k] = accr;
    yi[k] = acci;
  }
  /* restore the phase ramp, y * exp(-j 2 pi f_k t0), 8 RE at a time as above, then round to nearest,
   * saturate to int16 and interleave to c16_t */
  {
    float sr[8], si[8];
    for (int l = 0; l < 8; l++) {
      sr[l] = cos(dphi * l);
      si[l] = -sin(dphi * l);
    }
    const simde__m256 vsr = simde_mm256_loadu_ps(sr), vsi = simde_mm256_loadu_ps(si);
    const double wr = cos(8 * dphi), wi = -sin(8 * dphi);
    double br = 1.0, bi = 0.0;
    int k = 0;
    for (; k + 8 <= nre; k += 8) {
      const simde__m256 vbr = simde_mm256_set1_ps(br), vbi = simde_mm256_set1_ps(bi);
      const simde__m256 cr = simde_mm256_fmsub_ps(vbr, vsr, simde_mm256_mul_ps(vbi, vsi));
      const simde__m256 ci = simde_mm256_fmadd_ps(vbr, vsi, simde_mm256_mul_ps(vbi, vsr));
      const simde__m256 vr = simde_mm256_loadu_ps(yr + k), vi = simde_mm256_loadu_ps(yi + k);
      const simde__m256i zr = simde_mm256_cvtps_epi32(simde_mm256_fmsub_ps(vr, cr, simde_mm256_mul_ps(vi, ci)));
      const simde__m256i zi = simde_mm256_cvtps_epi32(simde_mm256_fmadd_ps(vr, ci, simde_mm256_mul_ps(vi, cr)));
      /* (r, i) pairs of RE k..k+3 in the low lane and k+4..k+7 in the high lane, saturated to int16 */
      const simde__m256i lo = simde_mm256_unpacklo_epi32(zr, zi), hi = simde_mm256_unpackhi_epi32(zr, zi);
      simde_mm256_storeu_si256((simde__m256i *)(out + k), simde_mm256_packs_epi32(lo, hi));
      const double t = br * wr - bi * wi;
      bi = br * wi + bi * wr;
      br = t;
    }
    for (; k < nre; k++) {
      const double cr = br * sr[k % 8] - bi * si[k % 8], ci = br * si[k % 8] + bi * sr[k % 8];
      const double zr = yr[k] * cr - yi[k] * ci, zi = yr[k] * ci + yi[k] * cr;
      out[k] = (c16_t){.r = (int16_t)max(-32768, min(32767, lrint(zr))), .i = (int16_t)max(-32768, min(32767, lrint(zi)))};
    }
  }
}

/* Timing estimate from pilot values on a comb of spacing sp: per-pilot LS of a single layer (sp = 2), or pair averages
 * for multi-layer (sp = 4). The N-point impulse response of the comb (zeros in between) is periodic in M = N / sp samples
 * and equals, up to a factor 1 / sqrt(sp), the M-point IDFT of the packed pilot values, which is computed instead: the
 * peak search covers delays in [-N / (2 sp), N / (2 sp)). For sp = 2 this covers the TA range (+-31 steps of 16 N / 2048
 * samples) without the images that the 4-RE hold of the box LS creates; for multi-layer the per-pilot values cannot be
 * used (the other port of the CDM group appears with an alternating sign, i.e. as a second peak N / 4 away), and sp = 4
 * gives the same +-N / 8 range as the box LS. Same peak criterion as nr_est_delay(). The input is shifted right so that
 * the unitary fixed-point IDFT cannot overflow at high input level: each output component is bounded by
 * sum(|re| + |im|) / sqrt(M) of the input (reached for phase-aligned input) and the shift adds at most 1 per component,
 * so s is the smallest shift with (sum >> s) + 2M <= 32767 sqrt(M) (same bound as the issue #691 fix). The argmax and
 * the peak-to-mean ratio are scale-invariant; delay_max_val is reported at the unshifted scale of an N-point IDFT (the
 * unit of nr_est_delay()) for the cross-antenna comparison. The M samples of the impulse response are written to t. */
static void ul_chest_pilot_delay(const c16_t *pls, const int npil, const int sp, const int N, delay_t *delay, c16_t *t)
{
  const int M = N / sp, n = min(npil, M);
  int64_t l1 = 0;
  for (int m = 0; m < n; m++)
    l1 += abs(pls[m].r) + abs(pls[m].i);
  int sh = 0;
  while (sh < 16) {
    const int64_t bound = (l1 >> sh) + 2 * (int64_t)M;
    if (bound * bound <= 32767LL * 32767 * M)
      break;
    sh++;
  }
  c16_t in[M] __attribute__((aligned(32)));
  for (int m = 0; m < n; m++)
    in[m] = (c16_t){.r = pls[m].r >> sh, .i = pls[m].i >> sh};
  memset(in + n, 0, (M - n) * sizeof(*in));
  idft(get_idft(M), (int16_t *)in, (int16_t *)t, 1);
  int max_pos = delay->delay_max_pos;
  int64_t max_val = ((int64_t)delay->delay_max_val * sp) >> (2 * sh);
  uint64_t mean_val = 0;
  for (int i = 0; i < M; i++) {
    const int v = c16amp2(t[i]) >> 1;
    mean_val += v;
    if (v > max_val) {
      max_pos = i;
      max_val = v;
    }
  }
  mean_val /= M;
  if (max_pos >= M / 2)
    max_pos -= M;
  delay->delay_max_pos = max_pos;
  delay->delay_max_val = (int)min((max_val << (2 * sh)) / sp, (int64_t)INT_MAX);
  delay->valid = mean_val > 0 && (uint64_t)max_val / mean_val > 15; /* PEAK_DETECT_THRESHOLD of nr_est_delay() */
  delay->est_delay = delay->valid ? max_pos : 0;
}

static void nr_pusch_antenna_processing(void *arg)
{
  puschAntennaProc_t *rdata = (puschAntennaProc_t *)arg;
  unsigned char Ns = rdata->Ns;
  int nl = rdata->nl;
  unsigned short p = rdata->p;
  unsigned char symbol = rdata->symbol;
  int aarx = rdata->aarx;
  int numAntennas = rdata->numAntennas;
  unsigned short bwp_start_subcarrier = rdata->bwp_start_subcarrier;
  const nfapi_nr_pusch_pdu_t *pusch_pdu = rdata->pusch_pdu;
  int *max_ch = rdata->max_ch;
  c16_t *pilot = rdata->pilot;
  uint64_t noise_amp2 = *(rdata->noise_amp2);
  int nest_count = *(rdata->nest_count);
  delay_t *delay = rdata->delay;

  const int chest_freq = rdata->chest_freq;
  c16_t(*ul_ch_estimates)[rdata->ch_est_size] = rdata->ul_ch_estimates;
  NR_DL_FRAME_PARMS *frame_parms = rdata->frame_parms;
  const int symbolSize = frame_parms->ofdm_symbol_size;
  const int slot_offset = (Ns % RU_RX_SLOT_DEPTH) * frame_parms->symbols_per_slot * symbolSize;
  const int delta = get_delta(p, pusch_pdu->dmrs_config_type);
  const int symbol_offset = symbolSize * symbol;
  const int k0 = bwp_start_subcarrier;
  const int nb_rb_pusch = pusch_pdu->rb_size;
  const int aa_start = rdata->ant_port_start;
  const uint8_t num_sp_streams = rdata->pusch_pdu->param_v4.numSpatialStreamIndices;
  for (int antenna = aarx; antenna < aarx + numAntennas; antenna++) {
    c16_t ul_ls_est[symbolSize] __attribute__((aligned(32)));
    memset(ul_ls_est, 0, sizeof(c16_t) * symbolSize);
    c16_t *rxdataF = (c16_t *)&rdata->rxdataF[aa_start + antenna][symbol_offset + slot_offset];
    c16_t *ul_ch = ul_ch_estimates[nl * num_sp_streams + antenna] + rdata->ch_row_offset;
    memset(ul_ch, 0, sizeof(*ul_ch) * symbolSize);

    LOG_D(PHY,
          "symbol_offset %d, slot_offset %d, OFDM size %d, Ns = %d, k0 = %d, symbol %d\n",
          symbol_offset,
          slot_offset,
          symbolSize,
          Ns,
          k0,
          symbol);

#ifdef DEBUG_PUSCH
    LOG_I(PHY, "symbol_offset %d, delta %d\n", symbol_offset, delta);
    LOG_I(PHY, "ch est pilot, N_RB_UL %d\n", frame_parms->N_RB_UL);
    LOG_I(PHY,
          "bwp_start_subcarrier %d, k0 %d, first_carrier %d, nb_rb_pusch %d\n",
          bwp_start_subcarrier,
          k0,
          frame_parms->first_carrier_offset,
          nb_rb_pusch);
    LOG_I(PHY, "ul_ch addr %p \n", ul_ch);
#endif

    if (pusch_pdu->dmrs_config_type == pusch_dmrs_type1
        && (chest_freq == CHEST_FREQ_BOX_INTERP || chest_freq == CHEST_FREQ_LMMSE_INTERP)) {
      c16_t *pil = pilot;
      int re_offset = k0;
      LOG_D(PHY, "PUSCH estimation DMRS type 1, Freq-domain interpolation");
      int pilot_cnt = 0;
#if T_TRACER
      int ch_est_cnt = 0; // To trace channel coefficients
#endif

      /* LMMSE needs W_K pilot values: per-pilot LS for a single layer (3 PRB), pair averages for multi-layer
       * (6 PRB). Smaller allocations use the box interpolation. */
      const int single_layer = pusch_pdu->nrOfLayers == 1;
      const int lmmse = chest_freq == CHEST_FREQ_LMMSE_INTERP && (single_layer ? 6 : 3) * nb_rb_pusch >= W_K;
      /* single-layer LMMSE input: per-pilot LS v = (pilot rx) >> 15, saturated to int16, no delay compensation (the
       * LMMSE interpolator estimates the first-path delay); the box LS is derived from it, (pilot rx) >> 16 = v >> 1 */
      c16_t pls[lmmse && single_layer ? 6 * nb_rb_pusch : 1];
      // the vectorised LS has no per-pilot output: LMMSE takes the scalar loop
      const int ls_mode = lmmse ? 1 : pusch_chest_mode();
#if T_TRACER
      const int ls_traced = T_ACTIVE(T_GNB_PHY_UL_FD_CHAN_EST_DMRS_POS) && nl == 0;
#else
      const int ls_traced = 0;
#endif
      c16_t fast_ls[ls_mode == 2 ? symbolSize : 1] __attribute__((aligned(32)));
      int fast_max = *max_ch;
      if (ls_mode == 0 && !ls_traced) {
        pusch_type1_ls_fast(pilot, &rxdataF[k0 + delta], 3 * nb_rb_pusch, ul_ls_est, max_ch);
      } else {
        if (ls_mode == 2)
          pusch_type1_ls_fast(pilot, &rxdataF[k0 + delta], 3 * nb_rb_pusch, fast_ls, &fast_max);
        for (int n = 0; n < 3 * nb_rb_pusch; n++) {
          // LS estimation
          c32_t ch = {0};

          for (int k_line = 0; k_line <= 1; k_line++) {
            re_offset = (k0 + (n << 2) + (k_line << 1) + delta);
            if (lmmse && single_layer) {
              const c32_t v = c32x16mulShift(*pil, rxdataF[re_offset], 15);
              pls[2 * n + k_line] = (c16_t){.r = (int16_t)max(-32768, min(32767, v.r)), .i = (int16_t)max(-32768, min(32767, v.i))};
              ch.r += v.r >> 1;
              ch.i += v.i >> 1;
            } else {
              ch = c32x16maddShift(*pil, rxdataF[re_offset], ch, 16);
            }
            pil++;
          }

          c16_t ch16 = {.r = (int16_t)ch.r, .i = (int16_t)ch.i};
          *max_ch = max(*max_ch, max(abs(ch.r), abs(ch.i)));
          for (int k = pilot_cnt << 1; k < (pilot_cnt << 1) + 4; k++) {
            ul_ls_est[k] = ch16;
          }
  //------------------Write channel parameters to Memory  for data recording ------------------//
  #if T_TRACER
          if (T_ACTIVE(T_GNB_PHY_UL_FD_CHAN_EST_DMRS_POS) && nl == 0) {
            // Trace channel coefficients
            c16_t *pusch_ch_est_dmrs_pos_slot_mem = rdata->pusch_ch_est_dmrs_pos_slot_mem;
            int dmrs_symbol_start_idx = rdata->dmrs_symbol_start_idx;
            pusch_ch_est_dmrs_pos_slot_mem[dmrs_symbol_start_idx + delta + ch_est_cnt] = ch16; // 0, 2, 4, 6, 8, location of REs
            pusch_ch_est_dmrs_pos_slot_mem[dmrs_symbol_start_idx + delta + ch_est_cnt + 2] = ch16; // 0, 2, 4, 6, 8, location of REs
          }
          ch_est_cnt += 4;
  #endif
          pilot_cnt += 2;
        }
        if (ls_mode == 2) {
          static int ls_checked, ls_bad;
          const int bad = memcmp(fast_ls, ul_ls_est, sizeof(c16_t) * 12 * nb_rb_pusch) != 0 || fast_max != *max_ch;
          ls_bad += bad;
          if (bad || (++ls_checked % 1000) == 0 || ls_checked == 1)
            fprintf(stderr, "OAI_CHEST_CHECK LS: %d checked, %d mismatched%s\n", ls_checked, ls_bad, bad ? " (MISMATCH)" : "");
        }
      }
      if (lmmse) {
        const int N = frame_parms->ofdm_symbol_size;
        c16_t t_delay[N / 2] __attribute__((aligned(32))); /* impulse response of the IDFT peak search, for the scope */
        int t_delay_len = 0;
        if (lmmse && single_layer) {
          const int npil = 6 * nb_rb_pusch;
          ul_chest_wiener_est_t est;
          ul_chest_wiener_interp(pls, npil, 2, 0, frame_parms->subcarrier_spacing, ul_ch, &est);
          /* timing advance input: the mean delay arg c(1) / (2 pi df1), in samples at the OFDM symbol rate, replaces
           * the IDFT peak search when it is accurate: c(1) has an SNR of about (npil - 1) snr^2 / (2 snr + 1), which
           * gives the delay standard deviation below; it must stay under a quarter of a TA step (16 N / 2048 samples)
           * and the allocation must have 10 PRB or more. Otherwise, at low SNR or on small allocations, the IDFT peak
           * search on the per-pilot LS is used. */
          const double snr = pow(10.0, est.snr_db / 10.0);
          const double snr_c1 = (npil - 1) * snr * snr / (2 * snr + 1);
          const double delay_std = N / (4 * M_PI * sqrt(2 * snr_c1));
          if (npil >= 60 && delay_std <= (16.0 * N / 2048) / 4 && est.S > 0) {
            /* cross-antenna metric in the unit of the IDFT peak search: peak power of a single path of power S */
            const double val = (double)npil * npil * est.S / (2 * N);
            if (val > delay->delay_max_val) {
              const int d = lrint(est.t_mean * N * frame_parms->subcarrier_spacing);
              delay->delay_max_pos = d;
              delay->delay_max_val = (int)min(val, (double)INT_MAX);
              delay->valid = true;
              delay->est_delay = d;
            }
          } else {
            ul_chest_pilot_delay(pls, npil, 2, N, delay, t_delay);
            t_delay_len = N / 2;
          }
        } else if (lmmse) {
          /* multi-layer: the pair average of each CDM pilot pair (box LS, held in ul_ls_est[4n .. 4n+3]) is the frequency
           * OCC despreading, so the LMMSE runs on these values, one per 4 RE at RE 4n + 1. Timing advance: IDFT peak
           * search on the same values (see ul_chest_pilot_delay). */
          const int nq = 3 * nb_rb_pusch;
          c16_t q[nq];
          for (int n = 0; n < nq; n++)
            q[n] = ul_ls_est[4 * n];
          ul_chest_wiener_est_t est;
          ul_chest_wiener_interp(q, nq, 4, 2, frame_parms->subcarrier_spacing, ul_ch, &est);
          ul_chest_pilot_delay(q, nq, 4, N, delay, t_delay);
          t_delay_len = N / 4;
        }
        if (t_delay_len && rdata->scope && antenna == 0) {
          metadata mt = {.slot = -1, .frame = -1};
          scopeData_t *tmp = rdata->scope;
          tmp->copyData(tmp, gNBulDelay, t_delay, sizeof(c16_t), 1, t_delay_len, 0, &mt);
        }

        // LMMSE output has no delay to revert: noise as the residual of the box LS
        ul_ch = ul_ch_estimates[nl * num_sp_streams + antenna] + rdata->ch_row_offset;
        for (int k = 0; k < 12 * nb_rb_pusch; k++)
          noise_amp2 += c16amp2(c16sub(ul_ls_est[k], ul_ch[k]));
        nest_count += 12 * nb_rb_pusch;
      } else {
        c16_t ch_estimates_time[frame_parms->ofdm_symbol_size] __attribute__((aligned(32)));
        nr_est_delay(frame_parms->ofdm_symbol_size, ul_ls_est, ch_estimates_time, delay);
        if (rdata->scope && antenna == 0) {
          metadata mt = {.slot = -1, .frame = -1};
          scopeData_t *tmp = rdata->scope;
          tmp->copyData(tmp, gNBulDelay, ch_estimates_time, sizeof(c16_t), 1, frame_parms->ofdm_symbol_size, 0, &mt);
        }
        int delay_idx = get_delay_idx(delay->est_delay, MAX_DELAY_COMP);
        c16_t *ul_delay_table = frame_parms->delay_table[delay_idx];

#ifdef DEBUG_PUSCH
        printf("Estimated delay = %i\n", delay->est_delay >> 1);
#endif

        const int chest_mode = pusch_chest_mode();
        c16_t *ul_ch0 = ul_ch;
        c16_t fast_ch[chest_mode == 2 ? symbolSize : 1] __attribute__((aligned(32)));
        uint64_t fast_noise = 0;
        int fast_nest = 0;
        const uint64_t noise_before = noise_amp2;
        const int nest_before = nest_count;
        if (chest_mode != 1) {
          int inv_idx = get_delay_idx(-delay->est_delay, MAX_DELAY_COMP);
          c16_t *dst = chest_mode == 2 ? fast_ch : ul_ch;
          if (chest_mode == 2)
            memset(fast_ch, 0, sizeof(c16_t) * symbolSize);
          pusch_type1_interp_fast(ul_ls_est, ul_delay_table, frame_parms->delay_table[inv_idx], nb_rb_pusch, dst, &fast_noise, &fast_nest);
        }
        if (chest_mode != 0) {
          pilot_cnt = 0;
          for (int n = 0; n < 3 * nb_rb_pusch; n++) {
            // Channel interpolation
            for (int k_line = 0; k_line <= 1; k_line++) {
              // Apply delay
              int k = pilot_cnt << 1;
              c16_t ch16 = c16mulShift(ul_ls_est[k], ul_delay_table[k], 8);

    #ifdef DEBUG_PUSCH
              re_offset = (k0 + (n << 2) + (k_line << 1)) % symbolSize;
              c16_t *rxF = &rxdataF[re_offset];
              printf("pilot %4d: pil -> (%6d,%6d), rxF -> (%4d,%4d), ch -> (%4d,%4d)\n",
                     pilot_cnt,
                     pil->r,
                     pil->i,
                     rxF->r,
                     rxF->i,
                     ch.r,
                     ch.i);
    #endif

              if (pilot_cnt == 0) {
                c16multaddVectRealComplex(filt16_ul_p0, &ch16, ul_ch, 16);
              } else if (pilot_cnt == 1 || pilot_cnt == 2) {
                c16multaddVectRealComplex(filt16_ul_p1p2, &ch16, ul_ch, 16);
              } else if (pilot_cnt == (6 * nb_rb_pusch - 1)) {
                c16multaddVectRealComplex(filt16_ul_last, &ch16, ul_ch, 16);
              } else {
                c16multaddVectRealComplex(filt16_ul_middle, &ch16, ul_ch, 16);
                if (pilot_cnt % 2 == 0) {
                  ul_ch += 4;
                }
              }

              pilot_cnt++;
            }
          }

          // Revert delay
          pilot_cnt = 0;
          ul_ch = ul_ch_estimates[nl * num_sp_streams + antenna] + rdata->ch_row_offset;
          int inv_delay_idx = get_delay_idx(-delay->est_delay, MAX_DELAY_COMP);
          c16_t *ul_inv_delay_table = frame_parms->delay_table[inv_delay_idx];
          for (int n = 0; n < 3 * nb_rb_pusch; n++) {
            for (int k_line = 0; k_line <= 1; k_line++) {
              int k = pilot_cnt << 1;
              ul_ch[k] = c16mulShift(ul_ch[k], ul_inv_delay_table[k], 8);
              ul_ch[k + 1] = c16mulShift(ul_ch[k + 1], ul_inv_delay_table[k + 1], 8);
              noise_amp2 += c16amp2(c16sub(ul_ls_est[k], ul_ch[k]));
              noise_amp2 += c16amp2(c16sub(ul_ls_est[k + 1], ul_ch[k + 1]));

    #ifdef DEBUG_PUSCH
              re_offset = (k0 + (n << 2) + (k_line << 1)) % symbolSize;
              printf("ch -> (%4d,%4d), ch_inter -> (%4d,%4d)\n", ul_ls_est[k].r, ul_ls_est[k].i, ul_ch[k].r, ul_ch[k].i);
    #endif
              pilot_cnt++;
              nest_count += 2;
            }
          }
        }
        if (chest_mode == 0) {
          noise_amp2 += fast_noise;
          nest_count += fast_nest;
        } else if (chest_mode == 2) {
          // compare with the filter loop over the REs it writes (12 nb_rb + 4) and the noise / estimate counters
          static int checked, mismatched;
          const uint64_t loop_noise = noise_amp2 - noise_before;
          const int bad = memcmp(fast_ch, ul_ch0, sizeof(c16_t) * (12 * nb_rb_pusch + 4)) != 0 || fast_noise != loop_noise
                          || fast_nest != nest_count - nest_before;
          mismatched += bad;
          if (bad || (++checked % 1000) == 0 || checked == 1)
            fprintf(stderr, "OAI_CHEST_CHECK: %d checked, %d mismatched%s\n", checked, mismatched, bad ? " (MISMATCH)" : "");
        }
      }
      // Align the channel estimates for the delta shift
      if (delta != 0) {
        c16_t *ul_ch_base = ul_ch_estimates[nl * num_sp_streams + antenna] + rdata->ch_row_offset;
        memmove(&ul_ch_base[delta], ul_ch_base, (nb_rb_pusch * 12 - delta) * sizeof(c16_t));
        for (int d = 0; d < delta; d++)
          ul_ch_base[d] = ul_ch_base[delta];
      }

    } else if (pusch_pdu->dmrs_config_type == pusch_dmrs_type2
               && (chest_freq == CHEST_FREQ_BOX_INTERP
                   || chest_freq == CHEST_FREQ_LMMSE_INTERP)) { // pusch_dmrs_type2  |p_r,p_l,d,d,d,d,p_r,p_l,d,d,d,d|
      LOG_D(PHY, "PUSCH estimation DMRS type 2, Freq-domain interpolation\n");
      c16_t *pil = pilot;
      c16_t *rx = &rxdataF[delta];
      for (int n = 0; n < nb_rb_pusch * NR_NB_SC_PER_RB; n += 6) {
        c16_t ch0 = c16mulShift(*pil, rx[(k0 + n) % symbolSize], 15);
        pil++;
        c16_t ch1 = c16mulShift(*pil, rx[(k0 + n + 1) % symbolSize], 15);
        pil++;
        c16_t ch = c16addShift(ch0, ch1, 1);
        *max_ch = max(*max_ch, max(abs(ch.r), abs(ch.i)));
        multadd_real_four_symbols_vector_complex_scalar(filt8_rep4, ch, &ul_ls_est[n]);
        ul_ls_est[n + 4] = ch;
        ul_ls_est[n + 5] = ch;
        noise_amp2 += c16amp2(c16sub(ch0, ch));
        nest_count += 1;
      }

      // Delay compensation
      c16_t ch_estimates_time[frame_parms->ofdm_symbol_size] __attribute__((aligned(32)));
      nr_est_delay(frame_parms->ofdm_symbol_size, ul_ls_est, ch_estimates_time, delay);
      if (rdata->scope && antenna == 0) {
        metadata mt = {.slot = -1, .frame = -1};
        scopeData_t *tmp = rdata->scope;
        tmp->copyData(tmp, gNBulDelay, ch_estimates_time, sizeof(c16_t), 1, frame_parms->ofdm_symbol_size, 0, &mt);
      }
      if (chest_freq == CHEST_FREQ_LMMSE_INTERP && 2 * nb_rb_pusch >= W_K) {
        /* LMMSE on the pair averages (the frequency OCC despreading, held in ul_ls_est[6n + 4]), one per 6 RE at RE
         * 6n + delta + 1/2 relative to k0; the output is aligned on k0. The timing advance keeps the IDFT peak
         * search above (arg c(1) at a 6-RE spacing wraps inside the TA range). */
        const int nq = 2 * nb_rb_pusch;
        c16_t q[nq];
        for (int n = 0; n < nq; n++)
          q[n] = ul_ls_est[6 * n + 4];
        ul_chest_wiener_est_t est;
        ul_chest_wiener_interp(q, nq, 6, 2 * delta + 1, frame_parms->subcarrier_spacing, ul_ch, &est);
      } else {
        int delay_idx = get_delay_idx(-delay->est_delay, MAX_DELAY_COMP);
        c16_t *ul_delay_table = frame_parms->delay_table[delay_idx];
        for (int n = 0; n < nb_rb_pusch * NR_NB_SC_PER_RB; n++) {
          ul_ch[n] = c16mulShift(ul_ls_est[n], ul_delay_table[n % 6], 8);
        }
      }

    }
    // this is case without frequency-domain linear interpolation, just take average of LS channel estimates of 6 DMRS REs and use a
    // common value for the whole PRB
    else if (pusch_pdu->dmrs_config_type == pusch_dmrs_type1) {
      LOG_D(PHY, "PUSCH estimation DMRS type 1, no Freq-domain interpolation\n");
      c16_t *rxF = &rxdataF[delta];
      int pil_offset = 0;
      int re_offset = k0;
      c16_t ch;

      // First PRB
      ch = c32x16cumulVectVectWithSteps(pilot, &pil_offset, 1, rxF, &re_offset, 2, symbolSize, 6);

#if NO_INTERP
      for (c16_t *end = ul_ch + 12; ul_ch < end; ul_ch++)
        *ul_ch = ch;
#else
      c16multaddVectRealComplex(filt8_avlip0, &ch, ul_ch, 8);
      ul_ch += 8;
      c16multaddVectRealComplex(filt8_avlip1, &ch, ul_ch, 8);
      ul_ch += 8;
      c16multaddVectRealComplex(filt8_avlip2, &ch, ul_ch, 8);
      ul_ch -= 12;
#endif

      for (int pilot_cnt = 6; pilot_cnt < 6 * (nb_rb_pusch - 1); pilot_cnt += 6) {
        ch = c32x16cumulVectVectWithSteps(pilot, &pil_offset, 1, rxF, &re_offset, 2, symbolSize, 6);
        *max_ch = max(*max_ch, max(abs(ch.r), abs(ch.i)));

#if NO_INTERP
        for (c16_t *end = ul_ch + 12; ul_ch < end; ul_ch++)
          *ul_ch = ch;
#else
        ul_ch[3].r += (ch.r * 1365) >> 15; // 1/12*16384
        ul_ch[3].i += (ch.i * 1365) >> 15; // 1/12*16384

        ul_ch += 4;
        c16multaddVectRealComplex(filt8_avlip3, &ch, ul_ch, 8);
        ul_ch += 8;
        c16multaddVectRealComplex(filt8_avlip4, &ch, ul_ch, 8);
        ul_ch += 8;
        c16multaddVectRealComplex(filt8_avlip5, &ch, ul_ch, 8);
        ul_ch -= 8;
#endif
      }
      // Last PRB
      ch = c32x16cumulVectVectWithSteps(pilot, &pil_offset, 1, rxF, &re_offset, 2, symbolSize, 6);

#if NO_INTERP
      for (c16_t *end = ul_ch + 12; ul_ch < end; ul_ch++)
        *ul_ch = ch;
#else
      ul_ch[3].r += (ch.r * 1365) >> 15; // 1/12*16384
      ul_ch[3].i += (ch.i * 1365) >> 15; // 1/12*16384

      ul_ch += 4;
      c16multaddVectRealComplex(filt8_avlip3, &ch, ul_ch, 8);
      ul_ch += 8;
      c16multaddVectRealComplex(filt8_avlip6, &ch, ul_ch, 8);
#endif
    } else { // this is case without frequency-domain linear interpolation, just take average of LS channel estimates of 4 DMRS REs
             // and use a common value for the whole PRB
      LOG_D(PHY, "PUSCH estimation DMRS type 2, no Freq-domain interpolation");
      c16_t *pil = pilot;
      int re_offset = (k0 + delta) % symbolSize;
      c32_t ch0 = {0};
      // First PRB
      ch0 = c32x16mulShift(*pil, rxdataF[re_offset], 15);
      pil++;
      re_offset = (re_offset + 1) % symbolSize;
      ch0 = c32x16maddShift(*pil, rxdataF[re_offset], ch0, 15);
      pil++;
      re_offset = (re_offset + 5) % symbolSize;
      ch0 = c32x16maddShift(*pil, rxdataF[re_offset], ch0, 15);
      re_offset = (re_offset + 1) % symbolSize;
      ch0 = c32x16maddShift(*pil, rxdataF[re_offset], ch0, 15);
      pil++;
      re_offset = (re_offset + 5) % symbolSize;

      c16_t ch = c16x32div(ch0, 4);
#if NO_INTERP
      for (c16_t *end = ul_ch + 12; ul_ch < end; ul_ch++)
        *ul_ch = ch;
#else
      c16multaddVectRealComplex(filt8_avlip0, &ch, ul_ch, 8);
      ul_ch += 8;
      c16multaddVectRealComplex(filt8_avlip1, &ch, ul_ch, 8);
      ul_ch += 8;
      c16multaddVectRealComplex(filt8_avlip2, &ch, ul_ch, 8);
      ul_ch -= 12;
#endif

      for (int pilot_cnt = 4; pilot_cnt < 4 * (nb_rb_pusch - 1); pilot_cnt += 4) {
        c32_t ch0;
        ch0 = c32x16mulShift(*pil, rxdataF[re_offset], 15);
        pil++;
        re_offset = (re_offset + 1) % symbolSize;

        ch0 = c32x16maddShift(*pil, rxdataF[re_offset], ch0, 15);
        pil++;
        re_offset = (re_offset + 5) % symbolSize;

        ch0 = c32x16maddShift(*pil, rxdataF[re_offset], ch0, 15);
        pil++;
        re_offset = (re_offset + 1) % symbolSize;

        ch0 = c32x16maddShift(*pil, rxdataF[re_offset], ch0, 15);
        pil++;
        re_offset = (re_offset + 5) % symbolSize;

        ch = c16x32div(ch0, 4);
        *max_ch = max(*max_ch, max(abs(ch.r), abs(ch.i)));

#if NO_INTERP
        for (c16_t *end = ul_ch + 12; ul_ch < end; ul_ch++)
          *ul_ch = ch;
#else
        ul_ch[3] = c16maddShift(ch, (c16_t){1365, 1365}, (c16_t){0, 0}, 15); // 1365 = 1/12*16384 (full range is +/- 32768)
        ul_ch += 4;
        c16multaddVectRealComplex(filt8_avlip3, &ch, ul_ch, 8);
        ul_ch += 8;
        c16multaddVectRealComplex(filt8_avlip4, &ch, ul_ch, 8);
        ul_ch += 8;
        c16multaddVectRealComplex(filt8_avlip5, &ch, ul_ch, 8);
        ul_ch -= 8;
#endif
      }

      // Last PRB
      ch0 = c32x16mulShift(*pil, rxdataF[re_offset], 15);
      pil++;
      re_offset = (re_offset + 1) % symbolSize;

      ch0 = c32x16maddShift(*pil, rxdataF[re_offset], ch0, 15);
      pil++;
      re_offset = (re_offset + 5) % symbolSize;

      ch0 = c32x16maddShift(*pil, rxdataF[re_offset], ch0, 15);
      pil++;
      re_offset = (re_offset + 1) % symbolSize;

      ch0 = c32x16maddShift(*pil, rxdataF[re_offset], ch0, 15);
      pil++;
      re_offset = (re_offset + 5) % symbolSize;

      ch = c16x32div(ch0, 4);
#if NO_INTERP
      for (c16_t *end = ul_ch + 12; ul_ch < end; ul_ch++)
        *ul_ch = ch;
#else
      ul_ch[3] = c16maddShift(ch, (c16_t){1365, 1365}, (c16_t){0, 0}, 15); // 1365 = 1/12*16384 (full range is +/- 32768)
      ul_ch += 4;
      c16multaddVectRealComplex(filt8_avlip3, &ch, ul_ch, 8);
      ul_ch += 8;
      c16multaddVectRealComplex(filt8_avlip6, &ch, ul_ch, 8);
#endif
    }

#ifdef DEBUG_PUSCH
    ul_ch = ul_ch_estimates[nl * num_sp_streams + aarx] + rdata->ch_row_offset;
    for (int idxP = 0; idxP < ceil((float)nb_rb_pusch * 12 / 8); idxP++) {
      for (int idxI = 0; idxI < 8; idxI++) {
        printf("%d\t%d\t", ul_ch[idxP * 8 + idxI].r, ul_ch[idxP * 8 + idxI].i);
      }
      printf("%d\n", idxP);
    }
#endif
    // update the values inside the arrays
    *(rdata->noise_amp2) = noise_amp2;
    *(rdata->nest_count) = nest_count;
  }
  completed_task_ans(rdata->ans);
}

int nr_pusch_channel_estimation(PHY_VARS_gNB *gNB,
                                unsigned char Ns,
                                int nl,
                                unsigned short p,
                                uint8_t lp,
                                unsigned char symbol,
                                NR_gNB_PUSCH *pusch_vars,
                                int nb_ch,
                                int ch_est_size,
                                c16_t ul_ch_estimates[nb_ch][ch_est_size],
                                int ch_row_offset,
                                uint16_t ant_port_start,
                                unsigned short bwp_start_subcarrier,
                                const nfapi_nr_pusch_pdu_t *pusch_pdu,
                                int *max_ch,
                                uint32_t *nvar,
                                c16_t *pusch_dmrs_slot_mem,
                                c16_t *pusch_ch_est_dmrs_pos_slot_mem)
{
  c16_t pilot[3280] __attribute__((aligned(32)));

#ifdef DEBUG_CH
  FILE *debug_ch_est;
  debug_ch_est = fopen("debug_ch_est.txt", "w");
#endif

  const int nb_rb_pusch = pusch_pdu->rb_size;

  //------------------generate DMRS------------------//
  NR_DL_FRAME_PARMS *fp = &gNB->frame_parms;
  if (pusch_pdu->transform_precoding == transformPrecoder_disabled) {
    // Note: pilot returned by the following function is already the complex conjugate of the transmitted DMRS
    const uint32_t *gold = nr_gold_pusch(fp->N_RB_UL,
                                         fp->symbols_per_slot,
                                         gNB->gNB_config.cell_config.phy_cell_id.value,
                                         pusch_pdu->scid,
                                         Ns,
                                         symbol);
    float beta_dmrs_pusch = get_beta_dmrs(pusch_pdu->num_dmrs_cdm_grps_no_data, pusch_pdu->dmrs_config_type == pusch_dmrs_type2);
    int16_t dmrs_scaling = (1 / beta_dmrs_pusch) * (1 << 14);
    nr_pusch_dmrs_rx(fp->Ncp,
                     gold,
                     pilot,
                     (1000 + p),
                     lp % 2,
                     nb_rb_pusch,
                     (pusch_pdu->bwp_start + pusch_pdu->rb_start) * NR_NB_SC_PER_RB,
                     pusch_pdu->dmrs_config_type,
                     dmrs_scaling);
  } else { // if transform precoding or SC-FDMA is enabled in Uplink
    // NR_SC_FDMA supports type1 DMRS so only 6 DMRS REs per RB possible
    const int index = get_index_for_dmrs_lowpapr_seq(nb_rb_pusch * (NR_NB_SC_PER_RB / 2));
    const uint8_t u = pusch_pdu->dfts_ofdm.low_papr_group_number;
    const uint8_t v = pusch_pdu->dfts_ofdm.low_papr_sequence_number;
    c16_t *dmrs_seq = gNB_dmrs_lowpaprtype1_sequence[u][v][index];
    LOG_D(PHY, "Transform Precoding params. u: %d, v: %d, index for dmrsseq: %d\n", u, v, index);
    AssertFatal(index >= 0,
                "Num RBs not configured according to 3GPP 38.211 section 6.3.1.4. For PUSCH with transform precoding, num RBs "
                "cannot be multiple of any other primenumber other than 2,3,5\n");
    AssertFatal(dmrs_seq != NULL, "DMRS low PAPR seq not found, check if DMRS sequences are generated");
    nr_pusch_lowpaprtype1_dmrs_rx(fp->Ncp, dmrs_seq, pilot, 1000, 0, nb_rb_pusch, 0, pusch_pdu->dmrs_config_type);
#ifdef DEBUG_PUSCH
    printf("NR_UL_CHANNEL_EST: index %d, u %d,v %d\n", index, u, v);
    LOG_M("gNb_DMRS_SEQ.m", "gNb_DMRS_SEQ", dmrs_seq, 6 * nb_rb_pusch, 1, 1);
#endif
  }
  //------------------------------------------------//

#ifdef DEBUG_PUSCH

  for (int i = 0; i < (6 * nb_rb_pusch); i++) {
    LOG_I(PHY, "In %s: %d + j*(%d)\n", __FUNCTION__, pilot[i].r, pilot[i].i);
  }

#endif

  //------------------Write DMRS to Memory for Data Recording ------------------//
  int dmrs_symbol_start_idx = symbol * pusch_pdu->nrOfLayers * nb_rb_pusch * NR_NB_SC_PER_RB + nl * nb_rb_pusch * NR_NB_SC_PER_RB;
#if T_TRACER
  if (T_ACTIVE(T_GNB_PHY_UL_FD_DMRS) && nl == 0) {
    // used by T-Tracer to trace DMRS slot grid
    int dmrs_delta = 0; // intialize it to zero currently, derive it later from above functions
    for (int i = 0; i < (6 * nb_rb_pusch); i++) {
      // the generated DMRs is a complex conjugate of mod table, so flip the sign of imag. part
      pusch_dmrs_slot_mem[dmrs_symbol_start_idx + dmrs_delta + i * 2].r = pilot[i].r; // 0, 2, 4, 6, 8, location of REs
      pusch_dmrs_slot_mem[dmrs_symbol_start_idx + dmrs_delta + i * 2].i = -pilot[i].i; // 0, 2, 4, 6, 8, location of REs
    }
  }
#endif

  int nest_count = 0;
  uint64_t noise_amp2 = 0;
  delay_t *delay = &pusch_vars->delay;
  memset(delay, 0, sizeof(*delay));

  int nb_antennas_rx = pusch_pdu->param_v4.numSpatialStreamIndices;
  delay_t delay_arr[nb_antennas_rx];
  uint64_t noise_amp2_arr[nb_antennas_rx];
  int max_ch_arr[nb_antennas_rx];
  int nest_count_arr[nb_antennas_rx];

  for (int i = 0; i < nb_antennas_rx; ++i) {
    max_ch_arr[i] = *max_ch;
    nest_count_arr[i] = nest_count;
    noise_amp2_arr[i] = noise_amp2;
    delay_arr[i] = *delay;
  }

  notifiedFIFO_t respPuschAarx;
  initNotifiedFIFO(&respPuschAarx);
  start_meas(&gNB->pusch_channel_estimation_antenna_processing_stats);
  int numAntennas = gNB->dmrs_num_antennas_per_thread;
  int num_jobs = CEILIDIV(nb_antennas_rx, numAntennas);
  puschAntennaProc_t rdatas[num_jobs];
  memset(rdatas, 0, sizeof(rdatas));
  task_ans_t ans;
  init_task_ans(&ans, num_jobs);
  for (int job_id = 0; job_id < num_jobs; job_id++) {
    puschAntennaProc_t *rdata = &rdatas[job_id];
    task_t task = {.func = nr_pusch_antenna_processing, .args = rdata};

    // Local init in the current loop
    rdata->Ns = Ns;
    rdata->nl = nl;
    rdata->p = p;
    rdata->symbol = symbol;
    rdata->aarx = job_id * numAntennas;
    rdata->numAntennas = numAntennas;
    rdata->bwp_start_subcarrier = bwp_start_subcarrier;
    rdata->pusch_pdu = pusch_pdu;
    rdata->max_ch = &max_ch_arr[rdata->aarx];
    rdata->pilot = pilot;
    rdata->nest_count = &nest_count_arr[rdata->aarx];
    rdata->noise_amp2 = &noise_amp2_arr[rdata->aarx];
    rdata->delay = &delay_arr[rdata->aarx];
    rdata->ant_port_start = ant_port_start;
    rdata->frame_parms = fp;
    rdata->ul_ch_estimates = ul_ch_estimates;
    rdata->ch_est_size = ch_est_size;
    rdata->ch_row_offset = ch_row_offset;
    rdata->chest_freq = gNB->chest_freq;
    rdata->rxdataF = gNB->common_vars.rxdataF;
    rdata->scope = gNB->scopeData;
    rdata->ans = &ans;
    rdata->pusch_ch_est_dmrs_pos_slot_mem = pusch_ch_est_dmrs_pos_slot_mem;
    rdata->dmrs_symbol_start_idx = dmrs_symbol_start_idx;
    // Call the nr_pusch_antenna_processing function
    if (job_id == num_jobs - 1) {
      // Run the last job inline
      nr_pusch_antenna_processing(rdata);
    } else {
      pushTpool(&gNB->threadPool, task);
    }
  } // Antenna Loop

  join_task_ans(&ans);

  stop_meas(&gNB->pusch_channel_estimation_antenna_processing_stats);
  for (int aarx = 0; aarx < nb_antennas_rx; aarx++) {
    *max_ch = max(*max_ch, max_ch_arr[aarx]);
    noise_amp2 += noise_amp2_arr[aarx];
    nest_count += nest_count_arr[aarx];
  }
  // get the maximum delay
  for (int aarx = 0; aarx < nb_antennas_rx; aarx++) {
    if (delay_arr[aarx].valid && delay_arr[aarx].delay_max_val > delay->delay_max_val) {
      *delay = delay_arr[aarx];
    }
  }

#ifdef DEBUG_CH
  fclose(debug_ch_est);
#endif

  if (nvar && nest_count > 0) {
    *nvar = (uint32_t)(noise_amp2 / nest_count);
  }

  return 0;
}

uint8_t get_srs_fd_cdm(const uint8_t N_ap, const uint8_t K_TC, uint8_t cyclic_shift)
{
  uint8_t fd_cdm = N_ap;
  switch (N_ap) {
    case 4:
      if ((K_TC == 2 && cyclic_shift >= 4) || (K_TC == 4 && cyclic_shift >= 6) || (K_TC == 8))
        fd_cdm = 2;
      break;
    case 8:
      if ((K_TC == 2 && cyclic_shift >= 4) || (K_TC == 4)) {
        fd_cdm = 4;
      } else if (K_TC == 8) {
        fd_cdm = 2;
      }
      break;
    default:
      break;
  }
  return fd_cdm;
}

// Hann window in Q15, from the cosine recurrence so there is one libm call per window.
static void srs_noise_window(int n, int16_t *w)
{
  const double c1 = cos(2.0 * M_PI / (n - 1));
  double a = 1.0;
  double b = c1;
  for (int i = 0; i < n; i++) {
    w[i] = (int16_t)lroundf((float)(16383.5 * (1.0 - a)));
    const double next = 2.0 * c1 * b - a;
    a = b;
    b = next;
  }
}

static int srs_noise_fft_size(int num_ls)
{
  int n = 64;
  while (n < num_ls && n < NR_SRS_NOISE_MAX_FFT)
    n <<= 1;
  return n;
}

// k-th smallest of a[0..n), reorders a so that a[0..k) <= a[k] <= a(k..n)
// Does not sort the whole array like qsort(), it only keeps splitting the part that contains the k-th value
static double quickselect_double(double *a, int n, int k)
{
  int lo = 0;
  int hi = n - 1;
  while (lo < hi) {
    const double pivot = a[(lo + hi) / 2];
    int l = lo;
    int h = hi;
    while (l <= h) {
      while (a[l] < pivot)
        l++;
      while (a[h] > pivot)
        h--;
      if (l <= h) {
        const double tmp = a[l];
        a[l++] = a[h];
        a[h--] = tmp;
      }
    }
    if (k <= h)
      hi = h;
    else if (k >= l)
      lo = l;
    else
      break;
  }
  return a[k];
}

// Compute noise from the delay domain.
// The channel sits in a few delay bins while the noise is spread over all of them, so the weakest bins hold noise only.
void nr_srs_noise_accumulate(int p_index,
                             const nfapi_nr_srs_pdu_t *srs_pdu,
                             const nr_srs_info_t *nr_srs_info,
                             const c16_t *srs_ls_estimated_channel,
                             nr_srs_noise_acc_t *acc,
                             nr_srs_prg_pow_t *pow_per_prg)
{
  const uint64_t subcarrier_offset = srs_pdu->bwp_start * NR_NB_SC_PER_RB;
  const uint8_t N_ap = 1 << srs_pdu->num_ant_ports;
  const uint8_t K_TC = 2 << srs_pdu->comb_size;
  const uint16_t m_SRS_b = get_m_srs(srs_pdu->config_index, srs_pdu->bandwidth_index);
  const uint16_t M_sc_b_SRS = m_SRS_b * NR_NB_SC_PER_RB / K_TC;
  const uint8_t fd_cdm = get_srs_fd_cdm(N_ap, K_TC, srs_pdu->cyclic_shift);
  const uint16_t prg_size = max((int)srs_pdu->beamforming.prg_size, 1);
  const uint16_t num_prgs = max((int)srs_pdu->beamforming.num_prgs, 1);

  const uint16_t first_sc = subcarrier_offset + nr_srs_info->k_0_p[p_index][0];
  const uint16_t step = K_TC * fd_cdm;

  // The per PRG signal power wants every estimate, the noise floor only needs enough delay bins
  const int all_ls = M_sc_b_SRS / fd_cdm;
  const int num_ls = min(all_ls, NR_SRS_NOISE_MAX_FFT);
  if (num_ls < NR_SRS_NOISE_MIN_ESTIMATES) {
    // every pair of the slot has the same num_ls, so no pair reads a floor and nr_srs_snr_per_prg returns early
    LOG_W(NR_PHY, "SRS: %d LS estimates, cannot read a noise floor\n", num_ls);
    return;
  }

  const int nfft = srs_noise_fft_size(num_ls);
  c16_t freq[NR_SRS_NOISE_MAX_FFT] __attribute__((aligned(32))) = {0};
  c16_t time[NR_SRS_NOISE_MAX_FFT] __attribute__((aligned(32))) = {0};
  if (acc->win_len != num_ls) {
    srs_noise_window(num_ls, acc->win);
    acc->win_sq = 0;
    for (int i = 0; i < num_ls; i++)
      acc->win_sq += (uint64_t)acc->win[i] * acc->win[i];
    acc->win_len = num_ls;
  }

  const int16_t *win = acc->win;
  const uint32_t prg_len = NR_NB_SC_PER_RB * prg_size;
  uint32_t offset = 0; // subcarrier distance from the first LS estimate
  int prg = 0;
  uint64_t pair_sum = 0; // LS power of this rx antenna and port pair, over the whole band
  uint64_t energy = 0; // sum of |ls*w|^2, the windowed power in LS units
  c16_t prev = {0, 0};
  for (int i = 0; i < all_ls; i++) {
    const c16_t ls = srs_ls_estimated_channel[first_sc + offset];

    // Neighbouring LS estimates sit K_TC*fd_cdm subcarriers apart, which is exactly the span the
    // despreading sums over, so their correlation measures how much of that sum is coherent.
    if (i) {
      acc->corr_re += (int64_t)ls.r * prev.r + (int64_t)ls.i * prev.i;
      acc->corr_im += (int64_t)ls.i * prev.r - (int64_t)ls.r * prev.i;
      acc->corr_cnt++;
    }
    prev = ls;
    while (prg < num_prgs - 1 && offset >= (prg + 1) * prg_len)
      prg++;
    const uint64_t p = c16amp2(ls);
    pow_per_prg[prg].pow += p;
    pow_per_prg[prg].cnt++;
    pair_sum += p;

    if (i < num_ls) {
      freq[i] = c16xmulConstShift(ls, win[i], 15);
      energy += c16amp2(freq[i]);
    }
    offset += step;
  }

  if (energy == 0 || acc->win_sq == 0)
    return;
  const uint64_t wsq = acc->win_sq;

  idft(get_idft(nfft), (int16_t *)freq, (int16_t *)time, 1);

  uint32_t power[NR_SRS_NOISE_MAX_FFT];
  uint64_t total = 0;
  for (int t = 0; t < nfft; t++) {
    power[t] = c16amp2(time[t]);
    total += power[t];
  }
  if (total == 0)
    return;

  // power[t]/total is the share of this bin, scaling it by the windowed energy puts it back in LS units,
  // so the bins of every antenna and port land on the same scale and simply add up.
  const double unit = (double)nfft * (double)energy * 1073741824.0 / (double)wsq / (double)total;
  for (int t = 0; t < nfft; t++)
    acc->spectrum[t] += power[t] * unit;

  if (acc->cnt < NR_SRS_NOISE_MAX_PAIRS && all_ls > 0)
    acc->pair_pow[acc->cnt] = (double)pair_sum / all_ls;

  acc->nfft = nfft;
  acc->num_ls = num_ls;
  acc->cnt++;
}

// A noise bin averaged over cnt antenna/port pairs follows Gamma(cnt)/cnt, so the percentile the floor
// is read at sits below its mean by a factor that depends only on cnt: Wilson-Hilferty in closed form.
static double srs_noise_quantile_scale(uint32_t cnt, int num_ls, double z)
{
  const double a = 1.0 / (9.0 * cnt);
  const double wh = 1.0 - a - z * sqrt(a);
  return wh * wh * wh * (1.0 + 5.0 / (num_ls * sqrt((double)cnt)));
}

// Fills the per PRG SNR and returns the wideband SNR
uint32_t nr_srs_snr_per_prg(const nfapi_nr_srs_pdu_t *srs_pdu,
                            uint32_t signal_power_avg,
                            nr_srs_noise_acc_t *acc,
                            const nr_srs_prg_pow_t *pow_per_prg,
                            int16_t *snr_per_prg)
{
  const uint8_t N_ap = 1 << srs_pdu->num_ant_ports;
  const uint8_t K_TC = 2 << srs_pdu->comb_size;
  const uint8_t fd_cdm = get_srs_fd_cdm(N_ap, K_TC, srs_pdu->cyclic_shift);
  const uint16_t num_prgs = max((int)srs_pdu->beamforming.num_prgs, 1);

  if (acc->cnt == 0 || acc->nfft == 0)
    return 1;

  // Only two percentiles are read. Dividing by cnt keeps the order, so only the two selected bins are divided
  double *bins = acc->spectrum;
  const int idx1 = (acc->nfft * NR_SRS_NOISE_QUANTILE_NUM) / NR_SRS_NOISE_QUANTILE_DEN;
  const int idx2 = (acc->nfft * NR_SRS_NOISE_QUANTILE2_NUM) / NR_SRS_NOISE_QUANTILE_DEN;
  // Two ranks are enough, no full sort: the first select leaves the 20th percentile bin at idx2 with all
  // smaller bins before it, the second finds the 10th percentile bin (the noise floor) among those
  const double q2_bin = quickselect_double(bins, acc->nfft, idx2) / acc->cnt;
  const double floor_bin = quickselect_double(bins, idx2, idx1) / acc->cnt;
  const double scale_q1 = srs_noise_quantile_scale(acc->cnt, acc->num_ls, NR_SRS_NOISE_QUANTILE_Z);

  // The percentile is noise plus whatever the channel leaves under that bin, so it is an upper
  // bound. A second percentile gives a second equation: the pedestal cancels in the difference,
  // leaving the noise times the difference of the two scales. That slope under-reads rather than
  // over-reads, so it is taken only when it lands below the bound, and never under a quarter of it.
  double noise = floor_bin / scale_q1;
  const double scale_q2 = srs_noise_quantile_scale(acc->cnt, acc->num_ls, NR_SRS_NOISE_QUANTILE2_Z);
  if (scale_q2 > scale_q1) {
    const double slope = (q2_bin - floor_bin) / (scale_q2 - scale_q1);
    if (slope < noise && slope > noise * NR_SRS_NOISE_SLOPE_MIN_RATIO)
      noise = slope;
  }
  const int noise_dB_x10 = dB_fixed_x10(max((uint32_t)noise, 1u));

  // The despreading only lifts the signal by fd_cdm where the channel is flat across the
  // K_TC*fd_cdm subcarriers it sums. Taking that on faith costs 10*log10(fd_cdm) over a channel
  // that decorrelates inside the span, so the gain is measured from the correlation between
  // neighbouring estimates: fd_cdm when the channel is flat, 1 when it is gone by the end.
  uint64_t band_pow = 0;
  uint32_t band_cnt = 0;
  for (int prg = 0; prg < num_prgs; prg++) {
    band_pow += pow_per_prg[prg].pow;
    band_cnt += pow_per_prg[prg].cnt;
  }
  double despread = fd_cdm;
  if (fd_cdm > 1 && acc->corr_cnt && band_cnt) {
    const double mean = (double)band_pow / band_cnt;
    const double r1 = sqrt((double)acc->corr_re * acc->corr_re + (double)acc->corr_im * acc->corr_im) / acc->corr_cnt;
    double rho = (mean > noise) ? r1 / (mean - noise) : 0.0;
    if (rho > 1.0)
      rho = 1.0;
    if (!(rho > 0.0))
      rho = 0.0;
    // rho is measured across the whole span, so one step inside it carries the fd_cdm-th root
    const double step_rho = pow(rho, 1.0 / fd_cdm);
    double sum = fd_cdm;
    for (int d = 1; d < fd_cdm; d++)
      sum += 2.0 * (fd_cdm - d) * pow(step_rho, d);
    despread = sum / fd_cdm;
    if (despread < 1.0)
      despread = 1.0;
    if (despread > fd_cdm)
      despread = fd_cdm;
  }

  // The LS power is averaged over every rx antenna and port pair, but a pair only carries signal
  // when its port reaches its antenna, so the mean is diluted by the ones that do not. Counting
  // the carriers against a threshold does not survive a timing offset: the delay breaks the port
  // orthogonality over the despreading span, and that leakage is a fixed fraction of the signal,
  // so it crosses any noise-referenced threshold once the SNR is high enough. The participation
  // ratio weighs each pair by its own power instead: C for C equal pairs, 1 when one holds
  // everything, and no threshold anywhere.
  double excess_sum = 0.0;
  double excess_sq = 0.0;
  if (acc->cnt <= NR_SRS_NOISE_MAX_PAIRS) {
    for (uint32_t i = 0; i < acc->cnt; i++) {
      const double s = acc->pair_pow[i] - noise;
      if (s > 0.0) {
        excess_sum += s;
        excess_sq += s * s;
      }
    }
  }
  double carrying = excess_sq > 0.0 ? excess_sum * excess_sum / excess_sq : (double)acc->cnt;
  if (!(carrying >= 1.0))
    carrying = 1.0;
  if (carrying > (double)acc->cnt)
    carrying = (double)acc->cnt;
  const double dilution = (double)acc->cnt / carrying;

  for (int prg = 0; prg < num_prgs; prg++) {
    nr_srs_prg_pow_t w = pow_per_prg[prg];
    int span = 0;

    while (w.cnt < NR_SRS_NOISE_PRG_MIN_ESTIMATES && span < num_prgs) {
      span++;
      for (int s = -1; s <= 1; s += 2) {
        const int j = prg + s * span;
        if (j >= 0 && j < num_prgs) {
          w.pow += pow_per_prg[j].pow;
          w.cnt += pow_per_prg[j].cnt;
        }
      }
    }

    // The LS estimates carry signal plus noise, so the noise comes off before the ratio. Around and
    // below 0 dB the two are within their own estimation error, so the difference can land at or
    // under zero, and dB_fixed_x10 takes a uint32_t. The floor is a ratio rather than an absolute
    // power, which also bounds what a single faded PRG can report.
    const double floor_signal = noise * NR_SRS_SNR_MIN_RATIO;
    const double raw = w.cnt ? ((double)w.pow / w.cnt - noise) * dilution / despread : 0.0;
    const double signal = max(raw, floor_signal);
    snr_per_prg[prg] = (dB_fixed_x10(signal) - noise_dB_x10) / 10.0;

    LOG_D(NR_PHY,
          "SRS-SNR-PRG: rnti %04x, prg %3d, noise %u (%d dB), snr %d\n",
          srs_pdu->rnti,
          prg,
          (uint32_t)noise,
          noise_dB_x10 / 10,
          snr_per_prg[prg]);
  }

  // The wideband SNR is the same estimator over the whole band: same mean, same noise, same scaling
  const double signal = max(((double)signal_power_avg - noise) * dilution / despread, noise * NR_SRS_SNR_MIN_RATIO);
  return (dB_fixed_x10(signal) - noise_dB_x10) / 10;
}

int nr_srs_ls_channel_estimation(int ant,
                                 int p_index,
                                 uint16_t ofdm_symbol_size,
                                 uint8_t N_symb_SRS,
                                 const nfapi_nr_srs_pdu_t *srs_pdu,
                                 const nr_srs_info_t *nr_srs_info,
                                 const c16_t *srs_generated_signal,
                                 c16_t srs_received_signal[ofdm_symbol_size * N_symb_SRS],
                                 c16_t srs_ls_estimated_channel[ofdm_symbol_size * N_symb_SRS],
                                 int64_t *ls_peak,
                                 delay_t *delay)
{
#ifdef SRS_DEBUG
  LOG_I(NR_PHY, "Calling %s function\n", __FUNCTION__);
#endif

  const uint64_t subcarrier_offset = srs_pdu->bwp_start * NR_NB_SC_PER_RB;

  const uint8_t N_ap = 1 << srs_pdu->num_ant_ports;
  const uint8_t K_TC = 2 << srs_pdu->comb_size;
  const uint16_t m_SRS_b = get_m_srs(srs_pdu->config_index, srs_pdu->bandwidth_index);
  const uint16_t M_sc_b_SRS = m_SRS_b * NR_NB_SC_PER_RB / K_TC;
  uint8_t fd_cdm = get_srs_fd_cdm(N_ap, K_TC, srs_pdu->cyclic_shift);

  int16_t scale = (N_ap > fd_cdm) ? N_ap / fd_cdm : 1;

  memset(srs_ls_estimated_channel, 0, ofdm_symbol_size * N_symb_SRS * sizeof(c16_t));

  for (int srs_symb = 0; srs_symb < N_symb_SRS; srs_symb++) {
    uint16_t srs_symbol_offset = srs_symb * ofdm_symbol_size;

#ifdef SRS_DEBUG
    LOG_I(NR_PHY, "====================== UE port %d --> gNB Rx antenna %i ======================\n", p_index, ant);
    LOG_I(NR_PHY, "============================== SRS symbol index %d ===========================\n", srs_symb);
#else
    UNUSED(ant);
#endif

    uint16_t subcarrier = subcarrier_offset + nr_srs_info->k_0_p[p_index][srs_symb];

    c16_t ls_estimated = {0};
    for (int k = 0; k < M_sc_b_SRS; k++) {
      if (k % fd_cdm == 0) {
        ls_estimated = (c16_t){0, 0};
        uint16_t subcarrier_cdm = subcarrier;

        // The despreading sums fd_cdm products and the result is then scaled by N_ap/fd_cdm, so the
        // value on the way out can be several bits wider than one product. Accumulating in 64 bits
        // and narrowing once at the end keeps every intermediate exact, and the narrowing clamps
        // rather than wraps: a wrap would send the estimate to the opposite rail and that impulse
        // spreads over every delay bin, which is what the noise floor is read from.
        int64_t acc_re = 0;
        int64_t acc_im = 0;

        for (int cdm_idx = 0; cdm_idx < fd_cdm; cdm_idx++) {
          c16_t generated_srs = srs_generated_signal[srs_symbol_offset + subcarrier_cdm];
          c16_t received_srs = srs_received_signal[srs_symbol_offset + subcarrier_cdm];
          acc_re += (int64_t)generated_srs.r * received_srs.r + (int64_t)generated_srs.i * received_srs.i;
          acc_im += (int64_t)generated_srs.r * received_srs.i - (int64_t)generated_srs.i * received_srs.r;

          // Subcarrier increment
          subcarrier_cdm += K_TC;
        }

        // We know that nr_srs_info->srs_generated_signal_bits bits are enough to represent the real and imaginary parts of
        // generated_srs, so that shift takes the products back to the scale of the received signal. The scaling for
        // N_ap > fd_cdm : N_ap/fd_cdm then follows, before the one narrowing to 16 bits.
        acc_re = (acc_re >> nr_srs_info->srs_generated_signal_bits) * scale;
        acc_im = (acc_im >> nr_srs_info->srs_generated_signal_bits) * scale;

        // The clamp keeps a wrap from turning one estimate into an impulse across every delay
        // bin; the peak says how close the received level came to needing it.
        const int64_t mag = max(llabs(acc_re), llabs(acc_im));
        if (mag > *ls_peak)
          *ls_peak = mag;
        ls_estimated = (c16_t){saturate_int16(acc_re), saturate_int16(acc_im)};
      }

      for (int ktc = 0; ktc < K_TC && srs_symbol_offset + subcarrier + ktc < ofdm_symbol_size * N_symb_SRS; ktc++) {
        srs_ls_estimated_channel[srs_symbol_offset + subcarrier + ktc] = ls_estimated;
      }

#ifdef SRS_DEBUG
      int subcarrier_log = subcarrier - subcarrier_offset;
      if (subcarrier_log % 12 == 0) {
        LOG_I(NR_PHY, "------------------------------------ %d ------------------------------------\n", subcarrier_log / 12);
        LOG_I(NR_PHY, "\t  __genRe________genIm__|____rxRe_________rxIm__|____lsRe________lsIm_\n");
      }
      LOG_I(NR_PHY,
            "(%4i) %6i\t%6i  |  %6i\t%6i  |  %6i\t%6i\n",
            subcarrier_log,
            srs_generated_signal[srs_symbol_offset + subcarrier].r,
            srs_generated_signal[srs_symbol_offset + subcarrier].i,
            srs_received_signal[srs_symbol_offset + subcarrier].r,
            srs_received_signal[srs_symbol_offset + subcarrier].i,
            ls_estimated.r,
            ls_estimated.i);
#endif

      // Subcarrier increment
      subcarrier += K_TC;
    } // for (int k = 0; k < M_sc_b_SRS; k++)

    // Delay estimation
    if (srs_symb == 0) {
      c16_t ch_estimates_time[ofdm_symbol_size] __attribute__((aligned(32)));
      nr_est_delay(ofdm_symbol_size, srs_ls_estimated_channel, ch_estimates_time, delay);
    }
  } // for (int srs_symb = 0; srs_symb < N_symb_SRS; srs_symb++)

  return 0;
}

int nr_srs_channel_interpolation(int p_index,
                                 uint16_t ofdm_symbol_size,
                                 uint16_t first_carrier_offset,
                                 uint8_t N_symb_SRS,
                                 const nfapi_nr_srs_pdu_t *srs_pdu,
                                 const nr_srs_info_t *nr_srs_info,
                                 const c16_t srs_ls_estimated_channel[ofdm_symbol_size * N_symb_SRS],
                                 int est_delay,
                                 c16_t srs_estimated_channel_freq[ofdm_symbol_size * N_symb_SRS],
                                 c16_t srs_estimated_channel_time[NR_SRS_IDFT_OVERSAMP_FACTOR * ofdm_symbol_size],
                                 c16_t srs_estimated_channel_time_shifted[NR_SRS_IDFT_OVERSAMP_FACTOR * ofdm_symbol_size],
                                 uint32_t *signal_power,
                                 int *idft_shift,
                                 c16_t delay_table[2 * MAX_DELAY_COMP + 1][NR_MAX_OFDM_SYMBOL_SIZE])
{
#ifdef SRS_DEBUG
  LOG_I(NR_PHY, "Calling %s function\n", __FUNCTION__);
#endif

  const uint64_t subcarrier_offset = srs_pdu->bwp_start * NR_NB_SC_PER_RB;

  const uint8_t K_TC = 2 << srs_pdu->comb_size;
  const uint16_t m_SRS_b = get_m_srs(srs_pdu->config_index, srs_pdu->bandwidth_index);
  const uint16_t M_sc_b_SRS = m_SRS_b * NR_NB_SC_PER_RB / K_TC;

  c16_t srs_estimated_channel_freq_avg[ofdm_symbol_size];
  memset(srs_estimated_channel_freq_avg, 0, ofdm_symbol_size * sizeof(c16_t));

  for (int srs_symb = 0; srs_symb < N_symb_SRS; srs_symb++) {
    uint16_t srs_symbol_offset = srs_symb * ofdm_symbol_size;

#ifdef SRS_DEBUG
    LOG_I(NR_PHY, "============================== UE port %d ====================================\n", p_index);
    LOG_I(NR_PHY, "============================== SRS symbol index %d ===========================\n", srs_symb);
#endif

    c16_t srs_est[ofdm_symbol_size] __attribute__((aligned(32)));
    memset(srs_est, 0, (ofdm_symbol_size) * sizeof(c16_t));

    // Start of buffer is 32 byte aligned.
    uint16_t subcarrier_abs = 0;
    c16_t *srs_estimated_channel16 = &srs_est[subcarrier_abs];

    uint16_t subcarrier = subcarrier_offset + nr_srs_info->k_0_p[p_index][srs_symb];

    int delay_idx = get_delay_idx(est_delay, MAX_DELAY_COMP);
    const c16_t *srs_delay_table = delay_table[delay_idx];

    // Delay table might be FFT shift sensitive. Not sure.
    uint16_t subcarrier_delay =
        CIRCULAR_INC(first_carrier_offset, subcarrier_offset + nr_srs_info->k_0_p[p_index][srs_symb], ofdm_symbol_size);
    for (int k = 0; k < M_sc_b_SRS; k++) {
      // Apply delay
      c16_t ls_estimated =
          c16mulShift(srs_ls_estimated_channel[srs_symbol_offset + subcarrier], srs_delay_table[subcarrier_delay], 8);

      // Channel interpolation
      if (srs_pdu->comb_size == 0) {
        if (k == 0) { // First subcarrier case
          // filt8_start is {12288,8192,4096,0,0,0,0,0}
          c16multaddVectRealComplex(filt8_start, &ls_estimated, srs_estimated_channel16, 8);
        } else if (k == (M_sc_b_SRS - 1)) { // End of OFDM symbol or last subcarrier cases
          // filt8_end is {4096,8192,12288,16384,0,0,0,0}
          c16multaddVectRealComplex(filt8_end, &ls_estimated, srs_estimated_channel16, 8);
        } else if (k % 2 == 1) { // 1st middle case
          // filt8_middle2 is {4096,8192,8192,8192,4096,0,0,0}
          c16multaddVectRealComplex(filt8_middle2, &ls_estimated, srs_estimated_channel16, 8);
        } else if (k % 2 == 0) { // 2nd middle case
          // filt8_middle4 is {0,0,4096,8192,8192,8192,4096,0}
          c16multaddVectRealComplex(filt8_middle4, &ls_estimated, srs_estimated_channel16, 8);
          srs_estimated_channel16 = &srs_est[subcarrier_abs];
        }
      } else {
        if (k == 0) { // First subcarrier case
          // filt16_start is {12288,8192,8192,8192,4096,0,0,0,0,0,0,0,0,0,0,0}
          c16multaddVectRealComplex(filt16_start, &ls_estimated, srs_estimated_channel16, 16);
        } else if (k == (M_sc_b_SRS - 1)) { // End of OFDM symbol or last subcarrier cases
          // filt16_end is {4096,8192,8192,8192,12288,16384,16384,16384,0,0,0,0,0,0,0,0}
          c16multaddVectRealComplex(filt16_end, &ls_estimated, srs_estimated_channel16, 16);
        } else { // Middle case
          // filt16_middle4 is {4096,8192,8192,8192,8192,8192,8192,8192,4096,0,0,0,0,0,0,0}
          c16multaddVectRealComplex(filt16_middle4, &ls_estimated, srs_estimated_channel16, 16);
          srs_estimated_channel16 = &srs_est[subcarrier_abs];
        }
      }

      // Subcarrier increment
      subcarrier += K_TC;
      subcarrier_delay = CIRCULAR_INC(subcarrier_delay, K_TC, ofdm_symbol_size);
      subcarrier_abs += K_TC;
    } // for (int k = 0; k < M_sc_b_SRS; k++)

    // Revert delay
    int inv_delay_idx = get_delay_idx(-est_delay, MAX_DELAY_COMP);
    const c16_t *srs_inv_delay_table = delay_table[inv_delay_idx];
    subcarrier_abs = 0;
    subcarrier_delay =
        CIRCULAR_INC(first_carrier_offset, subcarrier_offset + nr_srs_info->k_0_p[p_index][srs_symb], ofdm_symbol_size);

    for (int k = 0; k < K_TC * M_sc_b_SRS; k++) {
      srs_est[subcarrier_abs] = c16mulShift(srs_est[subcarrier_abs], srs_inv_delay_table[subcarrier_delay], 8);
      // Subcarrier increment
      subcarrier_delay = CIRCULAR_INC(subcarrier_delay, 1, ofdm_symbol_size);
      subcarrier_abs++;
    }

    // Copy as DC in center.
    const uint half_bw = ofdm_symbol_size - first_carrier_offset;
    const uint neg_start = (ofdm_symbol_size >> 1) - half_bw + subcarrier_offset + nr_srs_info->k_0_p[p_index][srs_symb];
    memset(&srs_estimated_channel_freq[srs_symbol_offset], 0, sizeof(c16_t) * neg_start);
    memcpy(&srs_estimated_channel_freq[srs_symbol_offset + neg_start],
           srs_est,
           (ofdm_symbol_size - neg_start) * sizeof(c16_t));

    // Average srs channel estimates over multiple symbols
    int16_t scale_factor = (1 << 15) / N_symb_SRS;
    multadd_complex_vector_real_scalar(&srs_estimated_channel_freq[srs_symbol_offset],
                                       scale_factor,
                                       srs_estimated_channel_freq_avg,
                                       ofdm_symbol_size);

#ifdef SRS_DEBUG
    subcarrier = subcarrier_offset + nr_srs_info->k_0_p[p_index][srs_symb];
    subcarrier_abs = neg_start;

    for (int k = 0; k < K_TC * M_sc_b_SRS; k++) {
      int subcarrier_log = subcarrier - subcarrier_offset;

      if (k == 0 || subcarrier_log % NR_NB_SC_PER_RB == 0) {
        LOG_I(NR_PHY,
              "---------------------------- PRB %d, symbol %d -------------------------------\n",
              subcarrier_log / NR_NB_SC_PER_RB,
              srs_symb);
        LOG_I(NR_PHY, "\t  __lsRe__________lsIm__|____intRe_______intIm__\n");
      }

      LOG_I(NR_PHY,
            "(%4i) %6i\t%6i  |  %6i\t%6i\n",
            subcarrier_log,
            srs_ls_estimated_channel[srs_symbol_offset + subcarrier].r,
            srs_ls_estimated_channel[srs_symbol_offset + subcarrier].i,
            srs_estimated_channel_freq[srs_symbol_offset + subcarrier_abs].r,
            srs_estimated_channel_freq[srs_symbol_offset + subcarrier_abs].i);

      // Subcarrier increment
      subcarrier++;
      subcarrier_abs++;
    }
#endif
  }

  // Convert to time domain
  int16_t ofdm_symbol_size_half = ofdm_symbol_size >> 1;
  int16_t ofdm_os_size = NR_SRS_IDFT_OVERSAMP_FACTOR * ofdm_symbol_size;
  int16_t ofdm_os_size_half = ofdm_os_size >> 1;
  int16_t start_offset = ofdm_os_size - ofdm_symbol_size_half;

  c16_t chF_interpol[ofdm_os_size] __attribute__((aligned(32)));
  memset(chF_interpol, 0, sizeof(chF_interpol));

  // Place SRS channel estimates in FFT shifted format for oversampling
  memcpy(&chF_interpol[0], &srs_estimated_channel_freq_avg[ofdm_symbol_size_half], ofdm_symbol_size_half * sizeof(c16_t));
  memcpy(&chF_interpol[start_offset], &srs_estimated_channel_freq_avg[0], ofdm_symbol_size_half * sizeof(c16_t));

  // Convert to time domain oversampled; the input is prescaled so that the IDFT cannot saturate
  *idft_shift = nr_idft_prescale_shift(chF_interpol, ofdm_os_size);
  if (*idft_shift)
    for (int i = 0; i < ofdm_os_size; i++)
      chF_interpol[i] = (c16_t){chF_interpol[i].r >> *idft_shift, chF_interpol[i].i >> *idft_shift};
  freq2time(ofdm_os_size, (int16_t *)chF_interpol, (int16_t *)srs_estimated_channel_time);

  // Do FFT shift
  memcpy(srs_estimated_channel_time_shifted, &srs_estimated_channel_time[ofdm_os_size_half], ofdm_os_size_half * sizeof(c16_t));

  memcpy(&srs_estimated_channel_time_shifted[ofdm_os_size_half], srs_estimated_channel_time, ofdm_os_size_half * sizeof(c16_t));

  // Compute wideband SNR on the symbol 0
  int tot_subcarriers = m_SRS_b * NR_NB_SC_PER_RB;
  uint16_t subcarrier_abs = subcarrier_offset + nr_srs_info->k_0_p[p_index][0];
  *signal_power = signal_energy_nodc(&srs_ls_estimated_channel[subcarrier_abs], tot_subcarriers);

  if (*signal_power == 0) {
    LOG_W(NR_PHY, "Received SRS signal power is 0\n");
    return -1;
  }

  return 0;
}
