/* Scalar reference of the dec_std algorithm, written directly on the lifted parity-check matrix in natural
 * indexing (no rotated buffers, no padding, no vector layout). Every SIMD implementation must be bit-exact
 * with it (same hard decisions, same a-posteriori LLRs, same iteration count).
 *
 * Algorithm (layered min-sum, int8):
 *   rows (layers) i = 0..M-1 in order; check t = 0..Z-1 of row i is connected, for every edge (i, c) of the base
 *   graph with shift P, to bit (t + P) mod Z of column c (H block = identity shifted right by P, 38.212 5.3.2).
 *   For each check:  Q_e = sat8(APP_c[(t+P) mod Z] - R_e[t])   (degree-1 columns: Q_e = channel LLR, exactly)
 *                    m1, m2 = two smallest |Q_e|, sgn = parity of (Q_e < 0)
 *                    (|Q_e| saturated to 127 as by SQABS: |-128| = 127; this differs from |-128| = 128 only if
 *                    corr(127) != corr(128), i.e. nshift = 0, beta >= 1 and rmax > 127 - beta, never with the
 *                    default parameters)
 *                    R_e[t] = (-1)^(sgn xor (Q_e<0)) * corr(|Q_e| == m1 ? m2 : m1)
 *                    APP_c[(t+P) mod Z] = sat8(Q_e + R_e[t])
 *   corr(m) = clamp(m - (m >> nshift) - beta, 0, rmax)   (normalised and/or offset min-sum, |messages| <= rmax)
 *   Channel LLRs are scaled and clamped: with the AGC (ldpc_agc_setup) L = clamp(sat8((x G + 2^(t-1)) >> t)), t = sh + s
 *   (fillers [kp, K) bypass the gain), else L = clamp((llr + 2^(s-1)) >> s, +-(deg*rmax - 1)), deg = 1 for
 *   degree-1 columns, 2 otherwise (ldpc_chan_max).
 *   Only rows 0 .. nact-1 are processed (and checked): nact = ldpc_active_rows(), rows whose degree-1 column lies
 *   beyond the last non-zero input LLR are skipped. Such a row is an exact no-op (its degree-1 Q is 0, so m1 = 0 and
 *   every message to the other columns is 0) and its check can always be satisfied by its own parity bit; the APP
 *   of its degree-1 column is reported as the channel value 0.
 *   After every iteration: syndrome of the hard decision (APP < 0) on all nact*Z checks; stop when zero.
 *   CRC mode (a CRC callback is given, early termination on): stop when the core rows 0..3 are satisfied and the CRC
 *   of the systematic bits passes, checked after every iteration and after the core rows of iterations >= 2; only for
 *   codes whose core rows hold at most 60 % of the edges (else the full-syndrome rule above).
 */
#include <stdlib.h>
#include <string.h>
#include "ldpc_internal.h"

static inline int8_t sat8(int x) { return (int8_t)(x > 127 ? 127 : x < -128 ? -128 : x); }

/* 1 if the hard decision satisfies rows 0 .. nrows-1 */
static int syn_rows(const int8_t *app, const int *ecol, const int *eshift, const int *erow0, int nrows, int z)
{
  for (int i = 0; i < nrows; i++)
    for (int t = 0; t < z; t++) {
      int par = 0;
      for (int e = erow0[i]; e < erow0[i + 1]; e++) par ^= app[ecol[e] * z + (t + eshift[e]) % z] < 0;
      if (par) return 0;
    }
  return 1;
}

int ldpc_ref_decode(int bg, int z, int rcode, int nact, int max_iter, int no_et, int nshift, int beta, int rmax, int in_shift,
                    const ldpc_agc_t *agc, const int8_t *llr_in, int8_t *app_out, int *ok,
                    int (*crc)(void *ctx, const int8_t *app), void *crc_ctx)
{
  const int ncol = ldpc_ncol(bg, rcode);
  const int kbcols = bg == 1 ? 22 : 10, nrow_all = ncol - kbcols;
  const int ils = ldpc_ils_of_z[z];
  const ldpc_bg_edge_t *tab = bg == 1 ? ldpc_bg1_edges : ldpc_bg2_edges;
  const int ntab = bg == 1 ? LDPC_BG1_EDGES : LDPC_BG2_EDGES;
  int ne = 0, ecol[LDPC_MAXEDGE], eshift[LDPC_MAXEDGE], erow0[LDPC_MAXROW + 1], cdeg[LDPC_MAXCOL] = {0};
  for (int i = 0; i < nrow_all; i++) {
    erow0[i] = ne;
    for (int k = 0; k < ntab; k++)
      if (tab[k].row == i) {
        ecol[ne] = tab[k].col;
        eshift[ne] = tab[k].v[ils] % z;
        cdeg[tab[k].col]++;
        ne++;
      }
  }
  erow0[nrow_all] = ne;
  /* CRC-mode rule decided on the full graph of the rate (as the plan does) */
  const int crcstop = crc && !no_et && nrow_all >= LDPC_NCORE && 10 * erow0[LDPC_NCORE] <= 6 * ne;
  const int nrow = nact > 0 && nact < nrow_all ? nact : nrow_all; /* rows beyond the last non-zero LLR: no-ops */
  int8_t *app = malloc((size_t)ncol * z), *R = calloc((size_t)ne * z, 1), *Q = malloc(32 * (size_t)1);
  int8_t *llr = malloc((size_t)ncol * z);
  for (int i = 0; i < ncol * z; i++) {
    const int cm = ldpc_chan_max(rmax, cdeg[i / z] == 1);
    if (agc && agc->G && !(i >= agc->kp && i < agc->K)) { /* AGC (in_shift folded into the gain); fillers bypass it */
      const int8_t v = ldpc_gain1(llr_in[i], agc->G, agc->sh + in_shift);
      llr[i] = (int8_t)(v > cm ? cm : v < -cm ? -cm : v);
    } else /* fillers with the AGC on: +127 (known bits; the gain is not applied) */
      llr[i] = ldpc_scale_in(agc && agc->G && i >= agc->kp && i < agc->K ? 127 : llr_in[i], in_shift, cm);
  }
  memcpy(app, llr, (size_t)ncol * z);
  int it = 0, good = 0;
  const int midcheck = crcstop;
  while (it < max_iter) {
    it++;
    int stop = 0;
    for (int i = 0; i < nrow && !stop; i++) {
      const int e0 = erow0[i], deg = erow0[i + 1] - e0;
      for (int t = 0; t < z; t++) {
        unsigned m1 = 255, m2 = 255;
        int sgn = 0;
        for (int e = 0; e < deg; e++) {
          const int c = ecol[e0 + e], j = (t + eshift[e0 + e]) % z;
          int q = cdeg[c] == 1 ? llr[c * z + j] : sat8(app[c * z + j] - R[(size_t)(e0 + e) * z + t]);
          Q[e] = (int8_t)q;
          unsigned a = (unsigned)(q < 0 ? -q : q);
          if (a > 127) a = 127;
          if (a < m1) { m2 = m1; m1 = a; }
          else if (a < m2) m2 = a;
          sgn ^= q < 0;
        }
        const int c1 = ldpc_corr(m1, nshift, beta, rmax), c2 = ldpc_corr(m2, nshift, beta, rmax);
        for (int e = 0; e < deg; e++) {
          const int c = ecol[e0 + e], j = (t + eshift[e0 + e]) % z;
          const int q = Q[e];
          unsigned a = (unsigned)(q < 0 ? -q : q);
          if (a > 127) a = 127;
          int r = a == m1 ? c2 : c1;
          if (sgn ^ (q < 0)) r = -r;
          R[(size_t)(e0 + e) * z + t] = (int8_t)r;
          app[c * z + j] = sat8(q + r);
        }
      }
      /* CRC mode: after the core rows of iteration >= 2: core parity + CRC */
      if (midcheck && it >= 2 && i == LDPC_NCORE - 1 && syn_rows(app, ecol, eshift, erow0, LDPC_NCORE, z) && crc(crc_ctx, app)) {
        good = 1;
        stop = 1;
      }
    }
    if (stop) break;
    if (crcstop) { /* after every iteration: core parity + CRC */
      good = syn_rows(app, ecol, eshift, erow0, LDPC_NCORE, z);
      if (good && crc(crc_ctx, app)) break;
      continue;
    }
    good = syn_rows(app, ecol, eshift, erow0, nrow, z);
    if (good && !no_et) break;
  }
  if (app_out) memcpy(app_out, app, (size_t)ncol * z);
  if (ok) *ok = good;
  free(app);
  free(llr);
  free(R);
  free(Q);
  return it;
}
