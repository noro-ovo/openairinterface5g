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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ldpc_internal.h"

static inline int8_t sat8(int x) { return (int8_t)(x > 127 ? 127 : x < -128 ? -128 : x); }

/* error-correction experiments (README section 11), reference implementation only, selected by environment:
   LDPC_REF_B2D="dc,de,d1"   offset deltas: core rows 0-3, extension rows, first iteration (2-D correction)
   LDPC_REF_TBL="t0,..,t7"   message of the non-minimum edges reduced by t[min(m2 - m1, 7)] (lambda-min-like)
   LDPC_REF_ABS=1            absorbed messages: R = APP_new - Q (the increment actually applied, after saturation)
   LDPC_REF_ORDER=1          iterations >= 2: rows in ascending degree order (stable) */
static int x_init, x_b2d[3], x_tbl[8], x_tblon, x_abs, x_order, x_app16;
static void x_read(void)
{
  if (x_init) return;
  const char *e = getenv("LDPC_REF_B2D");
  if (e) sscanf(e, "%d,%d,%d", &x_b2d[0], &x_b2d[1], &x_b2d[2]);
  e = getenv("LDPC_REF_TBL");
  if (e) x_tblon = sscanf(e, "%d,%d,%d,%d,%d,%d,%d,%d", &x_tbl[0], &x_tbl[1], &x_tbl[2], &x_tbl[3], &x_tbl[4], &x_tbl[5], &x_tbl[6], &x_tbl[7]) > 0;
  e = getenv("LDPC_REF_ABS");
  x_abs = e && atoi(e);
  e = getenv("LDPC_REF_ORDER");
  x_order = e ? atoi(e) : 0;
  e = getenv("LDPC_REF_APP16"); /* int16 a-posteriori LLRs, int8 messages (hybrid variant) */
  x_app16 = e && atoi(e);
  x_init = 1;
}

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
  /* whole filler columns [fc0, kbcols) (every bit a known 0) are removed from the graph: a known bit is an infinite
     positive LLR, it never changes a sign and is never a minimum */
  const int fc0 = ldpc_filler_col0(z, kbcols, agc ? agc->kp : 0, agc ? agc->K : 0);
  int ne_full = 0, ncore_full = 0;
  for (int k = 0; k < ntab; k++) {
    ne_full += tab[k].row < nrow_all;
    ncore_full += tab[k].row < LDPC_NCORE;
  }
  for (int i = 0; i < nrow_all; i++) {
    erow0[i] = ne;
    for (int k = 0; k < ntab; k++)
      if (tab[k].row == i && !(tab[k].col >= fc0 && tab[k].col < kbcols)) {
        ecol[ne] = tab[k].col;
        eshift[ne] = tab[k].v[ils] % z;
        cdeg[tab[k].col]++;
        ne++;
      }
  }
  erow0[nrow_all] = ne;
  /* CRC-mode rule decided on the full graph of the rate (as the plan does) */
  const int crcstop = crc && !no_et && nrow_all >= LDPC_NCORE && 10 * ncore_full <= 6 * ne_full;
  const int nrow = nact > 0 && nact < nrow_all ? nact : nrow_all; /* rows beyond the last non-zero LLR: no-ops */
  int8_t *app = malloc((size_t)ncol * z), *R = calloc((size_t)ne * z, 1), *Q = malloc(32 * (size_t)1);
  int8_t *llr = malloc((size_t)ncol * z);
  for (int i = 0; i < ncol * z; i++) {
    const int cm = ldpc_chan_max(rmax, cdeg[i / z] == 1);
    const int fill = agc && i >= agc->kp && i < agc->K;
    if (agc && agc->G && !fill) { /* AGC (in_shift folded into the gain); fillers bypass it */
      const int8_t v = ldpc_gain1(llr_in[i], agc->G, agc->sh + in_shift);
      llr[i] = (int8_t)(v > cm ? cm : v < -cm ? -cm : v);
    } else /* fillers: +127 whatever the caller passed (known bits; the gain is not applied) */
      llr[i] = ldpc_scale_in(fill ? 127 : llr_in[i], in_shift, cm);
  }
  memcpy(app, llr, (size_t)ncol * z);
  x_read();
  int16_t *a16 = x_app16 ? malloc(sizeof(int16_t) * (size_t)ncol * z) : NULL;
  if (a16) for (int i = 0; i < ncol * z; i++) a16[i] = llr[i];
  int qv[32];
  int it = 0, good = 0;
  x_read();
  /* layer order: ascending row degree (stable) in every iteration (ldpc_row_order(), default); the CRC-mode check
     after the core rows exists only in natural order (the core rows have the largest degrees: they come last) */
  const int ordered = ldpc_row_order(bg, rcode) || x_order == 2;
  const int midcheck = crcstop && !ordered && !x_order;
  int ord[LDPC_MAXROW];
  for (int i = 0; i < nrow; i++) ord[i] = i;
  if (ordered || x_order) /* ascending degree, stable */
    for (int a = 1; a < nrow; a++)
      for (int b = a; b > 0 && erow0[ord[b] + 1] - erow0[ord[b]] < erow0[ord[b - 1] + 1] - erow0[ord[b - 1]]; b--) {
        const int tt = ord[b]; ord[b] = ord[b - 1]; ord[b - 1] = tt;
      }
  while (it < max_iter) {
    it++;
    int stop = 0;
    for (int ii = 0; ii < nrow && !stop; ii++) {
      const int i = it >= 2 || ordered ? ord[ii] : ii; /* (LDPC_REF_ORDER=1: iterations >= 2 only; experiments) */
      const int e0 = erow0[i], deg = erow0[i + 1] - e0;
      int bt = beta + (i < LDPC_NCORE ? x_b2d[0] : x_b2d[1]) + (it == 1 ? x_b2d[2] : 0);
      if (bt < 0) bt = 0;
      for (int t = 0; t < z; t++) {
        unsigned m1 = 255, m2 = 255;
        int sgn = 0;
        for (int e = 0; e < deg; e++) {
          const int c = ecol[e0 + e], j = (t + eshift[e0 + e]) % z;
          int q = cdeg[c] == 1 ? llr[c * z + j] : a16 ? a16[c * z + j] - R[(size_t)(e0 + e) * z + t] : sat8(app[c * z + j] - R[(size_t)(e0 + e) * z + t]);
          qv[e] = q;
          unsigned a = (unsigned)(q < 0 ? -q : q);
          if (a > 127) a = 127;
          if (a < m1) { m2 = m1; m1 = a; }
          else if (a < m2) m2 = a;
          sgn ^= q < 0;
        }
        int c1 = ldpc_corr(m1, nshift, bt, rmax);
        const int c2 = ldpc_corr(m2, nshift, bt, rmax);
        if (x_tblon) { const int dd = (int)(m2 - m1) > 7 ? 7 : (int)(m2 - m1); c1 -= x_tbl[dd]; if (c1 < 0) c1 = 0; }
        for (int e = 0; e < deg; e++) {
          const int c = ecol[e0 + e], j = (t + eshift[e0 + e]) % z;
          const int q = qv[e];
          unsigned a = (unsigned)(q < 0 ? -q : q);
          if (a > 127) a = 127;
          int r = a == m1 ? c2 : c1;
          if (sgn ^ (q < 0)) r = -r;
          if (a16) { /* hybrid: the APP never saturates (|APP| < 32767 always) */
            const int an16 = q + r;
            a16[c * z + j] = (int16_t)(an16 > 32767 ? 32767 : an16 < -32767 ? -32767 : an16);
            R[(size_t)(e0 + e) * z + t] = (int8_t)r;
            app[c * z + j] = sat8(an16);
            continue;
          }
          const int8_t an = sat8(q + r);
          R[(size_t)(e0 + e) * z + t] = (int8_t)((x_abs || ldpc_absorb()) && cdeg[c] != 1 ? an - q : r);
          app[c * z + j] = an;
        }
      }
      /* CRC mode: after the core rows of iteration >= 2: core parity + CRC */
      if (midcheck && it >= 2 && ii == LDPC_NCORE - 1 && syn_rows(app, ecol, eshift, erow0, LDPC_NCORE, z) && crc(crc_ctx, app)) {
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
  free(a16);
  if (ok) *ok = good;
  free(app);
  free(llr);
  free(R);
  free(Q);
  return it;
}
