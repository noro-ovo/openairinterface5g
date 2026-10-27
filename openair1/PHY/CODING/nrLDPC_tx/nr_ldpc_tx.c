/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*
 * Packed LDPC transmit chain driver: plans, kernel dispatch, bit-sliced encoder, rate-matching window and output
 * passes (TS 38.212 5.3.2, 5.4.2.1, 5.4.2.2).
 *
 * Encoder (exploits the structure of H):
 *   H = [ A  B  0 ]   rows 0..3: A = systematic columns 0..kb-1, B = 4x4 core on columns kb..kb+3
 *       [ C  D  I ]   rows 4..m-1: extension, identity on column kb+r
 *   B column 0 has shift a in rows 0 and 3 and shift b in the "special" row (row 1 for BG1, row 2 for BG2),
 *   the other B entries are identities on the double diagonal.
 *   lambda_r = sum_j P_rj s_j (r = 0..3)
 *   sum of the 4 core rows: P^b p0 = lambda_0+..+lambda_3  ->  p0 = P^-b sigma
 *   p1 = lambda_0 + P^a p0,  p3 = lambda_3 + P^a p0,  p2 = lambda_2 + p3 (BG1) / lambda_1 + p1 (BG2)
 *   extension rows: p_{kb+r} = sum_{j < kb+4} P_rj x_j
 * Every term is a circulant-shifted block = unaligned load from a doubled column [x | x], in bytes holding one bit
 * of each of the 8 code blocks of a group. Only the codeword prefix d_0 .. d_{nout-1} that the circular-buffer
 * window reads is produced: extension rows past it are not computed.
 *
 * Output: the rate-matched sequence of a code block is e_k = d_{pos((rank0 + k) mod P)}, P = Ncb - F non-filler
 * positions, and the interleaved sequence is f_{Qm t + q} = e_{q E/Qm + t}. Both are read in chunks straight from
 * d (one contiguous span per stream and chunk, copied only across a filler or wrap boundary), the bits of each
 * code block are extracted from the bit-sliced bytes into 64-bit words and written at the code block's offset.
 */
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "nr_ldpc_tx.h"
#include "nr_ldpc_tx_internal.h"
#include "nr_ldpc_tx_tables.h"

#define MAXE 316

typedef struct {
  int bg, z, kb, m, S, a, b;
  int lam_ne, ext_ne;
  uint8_t lam_cnt[4], ext_cnt[42];
  uint16_t lam_off[MAXE], ext_off[MAXE];
  uint8_t lam_col[MAXE], ext_col[MAXE];
  uint16_t sig_off[4], p0_off[1], p13_off[4], p2_off[2];
  uint16_t dbl_off[22];
} plan_t;

static plan_t g_plan[2][51];
static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static const nr_ldpc_tx_kern_t *g_kern, *g_default;
static const nr_ldpc_tx_kern_t *g_avail[6];
static const char *const g_names[6] = {"scalar", "neon", "neon-eor3", "sve2", "avx2", "avx512"};

static int ils_of(int z)
{
  if (z < 2 || z > 384)
    return -1;
  for (int i = 0; i < 8; i++) {
    int a = nr_ldpc_ls_a[i];
    if (z % a == 0) {
      int q = z / a;
      if ((q & (q - 1)) == 0)
        return i;
    }
  }
  return -1;
}

static int8_t g_zidx[NR_LDPC_TX_ZMAX + 1]; /* Z -> index in nr_ldpc_zlist, -1 if not a lifting size */
static int zidx(int z)
{
  return (unsigned)z <= NR_LDPC_TX_ZMAX ? g_zidx[z] : -1;
}

static void build_plan(plan_t *p, int bg, int z)
{
  const nr_ldpc_edge_t *E = bg == 1 ? nr_ldpc_bg1_edges : nr_ldpc_bg2_edges;
  const int ne = bg == 1 ? NR_LDPC_BG1_NEDGES : NR_LDPC_BG2_NEDGES;
  const int ils = ils_of(z);
  memset(p, 0, sizeof(*p));
  p->bg = bg;
  p->z = z;
  p->kb = bg == 1 ? 22 : 10;
  p->m = bg == 1 ? 46 : 42;
  p->S = 2 * z;
  const int kb = p->kb, S = p->S, sp = bg == 1 ? 1 : 2;
  for (int e = 0; e < ne; e++) {
    const int r = E[e].row, c = E[e].col, P = E[e].v[ils] % z;
    if (c == kb && r == 0)
      p->a = P;
    if (c == kb && r == sp)
      p->b = P;
    if (r < 4) {
      if (c < kb) {
        p->lam_off[p->lam_ne] = c * S + P;
        p->lam_col[p->lam_ne++] = c;
        p->lam_cnt[r]++;
      }
    } else if (c != kb + r) {
      p->ext_off[p->ext_ne] = c * S + P;
      p->ext_col[p->ext_ne++] = c;
      p->ext_cnt[r - 4]++;
    }
  }
  for (int c = 0; c < kb; c++)
    p->dbl_off[c] = c * z;
  const int LAM = (kb + 4) * S, SIG = (kb + 8) * S, P0 = kb * S;
  for (int r = 0; r < 4; r++)
    p->sig_off[r] = LAM + r * S;
  p->p0_off[0] = SIG + (z - p->b) % z;
  p->p13_off[0] = LAM + 0 * S;
  p->p13_off[1] = P0 + p->a;
  p->p13_off[2] = LAM + 3 * S;
  p->p13_off[3] = P0 + p->a;
  if (bg == 1) {
    p->p2_off[0] = LAM + 2 * S;
    p->p2_off[1] = (kb + 3) * S;
  } else {
    p->p2_off[0] = LAM + 1 * S;
    p->p2_off[1] = (kb + 1) * S;
  }
}

static void init_once(void)
{
  memset(g_zidx, -1, sizeof(g_zidx));
  for (int i = 0; i < 51; i++)
    g_zidx[nr_ldpc_zlist[i]] = (int8_t)i;
  for (int bg = 1; bg <= 2; bg++)
    for (int i = 0; i < 51; i++)
      build_plan(&g_plan[bg - 1][i], bg, nr_ldpc_zlist[i]);
  g_avail[0] = nr_ldpc_tx_kern_scalar();
  g_avail[1] = nr_ldpc_tx_kern_neon();
  g_avail[2] = nr_ldpc_tx_kern_neon3();
  g_avail[3] = nr_ldpc_tx_kern_sve2();
  g_avail[4] = nr_ldpc_tx_kern_avx2();
  g_avail[5] = nr_ldpc_tx_kern_avx512();
  /* aarch64: at VL = 128 bits SVE2 has no width advantage and its one-vector loads cost more instructions than
   * NEON's ldp / ld1 {4 regs}, so NEON+EOR3 (FEAT_SHA3) is preferred and SVE2 is used only for VL > 128 */
  int pref[] = {5, 4, 3, 2, 1, 0};
  if (g_avail[3] && g_avail[3]->tb <= 16) {
    pref[2] = 2;
    pref[3] = 3;
  }
  for (unsigned i = 0; i < sizeof(pref) / sizeof(pref[0]); i++)
    if (g_avail[pref[i]]) {
      g_kern = g_avail[pref[i]];
      break;
    }
  const char *env = getenv("NR_LDPC_TX_VARIANT");
  if (env)
    for (int v = 0; v < 6; v++)
      if (!strcmp(env, g_names[v]) && g_avail[v])
        g_kern = g_avail[v];
  g_default = g_kern;
}

int nr_ldpc_tx_init(void)
{
  pthread_once(&g_once, init_once);
  return 0;
}

int nr_ldpc_tx_variant_available(const char *name)
{
  nr_ldpc_tx_init();
  if (!name || !strcmp(name, "auto"))
    return 1;
  for (int v = 0; v < 6; v++)
    if (!strcmp(name, g_names[v]))
      return g_avail[v] != NULL;
  return 0;
}

int nr_ldpc_tx_set_variant(const char *name)
{
  nr_ldpc_tx_init();
  if (!name || !strcmp(name, "auto")) {
    g_kern = g_default;
    return 0;
  }
  for (int v = 0; v < 6; v++)
    if (!strcmp(name, g_names[v]) && g_avail[v]) {
      g_kern = g_avail[v];
      return 0;
    }
  return -1;
}

const char *nr_ldpc_tx_variant(void)
{
  nr_ldpc_tx_init();
  return g_kern->name;
}

/* ------------------------------------------------------------------------------------------------- encoder */

/* drop the edges of all-filler systematic columns [clim, kb) */
static int filter_edges(const uint16_t *off,
                        const uint8_t *col,
                        const uint8_t *cnt,
                        int nrows,
                        int clim,
                        int kb,
                        uint16_t *ooff,
                        uint8_t *ocnt)
{
  int k = 0, n = 0;
  for (int r = 0; r < nrows; r++) {
    int c = 0;
    for (int e = 0; e < cnt[r]; e++, k++)
      if (col[k] < clim || col[k] >= kb) {
        ooff[n++] = off[k];
        c++;
      }
    ocnt[r] = c;
  }
  return n;
}

static const uint8_t g_zero[NR_LDPC_TX_ZMAX * 22 / 8 + 256] __attribute__((aligned(64)));

/* d_0 .. d_{nout-1} of ncb (0..8) code blocks, bit-sliced (see nr_ldpc_tx_encode_sliced) */
static void
encode_sliced(const nr_ldpc_tx_kern_t *kern, const plan_t *p, int kprime, int ncb, const uint8_t *const *in, uint8_t *out, int nout)
{
  const int z = p->z, kb = p->kb, K = kb * z, S = p->S;
  uint8_t work[(NR_LDPC_TX_MAXCOLS + 5) * 2 * NR_LDPC_TX_ZMAX + NR_LDPC_TX_PAD] __attribute__((aligned(64)));
  uint8_t lin[22 * NR_LDPC_TX_ZMAX + 8 * 256 + 64] __attribute__((aligned(64)));
  const uint8_t *ptr[8];
  for (int j = 0; j < 8; j++)
    ptr[j] = j < ncb ? in[j] : g_zero;

  const int nsys = K - 2 * z, clim = (kprime + z - 1) / z;
  /* 1. systematic columns: bit transposition straight into the doubled work columns (+ output) when possible */
  int c0 = 0, cw = 2;
  if (kern->tcols && (z & 7) == 0 && z >= kern->tcols_zmin && kprime >= z && (!kern->tcols_use || kern->tcols_use(z))) {
    c0 = kprime / z; /* columns without filler bits */
    cw = 2 + nout / z;
    if (cw > c0)
      cw = c0;
    if (cw < 2)
      cw = 2;
    kern->tcols(ptr, z, c0, work, S, out, cw);
  }
  if (c0 < clim) {
    /* remaining bits [c0*z, kprime) -> lin (zero padded), then doubled into the work columns c0..clim-1 */
    const uint8_t *pp[8];
    const int b0 = c0 * z / 8, nbits = kprime - c0 * z;
    for (int j = 0; j < 8; j++)
      pp[j] = ptr[j] + b0;
    const int tb = kern->tb, nbytes = nbits >> 3, rem = nbits & 7;
    const int nblk = nbytes / tb;
    if (nblk)
      kern->transpose(pp, nblk, lin);
    const int done = nblk * tb, tail = nbytes - done + (rem != 0);
    if (tail) {
      uint8_t stg[8][256] __attribute__((aligned(64)));
      const uint8_t *sp[8];
      for (int j = 0; j < 8; j++) {
        memset(stg[j], 0, tb);
        memcpy(stg[j], pp[j] + done, tail);
        if (rem)
          stg[j][tail - 1] &= (uint8_t)(0xff00 >> rem);
        sp[j] = stg[j];
      }
      kern->transpose(sp, 1, lin + 8 * done);
    }
    if (nbits < (clim - c0) * z)
      memset(lin + nbits, 0, (clim - c0) * z - nbits);
    static const uint8_t ones[22] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    kern->rows(lin, p->dbl_off, ones, clim - c0, z, work + c0 * S, S, 1);
  }
  /* 2. systematic output for the columns not written by the column transposer (filler columns are 0):
   *    columns [cw, c0) from the work area (only when nout cuts them), [max(c0,2), clim) from lin, the rest 0 */
  for (int c = cw; c < c0; c++) {
    const int pos = (c - 2) * z;
    if (pos >= nout)
      break;
    memcpy(out + pos, work + c * S, nout - pos < z ? nout - pos : z);
  }
  {
    const int s0 = (c0 > 2 ? c0 : 2) * z - 2 * z; /* first output position not yet written */
    int s1 = clim * z - 2 * z; /* end of the non-filler columns */
    if (s1 > nout)
      s1 = nout;
    if (s1 > s0)
      memcpy(out + s0, lin + s0 + 2 * z - c0 * z, s1 - s0);
    const int e = nsys < nout ? nsys : nout;
    const int f0 = s1 > s0 ? s1 : s0;
    if (e > f0)
      memset(out + f0, 0, e - f0);
  }
  if (nout <= nsys)
    return;
  memset(work + (kb + 9) * S, 0, NR_LDPC_TX_PAD);

  const uint16_t *lam_off = p->lam_off, *ext_off = p->ext_off;
  const uint8_t *lam_cnt = p->lam_cnt, *ext_cnt = p->ext_cnt;
  uint16_t flo[MAXE], feo[MAXE];
  uint8_t flc[4], fec[42];
  if (clim < kb) {
    filter_edges(p->lam_off, p->lam_col, p->lam_cnt, 4, clim, kb, flo, flc);
    filter_edges(p->ext_off, p->ext_col, p->ext_cnt, p->m - 4, clim, kb, feo, fec);
    lam_off = flo;
    lam_cnt = flc;
    ext_off = feo;
    ext_cnt = fec;
  }

  /* 3. core: lambda_0..3, sigma, p0, p1/p3, p2 (all kept doubled in the work area) */
  static const uint8_t c1[1] = {1}, c2[2] = {2, 2}, c4[1] = {4};
  const int LAM = (kb + 4) * S, SIG = (kb + 8) * S;
  kern->rows(work, lam_off, lam_cnt, 4, z, work + LAM, S, 1);
  kern->rows(work, p->sig_off, c4, 1, z, work + SIG, S, 1);
  kern->rows(work, p->p0_off, c1, 1, z, work + kb * S, S, 1);
  kern->rows(work, p->p13_off, c2, 2, z, work + (kb + 1) * S, 2 * S, 1);
  kern->rows(work, p->p2_off, c2, 1, z, work + (kb + 2) * S, S, 1);

  /* 4. parity output: core rows, then the extension rows that nout reaches, directly into out */
  const int npar = nout - nsys;
  uint8_t *op = out + nsys;
  for (int r = 0; r < 4 && r * z < npar; r++)
    memcpy(op + r * z, work + (kb + r) * S, npar - r * z < z ? npar - r * z : z);
  if (npar > 4 * z) {
    const int nfull = npar / z - 4, part = npar % z;
    if (nfull > 0)
      kern->rows(work, ext_off, ext_cnt, nfull, z, op + 4 * z, z, 0);
    if (part) {
      int skip = 0;
      for (int r = 0; r < nfull; r++)
        skip += ext_cnt[r];
      kern->rows(work, ext_off + skip, ext_cnt + nfull, 1, z, work + LAM, z, 0);
      memcpy(op + (4 + nfull) * z, work + LAM, part);
    }
  }
}

int nr_ldpc_tx_encode_sliced(int bg, int z, int kprime, int n, const uint8_t *const *c, uint8_t *out, int nout)
{
  nr_ldpc_tx_init();
  const int zi = zidx(z);
  if ((bg != 1 && bg != 2) || zi < 0 || n < 0 || n > 8)
    return -1;
  const plan_t *p = &g_plan[bg - 1][zi];
  const int N = (bg == 1 ? 66 : 50) * z;
  if (kprime <= 2 * z || kprime > p->kb * z || nout < 0 || nout > N)
    return -1;
  if (nout > 0)
    encode_sliced(g_kern, p, kprime, n, c, out, nout);
  return nout;
}

/* ------------------------------------------------------------------------------------------- output stage */

typedef struct {
  const uint8_t *d; /* bit-sliced d_0 .. */
  uint32_t P; /* non-filler positions of the circular buffer: Ncb - F */
  uint32_t foff; /* first filler position in d: K' - 2Z */
  uint32_t F;
} rm_t;

static inline uint32_t rm_pos(const rm_t *rm, uint32_t u)
{
  return u < rm->foff ? u : u + rm->F;
}

/* e_r .. e_{r+L-1} of the rate-matched sequence (r: rank in the circular buffer, any value): a pointer into d when
 * the span is contiguous there, otherwise the span is copied into scratch */
static const uint8_t *e_span(const rm_t *rm, uint32_t r, int L, uint8_t *scratch)
{
  uint32_t u = r % rm->P;
  uint32_t run = (rm->F && u < rm->foff) ? rm->foff - u : rm->P - u;
  if ((uint32_t)L <= run)
    return rm->d + rm_pos(rm, u);
  int k = 0;
  while (k < L) {
    run = (rm->F && u < rm->foff) ? rm->foff - u : rm->P - u;
    const int cnt = (uint32_t)(L - k) < run ? L - k : (int)run;
    memcpy(scratch + k, rm->d + rm_pos(rm, u), cnt);
    k += cnt;
    u += cnt;
    if (u == rm->P)
      u = 0;
  }
  return scratch;
}

static inline void or_word(uint64_t *p, uint64_t v)
{
  if (v)
    __atomic_fetch_or(p, v, __ATOMIC_RELAXED);
}

/* one pass over the interleaved sequence of length E for the code blocks of mask `planes` */
static void out_pass(const nr_ldpc_tx_kern_t *kern,
                     const rm_t *rm,
                     uint32_t rank0,
                     int qm,
                     uint32_t E,
                     unsigned planes,
                     const uint32_t *bit0, /* per code block: bit offset relative to out64 */
                     uint64_t *out64)
{
  uint8_t stage[NR_LDPC_TX_STAGE] __attribute__((aligned(64)));
  uint8_t scratch[NR_LDPC_TX_CH + 8 * NR_LDPC_TX_OVERREAD] __attribute__((aligned(64)));
  uint64_t W[8][NR_LDPC_TX_WS] __attribute__((aligned(64)));
  const uint32_t EQm = E / qm;
  const int sstride = NR_LDPC_TX_CH / qm + NR_LDPC_TX_OVERREAD;
  uint32_t w0[8], g_lo[8], g_hi[8];
  int sh[8];
  for (int j = 0; j < 8; j++) {
    W[j][0] = 0;
    if (!(planes >> j & 1))
      continue;
    w0[j] = bit0[j] >> 6;
    sh[j] = bit0[j] & 63;
    g_lo[j] = sh[j] != 0;
    g_hi[j] = (sh[j] + E) >> 6;
  }
  for (uint32_t i0 = 0; i0 < E; i0 += NR_LDPC_TX_CH) {
    const int n = E - i0 < NR_LDPC_TX_CH ? (int)(E - i0) : NR_LDPC_TX_CH;
    const int L = n / qm, nw = (n + 63) >> 6;
    const uint32_t t0 = i0 / qm, g0 = i0 >> 6;
    const uint8_t *src[8];
    for (int q = 0; q < qm; q++)
      src[q] = e_span(rm, rank0 + q * EQm + t0, L, scratch + q * sstride);
    kern->extract(qm, src, L, stage, &W[0][1], planes);
    const int last = i0 + n == E;
    for (int j = 0; j < 8; j++) {
      if (!(planes >> j & 1))
        continue;
      uint64_t *w = &W[j][1];
      if (last && (n & 63))
        w[nw - 1] &= ~0ULL >> (64 - (n & 63));
      /* words g0 .. g0+nw-1: interior run [a, b) stored by the kernel, the edge words ORed */
      uint32_t a = g_lo[j] > g0 ? g_lo[j] : g0, b = g_hi[j] < g0 + nw ? g_hi[j] : g0 + nw;
      if (b < a)
        b = a;
      for (uint32_t g = g0; g < a && g < g0 + nw; g++) {
        const int k = g - g0;
        or_word(out64 + w0[j] + g, (w[k] << sh[j]) | (sh[j] ? w[k - 1] >> (64 - sh[j]) : 0));
      }
      if (b > a)
        kern->write(out64 + w0[j] + a, w + (a - g0), b - a, sh[j]);
      for (uint32_t g = b; g < g0 + nw; g++) {
        const int k = g - g0;
        or_word(out64 + w0[j] + g, (w[k] << sh[j]) | (sh[j] ? w[k - 1] >> (64 - sh[j]) : 0));
      }
      if (last) {
        if (sh[j])
          or_word(out64 + w0[j] + g0 + nw, w[nw - 1] >> (64 - sh[j]));
      } else {
        W[j][0] = w[nw - 1];
      }
    }
  }
}

int nr_ldpc_tx_encode(const nr_ldpc_tx_params_t *prm,
                      int n,
                      const uint8_t *const *c,
                      const uint32_t *E,
                      const uint32_t *Eoff,
                      uint8_t *out)
{
  nr_ldpc_tx_init();
  const int bg = prm->BG, z = prm->Z, qm = prm->Qm;
  const int zi = zidx(z);
  if ((bg != 1 && bg != 2) || zi < 0 || n < 1 || n > 8 || !(qm == 1 || qm == 2 || qm == 4 || qm == 6 || qm == 8))
    return -1;
  const plan_t *p = &g_plan[bg - 1][zi];
  const uint32_t N = (bg == 1 ? 66 : 50) * z, K = prm->K, F = prm->F, Ncb = prm->Ncb;
  if (K != (uint32_t)p->kb * z || F >= K || K - F <= 2u * z || Ncb > N || prm->k0 >= Ncb)
    return -1;
  const uint32_t kprime = K - F, foff = kprime - 2 * z;
  /* same validity rule as nr_rate_matching_ldpc() */
  if (foff > Ncb || F > Ncb - foff || F == Ncb)
    return -1;
  uint32_t Emax = 0;
  for (int j = 0; j < n; j++) {
    if (E[j] % qm)
      return -1;
    if (E[j] > Emax)
      Emax = E[j];
  }
  if (Emax == 0)
    return 0;

  rm_t rm = {.P = Ncb - F, .foff = foff, .F = F};
  const uint32_t k0 = prm->k0;
  const uint32_t rank0 = k0 < foff ? k0 : (k0 < foff + F ? foff : k0 - F);
  /* codeword prefix that the window reads */
  const uint32_t nout = rank0 + Emax > rm.P ? rm_pos(&rm, rm.P - 1) + 1 : rm_pos(&rm, rank0 + Emax - 1) + 1;

  uint8_t d[66 * NR_LDPC_TX_ZMAX + NR_LDPC_TX_OVERREAD + 64] __attribute__((aligned(64)));
  encode_sliced(g_kern, p, kprime, n, c, d, nout);
  rm.d = d;

  /* bit offsets relative to out aligned down to 8 bytes */
  const uintptr_t mis = (uintptr_t)out & 7;
  uint64_t *out64 = (uint64_t *)(out - mis);
  uint32_t bit0[8] = {0};
  for (int j = 0; j < n; j++)
    bit0[j] = Eoff[j] + 8 * mis;
  /* one pass per distinct E (a transport block has at most two values; the 8-code-block extraction costs about the
   * same per pass as merging the two stream sets byte-wise would) */
  unsigned todo = 0;
  for (int j = 0; j < n; j++)
    if (E[j])
      todo |= 1u << j;
  while (todo) {
    const uint32_t e = E[__builtin_ctz(todo)];
    unsigned planes = 0;
    for (int j = 0; j < n; j++)
      if ((todo >> j & 1) && E[j] == e)
        planes |= 1u << j;
    out_pass(g_kern, &rm, rank0, qm, e, planes, bit0, out64);
    todo &= ~planes;
  }
  return 0;
}
