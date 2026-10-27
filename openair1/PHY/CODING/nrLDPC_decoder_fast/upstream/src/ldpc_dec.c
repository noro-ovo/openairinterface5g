/* dec_std: plan construction, run-time dispatch, workspace, input/output and the iteration driver. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "ldpc_internal.h"
#ifndef LDPC_DEC_AGC_STAT
#define LDPC_DEC_AGC_STAT 1
#endif
#ifndef LDPC_DEC_QMT_UNK /* AGC target multipliers (percent) per modulation order, see ldpc_agc_target() */
#define LDPC_DEC_QMT_UNK 100
#endif
#ifndef LDPC_DEC_QMT_2
#define LDPC_DEC_QMT_2 125
#endif
#ifndef LDPC_DEC_QMT_4
#define LDPC_DEC_QMT_4 100
#endif
#ifndef LDPC_DEC_QMT_6
#define LDPC_DEC_QMT_6 100
#endif
#ifndef LDPC_DEC_QMT_8
#define LDPC_DEC_QMT_8 150
#endif
#ifndef LDPC_DEC_QMTN_UNK /* the same with the CRC-mode NMS default */
#define LDPC_DEC_QMTN_UNK 100
#define LDPC_DEC_QMTN_2 80
#define LDPC_DEC_QMTN_4 100
#define LDPC_DEC_QMTN_6 100
#define LDPC_DEC_QMTN_8 200
#endif
#ifndef LDPC_DEC_CRCALG_DEFAULT /* CRC mode (OAI's PHY): NMS 7/8 + 1 by default */
#define LDPC_DEC_CRCALG_DEFAULT 1
#endif
#ifndef LDPC_DEC_RSLOPE_1_13 /* AGC target rate slopes (percent per unit code rate), see ldpc_agc_target_rate() */
#define LDPC_DEC_RSLOPE_1_13 300
#define LDPC_DEC_RSLOPE_1_23 430
#define LDPC_DEC_RSLOPE_1_89 0
#define LDPC_DEC_RSLOPE_2_15 150
#define LDPC_DEC_RSLOPE_2_13 450
#define LDPC_DEC_RSLOPE_2_23 250
#endif
#ifndef LDPC_DEC_AGC_DEFAULT
#define LDPC_DEC_AGC_DEFAULT 1
#endif
#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

int ldpc_ncol(int bg, int rcode)
{
  if (bg == 1) return rcode == 13 ? 68 : rcode == 23 ? 35 : rcode == 89 ? 27 : 0;
  if (bg == 2) return rcode == 15 ? 52 : rcode == 13 ? 32 : rcode == 23 ? 17 : 0;
  return 0;
}

static inline int rup(int x, int a) { return (x + a - 1) / a * a; }

/* buffer read by visit v (row order) of a column of degree deg; visit v writes rb(v + 1) (rb(deg) = 0). Odd degrees
   use a third buffer for the last two visits, so a visit never reads and writes the same buffer and every
   iteration starts and ends in buffer 0. */
static int rbuf(int v, int deg)
{
  if (v <= 0 || v >= deg) return 0;
  if ((deg & 1) && v == deg - 1) return 2;
  return v & 1;
}

void ldpc_plan_free(ldpc_plan_t *p)
{
  free(p->rdtab);
  p->rdtab = NULL;
  p->tabcap = 0;
}

int ldpc_plan_build(ldpc_plan_t *p, int bg, int z0, int rcode, int nact, int fc0, int nb, int W, int U, int nshift, int beta,
                    int rmax, const ldpc_kernels_t *kt)
{
  /* nb > 1: nb code blocks interleaved lane-wise (lane t nb + b), i.e. one code of lifting size z0 nb with shifts
     P nb (ldpc_decode_batch()) */
  if (nb < 1) nb = 1;
  const int z = z0 * nb;
  int32_t *tab = p->rdtab;
  const int tabcap = p->tabcap;
  memset(p, 0, sizeof *p);
  p->rdtab = tab;
  p->tabcap = tabcap;
  const int ncol = ldpc_ncol(bg, rcode);
  if (!ncol || z0 < 2 || z > 384 || ldpc_ils_of_z[z0] == 255) return -1;
  const int ils = ldpc_ils_of_z[z0];
  const ldpc_bg_edge_t *etab = bg == 1 ? ldpc_bg1_edges : ldpc_bg2_edges;
  const int ntab = bg == 1 ? LDPC_BG1_EDGES : LDPC_BG2_EDGES;
  p->bg = bg; p->z = z; p->z0 = z0; p->nb = nb; p->rcode = rcode; p->ncol = ncol;
  p->kbcols = bg == 1 ? 22 : 10;
  const int nrow_full = ncol - p->kbcols;
  /* active rows only (ldpc_active_rows()): the others are exact no-ops. Every column of rows 0..3 has >= 2 edges in
     rows 0..3 (BG1 and BG2), so the kind and clamp of every column read by an active row is the full graph's; the
     degree-1 columns of the skipped rows become unused columns (channel copy, all zero) */
  p->nrow = nact >= LDPC_NCORE && nact < nrow_full ? nact : nrow_full;
  int ne_full = 0, ncore_full = 0;
  for (int k = 0; k < ntab; k++) {
    ne_full += etab[k].row < nrow_full;
    ncore_full += etab[k].row < LDPC_NCORE;
  }
  p->fc0 = fc0 > 0 && fc0 < p->kbcols ? fc0 : p->kbcols;
#define REMOVED(c) ((c) >= p->fc0 && (c) < p->kbcols)
  p->W = W; p->U = U; p->B = W * U;
  p->zp = rup(z, p->B);
  p->nshift = nshift; p->beta = beta; p->rmax = rmax;
  p->lim = ldpc_corr_lim(nshift, beta, rmax);
  const int B = p->B, zp = p->zp, D = B < z ? B : z;
  /* layer order (README section 11): rows by ascending degree (stable), in every iteration; else natural */
  int ordv[LDPC_MAXROW], rdeg[LDPC_MAXROW] = {0}, rpos[LDPC_MAXROW];
  for (int k = 0; k < ntab; k++)
    if (etab[k].row < p->nrow && !REMOVED(etab[k].col)) rdeg[etab[k].row]++;
  for (int i = 0; i < p->nrow; i++) ordv[i] = i;
  p->ordered = ldpc_row_order(bg, rcode);
  p->absorb = ldpc_absorb();
  if (p->ordered)
    for (int a = 1; a < p->nrow; a++)
      for (int b = a; b > 0 && rdeg[ordv[b]] < rdeg[ordv[b - 1]]; b--) { const int tt = ordv[b]; ordv[b] = ordv[b - 1]; ordv[b - 1] = tt; }
  p->core0 = 0;
  for (int i = 0; i < p->nrow; i++) {
    rpos[ordv[i]] = i;
    if (ordv[i] == 0) p->core0 = i;
  }
  /* column degrees within the rows of this rate; frame of a column after an iteration = shift of its last row in
     the plan order */
  int lastrow_shift[LDPC_MAXCOL], lastpos[LDPC_MAXCOL];
  for (int c = 0; c < LDPC_MAXCOL; c++) lastpos[c] = -1;
  for (int k = 0; k < ntab; k++)
    if (etab[k].row < p->nrow && !REMOVED(etab[k].col)) {
      if (etab[k].col >= ncol) return -1; /* cannot happen with 38.212 base graphs */
      p->cdeg[etab[k].col]++;
      if (rpos[etab[k].row] > lastpos[etab[k].col]) {
        lastpos[etab[k].col] = rpos[etab[k].row];
        lastrow_shift[etab[k].col] = etab[k].v[ils] % z0 * nb;
      }
    }
  /* workspace layout */
  int off = 0;
  const int zpa = rup(zp, 64);
  p->cbsz = rup(z + 2 * B, 64);
  for (int c = 0; c < ncol; c++) {
    if (p->cdeg[c] >= 2) {
      p->ckind[c] = COL_MULTI;
      p->crho[c] = (uint16_t)lastrow_shift[c];
      p->nbuf[c] = (p->cdeg[c] & 1) ? 3 : 2;
      p->cbuf[c] = off;
      off += p->nbuf[c] * p->cbsz;
    } else if (p->cdeg[c] == 1) {
      p->ckind[c] = COL_DEG1;
      p->crho[c] = (uint16_t)lastrow_shift[c];
      p->cchan[c] = off; off += zpa;
    } else {
      p->ckind[c] = COL_UNUSED;
      p->cchan[c] = off; off += zpa;
    }
  }
  p->lanemask = off; off += zpa;
  /* edges: per row, multi edges first, then degree-1 edges */
  int ne = 0, ntb = 0, nyt = 0, nyt_core = 0, ncore_edges = 0;
  int prev_shift[LDPC_MAXCOL], visit[LDPC_MAXCOL] = {0}, ed[LDPC_MAXEDGE], erb[LDPC_MAXEDGE];
  for (int c = 0; c < ncol; c++) prev_shift[c] = p->crho[c]; /* frame at the start of an iteration */
  int zcol[LDPC_MAXCOL] = {0}; /* iteration 1: columns still exactly zero (punctured columns 0, 1 before any update) */
  zcol[0] = zcol[1] = 1;
  for (int i = 0; i < p->nrow; i++) {
    ldpc_row_t *rw = &p->row[i];
    const int nr = ordv[i]; /* natural row index */
    rw->nat = (uint8_t)nr;
    rw->e0 = (uint16_t)ne;
    for (int pass = 0; pass < 2; pass++)
      for (int k = 0; k < ntab; k++) {
        if (etab[k].row != nr || REMOVED(etab[k].col)) continue;
        const int c = etab[k].col, s = etab[k].v[ils] % z0 * nb;
        const int multi = p->ckind[c] == COL_MULTI;
        if (multi != (pass == 0)) continue;
        p->ecol[ne] = (uint8_t)c;
        p->eshift[ne] = (uint16_t)s;
        if (multi) {
          const int v = visit[c]++, deg = p->cdeg[c];
          ed[ne] = (s - prev_shift[c] + z) % z;
          erb[ne] = p->cbuf[c] + rbuf(v, deg) * p->cbsz;
          p->wo[ne] = p->cbuf[c] + rbuf(v + 1, deg) * p->cbsz;
          p->sb[ne] = p->cbuf[c];
          p->so[ne] = (s - p->crho[c] + z) % z;
          prev_shift[c] = s;
          rw->nn++;
        } else {
          p->wo[ne] = p->cchan[c];
          p->sb[ne] = p->cchan[c];
          p->so[ne] = 0;
          p->crow1[c] = (uint8_t)i;
          rw->n1++;
        }
        ne++;
      }
    if (rw->nn < 2 || rw->n1 > 1) return -1; /* cannot happen with 38.212 base graphs */
    { /* iteration-1 kind of the row (rows with a degree-1 edge never see a zero column in the 38.212 graphs) */
      int nz = 0, ze = 0;
      for (int e = 0; e < rw->nn; e++)
        if (zcol[p->ecol[rw->e0 + e]]) { nz++; ze = e; }
      rw->it1 = (uint8_t)(rw->n1 || !nz ? 0 : nz >= 2 ? 1 : 2);
      rw->ze = (uint8_t)ze;
      if (rw->it1 == 2) zcol[p->ecol[rw->e0 + ze]] = 0;
      if (!nz) /* a row without a zero column that writes a column: it is no longer guaranteed zero */
        for (int e = 0; e < rw->nn; e++) zcol[p->ecol[rw->e0 + e]] = 0;
      if (rw->n1) for (int e = 0; e < rw->nn; e++) zcol[p->ecol[rw->e0 + e]] = 0;
    }
    if (!p->ordered && i == LDPC_NCORE - 1) { /* view after the core rows (CRC-mode early stop) */
      for (int c = 0; c < ncol; c++)
        if (p->ckind[c] == COL_MULTI) {
          p->cbufm[c] = p->cbuf[c] + rbuf(visit[c], p->cdeg[c]) * p->cbsz;
          p->crhom[c] = (uint16_t)(visit[c] ? prev_shift[c] : p->crho[c]);
        }
      p->ncore = LDPC_NCORE;
      for (int r = 0; r <= i; r++)
        if (p->row[r].n1) p->ncore = 0; /* (cannot happen: the core rows have no degree-1 column) */
      ncore_edges = ne;
      nyt_core = nyt + zp / B * rw->nn;
    }
    int u = 1;
    if (kt) {
      u = kt->row_u(rw->nn, rw->n1, U);
      if (u < 1 || U % u) return -1;
    }
    const int Br = u * W;
    rw->U = (uint8_t)u;
    rw->nsc = (uint16_t)(zp / Br);
    rw->ndbl = (uint8_t)((D + Br - 1) / Br);
    rw->roff = (uint32_t)off;
    off += rw->nn * zp;
    if (rw->n1) {
      rw->soff = (uint32_t)off;
      off += 3 * zp;
    }
    rw->toff = (uint32_t)ntb;
    ntb += rw->nsc * rw->nn;
    rw->yoff = (uint32_t)nyt;
    nyt += zp / B * rw->nn;
    if (kt) {
      rw->fn[0] = kt->row_fn(rw->nn, rw->n1, u, 0);
      rw->fn[1] = kt->row_fn(rw->nn, rw->n1, u, 1);
      rw->fn1 = rw->it1 ? kt->row_fn(rw->nn, rw->n1, u, rw->it1 + 1) : rw->fn[1];
      if (!rw->fn1) { rw->it1 = 0; rw->fn1 = rw->fn[1]; }
      if (!rw->fn[0] || !rw->fn[1]) return -1;
    }
  }
  p->nedge = ne;
  /* CRC-mode early stop (core rows + CRC, also after the core rows of an iteration): worth it only if the core rows
     hold at most 60 % of the edges (BG1 R1/3, R2/3, BG2 R1/5, R1/3; not BG1 R8/9 (76 of 79 edges) or BG2 R2/3 (36 of
     52), which keep the full-syndrome rule) */
  (void)ncore_edges;
  if (p->ordered) { /* the core rows (natural 0..3, the largest degrees) must be consecutive: [core0, core0 + 4) */
    p->ncore = LDPC_NCORE;
    for (int r = 0; r < LDPC_NCORE; r++)
      if (p->core0 + r >= p->nrow || p->row[p->core0 + r].nat >= LDPC_NCORE || p->row[p->core0 + r].n1) p->ncore = 0;
  }
  p->crcmode = p->ncore > 0 && 10 * ncore_full <= 6 * ne_full; /* decided on the full graph of the rate */
  p->midcheck = p->crcmode && !p->ordered; /* the view after the core rows exists only in natural order */
#undef REMOVED
  p->ws_size = (size_t)rup(off, 64) + 64;
  if (!kt) return 0;
  /* read offsets: superchunk k of a multi edge reads the run buf[(d + k B_r) mod Z ..] of its read buffer */
  if (ntb + nyt + nyt_core > p->tabcap) {
    free(p->rdtab);
    p->tabcap = 0;
    p->rdtab = malloc((size_t)(ntb + nyt + nyt_core) * sizeof *p->rdtab);
    if (!p->rdtab) return -1;
    p->tabcap = ntb + nyt + nyt_core;
  }
  p->ntab = ntb;
  p->sytab = p->rdtab + ntb;
  p->sytabm = p->sytab + nyt;
  for (int i = 0; i < p->nrow; i++) {
    const ldpc_row_t *rw = &p->row[i];
    const int Br = rw->U * W;
    for (int k = 0; k < rw->nsc; k++)
      for (int e = 0; e < rw->nn; e++) {
        const int ge = rw->e0 + e;
        p->rdtab[rw->toff + (size_t)k * rw->nn + e] = erb[ge] + (int)(((long)ed[ge] + (long)k * Br) % z);
      }
    for (int k = 0; k < zp / B; k++)
      for (int e = 0; e < rw->nn; e++) {
        const int ge = rw->e0 + e;
        p->sytab[rw->yoff + (size_t)k * rw->nn + e] = p->sb[ge] + (int)(((long)p->so[ge] + (long)k * B) % z);
        if (!p->ordered && i < p->ncore) {
          const int c = p->ecol[ge], om = (p->eshift[ge] - p->crhom[c] + z) % z;
          p->sytabm[rw->yoff + (size_t)k * rw->nn + e] = p->cbufm[c] + (int)(((long)om + (long)k * B) % z);
        }
      }
  }
  return 0;
}

/* ---------------------------------------------------------------- dispatch */
#if defined(__x86_64__) || defined(__i386__)
static int cpu_avx2(void) { __builtin_cpu_init(); return __builtin_cpu_supports("avx2"); }
static int cpu_avx512(void)
{
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl");
}
#endif
#if defined(__aarch64__)
static int cpu_sve2(void)
{
#if defined(__linux__) && defined(HWCAP2_SVE2)
  return (getauxval(AT_HWCAP) & HWCAP_SVE) && (getauxval(AT_HWCAP2) & HWCAP2_SVE2);
#else
  return 0;
#endif
}
#endif

static const char *const impl_names[LDPC_IMPL_COUNT] = {"auto", "ref", "scalar", "neon", "sve2", "avx2", "avx512"};
const char *ldpc_impl_name(ldpc_impl_t impl) { return (unsigned)impl < LDPC_IMPL_COUNT ? impl_names[impl] : "?"; }

static int impl_available_raw(ldpc_impl_t impl);
int ldpc_impl_available(ldpc_impl_t impl)
{
  static int cache[LDPC_IMPL_COUNT]; /* 0 unknown, 1 yes, 2 no */
  if ((unsigned)impl >= LDPC_IMPL_COUNT) return 0;
  int c = __atomic_load_n(&cache[impl], __ATOMIC_RELAXED);
  if (!c) {
    c = impl_available_raw(impl) ? 1 : 2;
    __atomic_store_n(&cache[impl], c, __ATOMIC_RELAXED);
  }
  return c == 1;
}
static int impl_available_raw(ldpc_impl_t impl)
{
  ldpc_kernels_t k;
  int W;
  switch (impl) {
    case LDPC_IMPL_AUTO:
    case LDPC_IMPL_REF:
    case LDPC_IMPL_SCALAR: return 1;
#if defined(__aarch64__)
    case LDPC_IMPL_NEON: return ldpc_kern_neon(&k, &W);
    case LDPC_IMPL_SVE2: return cpu_sve2() && ldpc_kern_sve2(&k, &W);
#endif
#if defined(__x86_64__) || defined(__i386__)
    case LDPC_IMPL_AVX2: return cpu_avx2() && ldpc_kern_avx2(&k, &W);
    case LDPC_IMPL_AVX512: return cpu_avx512() && ldpc_kern_avx512(&k, &W);
#endif
    default: return 0;
  }
}

/* Automatic choice (cached): AVX-512 > AVX2 on x86; on aarch64 SVE2 whenever the CPU has it (also at VL 128:
   measured 3-6 % faster than NEON on Cortex-A720 with the previous kernel, and the SVE2 kernel now executes fewer
   instructions than the NEON one, README), else NEON. LDPC_DEC_IMPL=<name> overrides. */
static int g_best = -1;
ldpc_impl_t ldpc_impl_best(void)
{
  int b = __atomic_load_n(&g_best, __ATOMIC_RELAXED);
  if (b >= 0) return (ldpc_impl_t)b;
  b = LDPC_IMPL_SCALAR;
  const char *env = getenv("LDPC_DEC_IMPL");
  int found = 0;
  if (env)
    for (int i = 1; i < LDPC_IMPL_COUNT; i++)
      if (!strcmp(env, impl_names[i]) && ldpc_impl_available((ldpc_impl_t)i)) { b = i; found = 1; }
  if (!found) {
    if (ldpc_impl_available(LDPC_IMPL_AVX512)) b = LDPC_IMPL_AVX512;
    else if (ldpc_impl_available(LDPC_IMPL_AVX2)) b = LDPC_IMPL_AVX2;
    else if (ldpc_impl_available(LDPC_IMPL_SVE2)) b = LDPC_IMPL_SVE2;
    else if (ldpc_impl_available(LDPC_IMPL_NEON)) b = LDPC_IMPL_NEON;
  }
  __atomic_store_n(&g_best, b, __ATOMIC_RELAXED);
  return (ldpc_impl_t)b;
}

/* per lifting size: a wide vector unit gains nothing when Z fits a narrower one */
static ldpc_impl_t impl_for_z(ldpc_impl_t best, int z)
{
  if (best == LDPC_IMPL_AVX512 && z <= 32 && ldpc_impl_available(LDPC_IMPL_AVX2)) return LDPC_IMPL_AVX2;
  if (best == LDPC_IMPL_SVE2 && z <= 16 && ldpc_impl_available(LDPC_IMPL_NEON)) return LDPC_IMPL_NEON;
  return best;
}

static int get_kernels(ldpc_impl_t impl, ldpc_kernels_t *k, int *W)
{
  switch (impl) {
    case LDPC_IMPL_SCALAR: return ldpc_kern_scalar(k, W);
    case LDPC_IMPL_NEON: return ldpc_kern_neon(k, W);
    case LDPC_IMPL_SVE2: return ldpc_kern_sve2(k, W);
    case LDPC_IMPL_AVX2: return ldpc_kern_avx2(k, W);
    case LDPC_IMPL_AVX512: return ldpc_kern_avx512(k, W);
    default: return 0;
  }
}

/* maximum vectors per step: the most vectors per step that does not pad Z more than one vector does
   (amortises per-edge address and loop overhead; each row then uses the largest U <= this its register budget
   allows); override with LDPC_DEC_U for experiments */
static volatile int g_force_u = -1, g_gen = 0;
void ldpc_dec_force_unroll(int U)
{
  g_force_u = (U == 1 || U == 2 || U == 4) ? U : 0;
  __atomic_add_fetch(&g_gen, 1, __ATOMIC_SEQ_CST);
}
static int choose_u(int z, int W)
{
  if (g_force_u < 0) {
    const char *e = getenv("LDPC_DEC_U");
    int u = e ? atoi(e) : 0;
    g_force_u = (u == 1 || u == 2 || u == 4) ? u : 0;
  }
  if (g_force_u > 0) return g_force_u;
  const int nv = (z + W - 1) / W;
  for (int U = 4; U > 1; U >>= 1)
    if ((nv + U - 1) / U * U == nv || (nv >= 4 * U && (nv + U - 1) / U * U - nv <= 1)) return U;
  return 1;
}

/* ---------------------------------------------------------------- per-thread state */
#define NPLAN 8
typedef struct {
  ldpc_plan_t plan[NPLAN];
  ldpc_kernels_t kern[NPLAN];
  int valid[NPLAN], next, gen;
  int8_t *ws;
  size_t ws_cap;
  void *ws_raw;
} tstate_t;

static pthread_key_t tkey;
static pthread_once_t tonce = PTHREAD_ONCE_INIT;
static void tstate_free(void *v)
{
  tstate_t *t = v;
  if (t) {
    for (int i = 0; i < NPLAN; i++) ldpc_plan_free(&t->plan[i]);
    free(t->ws_raw);
    free(t);
  }
}
static void tkey_init(void) { pthread_key_create(&tkey, tstate_free); }
static tstate_t *tstate(void)
{
  pthread_once(&tonce, tkey_init);
  tstate_t *t = pthread_getspecific(tkey);
  if (!t) {
    t = calloc(1, sizeof *t);
    if (!t) return NULL;
    pthread_setspecific(tkey, t);
  }
  return t;
}

static int get_plan(tstate_t *t, ldpc_impl_t impl, int bg, int z, int rcode, int nact, int fc0, int nb, int nshift, int beta, int rmax,
                    const ldpc_plan_t **pp, const ldpc_kernels_t **kp)
{
  if (t->gen != g_gen) {
    memset(t->valid, 0, sizeof t->valid);
    t->gen = g_gen;
  }
  for (int i = 0; i < NPLAN; i++) {
    const ldpc_plan_t *p = &t->plan[i];
    if (t->valid[i] && p->impl == impl && p->bg == bg && p->z0 == z && p->nb == nb && p->rcode == rcode && p->nrow == nact && p->fc0 == fc0 && p->ordered == ldpc_row_order(bg, rcode) && p->absorb == ldpc_absorb() &&
        p->nshift == nshift && p->beta == beta && p->rmax == rmax) {
      *pp = p; *kp = &t->kern[i];
      return 0;
    }
  }
  const int slot = t->next;
  t->next = (t->next + 1) % NPLAN;
  t->valid[slot] = 0;
  int W;
  if (!ldpc_impl_available(impl) || !get_kernels(impl, &t->kern[slot], &W)) return -1;
  const int U = choose_u(z * nb, W);
  if (ldpc_plan_build(&t->plan[slot], bg, z, rcode, nact, fc0, nb, W, U, nshift, beta, rmax, &t->kern[slot])) return -1;
  t->plan[slot].impl = impl;
  t->valid[slot] = 1;
  *pp = &t->plan[slot]; *kp = &t->kern[slot];
  return 0;
}

static int8_t *get_ws(tstate_t *t, size_t sz)
{
  if (sz > t->ws_cap) {
    free(t->ws_raw);
    t->ws_cap = 0;
    t->ws_raw = malloc(sz + 64);
    if (!t->ws_raw) return NULL;
    t->ws_cap = sz;
    t->ws = (int8_t *)(((uintptr_t)t->ws_raw + 63) & ~(uintptr_t)63);
  }
  return t->ws;
}

/* ---------------------------------------------------------------- input / output */
#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

/* ---- AGC (automatic input gain; semantics of the dec_std follow-up, /home/user/ldpc_scratch/dec_std_followup):
   g = target / mean|x| over the transmitted non-filler positions, clamped to [1/8, 8], as G / 2^sh with G in [64, 127]
   (G * x fits int16); x' = sat8((x G + 2^(sh-1)) >> sh); fillers [kprime, K) bypass the gain (+127). The gain is
   chosen with integer arithmetic only (the follow-up used doubles; identical except at exact rounding ties). */
static void read_env(void);
static int g_agct, g_agcstat, g_agczero, g_allrows, g_noit1, g_nofill, g_agcstep, g_nosvegain, g_nopack, g_nobatch, g_batchn, g_svepack, g_order = -1, g_absorb;
/* AGC statistic: 0 = mean |x|, 1 = mean of |x| over the values below the int8 rails (|x| < 127) */
static int ldpc_agc_stat(void) { read_env(); return g_agcstat; }
/* target multipliers in percent per modulation order (index qm 0..8; 0 = unknown, 1 = BPSK-like / ldpctest, the
   calibration of the base targets); LDPC_DEC_QMT="u,q2,q4,q6,q8" overrides them (experiments) */
static int g_qmt[2][9] = {{LDPC_DEC_QMT_UNK, 100, LDPC_DEC_QMT_2, 100, LDPC_DEC_QMT_4, 100, LDPC_DEC_QMT_6, 100, LDPC_DEC_QMT_8},
                          /* with the CRC-mode (PHY) default NMS 7/8 + 1 */
                          {LDPC_DEC_QMTN_UNK, 100, LDPC_DEC_QMTN_2, 100, LDPC_DEC_QMTN_4, 100, LDPC_DEC_QMTN_6, 100, LDPC_DEC_QMTN_8}};
static int g_crcalg = -1;
/* rate slope of the target (percent per unit of code rate above the class's own rate, per (BG, R) class: BG1 R13,
   R23, R89, BG2 R15, R13, R23); LDPC_DEC_RSLOPE="a,b,c,d,e,f" overrides (experiments) */
static int g_rslope[6] = {LDPC_DEC_RSLOPE_1_13, LDPC_DEC_RSLOPE_1_23, LDPC_DEC_RSLOPE_1_89,
                          LDPC_DEC_RSLOPE_2_15, LDPC_DEC_RSLOPE_2_13, LDPC_DEC_RSLOPE_2_23};
static int rclass(int bg, int rcode) { return (bg == 1 ? 0 : 3) + (bg == 1 ? (rcode == 13 ? 0 : rcode == 23 ? 1 : 2) : (rcode == 15 ? 0 : rcode == 13 ? 1 : 2)); }
static int agc_target(int bg, int rcode, int qm, int rate_q16, int phy)
{
  read_env();
  const int t = bg == 1 ? (rcode == 13 ? 18 : rcode == 23 ? 25 : 34) : (rcode == 15 ? 16 : rcode == 13 ? 18 : 25);
  const int m = qm >= 0 && qm <= 8 ? g_qmt[phy][qm] : g_qmt[phy][0];
  /* the class's own rate Kb / (ncol - 2) in Q16; a block that leaves the tail of the class's graph untransmitted
     (OAI uses R13 for every rate < 2/3) runs at a higher SNR, where the offset needs a larger input scale */
  const int kb = bg == 1 ? 22 : 10, ncol = ldpc_ncol(bg, rcode);
  const long rnom = ((long)kb << 16) / (ncol - 2);
  const long dr = rate_q16 > rnom ? rate_q16 - rnom : 0;
  long mr = 100 + (g_rslope[rclass(bg, rcode)] * dr + 32768) / 65536; /* percent */
  if (mr > 400) mr = 400;
  return (int)(((long)t * g_agct * m / 100 * mr + 5000) / 10000);
}
int ldpc_agc_target_rate(int bg, int rcode, int qm, int rate_q16) { return agc_target(bg, rcode, qm, rate_q16, 0); }
int ldpc_agc_target(int bg, int rcode, int qm) { return agc_target(bg, rcode, qm, 0, 0); }
/* 1: the CRC-mode default algorithm is NMS 7/8 + 1 (LDPC_DEC_CRCALG=0: the offset defaults in every mode) */
int ldpc_crc_alg_default(void) { read_env(); return g_crcalg; }

#if defined(__x86_64__)
/* AVX2 statistic pass: groups of 32 bytes, the even 16-byte group of each (step 32) or both (step 16) */
__attribute__((target("avx2"))) static long abs_sum_avx2(const int8_t *x, int n, int unsat, int step, long *cnt, int *done)
{
  __m256i acc = _mm256_setzero_si256(), acn = _mm256_setzero_si256();
  const __m256i z = _mm256_setzero_si256(), lim = _mm256_set1_epi8(unsat ? 125 : -2), one = _mm256_set1_epi8(1);
  const __m256i keep = step == 32 ? _mm256_setr_epi64x(-1, -1, 0, 0) : _mm256_set1_epi8(-1);
  int i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i a = _mm256_and_si256(_mm256_abs_epi8(_mm256_loadu_si256((const __m256i *)(const void *)(x + i))), keep);
    const __m256i a1 = _mm256_sub_epi8(a, one); /* counted: 1 <= a (<= 126 with unsat), i.e. a - 1 <= lim unsigned */
    const __m256i ok = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_min_epu8(a1, lim), a1), keep);
    a = _mm256_and_si256(a, ok);
    acn = _mm256_add_epi64(acn, _mm256_sad_epu8(_mm256_and_si256(ok, one), z));
    acc = _mm256_add_epi64(acc, _mm256_sad_epu8(a, z));
  }
  const __m128i s2 = _mm_add_epi64(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
  const __m128i c2 = _mm_add_epi64(_mm256_castsi256_si128(acn), _mm256_extracti128_si256(acn, 1));
  *cnt = _mm_cvtsi128_si64(c2) + _mm_cvtsi128_si64(_mm_srli_si128(c2, 8));
  *done = i;
  return _mm_cvtsi128_si64(s2) + _mm_cvtsi128_si64(_mm_srli_si128(s2, 8));
}
#endif
/* sum of |x| and number of values over the samples of x[0..n): every value (step 16) or, with step 32, the values in
   the even 16-byte groups only (x[i], (i >> 4) even); only non-zero values (zeros carry no scale information:
   untransmitted positions); unsat: only |x| < 127 (values at the int8 rails are skipped).
   Integer sums: identical on every platform. */
static long abs_sum(const int8_t *x, int n, int unsat, int step, long *cnt)
{
  long s = 0, c = 0;
  int i = 0;
#if defined(__aarch64__)
  const uint8x16_t lim = vdupq_n_u8(unsat ? 126 : 255), one = vdupq_n_u8(1);
  while (i + 16 <= n) { /* u16 lanes: at most 128 steps of 2 * 255 before they are flushed */
    uint16x8_t acc = vdupq_n_u16(0), acn = vdupq_n_u16(0);
    for (int k = 0; k < 128 && i + 16 <= n; k++, i += step) {
      const uint8x16_t a = vreinterpretq_u8_s8(vabsq_s8(vld1q_s8(x + i)));
      const uint8x16_t m = vcltq_u8(vsubq_u8(a, one), lim); /* 1 <= |x| (<= 126 with unsat) */
      acc = vpadalq_u8(acc, unsat ? vandq_u8(a, m) : a);    /* zeros add nothing to the sum */
      acn = vpadalq_u8(acn, m);                             /* 255 per counted value */
    }
    s += vaddlvq_u16(acc);
    c += (long)vaddlvq_u16(acn) / 255;
  }
#elif defined(__x86_64__)
  static int avx2 = -1;
  if (avx2 < 0) avx2 = cpu_avx2();
  if (avx2 && step <= 32) {
    long c0;
    int done;
    s = abs_sum_avx2(x, n, unsat, step, &c0, &done);
    for (i = done; i < n; i++) {
      if ((i >> 4) & (step / 16 - 1)) continue;
      const int a = x[i] < 0 ? -x[i] : x[i];
      if (a > 0 && (!unsat || a < 127)) { s += a; c0++; }
    }
    *cnt = c0;
    return s;
  }
  __m128i acc = _mm_setzero_si128(), acn = _mm_setzero_si128();
  const __m128i lim = _mm_set1_epi8(unsat ? 125 : -2), z = _mm_setzero_si128(), one = _mm_set1_epi8(1);
  for (; i + 16 <= n; i += step) {
    const __m128i v = _mm_loadu_si128((const __m128i *)(const void *)(x + i)), m = _mm_cmpgt_epi8(z, v);
    __m128i a = _mm_sub_epi8(_mm_xor_si128(v, m), m); /* |x|, -128 -> 0x80 */
    const __m128i a1 = _mm_sub_epi8(a, one);   /* counted: 1 <= a (<= 126 with unsat) */
    const __m128i ok = _mm_cmpeq_epi8(_mm_min_epu8(a1, lim), a1);
    a = _mm_and_si128(a, ok);
    acn = _mm_add_epi64(acn, _mm_sad_epu8(_mm_and_si128(ok, one), z));
    acc = _mm_add_epi64(acc, _mm_sad_epu8(a, z));
  }
  s = _mm_cvtsi128_si64(acc) + _mm_cvtsi128_si64(_mm_srli_si128(acc, 8));
  c = _mm_cvtsi128_si64(acn) + _mm_cvtsi128_si64(_mm_srli_si128(acn, 8));
#endif
  for (; i < n; i++) {
    if ((i >> 4) & (step / 16 - 1)) continue;
    const int a = x[i] < 0 ? -x[i] : x[i];
    if (a > 0 && (!unsat || a < 127)) { s += a; c++; }
  }
  *cnt = c;
  return s;
}

long ldpc_test_abs_sum(const int8_t *x, int n, int unsat, int step, long *cnt) { return abs_sum(x, n, unsat, step, cnt); }

int ldpc_nend(const int8_t *llr, int n)
{
  int i = n;
  while (i >= 8) { /* 8 bytes at a time (scalar: the vector pipes stay free) */
    uint64_t w;
    memcpy(&w, llr + i - 8, 8);
    if (w) break;
    i -= 8;
  }
  while (i > 0 && !llr[i - 1]) i--;
  return i;
}

void ldpc_agc_setup(int z, int ncol, int K, int kprime, int nend, int target, const int8_t *llr, ldpc_agc_t *a)
{
  const int n = nend < ncol * z ? nend : ncol * z, kp = kprime > 0 && kprime <= K ? kprime : K;
  a->G = 0; a->sh = 0; a->kp = kp; a->K = K;
  if (target <= 0) return;
  long c1, c2;
  const int n2 = n > K ? n - K : 0; /* parity part up to the last non-zero LLR (the tail beyond it is all zero) */
  const int unsat = ldpc_agc_stat();
  const int nst = kp - 2 * z + n2; /* sampled statistic: every other 16-byte group from 8 KB */
  int step = g_agcstep == 1 ? (nst >= 8192 ? 32 : 16) /* v2 rule (experiments) */
           : g_agcstep ? g_agcstep : nst >= 8192 ? 32 : 16; /* every 4th group (64): not BLER-neutral, README 11 */
  if (g_agczero) step = ncol * z - 2 * z >= 8192 ? 32 : 16;
  const long den = abs_sum(llr + 2 * z, kp - 2 * z, unsat, step, &c1) + abs_sum(llr + K, n2, unsat, step, &c2);
  if (g_agczero) { /* experiments only (LDPC_DEC_AGCZERO=1): the v1 statistic, zeros counted over the whole input */
    const int seg[2][2] = {{2 * z, kp}, {K, ncol * z}};
    for (int g = 0; g < 2; g++)
      for (int i = seg[g][0]; i < seg[g][1]; i++)
        if (!llr[i] && !(step == 32 && (((i - seg[g][0]) >> 4) & 1))) c1++;
  }
  const long num = (long)target * (c1 + c2);
  if (den <= 0) return; /* all zero: nothing to scale */
  int G, sh;
  if (8 * num <= den) { G = 64; sh = 9; }      /* g = 1/8 */
  else if (num >= 8 * den) { G = 64; sh = 3; } /* g = 8 */
  else {
    sh = 6;
    while (sh < 9 && 2 * (num << sh) < 127 * den) sh++;  /* g 2^sh < 63.5 */
    while (sh > 3 && 2 * (num << sh) >= 255 * den) sh--; /* g 2^sh >= 127.5 */
    G = (int)((2 * (num << sh) + den) / (2 * den));
    if (G > 127) G = 127;
  }
  a->G = G; a->sh = sh;
}

/* ---- input conditioning: dst[i] = clamp(scale_in(gain(src[i]))) for i < n, 16 bytes per step; the tail is one
   overlapping step (dst = f(src) elementwise, so recomputing is harmless), only n < 16 runs scalar.
   aarch64: SMULL + SRSHL + SQXTN (gain), SRSHR (rounding shift, exact) + SMIN/SMAX; x86: 16-bit lanes (SSE4.1);
   elsewhere scalar. */
static inline int8_t cond1(int8_t x, int s, int cmax, int G, int sh)
{
  if (!G) return ldpc_scale_in(x, s, cmax);
  const int8_t v = ldpc_gain1(x, G, sh + s);
  return (int8_t)(v > cmax ? cmax : v < -cmax ? -cmax : v);
}
#if defined(__aarch64__)
/* no gain: SRSHR #s (exact rounding shift) + clamp; gain: SMULL / SMULL2 + SQRSHRN #t (one rounding, saturating; t > 8:
   SRSHR + SQXTN) + clamp */
#define SC16(sh_)                                                                                       \
  static void cond_run_##sh_(int8_t *restrict dst, const int8_t *restrict src, int n, int8x16_t c, int8x16_t nc) \
  {                                                                                                     \
    int i = 0;                                                                                          \
    for (;; i += 16) {                                                                                  \
      if (i + 16 > n) i = n - 16;                                                                       \
      int8x16_t v = vld1q_s8(src + i);                                                                  \
      if (sh_) v = vrshrq_n_s8(v, sh_ ? sh_ : 1);                                                       \
      vst1q_s8(dst + i, vmaxq_s8(vminq_s8(v, c), nc));                                                  \
      if (i + 16 >= n) break;                                                                           \
    }                                                                                                   \
  }
SC16(0) SC16(1) SC16(2) SC16(3)
#define GC16(t_)                                                                                        \
  static void gain_run_##t_(int8_t *restrict dst, const int8_t *restrict src, int n, int8x16_t c, int8x16_t nc, int G) \
  {                                                                                                     \
    const int8x8_t g8 = vdup_n_s8((int8_t)G);                                                           \
    const int8x16_t g16 = vdupq_n_s8((int8_t)G);                                                        \
    int i = 0;                                                                                          \
    for (;; i += 16) {                                                                                  \
      if (i + 16 > n) i = n - 16;                                                                       \
      const int8x16_t v = vld1q_s8(src + i);                                                            \
      const int8x16_t vs = t_ < 7 ? vqshlq_n_s8(v, t_ < 7 ? 7 - t_ : 0) : v; /* k = 7 - t >= 0: sat8(x << k) */  \
      const int16x8_t lo = vmull_s8(vget_low_s8(vs), g8), hi = vmull_high_s8(vs, g16);                  \
      int8x16_t y = vqrshrn_high_n_s16(vqrshrn_n_s16(lo, 7), hi, 7); /* rdmulh(., G) */                   \
      if (t_ > 7) y = vrshrq_n_s8(y, t_ > 7 ? t_ - 7 : 1);                                                \
      vst1q_s8(dst + i, vmaxq_s8(vminq_s8(y, c), nc));                                                  \
      if (i + 16 >= n) break;                                                                           \
    }                                                                                                   \
  }
GC16(3) GC16(4) GC16(5) GC16(6) GC16(7) GC16(8) GC16(9) GC16(10) GC16(11) GC16(12)
#endif
#if defined(__x86_64__)
static int cpu_avx2(void);
/* AVX2: 32 bytes per step in 16-bit lanes (same arithmetic as the SSE4.1 path below) */
__attribute__((target("avx2"))) static void copy_cond_avx2(int8_t *restrict dst, const int8_t *restrict src, int n, int s,
                                                          int cmax, int G, int sh)
{
  const __m256i c = _mm256_set1_epi8((int8_t)cmax), nc = _mm256_set1_epi8((int8_t)-cmax);
  const int t = G ? sh + s : s, kk = 7 - t; /* gain: the byte-SQRDMULH definition of ldpc_gain1() */
  const __m256i gv = _mm256_set1_epi16((int16_t)(G ? G : 1)), rg = _mm256_set1_epi16((int16_t)(t ? 1 << (t - 1) : 0));
  const __m128i ct = _mm_cvtsi32_si128(t), c7 = _mm_cvtsi32_si128(7);
  const __m128i ck = _mm_cvtsi32_si128(kk > 0 ? kk : 0), cr = _mm_cvtsi32_si128(kk < 0 ? -kk : 0);
  const __m256i r64 = _mm256_set1_epi16(64), rr = _mm256_set1_epi16((int16_t)(kk < 0 ? 1 << (-kk - 1) : 0));
  const __m256i s127 = _mm256_set1_epi16(127), sm128 = _mm256_set1_epi16(-128);
  int i = 0;
  for (;; i += 32) {
    if (i + 32 > n) i = n - 32;
    const __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(src + i));
    __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v)), hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1));
    if (G) {
      if (kk > 0) {
        lo = _mm256_max_epi16(_mm256_min_epi16(_mm256_sll_epi16(lo, ck), s127), sm128);
        hi = _mm256_max_epi16(_mm256_min_epi16(_mm256_sll_epi16(hi, ck), s127), sm128);
      }
      lo = _mm256_sra_epi16(_mm256_add_epi16(_mm256_mullo_epi16(lo, gv), r64), c7);
      hi = _mm256_sra_epi16(_mm256_add_epi16(_mm256_mullo_epi16(hi, gv), r64), c7);
      if (kk < 0) {
        lo = _mm256_sra_epi16(_mm256_add_epi16(lo, rr), cr);
        hi = _mm256_sra_epi16(_mm256_add_epi16(hi, rr), cr);
      }
    } else {
      lo = _mm256_sra_epi16(_mm256_add_epi16(lo, rg), ct);
      hi = _mm256_sra_epi16(_mm256_add_epi16(hi, rg), ct);
    }
    const __m256i y = _mm256_permute4x64_epi64(_mm256_packs_epi16(lo, hi), 0xD8);
    _mm256_storeu_si256((__m256i *)(void *)(dst + i), _mm256_max_epi8(_mm256_min_epi8(y, c), nc));
    if (i + 32 >= n) break;
  }
}
#endif
#if defined(__aarch64__)
static __thread int t_sve_gain;
#endif
static void copy_cond(int8_t *restrict dst, const int8_t *restrict src, int n, int s, int cmax, int G, int sh)
{
#if defined(__x86_64__)
  static int avx2 = -1;
  if (avx2 < 0) avx2 = cpu_avx2();
  if (n >= 32 && avx2) { copy_cond_avx2(dst, src, n, s, cmax, G, sh); return; }
#endif
  if (n >= 16) {
#if defined(__aarch64__)
    const int8x16_t c = vdupq_n_s8((int8_t)cmax), nc = vdupq_n_s8((int8_t)-cmax);
    if (G && t_sve_gain && !g_nosvegain) { ldpc_sve2_gain_run(dst, src, n, cmax, G, sh + s); return; }
    if (G) {
      switch (sh + s) {
        case 3: gain_run_3(dst, src, n, c, nc, G); return;
        case 4: gain_run_4(dst, src, n, c, nc, G); return;
        case 5: gain_run_5(dst, src, n, c, nc, G); return;
        case 6: gain_run_6(dst, src, n, c, nc, G); return;
        case 7: gain_run_7(dst, src, n, c, nc, G); return;
        case 8: gain_run_8(dst, src, n, c, nc, G); return;
        case 9: gain_run_9(dst, src, n, c, nc, G); return;
        case 10: gain_run_10(dst, src, n, c, nc, G); return;
        case 11: gain_run_11(dst, src, n, c, nc, G); return;
        case 12: gain_run_12(dst, src, n, c, nc, G); return;
        default: break; /* (cannot happen: sh in [3, 9], s in [0, 3]) */
      }
    } else
      switch (s) {
        case 0: cond_run_0(dst, src, n, c, nc); return;
        case 1: cond_run_1(dst, src, n, c, nc); return;
        case 2: cond_run_2(dst, src, n, c, nc); return;
        default: cond_run_3(dst, src, n, c, nc); return;
      }
#elif defined(__x86_64__) && defined(__SSE4_1__)
    const __m128i c = _mm_set1_epi8((int8_t)cmax), nc = _mm_set1_epi8((int8_t)-cmax);
    const int t = G ? sh + s : s, kk = 7 - t; /* gain: the byte-SQRDMULH definition of ldpc_gain1() */
    const __m128i gv = _mm_set1_epi16((int16_t)(G ? G : 1)), rg = _mm_set1_epi16((int16_t)(t ? 1 << (t - 1) : 0));
    const __m128i ct = _mm_cvtsi32_si128(t), c7 = _mm_cvtsi32_si128(7), r64 = _mm_set1_epi16(64);
    const __m128i ck = _mm_cvtsi32_si128(kk > 0 ? kk : 0), cr = _mm_cvtsi32_si128(kk < 0 ? -kk : 0);
    const __m128i rr = _mm_set1_epi16((int16_t)(kk < 0 ? 1 << (-kk - 1) : 0));
    const __m128i s127 = _mm_set1_epi16(127), sm128 = _mm_set1_epi16(-128);
    int i = 0;
    for (;; i += 16) {
      if (i + 16 > n) i = n - 16;
      const __m128i v = _mm_loadu_si128((const __m128i *)(const void *)(src + i));
      __m128i lo = _mm_cvtepi8_epi16(v), hi = _mm_cvtepi8_epi16(_mm_srli_si128(v, 8));
      if (G) {
        if (kk > 0) {
          lo = _mm_max_epi16(_mm_min_epi16(_mm_sll_epi16(lo, ck), s127), sm128);
          hi = _mm_max_epi16(_mm_min_epi16(_mm_sll_epi16(hi, ck), s127), sm128);
        }
        lo = _mm_sra_epi16(_mm_add_epi16(_mm_mullo_epi16(lo, gv), r64), c7);
        hi = _mm_sra_epi16(_mm_add_epi16(_mm_mullo_epi16(hi, gv), r64), c7);
        if (kk < 0) {
          lo = _mm_sra_epi16(_mm_add_epi16(lo, rr), cr);
          hi = _mm_sra_epi16(_mm_add_epi16(hi, rr), cr);
        }
      } else {
        lo = _mm_sra_epi16(_mm_add_epi16(lo, rg), ct);
        hi = _mm_sra_epi16(_mm_add_epi16(hi, rg), ct);
      }
      const __m128i y = _mm_packs_epi16(lo, hi); /* sat8 (no-gain path: values already in int8 range) */
      _mm_storeu_si128((__m128i *)(void *)(dst + i), _mm_max_epi8(_mm_min_epi8(y, c), nc));
      if (i + 16 >= n) break;
    }
    return;
#endif
  }
  for (int i = 0; i < n; i++) dst[i] = cond1(src[i], s, cmax, G, sh);
}

/* test hook: the vector conditioning kernel (tests compare it with the scalar definition) */
void ldpc_test_cond(int8_t *dst, const int8_t *src, int n, int s, int cmax, int G, int sh) { copy_cond(dst, src, n, s, cmax, G, sh); }

/* conditioned copy of the natural run src[a .. a+n) of column c into dst, honouring the filler range (fillers
   bypass the gain when the AGC is on) */
static void cond_col_run(int8_t *dst, const int8_t *colsrc, int c, int z, int a, int n, int s, int cmax, const ldpc_agc_t *ag)
{
  copy_cond(dst, colsrc + a, n, s, cmax, ag->G, ag->sh);
  const int f0 = ag->kp - c * z, f1 = ag->K - c * z; /* filler range in column coordinates */
  const int lo = f0 > a ? f0 : a, hi = f1 < a + n ? f1 : a + n;
  if (lo < hi) memset(dst + (lo - a), ldpc_scale_in(127, s, cmax), (size_t)(hi - lo));
}

__attribute__((noinline)) static void load_input(const ldpc_plan_t *p, int8_t *ws, const int8_t *llr, int in_shift, const ldpc_agc_t *ag)
{
#if defined(__aarch64__)
  t_sve_gain = p->impl == LDPC_IMPL_SVE2; /* SVE2 byte SQRDMULH gain and predicate bit packing (same results as NEON) */
#endif
  const int z = p->z, cm = ldpc_chan_max(p->rmax, 0), cm1 = ldpc_chan_max(p->rmax, 1);
  const int D = p->B < z ? p->B : z;
  for (int c = 0; c < p->ncol; c++) {
    const int8_t *src = llr + (size_t)c * z;
    const int rho = p->crho[c];
    if (p->ckind[c] == COL_MULTI) { /* buffer 0: buf[j] = L[(j + rho) mod Z], j in [0, Z + min(B, Z)) */
      int8_t *b = ws + p->cbuf[c];
      cond_col_run(b, src, c, z, rho, z - rho, in_shift, cm, ag);
      cond_col_run(b + z - rho, src, c, z, 0, rho, in_shift, cm, ag);
      memcpy(b + z, b, D);
    } else if (p->ckind[c] == COL_DEG1) {
      int8_t *b = ws + p->cchan[c];
      cond_col_run(b, src, c, z, rho, z - rho, in_shift, cm1, ag);
      cond_col_run(b + z - rho, src, c, z, 0, rho, in_shift, cm1, ag);
      memset(b + z, 0, p->zp - z);
    } else {
      cond_col_run(ws + p->cchan[c], src, c, z, 0, z, in_shift, cm, ag);
    }
  }
  int8_t *m = ws + p->lanemask;
  memset(m, -1, z);
  memset(m + z, 0, p->zp - z);
}

/* natural-order a-posteriori LLRs of column c; mid: the view after the core rows of an iteration */
static void get_column(const ldpc_plan_t *p, const int8_t *ws, int c, int8_t *dst, int mid)
{
  const int z = p->z, rho = mid && p->ckind[c] == COL_MULTI ? p->crhom[c] : p->crho[c];
  if (p->ckind[c] == COL_MULTI) { /* buffer 0 (mid: the current buffer), frame rho: APP[j] = buf[(j - rho) mod Z] */
    const int8_t *b = ws + (mid ? p->cbufm[c] : p->cbuf[c]);
    memcpy(dst, b + z - rho, rho);
    memcpy(dst + rho, b, z - rho);
  } else if (p->ckind[c] == COL_DEG1) { /* lazy: APP = Q + R from the channel and the state of its row */
    const ldpc_row_t *rw = &p->row[p->crow1[c]];
    const int Br = rw->U * p->W;
    const int8_t *ch = ws + p->cchan[c];
    for (int t = 0; t < z; t++) { /* check t <-> rotated position t, natural position (t + rho) mod Z */
      const int8_t *s = ws + rw->soff + (size_t)(t / Br) * 3 * Br + t % Br;
      const int8_t q = ch[t];
      const int j = t + rho < z ? t + rho : t + rho - z;
      dst[j] = ldpc_sat8(q + ldpc_msg1(q, (uint8_t)s[0], s[Br], s[2 * Br]));
    }
  } else {
    memcpy(dst, ws + p->cchan[c], z);
  }
}

/* sign bits of n int8 values, packed MSB first */
#if defined(__x86_64__)
static int cpu_avx512(void);
/* AVX-512BW: bytes reversed within each group of 8 (VPSHUFB), VPMOVB2M, 64 values -> 8 bytes per step */
__attribute__((target("avx512bw,avx512f"))) static int pack_bits_avx512(const int8_t *v, int n, uint8_t *out)
{
  const __m512i rev = _mm512_set_epi64(0x08090a0b0c0d0e0fLL, 0x0001020304050607LL, 0x08090a0b0c0d0e0fLL, 0x0001020304050607LL,
                                       0x08090a0b0c0d0e0fLL, 0x0001020304050607LL, 0x08090a0b0c0d0e0fLL, 0x0001020304050607LL);
  int i = 0;
  for (; i + 64 <= n; i += 64) {
    const __mmask64 m = _mm512_movepi8_mask(_mm512_shuffle_epi8(_mm512_loadu_si512((const void *)(v + i)), rev));
    const uint64_t x = (uint64_t)m;
    memcpy(out + (i >> 3), &x, 8);
  }
  return i;
}
#endif
static void pack_bits(const int8_t *v, int n, uint8_t *out)
{
  int i = 0;
#if defined(__x86_64__)
  static int a512 = -1;
  if (a512 < 0) a512 = cpu_avx512();
  if (a512 && !g_nopack) i = pack_bits_avx512(v, n, out);
#endif
#if defined(__aarch64__)
  if (t_sve_gain && g_svepack) i = ldpc_sve2_pack_bits(v, n, out); /* experiments only (README 11: no gain) */
  static const uint8_t wgt[16] = {128, 64, 32, 16, 8, 4, 2, 1, 128, 64, 32, 16, 8, 4, 2, 1};
  const uint8x16_t w = vld1q_u8(wgt);
  for (; i + 64 <= n; i += 64) { /* 64 values -> 8 bytes */
    const uint8x16_t a0 = vandq_u8(vcltzq_s8(vld1q_s8(v + i)), w), a1 = vandq_u8(vcltzq_s8(vld1q_s8(v + i + 16)), w);
    const uint8x16_t a2 = vandq_u8(vcltzq_s8(vld1q_s8(v + i + 32)), w), a3 = vandq_u8(vcltzq_s8(vld1q_s8(v + i + 48)), w);
    const uint8x16_t s = vpaddq_u8(vpaddq_u8(a0, a1), vpaddq_u8(a2, a3));
    vst1_u8(out + (i >> 3), vget_low_u8(vpaddq_u8(s, s)));
  }
#elif defined(__x86_64__) && defined(__SSSE3__)
  const __m128i rev = _mm_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
  for (; i + 16 <= n; i += 16) { /* bytes reversed within each group of 8: movemask bit 7 of a group = first value */
    const int m = _mm_movemask_epi8(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(const void *)(v + i)), rev));
    out[i >> 3] = (uint8_t)m;
    out[(i >> 3) + 1] = (uint8_t)(m >> 8);
  }
#endif
  for (; i + 8 <= n; i += 8) {
    uint64_t x;
    memcpy(&x, v + i, 8);
    x = (x >> 7) & 0x0101010101010101ULL;
    out[i >> 3] = (uint8_t)((x * 0x8040201008040201ULL) >> 56);
  }
  if (i < n) {
    uint8_t b = 0;
    for (int k = 0; i + k < n; k++) b |= (uint8_t)((v[i + k] < 0) << (7 - k));
    out[i >> 3] = b;
  }
}

/* test hook: bit packing (impl selects the SVE2 path on aarch64 like a decode with that implementation) */
void ldpc_test_pack(const int8_t *v, int n, uint8_t *out, int sve)
{
#if defined(__aarch64__)
  const int o = t_sve_gain;
  t_sve_gain = sve && ldpc_impl_available(LDPC_IMPL_SVE2);
  const int sp = g_svepack;
  g_svepack = sve;
  pack_bits(v, n, out);
  g_svepack = sp;
  t_sve_gain = o;
#else
  (void)sve;
  pack_bits(v, n, out);
#endif
}

__attribute__((noinline)) static void write_output(const ldpc_plan_t *p, const int8_t *ws, const int8_t *ref_app, ldpc_out_t mode,
                                                   uint8_t *out, int mid)
{
  const int z = p->z;
  const int ncols = mode == LDPC_OUT_LLR ? p->ncol : p->kbcols;
  int8_t tmp[LDPC_MAXCOL * 384];
  int8_t *dst = mode == LDPC_OUT_LLR ? (int8_t *)out : tmp;
  for (int c = 0; c < ncols; c++) {
    if (ref_app) memcpy(dst + c * z, ref_app + c * z, z);
    else get_column(p, ws, c, dst + c * z, mid);
  }
  if (mode == LDPC_OUT_BITS) pack_bits(tmp, ncols * z, out);
  else if (mode == LDPC_OUT_BYTES)
    for (int i = 0; i < ncols * z; i++) out[i] = tmp[i] < 0;
}

/* ---------------------------------------------------------------- driver */
/* Per (BG, R) defaults: offset min-sum, offset and channel scaling tuned at BLER ~1e-2 on ldpctest's LLR
   quantisation (README, "algorithm parameters"); the offset that works best decreases with the code rate.
   in_shift: 1 = channel LLRs halved, -1 = full resolution. */
/* experiments: LDPC_DEC_ALG="nshift,beta,rmax,in_shift" replaces the per-(BG, R) defaults, LDPC_DEC_AGC=0|1 the AGC
   default (read once) */
static int g_env_alg = -1, g_env_agc = -1, g_alg[4], g_agct = 100;
static void read_env(void)
{
  if (g_env_alg >= 0) return;
  const char *e = getenv("LDPC_DEC_ALG");
  int ok = e && sscanf(e, "%d,%d,%d,%d", &g_alg[0], &g_alg[1], &g_alg[2], &g_alg[3]) == 4;
  const char *a = getenv("LDPC_DEC_AGC");
  g_env_agc = a ? (atoi(a) != 0) : 2;
  const char *t = getenv("LDPC_DEC_AGCT"); /* experiments: AGC targets x percent */
  if (t && atoi(t) > 0) g_agct = atoi(t);
  const char *qt = getenv("LDPC_DEC_QMT");
  int q[5];
  for (int a = 0; a < 2; a++) {
    if (a) qt = getenv("LDPC_DEC_QMTN");
    if (qt && sscanf(qt, "%d,%d,%d,%d,%d", &q[0], &q[1], &q[2], &q[3], &q[4]) == 5) {
      g_qmt[a][0] = q[0]; g_qmt[a][2] = q[1]; g_qmt[a][4] = q[2]; g_qmt[a][6] = q[3]; g_qmt[a][8] = q[4];
    }
  }
  const char *ca = getenv("LDPC_DEC_CRCALG");
  g_crcalg = ca ? atoi(ca) != 0 : LDPC_DEC_CRCALG_DEFAULT;
  const char *rs = getenv("LDPC_DEC_RSLOPE");
  int v[6];
  if (rs && sscanf(rs, "%d,%d,%d,%d,%d,%d", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) == 6) memcpy(g_rslope, v, sizeof v);
  const char *abe = getenv("LDPC_DEC_ABS");
#ifdef LDPC_ABSORB
  g_absorb = abe && atoi(abe) != 0;
#else
  g_absorb = 0; (void)abe; /* the kernels are built without the option */
#endif
  const char *ore = getenv("LDPC_DEC_ORDER"); /* 0: natural layer order (v2 algorithm) */
  g_order = ore ? atoi(ore) != 0 : 1;
  const char *bnn = getenv("LDPC_DEC_BATCHN"); /* experiments: blocks per interleaved group */
  g_batchn = bnn ? atoi(bnn) : 0;
  const char *nbt = getenv("LDPC_DEC_NOBATCH"); /* measurements: ldpc_decode_batch() decodes one block at a time */
  g_nobatch = nbt && atoi(nbt) != 0;
  const char *spk = getenv("LDPC_DEC_SVEPACK"); /* experiments: SVE predicate bit packing */
  g_svepack = spk && atoi(spk) != 0;
  const char *npk = getenv("LDPC_DEC_NOPACK"); /* measurements: SSSE3 instead of AVX-512 bit packing */
  g_nopack = npk && atoi(npk) != 0;
  const char *nsg = getenv("LDPC_DEC_NOSVEGAIN"); /* measurements: NEON gain path also with SVE2 */
  g_nosvegain = nsg && atoi(nsg) != 0;
  const char *ase = getenv("LDPC_DEC_AGCSTEP"); /* experiments: fixed sampling step (16, 32, 64) */
  g_agcstep = ase ? atoi(ase) : 0;
  if (g_agcstep != 1 && g_agcstep != 16 && g_agcstep != 32 && g_agcstep != 64) g_agcstep = 0;
  const char *nfe = getenv("LDPC_DEC_NOFILL"); /* debugging only: keep the filler columns (not exact vs the reference) */
  g_nofill = nfe && atoi(nfe) != 0;
  const char *n1e = getenv("LDPC_DEC_NOIT1"); /* measurements: no iteration-1 shortcuts */
  g_noit1 = n1e && atoi(n1e) != 0;
  const char *ar = getenv("LDPC_DEC_ALLROWS");
  g_allrows = ar && atoi(ar) != 0;
  const char *az = getenv("LDPC_DEC_AGCZERO");
  g_agczero = az && atoi(az) != 0;
  const char *st = getenv("LDPC_DEC_AGCSTAT");
  g_agcstat = st ? atoi(st) : LDPC_DEC_AGC_STAT;
  __atomic_store_n(&g_env_alg, ok, __ATOMIC_RELEASE);
}
int ldpc_absorb(void) { read_env(); return g_absorb; }
int ldpc_row_order(int bg, int rcode) { read_env(); return g_order && !(bg == 1 && rcode == 89); }
int ldpc_agc_default(void)
{
  read_env();
  return g_env_agc == 2 ? LDPC_DEC_AGC_DEFAULT : g_env_agc;
}

void ldpc_default_alg(int bg, int rcode, int *nshift, int *beta, int *rmax, int *in_shift)
{
  read_env();
  if (g_env_alg == 1) {
    *nshift = g_alg[0] > 0 ? g_alg[0] : -1; *beta = g_alg[1] > 0 ? g_alg[1] : -1;
    *rmax = g_alg[2]; *in_shift = g_alg[3] > 0 ? g_alg[3] : -1;
    return;
  }
  int b = 2, s = 1;
  if (bg == 1 && rcode == 23) { b = 3; s = -1; }
  else if (bg == 1 && rcode == 89) { b = 1; s = -1; }
  else if (bg == 2 && rcode == 23) { b = 2; s = -1; }
  *nshift = -1; *beta = b; *rmax = LDPC_DEC_DEFAULT_RMAX; *in_shift = s;
}

/* reference path: CRC of the systematic hard decisions of natural-order APPs */
static int ref_crc(void *ctx, const int8_t *app)
{
  const ldpc_dec_params_t *prm = ctx;
  const int K = (prm->bg == 1 ? 22 : 10) * prm->z;
  uint8_t packed[22 * 384 / 8 + 8];
  pack_bits(app, K, packed);
  return prm->check_crc(packed, (uint32_t)(prm->kprime > 0 ? prm->kprime : K), prm->crc_type) != 0;
}

/* CRC of the systematic hard decisions of a view (mid: after the core rows); with packed-bit output they are
   written straight into out (the final output when the CRC passes) */
static int crc_view(const ldpc_dec_params_t *prm, const ldpc_plan_t *p, const int8_t *ws, uint8_t *out, int mid)
{
  uint8_t packed[22 * 384 / 8 + 8];
  uint8_t *pk = prm->out_mode == LDPC_OUT_BITS ? out : packed;
  write_output(p, ws, NULL, LDPC_OUT_BITS, pk, mid);
  return prm->check_crc(pk, (uint32_t)(prm->kprime > 0 ? prm->kprime : p->kbcols * p->z), prm->crc_type) != 0;
}

/* per-block setup shared by ldpc_decode() and ldpc_decode_batch(): parameters, AGC, active rows, filler columns */
typedef struct {
  int ncol, K, nshift, beta, rmax, in_shift, nact, fc0, pz, max_iter;
  ldpc_agc_t agc;
} dsetup_t;
static int dec_setup(const ldpc_dec_params_t *prm, const int8_t *llr, dsetup_t *d)
{
  const int ncol = ldpc_ncol(prm->bg, prm->rcode);
  if (!ncol || prm->z < 2 || prm->z > 384 || ldpc_ils_of_z[prm->z] == 255) return -1;
  int nshift = prm->nshift, beta = prm->beta, rmax = prm->rmax, in_shift = prm->in_shift;
  /* defaults: with a CRC callback (OAI's PHY) normalised + offset min-sum 7/8, 1 (scale-tolerant on demapper LLRs,
     README section 6), else the per-(BG, R) offset min-sum tuned on ldpctest's channel */
  int phy = 0;
  if (!nshift && !beta && !rmax && !in_shift) {
    ldpc_default_alg(prm->bg, prm->rcode, &nshift, &beta, &rmax, &in_shift);
    if (prm->check_crc && g_env_alg != 1 && ldpc_crc_alg_default()) {
      nshift = 3; beta = 1; rmax = LDPC_DEC_DEFAULT_RMAX; in_shift = 1;
      phy = 1;
    }
  }
  const int K = (prm->bg == 1 ? 22 : 10) * prm->z;
  ldpc_agc_t agc;
  const int agc_on = prm->agc > 0 || (prm->agc == 0 && ldpc_agc_default());
  const int nend = ldpc_nend(llr, ncol * prm->z);
  const int nact = g_allrows ? ncol - (prm->bg == 1 ? 22 : 10) /* measurement only: LDPC_DEC_ALLROWS=1 */
                             : ldpc_active_rows(ncol - (prm->bg == 1 ? 22 : 10), prm->bg == 1 ? 22 : 10, prm->z, nend);
  /* code rate of the transmitted part: K' / (non-zero span beyond the punctured 2Z, minus the fillers), Q16 */
  const int kpr = prm->kprime > 0 && prm->kprime <= K ? prm->kprime : K;
  const long span = (long)nend - 2 * prm->z - (K - kpr);
  const int rate_q16 = span > 0 ? (int)(((long)kpr << 16) / span) : 0;
  ldpc_agc_setup(prm->z, ncol, K, prm->kprime, nend,
                 agc_on ? agc_target(prm->bg, prm->rcode, prm->qm, rate_q16, phy) : 0, llr, &agc);
  if (nshift < 0) nshift = 0; /* explicit "no scaling" */
  if (beta < 0) beta = 0;
  if (rmax <= 0 || rmax > 127) rmax = LDPC_DEC_DEFAULT_RMAX;
  if (in_shift < 0) in_shift = 0;
  if (in_shift > 3) in_shift = 3;
  d->ncol = ncol; d->K = K; d->nshift = nshift; d->beta = beta; d->rmax = rmax; d->in_shift = in_shift; d->nact = nact;
  d->agc = agc;
  d->fc0 = g_nofill ? K / prm->z : ldpc_filler_col0(prm->z, K / prm->z, agc.kp, agc.K);
  /* punctured columns all zero (as OAI passes them): iteration-1 shortcuts (README section 11) */
  d->pz = g_noit1 ? 0 : ldpc_nend(llr, 2 * prm->z) == 0;
  d->max_iter = prm->max_iter < 1 ? 1 : prm->max_iter;
  return 0;
}

int ldpc_decode(const ldpc_dec_params_t *prm, const int8_t *llr, uint8_t *out, ldpc_dec_stats_t *st)
{
  ldpc_dec_stats_t dummy;
  if (!st) st = &dummy;
  memset(st, 0, sizeof *st);
  st->crc_ok = -1;
  dsetup_t d;
  if (dec_setup(prm, llr, &d)) return -1;
  const int K = d.K, nshift = d.nshift, beta = d.beta, rmax = d.rmax, in_shift = d.in_shift, nact = d.nact;
  ldpc_agc_t agc = d.agc;
  st->rows = nact;
  const int max_iter = d.max_iter;
  ldpc_impl_t impl = prm->impl == LDPC_IMPL_AUTO ? impl_for_z(ldpc_impl_best(), prm->z) : prm->impl;
  int it = 0, ok = 0, crcok = -1; /* crcok: 1 = the early stop already verified the CRC of the output */
  if (impl == LDPC_IMPL_REF) {
    static __thread int8_t app[LDPC_MAXCOL * 384];
    if (prm->abort_poll && prm->abort_poll(prm->abort_ctx)) { st->aborted = 1; return 0; }
    it = ldpc_ref_decode(prm->bg, prm->z, prm->rcode, nact, max_iter, prm->no_et, nshift, beta, rmax, in_shift, &agc, llr, app, &ok,
                         prm->check_crc ? ref_crc : NULL, (void *)prm);
    static __thread ldpc_plan_t p; /* geometry only (no table is allocated) */
    ldpc_plan_build(&p, prm->bg, prm->z, prm->rcode, 0, 0, 1, 8, 1, nshift, beta, rmax, NULL);
    write_output(&p, NULL, app, prm->out_mode, out, 0);
  } else {
    tstate_t *t = tstate();
    const ldpc_plan_t *p;
    const ldpc_kernels_t *k;
    if (!t || get_plan(t, impl, prm->bg, prm->z, prm->rcode, nact, d.fc0, 1, nshift, beta, rmax, &p, &k)) return -1;
    int8_t *ws = get_ws(t, p->ws_size);
    if (!ws) return -1;
    load_input(p, ws, llr, in_shift, &agc);
    const int crcstop = prm->check_crc && !prm->no_et && p->crcmode; /* CRC-mode early stop (README section 5) */
    int mid = 0;
    const int pz = d.pz;
    while (it < max_iter) {
      if (prm->abort_poll && prm->abort_poll(prm->abort_ctx)) { st->aborted = 1; st->iterations = it; return 0; }
      const int first = it == 0;
      int r = 0;
      if (crcstop && p->midcheck && !first) { /* core rows, then core parity + CRC */
        for (; r < p->ncore; r++) p->row[r].fn[pz && it == 1 && p->row[r].it1 == 1](p, &p->row[r], ws);
        if (k->syndrome(p, ws, 0, p->ncore, p->sytabm) && crc_view(prm, p, ws, out, 1)) { ok = 1; crcok = 1; mid = 1; it++; st->core_stop = 1; break; }
      }
      if (first) for (; r < p->nrow; r++) (pz ? p->row[r].fn1 : p->row[r].fn[1])(p, &p->row[r], ws);
      else for (; r < p->nrow; r++) p->row[r].fn[pz && it == 1 && p->row[r].it1 == 1](p, &p->row[r], ws);
      it++;
      if (prm->no_et) {
        if (it == max_iter) ok = k->syndrome(p, ws, 0, p->nrow, p->sytab);
      } else if (crcstop) {
        ok = k->syndrome(p, ws, p->core0, p->core0 + p->ncore, p->sytab);
        if (ok && crc_view(prm, p, ws, out, 0)) { crcok = 1; break; }
      } else if (k->syndrome(p, ws, 0, p->nrow, p->sytab)) {
        ok = 1;
        break;
      }
    }
    if (crcok != 1 || prm->out_mode != LDPC_OUT_BITS) write_output(p, ws, NULL, prm->out_mode, out, mid);
  }
  st->iterations = it;
  st->syndrome_ok = ok;
  if (prm->check_crc) {
    /* the CRC callback reads packed bits (OAI semantics); LLR/byte output modes get a packed copy */
    uint8_t packed[LDPC_MAXCOL * 384 / 8 + 8];
    uint8_t *pk = out;
    if (prm->out_mode != LDPC_OUT_BITS) {
      int8_t tmp[22 * 384];
      const int nb = (prm->bg == 1 ? 22 : 10) * prm->z;
      for (int i = 0; i < nb; i++)
        tmp[i] = prm->out_mode == LDPC_OUT_LLR ? ((int8_t *)out)[i] : (int8_t)(out[i] ? -1 : 0);
      pack_bits(tmp, nb, packed);
      pk = packed;
    }
    st->crc_ok = crcok == 1 ? 1 : prm->check_crc(pk, (uint32_t)(prm->kprime > 0 ? prm->kprime : K), prm->crc_type) ? 1 : 0;
    return st->crc_ok;
  }
  return ok;
}

/* ---------------------------------------------------------------- batch (small Z) */
/* natural-order output of block b of a batch plan (lane t nb + b of the interleaved columns) */
static void write_output_block(const ldpc_plan_t *p, const int8_t *ws, ldpc_out_t mode, uint8_t *out, int b)
{
  const int z0 = p->z0, nb = p->nb, ncols = mode == LDPC_OUT_LLR ? p->ncol : p->kbcols;
  int8_t col[384], tmp[LDPC_MAXCOL * 384];
  int8_t *dst = mode == LDPC_OUT_LLR ? (int8_t *)out : tmp;
  for (int c = 0; c < ncols; c++) {
    get_column(p, ws, c, col, 0);
    for (int t = 0; t < z0; t++) dst[c * z0 + t] = col[t * nb + b];
  }
  if (mode == LDPC_OUT_BITS) pack_bits(tmp, ncols * z0, out);
  else if (mode == LDPC_OUT_BYTES)
    for (int i = 0; i < ncols * z0; i++) out[i] = tmp[i] < 0;
}

int ldpc_decode_batch(const ldpc_dec_params_t *prm, int nblk, const int8_t *const *llr, uint8_t *const *out,
                      ldpc_dec_stats_t *st, int *ret)
{
  if (nblk <= 0) return 0;
  const int z = prm->z;
  ldpc_impl_t impl = prm->impl;
  if (impl == LDPC_IMPL_AUTO) impl = ldpc_impl_best(); /* (a group spans >= 16 lanes) */
  /* CRC mode, abort polling, the reference, or no room for 2 blocks: one block at a time (identical results) */
  if (prm->check_crc || prm->abort_poll || impl == LDPC_IMPL_REF || 2 * z > 384 || nblk == 1 || g_nobatch) {
    for (int b = 0; b < nblk; b++) {
      ldpc_dec_params_t q = *prm;
      q.impl = prm->impl;
      const int r = ldpc_decode(&q, llr[b], out[b], st ? &st[b] : NULL);
      if (ret) ret[b] = r;
    }
    return 0;
  }
  dsetup_t ds[384];
  if (nblk > 384) { /* (large batches: in slices) */
    for (int b0 = 0; b0 < nblk; b0 += 384) {
      const int n = nblk - b0 < 384 ? nblk - b0 : 384;
      if (ldpc_decode_batch(prm, n, llr + b0, out + b0, st ? st + b0 : NULL, ret ? ret + b0 : NULL)) return -1;
    }
    return 0;
  }
  for (int b = 0; b < nblk; b++)
    if (dec_setup(prm, llr[b], &ds[b])) return -1;
  tstate_t *t = tstate();
  if (!t) return -1;
  /* blocks per group: fill L lanes (qemu V-op counts on aarch64: one 16-byte vector, n z = 16, is the minimum;
     x86 timings: about 128 lanes; README section 11); Z >= L: no gain, one block at a time */
#if defined(__aarch64__)
  const int L = impl == LDPC_IMPL_SVE2 && ldpc_sve2_bytes() > 16 ? ldpc_sve2_bytes() : 16;
#else
  const int L = 128;
#endif
  int nbmax = L / z;
  if (nbmax > 384 / z) nbmax = 384 / z;
  if (g_batchn > 0) nbmax = g_batchn < 384 / z ? g_batchn : 384 / z;
  if (nbmax < 2) {
    for (int b = 0; b < nblk; b++) {
      const int r = ldpc_decode(prm, llr[b], out[b], st ? &st[b] : NULL);
      if (ret) ret[b] = r;
    }
    return 0;
  }
  static __thread int8_t xin[LDPC_MAXCOL * 384];
  static __thread int8_t mask[384 + 64];
  for (int b0 = 0; b0 < nblk;) {
    /* a group: consecutive blocks with the same active rows and filler columns (OAI: the code blocks of one TB) */
    int n = 1;
    while (b0 + n < nblk && n < nbmax && ds[b0 + n].nact == ds[b0].nact && ds[b0 + n].fc0 == ds[b0].fc0) n++;
    const dsetup_t *d0 = &ds[b0];
    const ldpc_plan_t *p;
    const ldpc_kernels_t *k;
    if (n == 1 || get_plan(t, impl, prm->bg, z, prm->rcode, d0->nact, d0->fc0, n, d0->nshift, d0->beta, d0->rmax, &p, &k)) {
      for (int b = b0; b < b0 + n; b++) { /* single blocks (or no plan): the normal path */
        const int r = ldpc_decode(prm, llr[b], out[b], st ? &st[b] : NULL);
        if (ret) ret[b] = r;
      }
      b0 += n;
      continue;
    }
    int8_t *ws = get_ws(t, p->ws_size);
    if (!ws) return -1;
    /* conditioned input per block (its own AGC gain, fillers, clamps), interleaved lane-wise */
    const int ncol = d0->ncol, Zi = p->z;
    int pz = 1;
    for (int b = 0; b < n; b++) {
      const dsetup_t *d = &ds[b0 + b];
      const int8_t *x = llr[b0 + b];
      pz &= d->pz;
      for (int c = 0; c < ncol; c++) {
        const int cm = ldpc_chan_max(d->rmax, p->ckind[c] == COL_DEG1);
        for (int tt = 0; tt < z; tt++) {
          const int i = c * z + tt;
          int8_t v;
          if (i >= d->agc.kp && i < d->agc.K) v = ldpc_scale_in(127, d->in_shift, cm);
          else if (d->agc.G) {
            v = ldpc_gain1(x[i], d->agc.G, d->agc.sh + d->in_shift);
            v = (int8_t)(v > cm ? cm : v < -cm ? -cm : v);
          } else v = ldpc_scale_in(x[i], d->in_shift, cm);
          xin[c * Zi + tt * n + b] = v;
        }
      }
    }
    ldpc_agc_t id = {0, 0, d0->K, d0->K}; /* already conditioned: no gain, no filler range */
    load_input(p, ws, xin, 0, &id);
    const int max_iter = d0->max_iter;
    int done[384] = {0}, ndone = 0, it = 0;
    while (it < max_iter && ndone < n) {
      const int first = it == 0;
      if (first) for (int r = 0; r < p->nrow; r++) (pz ? p->row[r].fn1 : p->row[r].fn[1])(p, &p->row[r], ws);
      else for (int r = 0; r < p->nrow; r++) p->row[r].fn[pz && it == 1 && p->row[r].it1 == 1](p, &p->row[r], ws);
      it++;
      if (prm->no_et && it < max_iter) continue;
      memset(mask, 0, (size_t)p->zp);
      /* rows one at a time; stop as soon as every unfinished block has a violated check (the usual case before
         convergence): per-block violation = OR of its lanes t n + b over t */
      for (int r = 0; r < p->nrow; r++) {
        k->synmask(p, ws, r, r + 1, p->sytab, mask);
        if (prm->no_et) continue;
        int8_t acc[384];
        memcpy(acc, mask, (size_t)n);
        for (int tt = 1; tt < z; tt++)
          for (int b = 0; b < n; b++) acc[b] |= mask[tt * n + b];
        int all = 1;
        for (int b = 0; b < n && all; b++) all = done[b] || acc[b] < 0;
        if (all) break;
      }
      for (int b = 0; b < n; b++) {
        if (done[b]) continue;
        int bad = 0;
        for (int tt = 0; tt < z && !bad; tt++) bad = mask[tt * n + b] < 0;
        if (!bad && !prm->no_et) {
          done[b] = it; ndone++;
          write_output_block(p, ws, prm->out_mode, out[b0 + b], b);
          if (st) { memset(&st[b0 + b], 0, sizeof st[0]); st[b0 + b].iterations = it; st[b0 + b].syndrome_ok = 1;
                    st[b0 + b].crc_ok = -1; st[b0 + b].rows = d0->nact; }
          if (ret) ret[b0 + b] = 1;
        } else if (prm->no_et && it == max_iter) {
          done[b] = -1;
          write_output_block(p, ws, prm->out_mode, out[b0 + b], b);
          if (st) { memset(&st[b0 + b], 0, sizeof st[0]); st[b0 + b].iterations = it; st[b0 + b].syndrome_ok = !bad;
                    st[b0 + b].crc_ok = -1; st[b0 + b].rows = d0->nact; }
          if (ret) ret[b0 + b] = !bad;
        }
      }
    }
    for (int b = 0; b < n; b++)
      if (!done[b]) { /* not converged within max_iter */
        write_output_block(p, ws, prm->out_mode, out[b0 + b], b);
        if (st) { memset(&st[b0 + b], 0, sizeof st[0]); st[b0 + b].iterations = it; st[b0 + b].syndrome_ok = 0;
                  st[b0 + b].crc_ok = -1; st[b0 + b].rows = d0->nact; }
        if (ret) ret[b0 + b] = 0;
      }
    b0 += n;
  }
  return 0;
}
