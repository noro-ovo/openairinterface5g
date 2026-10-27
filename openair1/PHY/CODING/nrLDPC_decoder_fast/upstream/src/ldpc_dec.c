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

int ldpc_plan_build(ldpc_plan_t *p, int bg, int z, int rcode, int nact, int W, int U, int nshift, int beta, int rmax,
                    const ldpc_kernels_t *kt)
{
  int32_t *tab = p->rdtab;
  const int tabcap = p->tabcap;
  memset(p, 0, sizeof *p);
  p->rdtab = tab;
  p->tabcap = tabcap;
  const int ncol = ldpc_ncol(bg, rcode);
  if (!ncol || z < 2 || z > 384 || ldpc_ils_of_z[z] == 255) return -1;
  const int ils = ldpc_ils_of_z[z];
  const ldpc_bg_edge_t *etab = bg == 1 ? ldpc_bg1_edges : ldpc_bg2_edges;
  const int ntab = bg == 1 ? LDPC_BG1_EDGES : LDPC_BG2_EDGES;
  p->bg = bg; p->z = z; p->rcode = rcode; p->ncol = ncol;
  p->kbcols = bg == 1 ? 22 : 10;
  const int nrow_full = ncol - p->kbcols;
  /* active rows only (ldpc_active_rows()): the others are exact no-ops. Every column of rows 0..3 has >= 2 edges in
     rows 0..3 (BG1 and BG2), so the kind and clamp of every column read by an active row is the full graph's; the
     degree-1 columns of the skipped rows become unused columns (channel copy, all zero) */
  p->nrow = nact >= LDPC_NCORE && nact < nrow_full ? nact : nrow_full;
  int ne_full = 0;
  for (int k = 0; k < ntab; k++) ne_full += etab[k].row < nrow_full;
  p->W = W; p->U = U; p->B = W * U;
  p->zp = rup(z, p->B);
  p->nshift = nshift; p->beta = beta; p->rmax = rmax;
  p->lim = ldpc_corr_lim(nshift, beta, rmax);
  const int B = p->B, zp = p->zp, D = B < z ? B : z;
  /* column degrees within the rows of this rate; the table is row-major */
  int lastrow_shift[LDPC_MAXCOL];
  for (int k = 0; k < ntab; k++)
    if (etab[k].row < p->nrow) {
      if (etab[k].col >= ncol) return -1; /* cannot happen with 38.212 base graphs */
      p->cdeg[etab[k].col]++;
      lastrow_shift[etab[k].col] = etab[k].v[ils] % z;
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
  for (int i = 0; i < p->nrow; i++) {
    ldpc_row_t *rw = &p->row[i];
    rw->e0 = (uint16_t)ne;
    for (int pass = 0; pass < 2; pass++)
      for (int k = 0; k < ntab; k++) {
        if (etab[k].row != i) continue;
        const int c = etab[k].col, s = etab[k].v[ils] % z;
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
    if (i == LDPC_NCORE - 1) { /* view after the core rows (CRC-mode early stop) */
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
      if (!rw->fn[0] || !rw->fn[1]) return -1;
    }
  }
  p->nedge = ne;
  /* CRC-mode early stop (core rows + CRC, also after the core rows of an iteration): worth it only if the core rows
     hold at most 60 % of the edges (BG1 R1/3, R2/3, BG2 R1/5, R1/3; not BG1 R8/9 (76 of 79 edges) or BG2 R2/3 (36 of
     52), which keep the full-syndrome rule) */
  p->midcheck = p->ncore > 0 && 10 * ncore_edges <= 6 * ne_full; /* decided on the full graph of the rate */
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
        if (i < p->ncore) {
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

static int get_plan(tstate_t *t, ldpc_impl_t impl, int bg, int z, int rcode, int nact, int nshift, int beta, int rmax,
                    const ldpc_plan_t **pp, const ldpc_kernels_t **kp)
{
  if (t->gen != g_gen) {
    memset(t->valid, 0, sizeof t->valid);
    t->gen = g_gen;
  }
  for (int i = 0; i < NPLAN; i++) {
    const ldpc_plan_t *p = &t->plan[i];
    if (t->valid[i] && p->impl == impl && p->bg == bg && p->z == z && p->rcode == rcode && p->nrow == nact &&
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
  const int U = choose_u(z, W);
  if (ldpc_plan_build(&t->plan[slot], bg, z, rcode, nact, W, U, nshift, beta, rmax, &t->kern[slot])) return -1;
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
static int g_agct, g_agcstat, g_agczero, g_allrows;
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
  if (avx2) {
    long c0;
    int done;
    s = abs_sum_avx2(x, n, unsat, step, &c0, &done);
    for (i = done; i < n; i++) {
      if (step == 32 && ((i >> 4) & 1)) continue;
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
    if (step == 32 && ((i >> 4) & 1)) continue;
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
  int step = kp - 2 * z + n2 >= 8192 ? 32 : 16; /* large blocks: every other 16-byte group */
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
      const int16x8_t lo = vmull_s8(vget_low_s8(v), g8), hi = vmull_high_s8(v, g16);                    \
      const int8x16_t y = t_ <= 8 ? vqrshrn_high_n_s16(vqrshrn_n_s16(lo, t_ <= 8 ? t_ : 8), hi, t_ <= 8 ? t_ : 8) \
                                  : vcombine_s8(vqmovn_s16(vrshrq_n_s16(lo, t_)), vqmovn_s16(vrshrq_n_s16(hi, t_))); \
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
  const int t = G ? sh + s : s;
  const __m256i gv = _mm256_set1_epi16((int16_t)(G ? G : 1)), rg = _mm256_set1_epi16((int16_t)(t ? 1 << (t - 1) : 0));
  const __m128i ct = _mm_cvtsi32_si128(t);
  int i = 0;
  for (;; i += 32) {
    if (i + 32 > n) i = n - 32;
    const __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(src + i));
    __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v)), hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1));
    if (G) { lo = _mm256_mullo_epi16(lo, gv); hi = _mm256_mullo_epi16(hi, gv); }
    lo = _mm256_sra_epi16(_mm256_add_epi16(lo, rg), ct);
    hi = _mm256_sra_epi16(_mm256_add_epi16(hi, rg), ct);
    const __m256i y = _mm256_permute4x64_epi64(_mm256_packs_epi16(lo, hi), 0xD8);
    _mm256_storeu_si256((__m256i *)(void *)(dst + i), _mm256_max_epi8(_mm256_min_epi8(y, c), nc));
    if (i + 32 >= n) break;
  }
}
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
    const int t = G ? sh + s : s;
    const __m128i gv = _mm_set1_epi16((int16_t)(G ? G : 1)), rg = _mm_set1_epi16((int16_t)(t ? 1 << (t - 1) : 0));
    const __m128i ct = _mm_cvtsi32_si128(t);
    int i = 0;
    for (;; i += 16) {
      if (i + 16 > n) i = n - 16;
      const __m128i v = _mm_loadu_si128((const __m128i *)(const void *)(src + i));
      __m128i lo = _mm_cvtepi8_epi16(v), hi = _mm_cvtepi8_epi16(_mm_srli_si128(v, 8));
      if (G) { lo = _mm_mullo_epi16(lo, gv); hi = _mm_mullo_epi16(hi, gv); }
      lo = _mm_sra_epi16(_mm_add_epi16(lo, rg), ct);
      hi = _mm_sra_epi16(_mm_add_epi16(hi, rg), ct);
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
  if (!ag->G) return;
  const int f0 = ag->kp - c * z, f1 = ag->K - c * z; /* filler range in column coordinates */
  const int lo = f0 > a ? f0 : a, hi = f1 < a + n ? f1 : a + n;
  if (lo < hi) memset(dst + (lo - a), ldpc_scale_in(127, s, cmax), (size_t)(hi - lo));
}

__attribute__((noinline)) static void load_input(const ldpc_plan_t *p, int8_t *ws, const int8_t *llr, int in_shift, const ldpc_agc_t *ag)
{
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
static void pack_bits(const int8_t *v, int n, uint8_t *out)
{
  int i = 0;
#if defined(__aarch64__)
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
  const char *ar = getenv("LDPC_DEC_ALLROWS");
  g_allrows = ar && atoi(ar) != 0;
  const char *az = getenv("LDPC_DEC_AGCZERO");
  g_agczero = az && atoi(az) != 0;
  const char *st = getenv("LDPC_DEC_AGCSTAT");
  g_agcstat = st ? atoi(st) : LDPC_DEC_AGC_STAT;
  __atomic_store_n(&g_env_alg, ok, __ATOMIC_RELEASE);
}
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

int ldpc_decode(const ldpc_dec_params_t *prm, const int8_t *llr, uint8_t *out, ldpc_dec_stats_t *st)
{
  ldpc_dec_stats_t dummy;
  if (!st) st = &dummy;
  memset(st, 0, sizeof *st);
  st->crc_ok = -1;
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
  st->rows = nact;
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
  const int max_iter = prm->max_iter < 1 ? 1 : prm->max_iter;
  ldpc_impl_t impl = prm->impl == LDPC_IMPL_AUTO ? impl_for_z(ldpc_impl_best(), prm->z) : prm->impl;
  int it = 0, ok = 0, crcok = -1; /* crcok: 1 = the early stop already verified the CRC of the output */
  if (impl == LDPC_IMPL_REF) {
    static __thread int8_t app[LDPC_MAXCOL * 384];
    if (prm->abort_poll && prm->abort_poll(prm->abort_ctx)) { st->aborted = 1; return 0; }
    it = ldpc_ref_decode(prm->bg, prm->z, prm->rcode, nact, max_iter, prm->no_et, nshift, beta, rmax, in_shift, &agc, llr, app, &ok,
                         prm->check_crc ? ref_crc : NULL, (void *)prm);
    static __thread ldpc_plan_t p; /* geometry only (no table is allocated) */
    ldpc_plan_build(&p, prm->bg, prm->z, prm->rcode, 0, 8, 1, nshift, beta, rmax, NULL);
    write_output(&p, NULL, app, prm->out_mode, out, 0);
  } else {
    tstate_t *t = tstate();
    const ldpc_plan_t *p;
    const ldpc_kernels_t *k;
    if (!t || get_plan(t, impl, prm->bg, prm->z, prm->rcode, nact, nshift, beta, rmax, &p, &k)) return -1;
    int8_t *ws = get_ws(t, p->ws_size);
    if (!ws) return -1;
    load_input(p, ws, llr, in_shift, &agc);
    const int crcstop = prm->check_crc && !prm->no_et && p->midcheck; /* CRC-mode early stop (README section 5) */
    int mid = 0;
    while (it < max_iter) {
      if (prm->abort_poll && prm->abort_poll(prm->abort_ctx)) { st->aborted = 1; st->iterations = it; return 0; }
      const int first = it == 0;
      int r = 0;
      if (crcstop && !first) { /* core rows, then core parity + CRC */
        for (; r < p->ncore; r++) p->row[r].fn[0](p, &p->row[r], ws);
        if (k->syndrome(p, ws, p->ncore, p->sytabm) && crc_view(prm, p, ws, out, 1)) { ok = 1; crcok = 1; mid = 1; it++; st->core_stop = 1; break; }
      }
      for (; r < p->nrow; r++) p->row[r].fn[first](p, &p->row[r], ws);
      it++;
      if (prm->no_et) {
        if (it == max_iter) ok = k->syndrome(p, ws, p->nrow, p->sytab);
      } else if (crcstop) {
        ok = k->syndrome(p, ws, p->ncore, p->sytab);
        if (ok && crc_view(prm, p, ws, out, 0)) { crcok = 1; break; }
      } else if (k->syndrome(p, ws, p->nrow, p->sytab)) {
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
