/* dec_std internal: decoding plan (per BG, Z, R, vector geometry) and kernel interface. */
#ifndef LDPC_INTERNAL_H
#define LDPC_INTERNAL_H
#include <stdint.h>
#include <stddef.h>
#include "ldpc_dec.h"
#include "ldpc_tables.h"

#define LDPC_MAXROW 46
#define LDPC_MAXCOL 68
#define LDPC_MAXEDGE 316
#define LDPC_NCORE 4 /* core rows: rows 0..3 contain every systematic and core parity column (BG1 and BG2) */

enum { COL_UNUSED = 0, COL_DEG1 = 1, COL_MULTI = 2 };

struct ldpc_plan;
struct ldpc_row;
/* processes one row (all its superchunks) */
typedef void (*ldpc_rowfn_t)(const struct ldpc_plan *p, const struct ldpc_row *rw, int8_t *ws);

typedef struct ldpc_row {
  uint16_t nn;   /* edges to columns of degree >= 2 ("multi" edges, listed first) */
  uint16_t n1;   /* edges to degree-1 columns (0 or 1, listed after the multi edges) */
  uint16_t e0;   /* index of the row's first edge in the per-edge arrays */
  uint8_t U;     /* vectors per step of this row (register budget: all Q of a step stay in registers) */
  uint8_t ndbl;  /* superchunks k < ndbl are also stored at +Z (second copy of the first min(B, Z) bytes) */
  uint16_t nsc;  /* superchunks of U*W checks */
  uint32_t roff; /* messages of the multi edges: nn * zp bytes, layout [k][e][U*W] */
  uint32_t soff; /* degree-1 rows: row state [k][m1c, c1, c2][U*W] (3 * zp bytes) for the lazy degree-1 APP */
  uint32_t toff; /* index of the row's read-offset table in rdtab: [k][e] for the multi edges */
  uint32_t yoff; /* index of the row's syndrome table in sytab: [k][e], superchunks of the plan's B = U*W bytes */
  ldpc_rowfn_t fn[2]; /* row kernel: [0] general, [1] first iteration (messages taken as 0) */
} ldpc_row_t;

/*
 * Workspace layout (offsets in bytes from the 64-byte aligned workspace base):
 *  - multi column c (degree >= 2): nbuf[c] = 2 (even degree) or 3 (odd degree) buffers of Z + 2B bytes (B = Umax*W).
 *    A buffer holds the column's a-posteriori LLRs in the frame of the row that wrote it, buf[j] = APP[(j + s) mod Z],
 *    plus a second copy of its first min(B, Z) bytes at buf[Z..]. Visit v (in row order) of the column reads buffer
 *    rb(v) and writes buffer wb(v) = rb(v + 1) != rb(v): reading and writing never overlap, so a row is processed in
 *    one pass per superchunk (Q stays in registers). rb(0) = wb(deg - 1) = 0: after every iteration the column is in
 *    buffer 0, frame crho[c]. A row with shift s reads its B_r-byte run for superchunk k contiguously at
 *    buf[(d + k B_r) mod Z], d = (s - s_prev) mod Z (rdtab, precomputed), and writes aligned at buf[k B_r].
 *  - degree-1 column c: cchan[c] = channel LLRs rotated by its shift (Q = channel LLR exactly); its APP is not stored:
 *    it is recomputed from the row state when needed (syndrome, LLR output).
 *  - unused column (no edge in the rows of this rate): cchan[c], natural order.
 *  - valid-lane mask (zp bytes), messages and row states per row.
 */
typedef struct ldpc_plan {
  int bg, z, rcode, ncol, nrow, kbcols;
  int W, U, B;     /* vector bytes, maximum vectors per step, B = W*U */
  int zp;          /* Z rounded up to B */
  int nedge;
  int nshift, beta, rmax, lim; /* lim: magnitudes are clamped to lim <= 127 without changing corr() */
  ldpc_row_t row[LDPC_MAXROW];
  int32_t wo[LDPC_MAXEDGE];   /* multi edge: write buffer (ws offset), write at wo + k*B_r; deg-1: cchan offset */
  int32_t sb[LDPC_MAXEDGE];   /* syndrome: multi edge: final buffer (buffer 0) offset; deg-1: cchan offset */
  int32_t so[LDPC_MAXEDGE];   /* syndrome: multi edge: rotation (s - crho) mod Z of the final buffer */
  uint16_t eshift[LDPC_MAXEDGE]; /* P_ij of the edge (reference / tests) */
  uint8_t ecol[LDPC_MAXEDGE];
  uint8_t ckind[LDPC_MAXCOL];
  uint8_t cdeg[LDPC_MAXCOL];
  uint8_t nbuf[LDPC_MAXCOL];
  uint8_t crow1[LDPC_MAXCOL]; /* degree-1 column: its row */
  uint16_t crho[LDPC_MAXCOL]; /* rotation of the stored column after a complete iteration */
  int32_t cbuf[LDPC_MAXCOL], cchan[LDPC_MAXCOL]; /* cbuf: buffer 0 of a multi column */
  int32_t cbsz;               /* bytes per multi-column buffer */
  int32_t lanemask;
  int32_t *rdtab;             /* read offsets (ws offsets) per row, superchunk and multi edge (heap, ntab entries) */
  int32_t *sytab;             /* syndrome read offsets into the final buffers (B-byte superchunks), after rdtab */
  /* CRC-mode early stop (README section 5): view of the multi columns after the core rows 0 .. ncore-1 of an
     iteration (buffer, frame) and the syndrome offsets of the core rows in that view (indexed like sytab) */
  int ncore, midcheck;        /* midcheck: CRC-mode early stop used (core rows <= 60 % of the edges) */
  int32_t cbufm[LDPC_MAXCOL];
  uint16_t crhom[LDPC_MAXCOL];
  int32_t *sytabm;
  int ntab, tabcap;
  size_t ws_size;
  ldpc_impl_t impl;
} ldpc_plan_t;

/* kernel table of one implementation */
typedef struct {
  /* 1 if the hard decision satisfies all checks */
  /* rows 0 .. nrows-1 with the syndrome table tab (p->sytab: end of an iteration; p->sytabm: after the core rows) */
  int (*syndrome)(const ldpc_plan_t *p, const int8_t *ws, int nrows, const int32_t *tab);
  /* largest U (1, 2, 4) <= umax the kernels support for a row with nn multi and n1 degree-1 edges (0: none) */
  int (*row_u)(int nn, int n1, int umax);
  /* row kernel for a shape (NULL: not provided) */
  ldpc_rowfn_t (*row_fn)(int nn, int n1, int u, int first);
} ldpc_kernels_t;

/* builds the plan; k = NULL: geometry only (U = 1 per row, no read-offset table, no row kernels) */
/* nact: rows processed (ldpc_active_rows(); 0 or >= the rate's rows: all) */
int ldpc_plan_build(ldpc_plan_t *p, int bg, int z, int rcode, int nact, int W, int U, int nshift, int beta, int rmax,
                    const ldpc_kernels_t *k);
void ldpc_plan_free(ldpc_plan_t *p);


/* per-ISA kernel providers (return 0 if not compiled in) */
int ldpc_kern_scalar(ldpc_kernels_t *k, int *W);
int ldpc_kern_neon(ldpc_kernels_t *k, int *W);
int ldpc_kern_sve2(ldpc_kernels_t *k, int *W);
int ldpc_kern_avx2(ldpc_kernels_t *k, int *W);
int ldpc_kern_avx512(ldpc_kernels_t *k, int *W);
int ldpc_sve2_bytes(void); /* SVE vector length in bytes; call only if SVE2 is available */

/* input conditioning shared by every implementation: AGC gain (G = 0: off) and filler range [kp, K) */
typedef struct {
  int G, sh, kp, K;
} ldpc_agc_t;
/* computes the gain of one block (integer arithmetic only, identical on every platform); a->G = 0 if target <= 0 */
/* nend: index after the last non-zero LLR (ldpc_nend(); the statistic ignores zeros anyway, nend only limits the scan) */
void ldpc_agc_setup(int z, int ncol, int K, int kprime, int nend, int target, const int8_t *llr, ldpc_agc_t *a);
/* index after the last non-zero LLR of llr[0..n) (0 if all zero) */
int ldpc_nend(const int8_t *llr, int n);
/* rows processed for a block whose LLRs end at nend: rows 0..3 always, row r >= 4 only if its degree-1 column
   (kbcols + r) holds a non-zero LLR position at or before nend - 1 (beyond the last non-zero LLR a row is an exact
   no-op: its degree-1 Q is 0, so every message to the other columns is 0) */
static inline int ldpc_active_rows(int nrow, int kbcols, int z, int nend)
{
  const int lastcol = nend > 0 ? (nend - 1) / z : -1;
  int r = lastcol - kbcols + 1;
  if (r < 4) r = 4;
  return r < nrow ? r : nrow;
}
/* conditioned channel LLR with the AGC on: in_shift is folded into the gain (one rounding):
   sat8((x G + 2^(t-1)) >> t), t = sh + in_shift, then the channel clamp */
static inline int8_t ldpc_gain1(int8_t x, int G, int t)
{
  const int v = ((int)x * G + (1 << (t - 1))) >> t;
  return (int8_t)(v > 127 ? 127 : v < -128 ? -128 : v);
}

void ldpc_test_cond(int8_t *dst, const int8_t *src, int n, int s, int cmax, int G, int sh); /* test hooks */
long ldpc_test_abs_sum(const int8_t *x, int n, int unsat, int step, long *cnt);

/* scalar reference decoder, natural indexing (src/ldpc_ref.c), rows 0 .. nact-1 (ldpc_active_rows()); returns iterations, *ok = syndrome ok (CRC mode: the
   parity of the core rows); app_out (ncol*Z) receives the final a-posteriori LLRs in input order.
   crc != NULL: CRC-mode early stop, crc(ctx, natural-order APPs) = 1 if the CRC of the systematic bits passes */
int ldpc_ref_decode(int bg, int z, int rcode, int nact, int max_iter, int no_et, int nshift, int beta, int rmax, int in_shift,
                    const ldpc_agc_t *agc, const int8_t *llr, int8_t *app_out, int *ok,
                    int (*crc)(void *ctx, const int8_t *app), void *crc_ctx);

/* channel LLR scaling used by every implementation */
/* channel LLR conditioning: rounded right shift by s, then clamp to +-cmax. cmax = deg * rmax - 1 with deg = 1
   for degree-1 columns and 2 otherwise: the messages of the column's checks (|R| <= rmax) can always overrule the
   channel value, without ties (else a strongly wrong degree-1 parity bit can never be corrected and the syndrome
   check fails although the information bits are right) */
static inline int8_t ldpc_scale_in(int8_t x, int s, int cmax)
{
  int v = s ? ((int)x + (1 << (s - 1))) >> s : x;
  return (int8_t)(v > cmax ? cmax : v < -cmax ? -cmax : v);
}
static inline int ldpc_chan_max(int rmax, int deg1) { const int m = (deg1 ? rmax : 2 * rmax) - 1; return m > 127 ? 127 : m; }

/* magnitude correction used by every implementation (bit-exact definition) */
static inline uint8_t ldpc_corr(unsigned m, int nshift, int beta, int rmax)
{
  if (nshift) m = m - (m >> nshift);
  int x = (int)m - beta;
  if (x < 0) x = 0;
  if (x > rmax) x = rmax;
  return (uint8_t)x;
}

/* smallest m in [0, 127] with corr(m') == corr(127) for all m' in [m, 127]: magnitudes (and |-128| = 128, and
   unsaturated |a - b| up to 255) can be clamped to it without changing any message, and corr(m) for m <= lim
   needs no rmax clamp: corr(m) = max(0, m - (m >> nshift) - beta) (README, section 5) */
static inline int ldpc_corr_lim(int nshift, int beta, int rmax)
{
  int m = 127;
  const uint8_t top = ldpc_corr(127, nshift, beta, rmax);
  while (m > 0 && ldpc_corr((unsigned)m - 1, nshift, beta, rmax) == top) m--;
  return m;
}

/* message of one edge from the row state (bit-exact definition shared by the scalar paths) */
static inline int8_t ldpc_msg1(int8_t q, uint8_t m1c, int8_t c1, int8_t c2)
{
  const unsigned a = (unsigned)(q < 0 ? -q : q);
  const int s = a == m1c ? c2 : c1;
  return (int8_t)(q < 0 ? -s : s);
}
static inline int8_t ldpc_sat8(int x) { return (int8_t)(x > 127 ? 127 : x < -128 ? -128 : x); }

#endif
