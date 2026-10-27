#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "tutil.h"
#include "ldpc_tables.h"

void rng_seed(rng_t *r, uint64_t seed)
{
  for (int i = 0; i < 4; i++) { /* splitmix64 */
    seed += 0x9E3779B97F4A7C15ULL;
    uint64_t z = seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    r->s[i] = z ^ (z >> 31);
  }
}
double rng_gauss(rng_t *r)
{ /* Marsaglia polar */
  for (;;) {
    double u = (rng_next(r) >> 11) * (2.0 / 9007199254740992.0) - 1.0;
    double v = (rng_next(r) >> 11) * (2.0 / 9007199254740992.0) - 1.0;
    double s = u * u + v * v;
    if (s > 0 && s < 1) return u * sqrt(-2.0 * log(s) / s);
  }
}
int t_ncol_full(int bg) { return bg == 1 ? 68 : 52; }
int t_kb(int bg) { return bg == 1 ? 22 : 10; }

static const ldpc_bg_edge_t *tab(int bg, int *n)
{
  *n = bg == 1 ? LDPC_BG1_EDGES : LDPC_BG2_EDGES;
  return bg == 1 ? ldpc_bg1_edges : ldpc_bg2_edges;
}

int t_syndrome(int bg, int z, int nrow, const uint8_t *cw)
{
  int n, bad = 0;
  const ldpc_bg_edge_t *t = tab(bg, &n);
  const int ils = ldpc_ils_of_z[z];
  for (int i = 0; i < nrow; i++)
    for (int c = 0; c < z; c++) {
      int par = 0;
      for (int k = 0; k < n; k++)
        if (t[k].row == i) par ^= cw[t[k].col * z + (c + t[k].v[ils] % z) % z] & 1;
      bad += par;
    }
  return bad;
}

/* inverse of the 4Z x 4Z core parity block, cached per (bg, z) */
typedef struct { int n, words; uint64_t *inv; } coreinv_t;
static coreinv_t cache[2][385];
static pthread_mutex_t cache_mx = PTHREAD_MUTEX_INITIALIZER;

static const coreinv_t *core_inverse(int bg, int z)
{
  pthread_mutex_lock(&cache_mx);
  coreinv_t *ci = &cache[bg - 1][z];
  if (ci->inv) { pthread_mutex_unlock(&cache_mx); return ci; }
  int ne;
  const ldpc_bg_edge_t *t = tab(bg, &ne);
  const int ils = ldpc_ils_of_z[z], kb = t_kb(bg), n = 4 * z, w = (2 * n + 63) / 64;
  uint64_t *m = calloc((size_t)n * w, 8);
  for (int k = 0; k < ne; k++) {
    if (t[k].row >= 4 || t[k].col < kb || t[k].col >= kb + 4) continue;
    const int P = t[k].v[ils] % z;
    for (int c = 0; c < z; c++) {
      const int r = t[k].row * z + c, col = (t[k].col - kb) * z + (c + P) % z;
      m[(size_t)r * w + col / 64] ^= 1ULL << (col % 64);
    }
  }
  for (int r = 0; r < n; r++) m[(size_t)r * w + (n + r) / 64] |= 1ULL << ((n + r) % 64);
  for (int col = 0; col < n; col++) { /* Gauss-Jordan over GF(2) */
    int piv = -1;
    for (int r = col; r < n; r++)
      if (m[(size_t)r * w + col / 64] >> (col % 64) & 1) { piv = r; break; }
    if (piv < 0) { free(m); pthread_mutex_unlock(&cache_mx); return NULL; }
    if (piv != col)
      for (int k = 0; k < w; k++) { uint64_t x = m[(size_t)piv * w + k]; m[(size_t)piv * w + k] = m[(size_t)col * w + k]; m[(size_t)col * w + k] = x; }
    for (int r = 0; r < n; r++)
      if (r != col && (m[(size_t)r * w + col / 64] >> (col % 64) & 1))
        for (int k = 0; k < w; k++) m[(size_t)r * w + k] ^= m[(size_t)col * w + k];
  }
  ci->n = n; ci->words = w; ci->inv = m;
  pthread_mutex_unlock(&cache_mx);
  return ci;
}

int t_encode(int bg, int z, const uint8_t *info, uint8_t *cw)
{
  int ne;
  const ldpc_bg_edge_t *t = tab(bg, &ne);
  const int ils = ldpc_ils_of_z[z], kb = t_kb(bg), ncol = t_ncol_full(bg), nrow = ncol - kb;
  const coreinv_t *ci = core_inverse(bg, z);
  if (!ci) return -1;
  memset(cw, 0, (size_t)ncol * z);
  for (int i = 0; i < kb * z; i++) cw[i] = info[i] & 1;
  /* s = H_core_info * c */
  uint8_t *s = calloc(4 * (size_t)z, 1);
  for (int k = 0; k < ne; k++)
    if (t[k].row < 4 && t[k].col < kb)
      for (int c = 0; c < z; c++) s[t[k].row * z + c] ^= cw[t[k].col * z + (c + t[k].v[ils] % z) % z];
  /* p = inv * s : inv is in the right half of the augmented matrix */
  for (int r = 0; r < 4 * z; r++) {
    int b = 0;
    for (int j = 0; j < 4 * z; j++)
      if (ci->inv[(size_t)r * ci->words + (4 * z + j) / 64] >> ((4 * z + j) % 64) & 1) b ^= s[j];
    cw[kb * z + r] = (uint8_t)b;
  }
  free(s);
  /* extension rows: exactly one column of degree 1 (col kb+i) per row i >= 4 */
  for (int i = 4; i < nrow; i++) {
    int unk = -1, unkP = 0;
    for (int k = 0; k < ne; k++)
      if (t[k].row == i && t[k].col == kb + i) { unk = k; unkP = t[k].v[ils] % z; }
    if (unk < 0) return -2;
    for (int c = 0; c < z; c++) {
      int b = 0;
      for (int k = 0; k < ne; k++)
        if (t[k].row == i && k != unk) b ^= cw[t[k].col * z + (c + t[k].v[ils] % z) % z];
      cw[(kb + i) * z + (c + unkP) % z] = (uint8_t)b;
    }
  }
  return 0;
}

void t_channel(rng_t *r, int bg, int z, int ncol, int kprime, const uint8_t *cw, double ebn0, int8_t *llr)
{
  const int kb = t_kb(bg), K = kb * z;
  const double rate = (double)kb / (ncol - 2);
  const double sigma = 1.0 / sqrt(2.0 * rate * pow(10.0, ebn0 / 10.0)), D = sigma / 16.0;
  memset(llr, 0, 2 * (size_t)z);
  for (int i = 2 * z; i < ncol * z; i++) {
    if (i >= kprime && i < K) { llr[i] = 127; continue; }
    const double y = (cw[i] ? -1.0 : 1.0) + sigma * rng_gauss(r);
    double q = floor(y / D);
    if (q <= -128) q = -128;
    else if (q >= 128) q = 127;
    llr[i] = (int8_t)q;
  }
}

void t_channel_s(rng_t *r, int bg, int z, int ncol, int kprime, const uint8_t *cw, double ebn0, int mode, double f, int sat,
                 int8_t *llr)
{
  const int kb = t_kb(bg), K = kb * z;
  const double rate = (double)kb / (ncol - 2);
  const double sigma = 1.0 / sqrt(2.0 * rate * pow(10.0, ebn0 / 10.0));
  const double lo = sat == 127 ? -128 : -sat, hi = sat;
  memset(llr, 0, 2 * (size_t)z);
  for (int i = 2 * z; i < ncol * z; i++) {
    if (i >= kprime && i < K) { llr[i] = 127; continue; }
    const double y = (cw[i] ? -1.0 : 1.0) + sigma * rng_gauss(r);
    double q = mode == 0 ? floor(f * y / (sigma / 16.0)) : floor(f * 10.0 * y / (sigma * sigma) + 0.5);
    if (q < lo) q = lo;
    if (q > hi) q = hi;
    llr[i] = (int8_t)q;
  }
}
