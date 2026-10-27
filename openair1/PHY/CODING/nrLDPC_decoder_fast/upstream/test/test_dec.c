/* dec_std correctness tests.
 *   test_dec [--quick] [--seed S] [--blocks N] [--impl name]
 * 1. tables/encoder: every (BG, Z): random info bits -> spec encoder -> independent syndrome of all rows = 0.
 * 2. bit-exactness of every available implementation (and every unroll factor) against the scalar reference:
 *    all 51 Z x 6 (BG, R), random K' (fillers), SNR from noise-only to clean, pure-noise LLRs incl. -128,
 *    saturated LLRs (+-127/-128), max iterations 1..20, early termination on/off, all output modes.
 *    Compared: output buffer (incl. untouched tail), iterations, success flag; LLR mode compares all APP LLRs.
 * 3. sanity: clean/high-SNR blocks decode to the transmitted bits, decoded codewords satisfy H c = 0.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ldpc_dec.h"
#include "tutil.h"
#include "ldpc_tables.h"
#include "ldpc_internal.h"

static const int R1[3] = {13, 23, 89}, R2[3] = {15, 13, 23};
#define OUTSZ (68 * 384 + 64)

static int fails;
#define CHECK(cond, ...)                         \
  do {                                           \
    if (!(cond)) {                               \
      fails++;                                   \
      if (fails < 30) { printf("FAIL: " __VA_ARGS__); printf("\n"); } \
    }                                            \
  } while (0)

/* CRC callbacks for the CRC-mode early stop: oracle (= the transmitted information bits) and a deterministic hash
   that passes for about 1 in 3 bit patterns (exercises "core rows satisfied, CRC failed, continue") */
static const uint8_t *g_info;
static int crc_oracle(uint8_t *b, uint32_t n, uint8_t t)
{
  (void)t;
  for (uint32_t i = 0; i < n; i++)
    if (((b[i >> 3] >> (7 - (i & 7))) & 1) != g_info[i]) return 0;
  return 1;
}
static int crc_hash(uint8_t *b, uint32_t n, uint8_t t)
{
  (void)t;
  uint32_t h = 2166136261u;
  for (uint32_t i = 0; i < (n + 7) / 8; i++) h = (h ^ (i + 1 == (n + 7) / 8 && (n & 7) ? b[i] & (0xFF00 >> (n & 7)) : b[i])) * 16777619u;
  return h % 3 == 0;
}

static void make_block(rng_t *r, int bg, int z, int ncol, int kprime, int kind, double snr, uint8_t *cw, int8_t *llr)
{
  const int K = t_kb(bg) * z;
  static uint8_t info[22 * 384];
  for (int i = 0; i < K; i++) info[i] = i < kprime ? (uint8_t)(rng_next(r) & 1) : 0;
  if (t_encode(bg, z, info, cw)) { printf("encode failed\n"); exit(2); }
  if (kind == 0) {
    t_channel(r, bg, z, ncol, kprime, cw, snr, llr);
  } else {
    memset(llr, 0, 2 * (size_t)z);
    for (int i = 2 * z; i < ncol * z; i++) {
      if (i >= kprime && i < K) { llr[i] = 127; continue; }
      if (kind == 1) llr[i] = (int8_t)(rng_next(r) & 0xFF); /* pure noise, includes -128 */
      else { /* saturated, mostly correct, some flipped */
        int v = (rng_next(r) & 3) ? 127 : 128;
        if (cw[i]) v = -v;
        if ((rng_next(r) % 16) == 0) v = -v;
        llr[i] = (int8_t)(v > 127 ? 127 : v < -128 ? -128 : v);
      }
    }
  }
}

int main(int argc, char **argv)
{
  int quick = 0, blocks = 3000;
  uint64_t seed = 1;
  const char *only = NULL;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--quick")) quick = 1;
    else if (!strcmp(argv[i], "--seed")) seed = strtoull(argv[++i], NULL, 0);
    else if (!strcmp(argv[i], "--blocks")) blocks = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--impl")) only = argv[++i];
  }
  rng_t rng;
  rng_seed(&rng, seed);
  static uint8_t info[22 * 384], cw[68 * 384];
  static int8_t llr[68 * 384];
  /* ---- 1. encoder / tables */
  int nenc = 0;
  for (int bg = 1; bg <= 2; bg++)
    for (int zi = 0; zi < 51; zi++) {
      const int z = ldpc_lifting_sizes[zi], K = t_kb(bg) * z;
      for (int rep = 0; rep < (quick ? 1 : 3); rep++) {
        for (int i = 0; i < K; i++) info[i] = rng_next(&rng) & 1;
        CHECK(t_encode(bg, z, info, cw) == 0, "encode BG%d Z%d", bg, z);
        CHECK(t_syndrome(bg, z, t_ncol_full(bg) - t_kb(bg), cw) == 0, "syndrome BG%d Z%d", bg, z);
        nenc++;
      }
    }
  printf("[1] spec encoder + independent syndrome: %d codewords, all 51 Z x 2 BG, fails so far %d\n", nenc, fails);

  /* ---- 2. bit-exactness vs reference */
  ldpc_impl_t impls[LDPC_IMPL_COUNT];
  int nimpl = 0;
  for (int i = LDPC_IMPL_SCALAR; i < LDPC_IMPL_COUNT; i++)
    if (ldpc_impl_available((ldpc_impl_t)i) && (!only || !strcmp(only, ldpc_impl_name((ldpc_impl_t)i))))
      impls[nimpl++] = (ldpc_impl_t)i;
  printf("[2] implementations under test:");
  for (int i = 0; i < nimpl; i++) printf(" %s", ldpc_impl_name(impls[i]));
  printf(" (unroll factors 1, 2, 4 each)\n");
  static uint8_t out_ref[OUTSZ], out_x[OUTSZ];
  int ncase = 0, nconv = 0, nmax = 0, ncrc = 0;
  const int total = quick ? blocks / 4 : blocks;
  for (int b = 0; b < total; b++) {
    const int bg = 1 + (int)(rng_next(&rng) & 1);
    const int rc = bg == 1 ? R1[rng_next(&rng) % 3] : R2[rng_next(&rng) % 3];
    /* every Z appears; first pass is systematic over Z */
    const int z = ldpc_lifting_sizes[b < 51 * 6 ? b % 51 : (int)(rng_next(&rng) % 51)];
    const int bgx = b < 51 * 6 ? 1 + (b / 51) % 2 : bg;
    const int rcx = b < 51 * 6 ? (bgx == 1 ? R1[(b / 102) % 3] : R2[(b / 102) % 3]) : rc;
    const int ncol = ldpc_ncol(bgx, rcx), K = t_kb(bgx) * z;
    int kprime = (rng_next(&rng) & 1) ? K : K - (int)(rng_next(&rng) % (K / 4 + 1));
    if (rng_next(&rng) % 8 == 0) kprime = K * 55 / 100 + (int)(rng_next(&rng) % (K / 10 + 1)); /* many filler columns */
    const int kind = (rng_next(&rng) % 10) < 7 ? 0 : (rng_next(&rng) & 1) ? 1 : 2;
    const double snr = -1.0 + (rng_next(&rng) % 81) / 10.0;
    static const int its[8] = {1, 2, 3, 4, 5, 8, 12, 20};
    const int maxit = its[rng_next(&rng) % 8];
    const int mode = (int)(rng_next(&rng) % 3);
    const int no_et = (rng_next(&rng) % 5) == 0;
    int nshift = 0, beta = 0, rmax = 0, insh = 0; /* defaults */
    if (rng_next(&rng) % 4 == 0) { /* other corrections too */
      static const int ns[4] = {-1, 2, 3, 4}, rm[4] = {31, 63, 100, 127};
      nshift = ns[rng_next(&rng) % 4];
      beta = (int)(rng_next(&rng) % 4);
      rmax = rm[rng_next(&rng) % 4];
      static const int is[4] = {-1, 1, 2, 3};
      insh = is[rng_next(&rng) % 4];
    }
    make_block(&rng, bgx, z, ncol, kprime, kind, snr, cw, llr);
    /* other LLR scales (AGC gains 1/8 .. 8): x f, f = 1/16 .. 8, saturated, fillers stay +127 */
    if (kind == 0 && rng_next(&rng) % 3 == 0) {
      static const double fs[8] = {0.0625, 0.25, 0.5, 0.7, 1.5, 2, 4, 8};
      const double fsc = fs[rng_next(&rng) % 8];
      for (int i = 2 * z; i < ncol * z; i++) {
        if (i >= kprime && i < K) continue;
        const double v = llr[i] * fsc;
        llr[i] = (int8_t)(v > 127 ? 127 : v < -128 ? -128 : v);
      }
    }
    if (kprime < K && rng_next(&rng) % 4 == 0) /* fillers not at +127 (the AGC bypass sets them to +127) */
      for (int i = kprime; i < K; i++) llr[i] = (int8_t)(rng_next(&rng) & 0x7F);
    /* zero-filled tail (OAI: untransmitted positions beyond E with R13 for every rate < 2/3): the LLRs end at a
       random position, also inside the core parity columns; the skipped rows and the AGC statistic depend on it */
    int ntail = 0;
    if (rng_next(&rng) % 3 == 0) {
      const int lo = K + 2 * z, nend = lo + (int)(rng_next(&rng) % (unsigned)(ncol * z - lo + 1));
      for (int i = nend; i < ncol * z; i++) llr[i] = 0;
      ntail = ncol * z - nend;
    }
    if (rng_next(&rng) % 8 == 0) /* punctured positions not zero: no iteration-1 shortcuts (they need exact zeros) */
      for (int i = 0; i < 2 * z; i += 1 + (int)(rng_next(&rng) % 64)) llr[i] = (int8_t)(rng_next(&rng) & 0xFF);
    static const uint8_t qms[6] = {0, 1, 2, 4, 6, 8};
    const uint8_t qmv = qms[rng_next(&rng) % 6];
    static const int8_t agcs[3] = {0, 1, -1};
    const int8_t agcv = agcs[rng_next(&rng) % 3];
    const int kpp = (rng_next(&rng) & 3) ? kprime : 0; /* kprime not always passed to the decoder */
    ldpc_dec_params_t p = {.bg = (uint8_t)bgx, .z = (uint16_t)z, .rcode = (uint8_t)rcx, .max_iter = (uint8_t)maxit,
                           .no_et = (uint8_t)no_et, .out_mode = (ldpc_out_t)mode, .impl = LDPC_IMPL_REF,
                           .nshift = (int8_t)nshift, .beta = (int8_t)beta, .rmax = (int8_t)rmax, .in_shift = (int8_t)insh,
                           .agc = agcv, .kprime = kpp, .qm = qmv};
    const int crcm = (int)(rng_next(&rng) % 4); /* 0, 1: no CRC; 2: oracle CRC; 3: hash CRC */
    static uint8_t infob[22 * 384];
    for (int i = 0; i < K; i++) infob[i] = cw[i];
    g_info = infob;
    if (crcm == 2) p.check_crc = crc_oracle;
    if (crcm == 3) p.check_crc = crc_hash;
    ldpc_dec_stats_t sr, sx;
    memset(out_ref, 0xA5, OUTSZ);
    int okr = ldpc_decode(&p, llr, out_ref, &sr);
    ncase++;
    nconv += sr.syndrome_ok;
    nmax += sr.iterations == maxit;
    for (int ii = 0; ii < nimpl; ii++)
      for (int u = 1; u <= 4; u <<= 1) {
        ldpc_dec_force_unroll(u);
        p.impl = impls[ii];
        memset(out_x, 0xA5, OUTSZ);
        int okx = ldpc_decode(&p, llr, out_x, &sx);
        const int same = okx == okr && sx.iterations == sr.iterations && sx.syndrome_ok == sr.syndrome_ok &&
                         sx.crc_ok == sr.crc_ok && !memcmp(out_ref, out_x, OUTSZ);
        if (!(same && sx.rows == sr.rows) && getenv("TDUMP")) {
          FILE *fd = fopen(getenv("TDUMP"), "wb");
          fwrite(&p, sizeof p, 1, fd); fwrite(llr, 1, ncol * z, fd); fclose(fd);
          fprintf(stderr, "rows %d/%d out diff at", sr.rows, sx.rows);
          for (int q = 0; q < OUTSZ; q++) if (out_ref[q] != out_x[q]) { fprintf(stderr, " %d", q); if (q > 2000) break; }
          fprintf(stderr, "\n"); exit(1);
        }
        CHECK(same && sx.rows == sr.rows, "%s U%d BG%d Z%d R%d K'%d kind %d snr %.1f it %d mode %d no_et %d ns %d beta %d rmax %d insh %d agc %d kp %d crc %d tail %d qm %d: ok %d/%d it %d/%d",
              ldpc_impl_name(impls[ii]), u, bgx, z, rcx, kprime, kind, snr, maxit, mode, no_et, nshift, beta, rmax, insh, agcv, kpp, crcm, ntail, qmv, okr, okx,
              sr.iterations, sx.iterations);
      }
    /* sanity: success means the hard decision on the systematic part is consistent with a codeword */
    ncrc += crcm >= 2;
    if (okr && mode == LDPC_OUT_LLR && crcm < 2) {
      static uint8_t hd[68 * 384];
      for (int i = 0; i < ncol * z; i++) hd[i] = ((int8_t *)out_ref)[i] < 0;
      CHECK(t_syndrome(bgx, z, sr.rows, hd) == 0, "ref success but syndrome != 0 (active rows)");
    }
    if (kind == 0 && !ntail && snr >= 6.0 && mode == LDPC_OUT_BYTES && maxit >= 5 && rcx != 89 && !nshift && !beta && !rmax && !insh && agcv >= 0 && crcm != 3) {
      int err = 0;
      for (int i = 0; i < kprime; i++) err += out_ref[i] != cw[i];
      CHECK(err == 0 && okr, "high-SNR block not decoded: BG%d Z%d R%d snr %.1f errors %d", bgx, z, rcx, snr, err);
    }
  }
  ldpc_dec_force_unroll(0);
  printf("[2] %d blocks x %d impls x 3 unroll factors vs reference (%d converged, %d at max iterations, %d in CRC mode)\n", ncase,
         nimpl, nconv, nmax, ncrc);
  /* ---- 3. degenerate inputs: all 0, all -128, all +127, alternating -128/+127, every (BG, R), a few Z, 1/20 it */
  int ndeg = 0;
  for (int bg = 1; bg <= 2; bg++)
    for (int ri = 0; ri < 3; ri++)
      for (int zi = 0; zi < 51; zi += 10)
        for (int pat = 0; pat < 4; pat++)
          for (int mi = 0; mi < 2; mi++) {
            const int z = ldpc_lifting_sizes[zi], rc = bg == 1 ? R1[ri] : R2[ri], ncol = ldpc_ncol(bg, rc);
            for (int i = 0; i < ncol * z; i++)
              llr[i] = (int8_t)(i < 2 * z ? 0 : pat == 0 ? 0 : pat == 1 ? -128 : pat == 2 ? 127 : (i & 1) ? -128 : 127);
            ldpc_dec_params_t p = {.bg = (uint8_t)bg, .z = (uint16_t)z, .rcode = (uint8_t)rc, .max_iter = (uint8_t)(mi ? 20 : 1),
                                   .out_mode = LDPC_OUT_LLR, .impl = LDPC_IMPL_REF, .agc = (int8_t)(pat & 1 ? 1 : -1)};
            ldpc_dec_stats_t sr, sx;
            memset(out_ref, 0xA5, OUTSZ);
            const int okr = ldpc_decode(&p, llr, out_ref, &sr);
            for (int ii = 0; ii < nimpl; ii++) {
              p.impl = impls[ii];
              memset(out_x, 0xA5, OUTSZ);
              const int okx = ldpc_decode(&p, llr, out_x, &sx);
              CHECK(okx == okr && sx.iterations == sr.iterations && !memcmp(out_ref, out_x, OUTSZ),
                    "degenerate pattern %d %s BG%d Z%d R%d", pat, ldpc_impl_name(impls[ii]), bg, z, rc);
            }
            if (pat == 2) CHECK(okr == 1, "all +127 input must decode (all-zero codeword) BG%d Z%d R%d", bg, z, rc);
            ndeg++;
          }
  printf("[3] degenerate inputs (all 0 / -128 / +127 / alternating), 1 and 20 iterations: %d cases\n", ndeg);
  /* ---- 4. input conditioning kernel (AGC gain + in_shift + clamp) vs its scalar definition: every int8 value,
     every gain G in [64, 127] x sh in [3, 9] and no gain, in_shift 0..3, clamps 46 / 93 / 127, several lengths */
  {
    static int8_t src[300], dst[300];
    for (int i = 0; i < 300; i++) src[i] = (int8_t)(i * 37 + 11);
    for (int i = 0; i < 256; i++) src[i] = (int8_t)i;
    int nbad = 0, ncfg = 0;
    static const int cms[3] = {46, 93, 127}, lens[4] = {300, 17, 16, 5};
    for (int G = 63; G <= 127; G++)
      for (int sh = 3; sh <= 9; sh++)
        for (int s = 0; s <= 3; s++)
          for (int ci = 0; ci < 3; ci++)
            for (int li = 0; li < 4; li++) {
              const int g = G == 63 ? 0 : G;
              if (!g && sh != 3) continue;
              ldpc_test_cond(dst, src, lens[li], s, cms[ci], g, sh);
              for (int i = 0; i < lens[li]; i++) {
                const int8_t x = src[i];
                int v;
                if (g) v = ldpc_gain1(x, g, sh + s); /* (v3: byte-SQRDMULH definition) */
                else v = s ? ((int)x + (1 << (s - 1))) >> s : x;
                v = v > cms[ci] ? cms[ci] : v < -cms[ci] ? -cms[ci] : v;
                nbad += dst[i] != (int8_t)v;
              }
              ncfg++;
            }
    /* AGC statistic pass (vector) vs scalar: sums and counts, all values / below the rails, every / every other group */
    static int8_t big[20000];
    for (int i = 0; i < 20000; i++) big[i] = (int8_t)(rng_next(&rng) & 0xFF);
    for (int i = 0; i < 3000; i++) big[rng_next(&rng) % 20000] = (rng_next(&rng) & 1) ? 127 : -128;
    for (int i = 0; i < 3000; i++) big[rng_next(&rng) % 20000] = 0;          /* zeros: not counted */
    for (int i = 12000; i < 12500; i++) big[i] = 0;                         /* a zero run */
    for (int len = 1; len < 20000; len = len * 3 + 7)
      for (int un = 0; un <= 1; un++)
        for (int st = 16; st <= 64; st *= 2) {
          long c, cs = 0, ss = 0;
          const long sv = ldpc_test_abs_sum(big, len, un, st, &c);
          for (int i = 0; i < len; i++) {
            if ((i >> 4) & (st / 16 - 1)) continue;
            const int a = big[i] < 0 ? -big[i] : big[i];
            if (a > 0 && (!un || a < 127)) { ss += a; cs++; }
          }
          nbad += sv != ss || c != cs;
          ncfg++;
        }
    /* hard-decision bit packing (MSB first) vs scalar, all lengths up to 1000, NEON/SSE and SVE2/AVX-512 paths */
    for (int len = 1; len <= 1000; len += 1 + len / 7)
      for (int sv = 0; sv <= 1; sv++) {
        static uint8_t pk[200];
        memset(pk, 0xA5, sizeof pk);
        ldpc_test_pack(big, len, pk, sv);
        for (int i = 0; i < len; i++) nbad += ((pk[i >> 3] >> (7 - (i & 7))) & 1) != (big[i] < 0);
        ncfg++;
      }
    CHECK(nbad == 0, "conditioning kernel / AGC statistic / bit packing: %d mismatches", nbad);
    printf("[4] input conditioning kernel (AGC gain with folded in_shift, in_shift, clamp) and AGC statistic pass vs scalar definitions: %d configurations, %d mismatches\n", ncfg, nbad);
  }
  /* ---- 5. skipped rows are exact no-ops: the reference with the active rows only vs all rows, fixed iterations
     (no early termination), zero tail at several transmitted lengths: every APP except the skipped rows' degree-1
     columns (all-zero channel) must be identical */
  {
    int ncmp = 0, nbad = 0;
    static int8_t appa[68 * 384], appb[68 * 384];
    for (int bg = 1; bg <= 2; bg++)
      for (int ri = 0; ri < 3; ri++)
        for (int zi = 0; zi < 51; zi += (quick ? 17 : 7)) {
          const int z = ldpc_lifting_sizes[zi], rc = bg == 1 ? R1[ri] : R2[ri], ncol = ldpc_ncol(bg, rc), K = t_kb(bg) * z;
          const int nrow = ncol - t_kb(bg);
          for (int f = 0; f < 4; f++) {
            make_block(&rng, bg, z, ncol, K, 0, 1.0 + 0.5 * f, cw, llr);
            const int nend = K + 4 * z + (int)((long)(ncol * z - K - 4 * z) * f / 4) + (f & 1) * (z / 3);
            for (int i = nend; i < ncol * z; i++) llr[i] = 0;
            const int nact = ldpc_active_rows(nrow, t_kb(bg), z, ldpc_nend(llr, ncol * z));
            for (int it = 1; it <= 6; it += 5) {
              int oka, okb;
              ldpc_ref_decode(bg, z, rc, nact, it, 1, 0, 2, 47, 1, NULL, llr, appa, &oka, NULL, NULL);
              ldpc_ref_decode(bg, z, rc, nrow, it, 1, 0, 2, 47, 1, NULL, llr, appb, &okb, NULL, NULL);
              for (int c = 0; c < ncol; c++) {
                if (c >= t_kb(bg) + 4 && c - t_kb(bg) >= nact) continue; /* degree-1 column of a skipped row */
                nbad += memcmp(appa + c * z, appb + c * z, z) != 0;
              }
              ncmp++;
            }
          }
        }
    CHECK(nbad == 0, "skipped rows are not no-ops: %d column mismatches", nbad);
    printf("[5] rows beyond the last non-zero LLR are exact no-ops (reference, active rows vs all rows): %d cases, %d mismatches\n", ncmp, nbad);
  }
  /* ---- 6. batch decoding (small Z, lane-interleaved) == block-by-block decoding: outputs, iterations, syndrome flag */
  {
    int nb_cases = 0, nbad = 0;
    static int8_t bl[24][68 * 384];
    static uint8_t bo[24][OUTSZ], so[OUTSZ];
    for (int tc = 0; tc < (quick ? 60 : 240); tc++) {
      const int bg = 1 + (int)(rng_next(&rng) & 1);
      const int rc = bg == 1 ? R1[rng_next(&rng) % 3] : R2[rng_next(&rng) % 3];
      const int z = ldpc_lifting_sizes[rng_next(&rng) % 26]; /* Z <= 48 */
      const int ncol = ldpc_ncol(bg, rc), K = t_kb(bg) * z;
      const int nblk = 1 + (int)(rng_next(&rng) % 24);
      const int kprime = (rng_next(&rng) & 1) ? K : K - (int)(rng_next(&rng) % (K / 3 + 1));
      const int tail = (rng_next(&rng) % 3 == 0) ? (int)(rng_next(&rng) % (ncol * z - K - 2 * z)) : 0;
      const int mode = (int)(rng_next(&rng) % 3), maxit = 1 + (int)(rng_next(&rng) % 12), no_et = rng_next(&rng) % 6 == 0;
      const int8_t *lp[24];
      uint8_t *op[24];
      for (int b = 0; b < nblk; b++) {
        make_block(&rng, bg, z, ncol, kprime, 0, -1.0 + (rng_next(&rng) % 70) / 10.0, cw, bl[b]);
        if (rng_next(&rng) % 3 == 0) /* other scales: other AGC gains per block */
          for (int i = 2 * z; i < ncol * z; i++) { const int v = bl[b][i] * (int)(1 + rng_next(&rng) % 4) / 2; if (i < kprime || i >= K) bl[b][i] = (int8_t)(v > 127 ? 127 : v < -128 ? -128 : v); }
        if (tail && (b % 4 != 3)) memset(bl[b] + ncol * z - tail, 0, (size_t)tail); /* most blocks share the tail */
        if (rng_next(&rng) % 10 == 0) bl[b][rng_next(&rng) % (2 * z)] = 5; /* punctured not zero: no shortcuts */
        lp[b] = bl[b]; op[b] = bo[b];
        memset(bo[b], 0xA5, OUTSZ);
      }
      ldpc_dec_params_t p = {.bg = (uint8_t)bg, .z = (uint16_t)z, .rcode = (uint8_t)rc, .max_iter = (uint8_t)maxit,
                             .no_et = (uint8_t)no_et, .out_mode = (ldpc_out_t)mode, .impl = LDPC_IMPL_AUTO, .kprime = kprime,
                             .qm = (uint8_t)(rng_next(&rng) % 2 ? 0 : 6)};
      ldpc_dec_stats_t sb[24], s1;
      int rb[24];
      CHECK(ldpc_decode_batch(&p, nblk, lp, op, sb, rb) == 0, "batch returned an error");
      for (int b = 0; b < nblk; b++) {
        memset(so, 0xA5, OUTSZ);
        const int r1 = ldpc_decode(&p, bl[b], so, &s1);
        const int same = r1 == rb[b] && s1.iterations == sb[b].iterations && s1.syndrome_ok == sb[b].syndrome_ok && !memcmp(so, bo[b], OUTSZ);
        nbad += !same;
        if (!same && nbad < 5) printf("FAIL: batch BG%d Z%d R%d nblk %d b %d mode %d it %d no_et %d tail %d kp %d: ret %d/%d it %d/%d\n", bg, z, rc,
                                      nblk, b, mode, maxit, no_et, tail, kprime, r1, rb[b], s1.iterations, sb[b].iterations);
        nb_cases++;
      }
    }
    CHECK(nbad == 0, "batch != single: %d blocks", nbad);
    printf("[6] ldpc_decode_batch == ldpc_decode block by block (Z <= 48, 1-24 blocks, per-block scales, tails, fillers): %d blocks, %d mismatches\n", nb_cases, nbad);
  }
  printf("RESULT: %s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
  return fails != 0;
}
