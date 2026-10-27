/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*! \file PHY/CODING/nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_decoder_select.c
 * \brief LDPC decoder selection (see nrLDPC_decoder_select.h).
 *
 * Compiled twice:
 *  - with NR_LDPC_HAVE_FAST_DECODER (libldpc.so): exports LDPCinit / LDPCshutdown / LDPCdecoder, which forward to the
 *    fast decoder (nrLDPC_fast_*, nrLDPC_decoder_fast/) or to OAI's decoder (nrLDPC_oai_*,
 *    nrLDPC_decoder_oai_renamed.c), plus the optional statistics;
 *  - without (libldpc_oaidec.so, libldpc_orig.so): OAI's decoder is linked under its own names; this file only answers
 *    "oai" and warns if the fast decoder was requested.
 */

#include "nrLDPC_decoder_select.h"
#include "nrLDPC_llr_prescale.h"
#include "PHY/CODING/nrLDPC_decoder/nrLDPC_types.h"
#include "PHY/CODING/coding_defs.h"
#include "common/utils/LOG/log.h"
#include "common/config/config_userapi.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

#ifndef NR_LDPC_DECODER_DEFAULT
#define NR_LDPC_DECODER_DEFAULT "fast"
#endif

static nrLDPC_decoder_id_t dec_selected = NR_LDPC_DECODER_OAI;
static pthread_once_t dec_once = PTHREAD_ONCE_INIT;

const char *nrLDPC_decoder_name(nrLDPC_decoder_id_t id)
{
  return id == NR_LDPC_DECODER_FAST ? "fast" : "oai";
}

static int parse_decoder(const char *s, nrLDPC_decoder_id_t *id)
{
  if (!strcasecmp(s, "fast") || !strcasecmp(s, "new"))
    *id = NR_LDPC_DECODER_FAST;
  else if (!strcasecmp(s, "oai") || !strcasecmp(s, "orig") || !strcasecmp(s, "legacy"))
    *id = NR_LDPC_DECODER_OAI;
  else
    return -1;
  return 0;
}

#ifdef NR_LDPC_HAVE_FAST_DECODER
#include "ldpc_dec.h" /* fast decoder's own API: ldpc_impl_best(), ldpc_impl_name() (hidden symbols of this library) */

int32_t nrLDPC_fast_LDPCinit(void);
int32_t nrLDPC_fast_LDPCshutdown(void);
int32_t nrLDPC_fast_LDPCdecoder(t_nrLDPC_dec_params *p, int8_t *llr, uint8_t *out, t_nrLDPC_time_stats *ts, decode_abort_t *ab);
int32_t nrLDPC_oai_LDPCinit(void);
int32_t nrLDPC_oai_LDPCshutdown(void);
int32_t nrLDPC_oai_LDPCdecoder(t_nrLDPC_dec_params *p, int8_t *llr, uint8_t *out, t_nrLDPC_time_stats *ts, decode_abort_t *ab);
static void stats_init(void);
#endif

static void select_init(void)
{
  char *cfg = NULL;
  const char *src = "build default";
  if (config_get_if() != NULL) {
    paramdef_t params[] = {
        {"decoder",
         "LDPC decoder of the segment coding library: \"fast\" (layered min-sum, Qm-aware AGC; default) or \"oai\" (OAI's original decoder)\n",
         0,
         .strptr = &cfg,
         .defstrval = NULL,
         TYPE_STRING,
         0,
         NULL},
    };
    config_get(config_get_if(), params, sizeofArray(params), "nrLDPC_coding_segment");
  }
  const char *env = getenv("OAI_LDPC_DECODER");
  const char *str = NR_LDPC_DECODER_DEFAULT;
  int requested = 1;
  if (env && *env) {
    str = env;
    src = "env OAI_LDPC_DECODER";
  } else if (cfg && *cfg && strcasecmp(cfg, "default")) {
    str = cfg;
    src = "config nrLDPC_coding_segment.decoder";
  } else {
    requested = 0;
  }
  nrLDPC_decoder_id_t id;
  AssertFatal(parse_decoder(str, &id) == 0, "invalid LDPC decoder \"%s\" (%s): expected \"fast\" or \"oai\"\n", str, src);
#ifdef NR_LDPC_HAVE_FAST_DECODER
  (void)requested;
  dec_selected = id;
  if (id == NR_LDPC_DECODER_FAST)
    LOG_I(PHY, "LDPC decoder: fast (%s; implementation %s)\n", src, ldpc_impl_name(ldpc_impl_best()));
  else
    LOG_I(PHY, "LDPC decoder: oai (%s)\n", src);
  stats_init();
#else
  dec_selected = NR_LDPC_DECODER_OAI;
  if (id != NR_LDPC_DECODER_OAI && requested)
    LOG_W(PHY, "LDPC decoder \"%s\" requested (%s), but this library contains OAI's decoder only: using oai\n", str, src);
#endif
}

nrLDPC_decoder_id_t nrLDPC_decoder_selected(void)
{
  pthread_once(&dec_once, select_init);
  return dec_selected;
}

const char *nrLDPC_decoder_llr_prescale_default(void)
{
  return nrLDPC_decoder_selected() == NR_LDPC_DECODER_FAST ? NR_LDPC_LLR_PRESCALE_DEFAULT_FAST : NR_LDPC_LLR_PRESCALE_DEFAULT;
}

#ifdef NR_LDPC_HAVE_FAST_DECODER

/* ------------------------------------------------------------------------------------------------------------------
 * optional statistics (OAI_LDPC_DECODER_STATS / OAI_LDPC_DECODER_TRACE)
 * ------------------------------------------------------------------------------------------------------------------ */
#define ST_MAXKEY 64
#define ST_TBINS 4096 /* decode time histogram, 0.5 us bins up to 2 ms */
typedef struct {
  int used;
  uint8_t dec, Qm, nl, BG, R;
  uint64_t calls, ok, iters, it_max, t_ns, t_max;
  uint32_t th[ST_TBINS + 1];
  uint64_t ith[64];
} st_key_t;
static struct {
  int on;
  const char *path;
  FILE *trace;
  pthread_mutex_t mtx;
  uint64_t calls, digest;
  st_key_t k[ST_MAXKEY];
} S = {.mtx = PTHREAD_MUTEX_INITIALIZER};

static uint64_t fnv(uint64_t h, const void *d, size_t n)
{
  const uint8_t *b = d;
  for (size_t i = 0; i < n; i++) {
    h ^= b[i];
    h *= 0x100000001b3ull;
  }
  return h;
}

static int ncol_of(int BG, int R)
{
  if (BG == 1)
    return R == 13 ? 68 : R == 23 ? 35 : R == 89 ? 27 : 0;
  if (BG == 2)
    return R == 15 ? 52 : R == 13 ? 32 : R == 23 ? 17 : 0;
  return 0;
}

static double tpct(const st_key_t *k, double q)
{
  uint64_t n = 0, lim = (uint64_t)(q * k->calls);
  for (int i = 0; i <= ST_TBINS; i++) {
    n += k->th[i];
    if (n > lim)
      return (i + 0.5) * 0.5;
  }
  return ST_TBINS * 0.5;
}

static void stats_write(void)
{
  if (!S.on)
    return;
  pthread_mutex_lock(&S.mtx);
  if (S.trace)
    fclose(S.trace);
  S.trace = NULL;
  FILE *f = S.calls ? fopen(S.path, "a") : NULL;
  if (f) {
    for (int j = 0; j < ST_MAXKEY && S.k[j].used; j++) {
      const st_key_t *k = &S.k[j];
      fprintf(f,
              "dec=%s Qm=%d nl=%d BG=%d R=%d calls=%llu ok=%llu it_mean=%.3f it_max=%llu t_mean_us=%.3f t_p50_us=%.2f t_p90_us=%.2f "
              "t_p99_us=%.2f t_max_us=%.2f it_hist=",
              nrLDPC_decoder_name(k->dec), k->Qm, k->nl, k->BG, k->R, (unsigned long long)k->calls, (unsigned long long)k->ok,
              (double)k->iters / k->calls, (unsigned long long)k->it_max, k->t_ns / 1e3 / k->calls, tpct(k, 0.5), tpct(k, 0.9),
              tpct(k, 0.99), k->t_max / 1e3);
      for (int i = 0; i < 64; i++)
        if (k->ith[i])
          fprintf(f, "%d:%llu,", i, (unsigned long long)k->ith[i]);
      fprintf(f, "\n");
    }
    fprintf(f, "digest calls=%llu digest=%016llx\n", (unsigned long long)S.calls, (unsigned long long)S.digest);
    fclose(f);
  }
  S.on = 0;
  pthread_mutex_unlock(&S.mtx);
}

static void stats_init(void)
{
  const char *p = getenv("OAI_LDPC_DECODER_STATS");
  const char *t = getenv("OAI_LDPC_DECODER_TRACE");
  if ((!p || !*p) && (!t || !*t))
    return;
  S.path = p && *p ? p : "/dev/null";
  S.trace = t && *t ? fopen(t, "w") : NULL;
  S.on = 1;
  atexit(stats_write);
}

static void stats_record(nrLDPC_decoder_id_t dec,
                         const t_nrLDPC_dec_params *p,
                         const int8_t *llr,
                         const uint8_t *out,
                         int32_t ret,
                         uint64_t dt)
{
  const int ok = ret < p->numMaxIter;
  /* iterations executed: fast = layered iterations (a core-row stop counts as one); oai = flooding iterations incl. the
     one before its loop (CRC mode: the CRC is first checked in the second loop pass) */
  int it;
  if (dec == NR_LDPC_DECODER_FAST)
    it = ok ? ret + 1 : p->numMaxIter;
  else
    it = ok ? ret + (p->check_crc ? 2 : 1) : p->numMaxIter + 1;
  /* digest of the call, same definition as the llrw wrapper used for the base-branch comparisons */
  const int n_llr = ncol_of(p->BG, p->R) * p->Z;
  uint64_t hh = 0xcbf29ce484222325ull;
  const int32_t hdr[6] = {p->BG, p->Z, p->R, p->Kprime, p->numMaxIter, ret};
  hh = fnv(hh, hdr, sizeof hdr);
  if (n_llr)
    hh = fnv(hh, llr, n_llr);
  if (p->outMode == nrLDPC_outMode_BIT)
    hh = fnv(hh, out, (size_t)(p->Kprime >> 3));
  pthread_mutex_lock(&S.mtx);
  if (!S.on) {
    pthread_mutex_unlock(&S.mtx);
    return;
  }
  S.calls++;
  S.digest += hh;
  if (S.trace)
    fputc(ok ? '1' : '0', S.trace);
  int j;
  for (j = 0; j < ST_MAXKEY; j++) {
    st_key_t *k = &S.k[j];
    if (!k->used) {
      k->used = 1;
      k->dec = dec;
      k->Qm = p->Qm;
      k->nl = p->nb_layers;
      k->BG = p->BG;
      k->R = p->R;
      break;
    }
    if (k->dec == dec && k->Qm == p->Qm && k->nl == p->nb_layers && k->BG == p->BG && k->R == p->R)
      break;
  }
  if (j < ST_MAXKEY) {
    st_key_t *k = &S.k[j];
    k->calls++;
    k->ok += ok;
    k->iters += it;
    if ((uint64_t)it > k->it_max)
      k->it_max = it;
    k->ith[it < 63 ? it : 63]++;
    k->t_ns += dt;
    if (dt > k->t_max)
      k->t_max = dt;
    const uint64_t b = dt / 500;
    k->th[b < ST_TBINS ? b : ST_TBINS]++;
  }
  pthread_mutex_unlock(&S.mtx);
}

static inline uint64_t now_ns(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

/* ------------------------------------------------------------------------------------------------------------------
 * exported decoder interface of libldpc.so
 * ------------------------------------------------------------------------------------------------------------------ */
int32_t LDPCinit(void)
{
  nrLDPC_decoder_selected();
  const int32_t a = nrLDPC_oai_LDPCinit();
  const int32_t b = nrLDPC_fast_LDPCinit();
  return a ? a : b;
}

int32_t LDPCshutdown(void)
{
  const int32_t a = nrLDPC_oai_LDPCshutdown();
  const int32_t b = nrLDPC_fast_LDPCshutdown();
  return a ? a : b;
}

int32_t LDPCdecoder(t_nrLDPC_dec_params *p_decParams, int8_t *p_llr, uint8_t *p_out, t_nrLDPC_time_stats *p_profiler, decode_abort_t *ab)
{
  const nrLDPC_decoder_id_t dec = nrLDPC_decoder_selected();
  if (__builtin_expect(!S.on, 1))
    return dec == NR_LDPC_DECODER_FAST ? nrLDPC_fast_LDPCdecoder(p_decParams, p_llr, p_out, p_profiler, ab)
                                       : nrLDPC_oai_LDPCdecoder(p_decParams, p_llr, p_out, p_profiler, ab);
  const uint64_t t0 = now_ns();
  const int32_t ret = dec == NR_LDPC_DECODER_FAST ? nrLDPC_fast_LDPCdecoder(p_decParams, p_llr, p_out, p_profiler, ab)
                                                  : nrLDPC_oai_LDPCdecoder(p_decParams, p_llr, p_out, p_profiler, ab);
  const uint64_t dt = now_ns() - t0;
  stats_record(dec, p_decParams, p_llr, p_out, ret, dt);
  return ret;
}

#endif /* NR_LDPC_HAVE_FAST_DECODER */
