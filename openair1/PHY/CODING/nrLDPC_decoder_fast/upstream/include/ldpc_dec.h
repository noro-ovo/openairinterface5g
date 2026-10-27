/* dec_std: 5G NR LDPC decoder (3GPP TS 38.212 Sec. 5.3.2), layered normalized/offset min-sum, int8.
 *
 * Input convention (identical to OAI's LDPCdecoder): int8 LLRs, positive = bit 0, one LLR per column bit of the
 * lifted base graph in natural order, i.e. llr[c*Z + j] for column c in 0..ncol-1 (ncol = 68/35/27 for BG1 R 1/3,
 * 2/3, 8/9 and 52/32/17 for BG2 R 1/5, 1/3, 2/3); the first 2Z LLRs (punctured columns) are 0, filler bits +127.
 */
#ifndef LDPC_DEC_H
#define LDPC_DEC_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  LDPC_IMPL_AUTO = 0, /* best available on this CPU */
  LDPC_IMPL_REF,      /* scalar reference of the algorithm (natural indexing, independent of the SIMD layout) */
  LDPC_IMPL_SCALAR,   /* portable C of the SIMD data layout (vector width 8, plain C) */
  LDPC_IMPL_NEON,
  LDPC_IMPL_SVE2,
  LDPC_IMPL_AVX2,
  LDPC_IMPL_AVX512,
  LDPC_IMPL_COUNT
} ldpc_impl_t;

typedef enum { LDPC_OUT_BITS = 0, LDPC_OUT_BYTES = 1, LDPC_OUT_LLR = 2 } ldpc_out_t;

typedef struct {
  uint8_t bg;        /* 1 or 2 */
  uint16_t z;        /* lifting size, Table 5.3.2-1 */
  uint8_t rcode;     /* BG1: 13, 23, 89; BG2: 15, 13, 23 (OAI's R codes) */
  uint8_t max_iter;  /* maximum number of layered iterations, >= 1 */
  uint8_t no_et;     /* 1: no early termination (always max_iter iterations), for timing */
  ldpc_out_t out_mode;
  ldpc_impl_t impl;
  /* min-sum magnitude correction m' = min(rmax, max(0, (m - (m >> nshift)) - beta)) (nshift <= 0: no scaling)
     and channel LLR scaling llr' = (llr + 2^(s-1)) >> s, s = in_shift (<= 0: none, max 3).
     All four 0: per-(BG, R) defaults of ldpc_default_alg(). rmax 0: LDPC_DEC_DEFAULT_RMAX. */
  int8_t nshift, beta, rmax, in_shift;
  /* automatic input gain (AGC): the LLRs are rescaled per block so that their mean magnitude over the transmitted,
     non-filler positions equals the per-(BG, R) target the parameters were tuned at (ldpc_agc_target()); the gain
     G / 2^sh (G 7 bits, gain 1/8 .. 8) is applied before in_shift, fillers [kprime, K) bypass it (+127).
     0: default (ldpc_agc_default()), 1: on, -1: off. kprime <= 0 means K (no fillers).
     The statistic counts the non-zero values below the int8 rails only (zeros, e.g. untransmitted positions, carry
     no scale information). */
  int8_t agc;
  /* modulation order of the code block's LLRs (AGC target, ldpc_agc_target()): 0 unknown, 1 BPSK-like (ldpctest
     channel, the calibration of the targets), 2, 4, 6, 8 (OAI PHY demapper output for QPSK .. 256QAM) */
  uint8_t qm;
  /* optional CRC callback (OAI semantics): called on the packed output (kprime bits) */
  int (*check_crc)(uint8_t *bytes, uint32_t nbits, uint8_t crc_type);
  int kprime;
  uint8_t crc_type;
  /* optional abort poll (checked before every iteration); non-zero = stop */
  int (*abort_poll)(void *ctx);
  void *abort_ctx;
} ldpc_dec_params_t;

typedef struct {
  int iterations;   /* layered iterations executed */
  int syndrome_ok;  /* hard decision is a codeword */
  int crc_ok;       /* -1 no CRC check, else result */
  int aborted;
  int core_stop;    /* CRC mode: decoding stopped after the core rows of the last iteration (counted in iterations) */
  int rows;         /* rows (layers) processed: rows beyond the last non-zero LLR are exact no-ops and skipped */
} ldpc_dec_stats_t;

/* Decodes one code block. out: LDPC_OUT_BITS: ceil(Kb*Z/8) bytes, MSB first (byte b bit 7 = bit 8b),
   Kb*Z = 22Z (BG1) or 10Z (BG2) systematic bits incl. filler positions; LDPC_OUT_BYTES: Kb*Z bytes 0/1;
   LDPC_OUT_LLR: ncol*Z a-posteriori int8 LLRs in input order.
   Returns 1 on success (syndrome ok, and CRC ok if a CRC callback is given; or CRC ok at the last iteration),
   0 on failure, -1 on invalid parameters. */
int ldpc_decode(const ldpc_dec_params_t *p, const int8_t *llr, uint8_t *out, ldpc_dec_stats_t *st);

/* name of an implementation; whether it can run on this CPU */
const char *ldpc_impl_name(ldpc_impl_t impl);
int ldpc_impl_available(ldpc_impl_t impl);
ldpc_impl_t ldpc_impl_best(void);

int ldpc_ncol(int bg, int rcode); /* 0 if invalid */

/* testing/benchmarking: force the number of vectors per step (1, 2, 4; 0 = automatic choice) */
void ldpc_dec_force_unroll(int U);

/* |messages| <= 47 keeps a saturated a-posteriori LLR stable: 127 - 47 - beta >= 47 (README) */
#define LDPC_DEC_DEFAULT_RMAX 47
/* AGC target mean |LLR| (input units, before in_shift, over the non-zero values below the rails) of a (BG, R code)
   and modulation order qm (see ldpc_dec_params_t.qm); whether the AGC is on by default */
int ldpc_agc_target(int bg, int rcode, int qm);
/* the same for a block of code rate rate_q16 (Q16, K' over the transmitted non-zero span): blocks that leave part of
   the class's graph untransmitted get a larger target (percent slope per class, see README) */
int ldpc_agc_target_rate(int bg, int rcode, int qm, int rate_q16);
int ldpc_agc_default(void);
/* 1 if a call with a CRC callback (OAI's PHY) and default parameters uses NMS 7/8 + offset 1 (scale-tolerant on
   demapper LLRs) instead of the per-(BG, R) offset min-sum; LDPC_DEC_CRCALG=0 turns it off */
int ldpc_crc_alg_default(void);
/* default (nshift, beta, rmax, in_shift) for a (BG, R code) */
void ldpc_default_alg(int bg, int rcode, int *nshift, int *beta, int *rmax, int *in_shift);

#ifdef __cplusplus
}
#endif
#endif
