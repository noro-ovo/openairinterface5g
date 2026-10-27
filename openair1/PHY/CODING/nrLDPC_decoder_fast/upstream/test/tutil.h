/* test utilities: spec encoder (GF(2) elimination of the core parity block, row-by-row extension parity),
 * independent syndrome, BPSK/AWGN channel with ldpctest's LLR quantisation, RNG. */
#ifndef TUTIL_H
#define TUTIL_H
#include <stdint.h>
#include <math.h>

typedef struct { uint64_t s[4]; } rng_t;
static inline uint64_t rng_next(rng_t *r)
{
  uint64_t *s = r->s, x = s[1] * 5, res = ((x << 7) | (x >> 57)) * 9, t = s[1] << 17;
  s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = (s[3] << 45) | (s[3] >> 19);
  return res;
}
void rng_seed(rng_t *r, uint64_t seed);
double rng_gauss(rng_t *r);

int t_ncol_full(int bg);              /* 68 / 52 */
int t_kb(int bg);                     /* 22 / 10 */
/* info: K = kb*z bits (one per byte, fillers must already be 0); cw: ncol_full*z bits (one per byte), full
   codeword of 38.212 5.3.2 incl. the 2Z punctured systematic bits; returns 0 on success */
int t_encode(int bg, int z, const uint8_t *info, uint8_t *cw);
/* number of unsatisfied checks of rows 0..nrow-1 (nrow = 46/42 for all) on cw (ncol*z bits, one per byte),
   directly from the spec tables (independent of the decoder plan) */
int t_syndrome(int bg, int z, int nrow, const uint8_t *cw);
/* LLRs for the decoder: llr[0..2z) = 0, filler positions [kprime, K) = +127, others BPSK + AWGN with
   Eb/N0 ebn0 (rate kb/(ncol-2)), ldpctest quantisation floor(y/(sigma/16)) saturated to int8 */
void t_channel(rng_t *r, int bg, int z, int ncol, int kprime, const uint8_t *cw, double ebn0, int8_t *llr);
/* scaled channel (from the dec_std follow-up): mode 0: floor(f * y / (sigma/16)) (f = 1: ldpctest); mode 1:
   round(f * 5 * 2y/sigma^2) (OCUDU-like demapper scale 5 L); saturation: sat = 127 -> [-128, 127], else [-sat, sat] */
void t_channel_s(rng_t *r, int bg, int z, int ncol, int kprime, const uint8_t *cw, double ebn0, int mode, double f, int sat,
                 int8_t *llr);
#endif
