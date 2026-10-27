/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*! \file PHY/CODING/nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_llr_prescale.h
 * \brief Modulation-order (Qm) dependent scaling of the int16 channel LLRs before the saturating int16 -> int8
 * conversion that feeds the LDPC decoder.
 *
 * The PHY produces int16 LLRs whose magnitude depends on the modulation order: 16QAM and 64QAM LLRs are 3-6x larger
 * than the 8-bit decoders expect and 20-50 % of them end up at the int8 rails after the pack, which costs OAI's own
 * decoder up to 0.45 dB (1 layer 16QAM) and makes offset/normalised min-sum decoders fail. The rule is a per-Qm gain
 * applied in int16 *before* the saturating pack (so it also un-clips), see nrLDPC_llr_prescale_parse().
 *
 * Gains are applied with symmetric rounding (round half away from zero):
 *   gain < 1        : y = sign(x) * ((|x| * q15 + 2^14) >> 15), q15 = round(gain * 2^15)  (exact /2^s for gain 2^-s)
 *   gain = 2, 4, 8  : y = saturate16(x * gain)
 *   gain = 1        : the unmodified pack (bit-exact legacy behaviour)
 * Allowed gains: [1/16, 1) or 1, 2, 4, 8. With gain >= 1/16, the filler LLRs (int16 0x7f7f) still saturate to +127
 * and the punctured/untransmitted zeros stay 0, so the filler/puncturing handling is unchanged.
 */

#ifndef __NRLDPC_LLR_PRESCALE__H__
#define __NRLDPC_LLR_PRESCALE__H__

#include <stdint.h>

/// default rule (measured on nr_ulsim, OAI develop 9313cbea + PR 356): QPSK x1, 16QAM x1/2, 64QAM x1/2, 256QAM x1
#define NR_LDPC_LLR_PRESCALE_DEFAULT "1,0.5,0.5,1"
/// maximum modulation order handled by the table (index = Qm, 0..8)
#define NR_LDPC_LLR_PRESCALE_MAX_QM 8

typedef enum { NR_LDPC_LLR_PRESCALE_NONE = 0, NR_LDPC_LLR_PRESCALE_FRAC, NR_LDPC_LLR_PRESCALE_LSHIFT } nr_ldpc_llr_prescale_mode_t;

typedef struct {
  uint8_t mode; ///< nr_ldpc_llr_prescale_mode_t
  uint8_t lshift; ///< mode LSHIFT: gain = 2^lshift
  int16_t q15; ///< mode FRAC: gain in Q15
  uint16_t q8; ///< gain in Q8 (reported to the decoder in t_nrLDPC_dec_params.llr_prescale_q8)
} nr_ldpc_llr_prescale_t;

/// per-Qm table, index = Qm (0..8); entries for Qm not in {2,4,6,8} are always x1
typedef struct {
  nr_ldpc_llr_prescale_t qm[NR_LDPC_LLR_PRESCALE_MAX_QM + 1];
} nr_ldpc_llr_prescale_rule_t;

/**
 * \brief parse a rule string into a table
 * \param str "g2,g4,g6,g8" (gains for QPSK, 16QAM, 64QAM, 256QAM, e.g. "1,0.5,0.5,1"), "default" (= NR_LDPC_LLR_PRESCALE_DEFAULT),
 *            or "off" / "none" / "0" (all x1 = legacy pack, bit-exact)
 * \param rule output table
 * \return 0 on success, -1 on a syntax error or a gain outside the allowed set (rule then holds all x1)
 */
int nrLDPC_llr_prescale_parse(const char *str, nr_ldpc_llr_prescale_rule_t *rule);

/// print the rule as "QPSK x.. 16QAM x.. 64QAM x.. 256QAM x.." into buf
void nrLDPC_llr_prescale_str(const nr_ldpc_llr_prescale_rule_t *rule, char *buf, int len);

/**
 * \brief saturating int16 -> int8 conversion with the gain of one modulation order
 * Converts nvec * 16 values (z and l must be 16-byte aligned; z must hold nvec*16 readable values).
 * With ps == NULL or mode NONE this is exactly the legacy loop (simde_mm_packs_epi16 per 16 values).
 */
void nrLDPC_llr_pack_int8(const int16_t *z, int8_t *l, int nvec, const nr_ldpc_llr_prescale_t *ps);

#endif
