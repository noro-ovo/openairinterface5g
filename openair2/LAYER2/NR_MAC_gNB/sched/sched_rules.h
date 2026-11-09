/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* Pure 3GPP rules of the scheduler: no global state, no MAC structure, each
 * function cites its clause. Times are in Tc (38.211 4.1), symbols are counted
 * from SFN 0 symbol 0 in the numerology of the channel. */

#ifndef SCHED_RULES_H_
#define SCHED_RULES_H_

#include <stdbool.h>
#include <stdint.h>

/// 38.211 4.1: kappa = Ts / Tc
#define RULES_KAPPA 64
/// 1 ms in Tc, Tc = 1 / (480000 x 4096) s
#define RULES_TC_PER_MS 1966080
/// 38.211 4.3.2: symbols per slot, normal cyclic prefix
#define RULES_SYMBOLS_PER_SLOT 14

/// 38.211 5.3.1: start of a symbol, normal cyclic prefix
int64_t rules_symbol_start_tc(int mu, int64_t symbol);
/// first symbol starting at or after t
int64_t rules_first_symbol_from(int mu, int64_t t_tc);

/// 38.214 Table 5.3-1 and 5.3-2: twice N1 (capability 2 has 4.5), -1 if not defined
int rules_n1_x2(int mu, bool cap2, bool dmrs_pos0, bool dmrs_l1_12);
/// 38.214 Table 6.4-1 and 6.4-2: twice N2 (capability 2 has 5.5), -1 if not defined
int rules_n2_x2(int mu, bool cap2);
/// N symbols of numerology mu given twice, (N)(2048+144)kappa 2^-mu Tc
int64_t rules_symbols_x2_tc(int mu, int n_x2);

/// 38.214 5.3: d1,1
int rules_d11(bool cap2, bool type_a, int start_symbol, int length, int pdcch_overlap, bool cset3_same_start);

typedef struct {
  int mu_pdcch, mu_pdsch, mu_ul;
  bool cap2; ///< processingType2Enabled for the PDSCH
  bool dmrs_pos0; ///< dmrs-AdditionalPosition pos0 in every configured DMRS-DownlinkConfig
  bool dmrs_l1_12; ///< additional DM-RS at l1 = 12 (N1,0 = 14)
  bool type_a; ///< PDSCH mapping type A
  int start_symbol, length; ///< PDSCH in its slot
  int pdcch_overlap; ///< symbols of the scheduling PDCCH that are also PDSCH symbols
  bool cset3_same_start; ///< 3-symbol CORESET starting with the PDSCH
  int n_rb; ///< PDSCH RBs: capability 2 at 30 kHz falls back to capability 1 above 136
  int d2, d3; ///< reported by the UE, 0 otherwise
  int64_t text_tc; ///< T_ext, shared spectrum in FR1, 0 otherwise
} rules_tproc1_t;

/// 38.214 5.3: Tproc,1, the numerology giving the largest value
int64_t rules_tproc1_tc(const rules_tproc1_t *in);

typedef struct {
  int mu_dl, mu_ul;
  bool cap2; ///< processingType2Enabled for the PUSCH
  bool first_symbol_dmrs_only; ///< d2,1 = 0
  int d2, d3, d4; ///< reported by the UE, 0 otherwise
  int64_t text_tc, tswitch_tc; ///< T_ext and T_switch, 0 if not applicable
  int64_t d22_tc; ///< BWP switching time, 0 without switch
} rules_tproc2_t;

/// 38.214 6.4: Tproc,2, the numerology giving the largest value
int64_t rules_tproc2_tc(const rules_tproc2_t *in);

/// 38.214 5.3 and 6.4: the first UL symbol (numerology mu_ul) whose CP, advanced by T_TA, starts
/// at least delay after end. ta_tc is (N_TA + N_TA,offset) Tc, 38.211 4.3.1
int64_t rules_earliest_ul_symbol(int mu_ul, int64_t end_tc, int64_t delay_tc, int64_t ta_tc);

/// 38.213 8.3: minimum time from the end of the RAR PDSCH to the Msg3 PUSCH, N_T,1 + N_T,2 + 0.5 ms;
/// mu is the smaller of the PDSCH and PUSCH numerologies
int64_t rules_msg3_delay_tc(int mu);
/// 38.213 8.4: minimum time from the end of the Msg4 PDSCH to its HARQ-ACK, N_T,1 + 0.5 ms
int64_t rules_msg4_harq_delay_tc(int mu);

/// MCS index tables: 38.214 Table 5.1.3.1-1 to -4, and 6.1.4.1-1, -2 with transform precoding
typedef enum {
  RULES_MCS_QAM64,
  RULES_MCS_QAM256,
  RULES_MCS_QAM64_LOWSE,
  RULES_MCS_QAM1024,
  RULES_MCS_TP_QAM64,
  RULES_MCS_TP_QAM64_LOWSE,
} rules_mcs_table_t;

/// modulation order and target code rate x 2048 of an MCS; q is 1 with tp-pi2BPSK, 2 otherwise
/// (transform precoding tables only); false for a reserved index
bool rules_mcs(rules_mcs_table_t table, int mcs, int q, int *qm, int *r_x2048);

/// 38.214 5.1.3.2 step 1: N_RE = min(156, 12 N_symb - N_DMRS - N_oh) n_PRB
int rules_tbs_n_re(int n_symb, int dmrs_re_per_prb, int x_overhead, int n_prb);
/// 38.214 5.1.3.2 steps 2 to 4 (6.1.4.2 for PUSCH): TBS in bits
int rules_tbs(int n_re, int r_x2048, int qm, int layers);

/// 38.212 7.2.2: LDPC base graph 2
bool rules_ldpc_bg2(int tbs, int r_x2048);
/// TBS plus TB CRC plus code block CRCs (38.212 7.2.1, 5.2.2)
int rules_coded_info_bits(int tbs, int r_x2048);
/// 38.214 5.1.3: effective code rate, info bits including CRC over the physical channel bits, at most 0.95.
/// n_re_data counts the REs really carrying the PDSCH (DM-RS, CSI-RS, PT-RS, SSB and rate matching removed)
bool rules_rate_ok(int tbs, int r_x2048, int qm, int layers, int n_re_data);

typedef struct {
  rules_mcs_table_t table;
  int q; ///< transform precoding tables
  int max_mcs;
  int n_symb, dmrs_re_per_prb, x_overhead, n_prb, layers; ///< TBS inputs
  int n_re_data; ///< REs carrying data, for the 0.95 limit
} rules_alloc_t;

/// the highest MCS up to max_mcs whose TB respects the 0.95 limit on this allocation, -1 if none;
/// its TBS in *tbs
int rules_max_mcs(const rules_alloc_t *a, int *tbs);

/// 38.214 5.1.2.1: SLIV of a start S and length L, -1 if 0 < L <= 14 - S does not hold
int rules_sliv(int start, int length);
/// 38.214 5.1.2.1: S and L of a SLIV, false if the value is not a SLIV
bool rules_sliv_decode(int sliv, int *start, int *length);
/// 38.214 Table 5.1.2.1-1: valid PDSCH S and L
bool rules_pdsch_sl_valid(bool type_a, bool extended_cp, int dmrs_typea_pos, int start, int length);
/// 38.214 Table 6.1.2.1-1: valid PUSCH S and L, repetition type A
bool rules_pusch_sl_valid(bool type_a, bool extended_cp, int start, int length);

/// 38.214 5.1.2.2.2 and 6.1.2.2.2: RIV of resource allocation type 1 in a BWP of n_bwp RBs, -1 if invalid
int rules_riv(int n_bwp, int rb_start, int n_rb);

/// 38.211 6.3.1.4: with transform precoding the PUSCH has 2^a 3^b 5^c RBs and one layer
bool rules_tp_alloc_valid(int n_rb, int layers);
/// the largest number of RBs at most n_rb that transform precoding allows, 0 if none
int rules_tp_max_rb(int n_rb);

/// 38.213 Table 10.1-2: monitored PDCCH candidates per slot and serving cell, -1 if not per slot
int rules_pdcch_max_candidates(int mu);
/// 38.213 Table 10.1-3: non-overlapped CCEs per slot and serving cell, -1 if not per slot
int rules_pdcch_max_cces(int mu);

#endif /* SCHED_RULES_H_ */
