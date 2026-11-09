/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* Conformance checker for the output of a gNB MAC scheduler. It reads FAPI
 * messages and a small description of the cell only, and shares no code with
 * any scheduler, so it applies to the legacy one and to its replacement. */

#ifndef SCHED_CHECK_H_
#define SCHED_CHECK_H_

#include "common/openairinterface5g_limits.h"
#include <stdbool.h>
#include <stdint.h>

#include "nfapi_nr_interface_scf.h"

#define SCHED_CHECK_MAX_SLOTS_PERIOD 320
#define SCHED_CHECK_MAX_SSB 64
#define SCHED_CHECK_MAX_ITEMS 64

typedef enum {
  SCHED_RULE_PDCCH_ON_UL,
  SCHED_RULE_PDSCH_ON_UL,
  SCHED_RULE_PUSCH_ON_DL,
  SCHED_RULE_PUCCH_ON_DL,
  SCHED_RULE_SSB_ON_UL,
  SCHED_RULE_PDSCH_ON_SSB,
  SCHED_RULE_PDSCH_RATE,
  SCHED_RULE_PDSCH_OVERLAP,
  SCHED_RULE_PUSCH_OVERLAP,
  SCHED_RULE_TP_PRB,
  SCHED_RULE_TP_LAYERS,
  SCHED_RULE_CCE_OVERLAP,
  SCHED_RULE_DCI_PER_RNTI,
  SCHED_RULE_HARQ_TPROC1,
  SCHED_RULE_PUSCH_TPROC2,
  SCHED_RULE_MSG3_DELAY,
  SCHED_RULE_MSG4_HARQ,
  SCHED_RULE_MSG4_TIMER,
  SCHED_RULE_RA_UNANSWERED,
  SCHED_RULE_UL_ON_PRACH,
  SCHED_RULE_PUSCH_CAPACITY,
  SCHED_RULE_COUNT
} sched_rule_t;

typedef enum {
  SCHED_SYM_DL,
  SCHED_SYM_UL,
  SCHED_SYM_FLEX,
} sched_sym_t;

typedef struct {
  /// length of the TDD pattern in slots, 0 for FDD
  int slots_per_period;
  /// direction of each symbol, for TDD
  uint8_t sym[SCHED_CHECK_MAX_SLOTS_PERIOD][NR_SYMBOLS_PER_SLOT];
  /// first symbol of each SSB in its half frame, as in 38.213 4.1
  int16_t ssb_start_symbol[SCHED_CHECK_MAX_SSB];
  /// SSB subcarrier spacing as a numerology
  int ssb_mu;
  bool fr2;
  /// PRB overlaps are legal between beams, and FAPI does not say which is which
  int num_beams;
  /// slots in a frame, to count time across slots
  int slots_per_frame;
  /// PDSCH decoding time N1 and PUSCH preparation time N2, in symbols, see sched_check_set_proc_time()
  int n1;
  int n2;
  /// numerology of the active BWPs, and N_T,1 of 38.213 8.3: N1 with additional DM-RS
  int mu;
  int nt1;
  /// ra-ContentionResolutionTimer in ms, 0 to skip the rule
  int ra_cr_timer_ms;
  /// UL timing advance N_TA,offset + N_TA in symbols, rounded up: Tproc,1 and Tproc,2 include it (38.214 5.3, 6.4)
  int ta_symbols;
  /// longest of ra-ResponseWindow and msgB-ResponseWindow, in slots
  int ra_window_slots;
  /// PRACH occasions: first PRB (absolute), PRBs of all FDM occasions, symbols of one occasion; 0 PRBs if unknown
  int prach_first_prb, prach_n_prb, prach_dur;
  /// PUSCHs the PHY receives in one slot, 0 when unknown
  int max_pusch_per_slot;
} sched_check_cfg_t;

typedef struct {
  sched_rule_t rule;
  int frame;
  int slot;
  int rnti;
} sched_check_item_t;

typedef enum {
  SCHED_PDU_SSB,
  SCHED_PDU_PDCCH,
  SCHED_PDU_PDSCH,
  SCHED_PDU_PUSCH,
  SCHED_PDU_PUCCH,
  SCHED_PDU_COUNT
} sched_pdu_t;

typedef struct {
  /// PDUs inspected, to tell a clean run from a blind checker
  uint64_t checked[SCHED_PDU_COUNT];
  /// initial PDSCH by number of symbols, and highest effective code rate seen, in 1/1000
  uint64_t pdsch_symbols[NR_SYMBOLS_PER_SLOT + 1];
  int max_rate_permille;
  /// total per rule, never saturates
  uint64_t count[SCHED_RULE_COUNT];
  /// first findings, for the log
  sched_check_item_t item[SCHED_CHECK_MAX_ITEMS];
  int num_items;
} sched_check_report_t;

/* connected UEs plus UEs in random access */
#define SCHED_CHECK_MAX_UES (MAX_MOBILES_PER_GNB + 32)
#define SCHED_CHECK_QUEUE 32

/* 38.214 5.1 and 6.1 forbid out of order HARQ-ACK and PUSCH, so the n-th DCI
 * of a UE matches its n-th PUSCH and the n-th PDSCH its n-th HARQ-ACK bit */
typedef struct {
  int rnti;
  int64_t last_slot;
  /// earliest legal first symbol of the HARQ-ACK of each pending PDSCH, in symbols since the start
  int64_t harq_earliest[SCHED_CHECK_QUEUE];
  /// for Msg4 only, the earliest symbol after N_T,1 + 0.5 ms (38.213 8.4), else -1
  int64_t harq_earliest_msg4[SCHED_CHECK_QUEUE];
  int harq_head;
  int harq_len;
  /// symbol after the PDCCH of each pending UL DCI, in symbols since the start
  int64_t dci_end[SCHED_CHECK_QUEUE];
  int dci_head;
  int dci_len;
  /// symbol after the RAR that gave this TC-RNTI, until its Msg3, else -1
  int64_t rar_end;
  /// between RAR and Msg4: Msg3 (re)transmissions restart the contention resolution timer
  bool in_ra;
  /// symbol after the last Msg3 (re)transmission, else -1
  int64_t msg3_end;
} sched_check_ue_t;

#define SCHED_CHECK_MAX_PRACH 16

/// a PRACH slot, to tell RA-RNTIs from other RNTIs
typedef struct {
  int slot;
  int first_ra;
  int64_t until;
} sched_check_prach_t;

#define SCHED_CHECK_MAX_PREAMBLES 16

/// a detected preamble waiting for its RAR or MsgB
typedef struct {
  /// last slot of the response window, -1 when free
  int64_t deadline;
  int rapid;
  int s_id;
  int f_id;
  int slot;
} sched_check_preamble_t;

typedef struct {
  bool started;
  int64_t slot;
  sched_check_prach_t prach[SCHED_CHECK_MAX_PRACH];
  int prach_next;
  int last_frame;
  int last_slot;
  sched_check_ue_t ue[SCHED_CHECK_MAX_UES];
  /// HARQ-ACK bits without a pending PDSCH, PUSCH without a pending DCI (Msg3), entries lost to full queues
  uint64_t harq_unmatched;
  uint64_t pusch_unmatched;
  uint64_t dropped;
  /// smallest margin over Tproc,1 and Tproc,2 seen, in symbols
  int min_harq_margin;
  int min_pusch_margin;
  /// random access: RARs, Msg3 and Msg4 seen, and the smallest margins over their minimum delays, in symbols
  uint64_t rars;
  uint64_t msg3s;
  uint64_t msg4s;
  int min_msg3_margin;
  int min_msg4_harq_margin;
  int min_msg4_timer_margin;
  /// detected preambles, and those answered by a RAR with their RAPID or by a MsgB
  sched_check_preamble_t preamble[SCHED_CHECK_MAX_PREAMBLES];
  uint64_t preambles;
  uint64_t preambles_answered;
} sched_check_timing_t;

/// audit identifier of a rule
const char *sched_rule_id(sched_rule_t rule);
/// clause a rule comes from, "-" for a physical conflict without a clause
const char *sched_rule_clause(sched_rule_t rule);

/// check the DL TTI and UL DCI requests of a slot, returns the number of findings
int sched_check_dl_slot(const sched_check_cfg_t *cfg,
                        const nfapi_nr_dl_tti_request_t *dl,
                        const nfapi_nr_ul_dci_request_t *ul_dci,
                        sched_check_report_t *report);
/// check the UL TTI request of a slot, returns the number of findings
int sched_check_ul_slot(const sched_check_cfg_t *cfg, const nfapi_nr_ul_tti_request_t *ul, sched_check_report_t *report);

/// N1 and N2 of UE processing capability 1, 38.214 Tables 5.3-1 and 6.4-1
void sched_check_set_proc_time(sched_check_cfg_t *cfg, int mu, bool dmrs_pos0);

/// check the time between PDSCH and HARQ-ACK, UL DCI and PUSCH, RAR and Msg3, Msg3 and Msg4; call once per slot, in order;
/// the MAC PDUs in tx identify RAR and Msg4
int sched_check_timing_slot(const sched_check_cfg_t *cfg,
                            sched_check_timing_t *timing,
                            const nfapi_nr_dl_tti_request_t *dl,
                            const nfapi_nr_ul_dci_request_t *ul_dci,
                            const nfapi_nr_ul_tti_request_t *ul,
                            const nfapi_nr_tx_data_request_t *tx,
                            sched_check_report_t *report);

/// a preamble detected in the PRACH occasion at frame.slot, symbol s_id, frequency f_id; its RAR or MsgB is
/// due within the response window (38.321 5.1.4, 5.1.4a); call between sched_check_timing_slot() calls
void sched_check_preamble(const sched_check_cfg_t *cfg, sched_check_timing_t *timing, int frame, int slot, int rapid, int s_id, int f_id);

#endif /* SCHED_CHECK_H_ */
