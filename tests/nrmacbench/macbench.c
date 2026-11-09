/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 *
 * nr_macbench: gNB MAC scheduler bench without PHY
 *
 * Runs the real DL and UL schedulers for N connected UEs. The PHY is replaced
 * by a FAPI loopback: each PUSCH of the slot comes back as a CRC and an
 * RX_DATA carrying a short BSR, each PUCCH as a UCI with ACK/NACK drawn from
 * the DL BLER and an SR when the UE has UL data. With OAI_SCHED_CHECK set
 * (the default here), the conformance checker runs on every slot.
 *
 * Usage:
 *   nr_macbench -u <num_ues> -n <num_slots> -B <dl_bler> -b <ul_bler>
 *               -D <dl_bytes> -U <ul_bytes> -R <N_RB> -m <numerology> -F
 *               -C <bytes>: every UE gets an RRCReconfiguration of <bytes> on SRB1
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common/utils/assertions.h"
#include "common/utils/nr/nr_common.h"
#include "executables/nr-uesoftmodem.h"
#include "executables/softmodem-common.h"
#include "LAYER2/NR_MAC_gNB/gNB_scheduler_dlsch_default_policies.h"
#include "LAYER2/NR_MAC_gNB/mac_proto.h"
#include "LAYER2/NR_MAC_gNB/mac_rrc_dl_handler.h"
#include "LAYER2/NR_MAC_gNB/nr_mac_gNB.h"
#include "LAYER2/NR_MAC_gNB/nr_radio_config.h"
#include "LAYER2/NR_MAC_gNB/sched/sched_check_oai.h"
#include "LAYER2/NR_MAC_gNB/sched/sched_dl.h"
#include "LAYER2/NR_MAC_gNB/sched/sched_ul.h"
#include "LAYER2/NR_MAC_COMMON/nr_mac.h"
#include "LAYER2/NR_MAC_COMMON/nr_mac_common.h"
#include "NR_CellGroupConfig.h"
#include "NR_ReconfigurationWithSync.h"
#include "NR_RLC-BearerConfig.h"
#include "PHY/defs_gNB.h"
#include "common/utils/oai_asn1.h"
#include "common/config/config_load_configmodule.h"
#include "common/ngran_types.h"
#include "common/ran_context.h"
#include "e1ap_messages_types.h"
#include "nfapi_nr_interface_scf.h"
#include "openair1/SIMULATION/NR_PHY/nr_unitary_defs.h"
#include "openair1/SIMULATION/TOOLS/sim.h"
#include "openair2/LAYER2/nr_rlc/nr_rlc_oai_api.h"

/* ── Globals required by OAI framework ── */
PHY_VARS_gNB *gNB;
RAN_CONTEXT_t RC;
int64_t uplink_frequency_offset[MAX_NUM_CCs][4];
char *uecap_file;
uint64_t downlink_frequency[MAX_NUM_CCs][4];
THREAD_STRUCT thread_struct;
nfapi_ue_release_request_body_t release_rntis;
instance_t DUuniqInstance = 0;
instance_t CUuniqInstance = 0;
unsigned int NTN_UE_Koffset = 0;
configmodule_interface_t *uniqCfg = NULL;
void *PHY_vars_UE_g = NULL;

void signal_rrc_msg(void) { abort(); }
void signal_rrc_state_changed_to(void) { abort(); }
void signal_ue_id(void) { abort(); }
void e1_bearer_context_setup(const e1ap_bearer_setup_req_t *req) { abort(); }
void e1_bearer_context_modif(const e1ap_bearer_mod_req_t *req) { abort(); }
struct e1ap_bearer_mod_confirm_s;
void e1_bearer_context_mod_confirm(const struct e1ap_bearer_mod_confirm_s *conf) { abort(); }
struct e1ap_bearer_release_cmd_s;
void e1_bearer_release_cmd(const struct e1ap_bearer_release_cmd_s *cmd) { abort(); }
nrUE_params_t nrUE_params;
nrUE_params_t *get_nrUE_params(void) { return &nrUE_params; }

/* no PHY: the configuration is not sent anywhere */
static void phy_config_none(NR_PHY_Config_t *cfg)
{
  (void)cfg;
}

#define BENCH_FIRST_RNTI 0x1234
/* SNR 20 dB in the FAPI 0.5 dB steps from -64 dB, above the 10 dB the MAC wants for an SR */
#define BENCH_UL_CQI 168
#define BENCH_RSSI 980
#define BENCH_TA 31

static const int bench_drb_id = 1;
static const int bench_drb_lcid = 4;

typedef struct {
  rnti_t rnti; /* 0: free */
  bool has_drb; /* preconfigured with DRB 1; UEs from random access only have SRB1 */
  uint64_t ul_backlog; /* bytes the UE still has to send */
  double mcs_target; /* MCS index with a 10 % BLER on the first transmission, < 0 without link model */
  double ph_1rb; /* -P: power headroom in dB of a 1-RB PUSCH (PH for M RBs = ph_1rb - 10 log10(2^mu M)), NAN without */
  uint64_t dl_mcs, dl_n, ul_mcs, ul_n;
  uint64_t dl_new_bytes;
  bool sr_failed; /* its SRs reached sr_TransMax without a grant */
} bench_ue_t;

typedef struct {
  uint64_t dl_tbs, dl_bytes, dl_ack, dl_nack, dl_mcs, dl_rbs;
  uint64_t ul_tbs, ul_bytes, ul_crc_err, ul_mcs, ul_rbs;
  uint64_t sr, pucch_csi;
  uint64_t dup_pusch;
  uint64_t ul_power_limited; /* -P: PUSCH needing more power than the UE has */ /* PUSCH PDUs of a UE that already has one in the slot: a UE sends one PUSCH per slot */
} bench_stats_t;

static int cmp_double(const void *a, const void *b)
{
  double da = *(const double *)a, db = *(const double *)b;
  return (da > db) - (da < db);
}

static double draw(void)
{
  return uniformrandom();
}

/* BLER of a TB: a logistic curve of the MCS, 10 % at the UE target, about 2 % one MCS below and 33 %
 * one above; each retransmission gains 2 MCS (HARQ combining); the floor is a residual BLER on top */
static double link_bler(const bench_ue_t *ue, double mcs, int round, double floor)
{
  if (ue == NULL || ue->mcs_target < 0)
    return floor;
  const double model = 1 / (1 + 9 * exp(-1.5 * (mcs - 2 * round - ue->mcs_target)));
  return 1 - (1 - model) * (1 - floor);
}

static bench_ue_t *bench_ue(bench_ue_t *ues, int num_ues, rnti_t rnti)
{
  for (int u = 0; u < num_ues; u++)
    if (ues[u].rnti == rnti)
      return &ues[u];
  return NULL;
}

/* random access: the UE side of 38.321 5.1 and the CU side of the RRC setup */
typedef struct {
  int to_start; /* UEs that still have to send a preamble */
  int per_occasion; /* preambles sent in one PRACH occasion (-A) */
  bool sib1_seen; /* a UE sends a preamble only after reading SIB1 */
  uint64_t preambles, msg3s, setups, released;
  uint64_t collisions, collided_msga, collided_ue; /* PUSCH lost because another PUSCH used the same PRBs and symbols */
  rnti_t msga_sent[MAX_MOBILES_PER_GNB]; /* TC-RNTI of UEs that sent their MsgA PUSCH: once per attempt */
  int num_msga_sent;
  rnti_t setup_rnti[MAX_MOBILES_PER_GNB]; /* RRCSetupRequest received, RRCSetup to send */
  int num_setup;
} bench_ra_t;

static bench_ra_t bench_ra;

/* F1 Initial UL RRC Message Transfer: the CU answers with an RRCSetup on SRB0 */
static void cu_initial_ul_rrc(module_id_t module_id, const f1ap_initial_ul_rrc_message_t *ul_rrc)
{
  (void)module_id;
  AssertFatal(bench_ra.num_setup < MAX_MOBILES_PER_GNB, "too many pending RRC setups\n");
  bench_ra.setup_rnti[bench_ra.num_setup++] = ul_rrc->gNB_DU_ue_id;
}

static void cu_ue_context_release_request(const f1ap_ue_context_rel_req_t *req)
{
  (void)req;
  bench_ra.released++;
}

static void cu_send_rrc_setups(void)
{
  /* the content is not decoded by the MAC: any SRB0 SDU of the size of an RRCSetup */
  static uint8_t rrc_setup[40];
  for (int i = 0; i < bench_ra.num_setup; i++) {
    const rnti_t rnti = bench_ra.setup_rnti[i];
    f1ap_dl_rrc_message_t dl = {.gNB_CU_ue_id = rnti, .gNB_DU_ue_id = rnti, .srb_id = 0, .rrc_container = rrc_setup, .rrc_container_length = sizeof(rrc_setup)};
    dl_rrc_message_transfer(&dl);
    bench_ra.setups++;
  }
  bench_ra.num_setup = 0;
}

/* common search spaces as configured by the gNB (gnb_config.c): RAR in 1, paging in 2, other SI in 3 */
static void add_common_search_spaces(NR_ServingCellConfigCommon_t *scc)
{
  NR_PDCCH_ConfigCommon_t *pcc = scc->downlinkConfigCommon->initialDownlinkBWP->pdcch_ConfigCommon->choice.setup;
  pcc->commonSearchSpaceList = calloc_or_fail(1, sizeof(*pcc->commonSearchSpaceList));
  int candidates[NUM_PDCCH_AGG_LEVELS] = {[PDCCH_AGG_LEVEL4] = NR_SearchSpace__nrofCandidates__aggregationLevel4_n2};
  for (int id = 1; id <= 3; id++)
    asn1cSeqAdd(&pcc->commonSearchSpaceList->list, rrc_searchspace_config(true, id, 0, candidates));
  asn1cCallocOne(pcc->searchSpaceSIB1, 0);
  asn1cCallocOne(pcc->ra_SearchSpace, 1);
  asn1cCallocOne(pcc->pagingSearchSpace, 2);
  asn1cCallocOne(pcc->searchSpaceOtherSystemInformation, 3);
}

/* 2-step RA as in gnb.sa.band78.106PRB.usrpb210.RA_2-Step.conf */
static void add_msga_config(NR_ServingCellConfigCommon_t *scc)
{
  NR_BWP_UplinkCommon_t *bwp = scc->uplinkConfigCommon->initialUplinkBWP;
  bwp->ext1 = calloc_or_fail(1, sizeof(*bwp->ext1));
  bwp->ext1->msgA_ConfigCommon_r16 = calloc_or_fail(1, sizeof(*bwp->ext1->msgA_ConfigCommon_r16));
  bwp->ext1->msgA_ConfigCommon_r16->present = NR_SetupRelease_MsgA_ConfigCommon_r16_PR_setup;
  NR_MsgA_ConfigCommon_r16_t *m = calloc_or_fail(1, sizeof(*m));
  bwp->ext1->msgA_ConfigCommon_r16->choice.setup = m;
  NR_RACH_ConfigCommonTwoStepRA_r16_t *ra = &m->rach_ConfigCommonTwoStepRA_r16;
  asn1cCallocOne(ra->rach_ConfigGenericTwoStepRA_r16.msgB_ResponseWindow_r16, 7); /* sl80 */
  asn1cCallocOne(ra->msgA_RSRP_Threshold_r16, 19);
  asn1cCallocOne(ra->rach_ConfigGenericTwoStepRA_r16.msgA_PRACH_ConfigurationIndex_r16, 148);
  ra->msgA_SSB_PerRACH_OccasionAndCB_PreamblesPerSSB_r16 = calloc_or_fail(1, sizeof(*ra->msgA_SSB_PerRACH_OccasionAndCB_PreamblesPerSSB_r16));
  ra->msgA_SSB_PerRACH_OccasionAndCB_PreamblesPerSSB_r16->present = 4; /* one */
  ra->msgA_SSB_PerRACH_OccasionAndCB_PreamblesPerSSB_r16->choice.one = 15;
  m->msgA_PUSCH_Config_r16 = calloc_or_fail(1, sizeof(*m->msgA_PUSCH_Config_r16));
  NR_MsgA_PUSCH_Resource_r16_t *r = calloc_or_fail(1, sizeof(*r));
  m->msgA_PUSCH_Config_r16->msgA_PUSCH_ResourceGroupA_r16 = r;
  r->msgA_MCS_r16 = 2;
  r->nrofSlotsMsgA_PUSCH_r16 = 1;
  r->nrofMsgA_PO_PerSlot_r16 = NR_MsgA_PUSCH_Resource_r16__nrofMsgA_PO_PerSlot_r16_one;
  r->msgA_PUSCH_TimeDomainOffset_r16 = 8;
  asn1cCallocOne(r->startSymbolAndLengthMsgA_PO_r16, 38);
  asn1cCallocOne(r->mappingTypeMsgA_PUSCH_r16, 1);
  r->guardBandMsgA_PUSCH_r16 = 0;
  r->frequencyStartMsgA_PUSCH_r16 = 0;
  r->nrofPRBs_PerMsgA_PO_r16 = 8;
  r->nrofMsgA_PO_FDM_r16 = NR_MsgA_PUSCH_Resource_r16__nrofMsgA_PO_FDM_r16_one;
  asn1cCallocOne(r->msgA_DMRS_Config_r16.msgA_PUSCH_NrofPorts_r16, 1);
  r->nrofDMRS_Sequences_r16 = 1;
  asn1cCallocOne(m->msgA_PUSCH_Config_r16->msgA_TransformPrecoder_r16, NR_MsgA_PUSCH_Config_r16__msgA_TransformPrecoder_r16_disabled);
}

/* the UE of a MsgA PUSCH occasion: an RA UE found by its RA-RNTI that has not sent its MsgA yet */
static NR_UE_info_t *msga_sender(gNB_MAC_INST *mac, rnti_t rnti)
{
  NR_UE_info_t *UE = find_ra_ue(&mac->UE_info, rnti);
  if (UE == NULL || UE->ra == NULL || UE->ra->ra_type != RA_2_STEP || UE->ra->ra_state != nrRA_WAIT_MsgA_PUSCH || UE->ra->RA_rnti != rnti)
    return NULL;
  for (int i = 0; i < bench_ra.num_msga_sent; i++)
    if (bench_ra.msga_sent[i] == UE->rnti)
      return NULL;
  return UE;
}

static bool pusch_overlap(const nfapi_nr_pusch_pdu_t *a, const nfapi_nr_pusch_pdu_t *b)
{
  const int a0 = a->bwp_start + a->rb_start, b0 = b->bwp_start + b->rb_start;
  const bool prbs = a0 < b0 + b->rb_size && b0 < a0 + a->rb_size;
  const bool symbols = a->start_symbol_index < b->start_symbol_index + b->nr_of_symbols
                       && b->start_symbol_index < a->start_symbol_index + a->nr_of_symbols;
  return prbs && symbols;
}

/* a PRACH occasion of the slot: the next UE sends a preamble, reported as a RACH indication */
static void loopback_prach(const nfapi_nr_prach_pdu_t *p, int frame, int slot)
{
  if (bench_ra.to_start == 0 || !bench_ra.sib1_seen)
    return;
  /* contention-based preambles start at 0; distinct RAPIDs, no collision */
  for (int k = 0; k < max(bench_ra.per_occasion, 1) && bench_ra.to_start > 0; k++) {
    bench_ra.to_start--;
    const int rapid = bench_ra.preambles % 8;
    bench_ra.preambles++;
    nr_initiate_ra_proc(0, 0, frame, slot, rapid, p->num_ra, p->prach_start_symbol, 0, 0);
  }
}

/* Msg3: UL-CCCH with a 48-bit RRCSetupRequest, whose first 6 bytes are the contention resolution identity */
static void loopback_msg3(const nfapi_nr_pusch_pdu_t *p, int frame, int slot, double ul_bler, bool collided)
{
  static uint8_t pdu[1 << 12];
  const uint32_t len = p->pusch_data.tb_size;
  AssertFatal(len >= 7 && len <= sizeof(pdu), "Msg3 TB of %u bytes\n", len);
  if (collided || (ul_bler > 0 && draw() < ul_bler)) {
    nr_rx_sdu(0, 0, frame, slot, p->rnti, NULL, len, p->pusch_data.harq_process_id, BENCH_TA, BENCH_UL_CQI, BENCH_RSSI);
    return;
  }
  memset(pdu, 0, len);
  pdu[0] = UL_SCH_LCID_CCCH_48_BITS;
  for (int i = 1; i <= 6; i++)
    pdu[i] = (p->rnti >> (i & 1 ? 8 : 0)) ^ (0x5a + i);
  if (len > 7)
    pdu[7] = UL_SCH_LCID_PADDING;
  bench_ra.msg3s++;
  nr_rx_sdu(0, 0, frame, slot, p->rnti, pdu, len, p->pusch_data.harq_process_id, BENCH_TA, BENCH_UL_CQI, BENCH_RSSI);
}

/* smallest index of 38.321 Table 6.1.3.1-1 whose value covers the buffer */
static int short_bsr_index(uint64_t bytes)
{
  for (int i = 0; i < NR_SHORT_BSR_TABLE_SIZE - 1; i++)
    if (bytes <= get_short_bsr_value(i))
      return i;
  return NR_SHORT_BSR_TABLE_SIZE - 1;
}

static void refill_rlc_buffers(gNB_MAC_INST *mac, bench_ue_t *ues, int num_ues, int bytes_per_ue)
{
  FOR_EACH_CONNECTED_UE(&mac->UE_info, UE) {
    const bench_ue_t *ue = bench_ue(ues, num_ues, UE->rnti);
    if (ue == NULL || !ue->has_drb)
      continue;
    /* keep the backlog bounded (full buffer for the scheduler, RAM flat over long runs) */
    if (UE->UE_sched_ctrl.rlc_status[bench_drb_lcid].bytes_in_buffer > max(20000, 10 * bytes_per_ue))
      continue;
    uint8_t *sdu = malloc(bytes_per_ue);
    AssertFatal(sdu, "malloc failed\n");
    memset(sdu, 0xAB, bytes_per_ue);
    protocol_ctxt_t ctxt = {.rntiMaybeUEid = UE->rnti, .enb_flag = 1, .module_id = 0};
    nr_rlc_data_req(&ctxt, 0, bench_drb_id, 0, bytes_per_ue, sdu);
  }
}

static int bench_mu = 1;

static const int bench_srb1 = 1;

/* -C: the RRCReconfiguration of every preconfigured UE, a new CellGroup and its SRB1 SDU */
static void start_reconfigs(gNB_MAC_INST *mac, bench_ue_t *ues, int num_ues, int bytes)
{
  FOR_EACH_CONNECTED_UE(&mac->UE_info, UE) {
    if (bench_ue(ues, num_ues, UE->rnti) == NULL)
      continue;
    AssertFatal(asn_copy(&asn_DEF_NR_CellGroupConfig, (void **)&UE->reconfigCellGroup, UE->CellGroup) == 0, "copy failed\n");
    uint8_t *sdu = malloc(bytes);
    AssertFatal(sdu, "malloc failed\n");
    memset(sdu, 0xCD, bytes);
    protocol_ctxt_t ctxt = {.rntiMaybeUEid = UE->rnti, .enb_flag = 1, .module_id = 0};
    nr_rlc_data_req(&ctxt, 1, bench_srb1, 0, bytes, sdu);
  }
}

/* -C: the MAC switches to the new CellGroup on the HARQ ACK of a PDSCH (sched_pdsch.action), the UE when it has
 * the whole RRCReconfiguration: a PDSCH waiting for its ACK with the action while SRB1 bytes of the message are
 * still in the RLC makes both use different configurations until the end arrives */
static uint64_t check_reconfig_action(gNB_MAC_INST *mac, int frame, uint64_t *seen)
{
  uint64_t early = 0;
  FOR_EACH_CONNECTED_UE(&mac->UE_info, UE) {
    const NR_UE_sched_ctrl_t *sc = &UE->UE_sched_ctrl;
    bool action = false;
    for (int pid = 0; pid < NR_MAX_HARQ_PROCESSES; pid++)
      action |= sc->harq_processes[pid].is_waiting && sc->harq_processes[pid].sched_pdsch.action != NULL;
    if (!action)
      continue;
    logical_chan_id_t lcid = bench_srb1;
    mac_rlc_status_resp_t st = {0};
    nr_mac_rlc_status_ind(UE->rnti, frame, 1, &lcid, &st);
    if (st.bytes_in_buffer > 0) {
      early++;
      if ((*seen)++ < 5)
        printf("  RNTI %04x: PDSCH with the CellGroup switch waits for its ACK, %d SRB1 bytes not sent yet\n", UE->rnti, st.bytes_in_buffer);
    }
  }
  return early;
}

/* -P: a PUSCH needing more power than the UE has is sent at its maximum power, its SNR lower by the missing dB: with
 * the link model, about 1 dB per MCS index */
static double power_deficit_db(const bench_ue_t *ue, int n_rb)
{
  if (ue == NULL || isnan(ue->ph_1rb))
    return 0;
  const double need = 10 * log10((double)(n_rb << bench_mu));
  return need > ue->ph_1rb ? need - ue->ph_1rb : 0;
}

/* A decoded PUSCH: short BSR with what is left after this TB, a single entry PHR with -P (38.321 6.1.3.8), then
 * padding */
static void loopback_pusch(const nfapi_nr_pusch_pdu_t *p, int frame, int slot, double ul_bler, bool collided, bench_ue_t *ue, bench_stats_t *st)
{
  static uint8_t pdu[1 << 18];
  const uint32_t len = p->pusch_data.tb_size;
  AssertFatal(len <= sizeof(pdu), "TB of %u bytes\n", len);
  /* round from the redundancy version sequence 0, 2, 3, 1 */
  static const int rv_round[4] = {0, 3, 1, 2};
  const double deficit = power_deficit_db(ue, p->rb_size);
  st->ul_power_limited += deficit > 0;
  const double bler = link_bler(ue, p->mcs_index + deficit, rv_round[p->pusch_data.rv_index & 3], ul_bler);
  const bool crc_err = collided || (bler > 0 && draw() < bler);
  ue->ul_mcs += p->mcs_index;
  ue->ul_n++;
  st->ul_tbs++;
  st->ul_mcs += p->mcs_index;
  st->ul_rbs += p->rb_size;
  if (crc_err) {
    st->ul_crc_err++;
    nr_rx_sdu(0, 0, frame, slot, p->rnti, NULL, len, p->pusch_data.harq_process_id, BENCH_TA, BENCH_UL_CQI, BENCH_RSSI);
    return;
  }
  memset(pdu, 0, len);
  uint32_t used = 0;
  const bool phr = ue && !isnan(ue->ph_1rb) && len >= 6;
  if (phr) {
    /* 38.133 Table 10.1.17.1-1: index PH + 32 from -32 to 22 dB, then 2 dB steps; PCMAX 23 dBm (index + 29) */
    const int ph = (int)lround(ue->ph_1rb - 10 * log10((double)(p->rb_size << bench_mu)));
    const int ph_idx = ph < -32 ? 0 : ph <= 22 ? ph + 32 : ph >= 38 ? 63 : 55 + (ph - 24) / 2;
    NR_SINGLE_ENTRY_PHR_MAC_CE ce = {.PH = ph_idx, .PCMAX = 23 + 29};
    pdu[used++] = UL_SCH_LCID_SINGLE_ENTRY_PHR;
    memcpy(&pdu[used], &ce, sizeof(ce));
    used += sizeof(ce);
  }
  if (len >= used + 2) {
    const uint32_t payload = len > used + 3 ? len - used - 3 : 0;
    const uint64_t sent = ue->ul_backlog < payload ? ue->ul_backlog : payload;
    ue->ul_backlog -= sent;
    st->ul_bytes += sent;
    NR_BSR_SHORT bsr = {.LcgID = 0, .Buffer_size = short_bsr_index(ue->ul_backlog)};
    pdu[used++] = UL_SCH_LCID_S_BSR;
    memcpy(&pdu[used++], &bsr, 1);
  }
  if (used < len)
    pdu[used] = UL_SCH_LCID_PADDING;
  nr_rx_sdu(0, 0, frame, slot, p->rnti, pdu, len, p->pusch_data.harq_process_id, BENCH_TA, BENCH_UL_CQI, BENCH_RSSI);
}

/* ACK/NACK drawn from the DL BLER, true for an ACK */
static bool draw_ack(double dl_bler, bench_stats_t *st)
{
  const bool nack = dl_bler > 0 && draw() < dl_bler;
  st->dl_nack += nack;
  st->dl_ack += !nack;
  return !nack;
}

/* BLER of the TB answered by HARQ bit b: the MAC pops its feedback list in this order */
static double harq_bit_bler(const NR_UE_info_t *UE, int b, const bench_ue_t *ue, double dl_bler)
{
  if (UE == NULL || ue == NULL || ue->mcs_target < 0)
    return dl_bler;
  const NR_UE_sched_ctrl_t *sc = &UE->UE_sched_ctrl;
  int pid = sc->feedback_dl_harq.head;
  for (int i = 0; i < b && pid >= 0; i++)
    pid = sc->feedback_dl_harq.next[pid];
  if (pid < 0)
    return dl_bler;
  const NR_UE_harq_t *h = &sc->harq_processes[pid];
  return link_bler(ue, h->sched_pdsch.mcs, h->round, dl_bler);
}

static void loopback_pucch(const nfapi_nr_pucch_pdu_t *p, int frame, int slot, double dl_bler, const bench_ue_t *ue, bench_stats_t *st)
{
  const NR_UE_info_t *UE = find_connected_nr_ue(&RC.nrmac[0]->UE_info, p->rnti);
  const bool sr = p->sr_flag && ue && ue->ul_backlog > 0;
  st->sr += sr;
  if (p->format_type <= 1) {
    nfapi_nr_uci_pucch_pdu_format_0_1_t uci = {
        .pduBitmap = (p->sr_flag != 0) | ((p->bit_len_harq > 0) << 1),
        .handle = p->handle,
        .rnti = p->rnti,
        .pucch_format = p->format_type,
        .ul_cqi = BENCH_UL_CQI,
        .timing_advance = 0xffff,
        .rssi = BENCH_RSSI,
        .sr.sr_indication = sr,
        .harq.num_harq = p->bit_len_harq,
    };
    /* format 0/1: 0 is an ACK */
    for (int b = 0; b < p->bit_len_harq; b++)
      uci.harq.harq_list[b].harq_value = !draw_ack(harq_bit_bler(UE, b, ue, dl_bler), st);
    handle_nr_uci_pucch_0_1(0, 0, frame, slot, &uci);
    return;
  }
  /* format 2 to 4: the MAC frees the payloads */
  if (p->bit_len_csi_part1 > 0)
    st->pucch_csi++;
  nfapi_nr_uci_pucch_pdu_format_2_3_4_t uci = {
      .pduBitmap = (p->sr_flag != 0) | ((p->bit_len_harq > 0) << 1),
      .handle = p->handle,
      .rnti = p->rnti,
      .pucch_format = p->format_type - 2,
      .ul_cqi = BENCH_UL_CQI,
      .timing_advance = 0xffff,
      .rssi = BENCH_RSSI,
  };
  if (p->sr_flag) {
    uci.sr.sr_bit_len = p->sr_flag;
    uci.sr.sr_payload = calloc(1, (p->sr_flag + 7) / 8);
    uci.sr.sr_payload[0] = sr;
  }
  if (p->bit_len_harq > 0) {
    uci.harq.harq_bit_len = p->bit_len_harq;
    uci.harq.harq_payload = calloc(1, (p->bit_len_harq + 7) / 8);
    /* first HARQ bit in the most significant position, 1 is an ACK */
    for (int b = 0; b < p->bit_len_harq; b++) {
      const int pos = p->bit_len_harq - 1 - b;
      uci.harq.harq_payload[pos >> 3] |= draw_ack(harq_bit_bler(UE, b, ue, dl_bler), st) << (pos & 7);
    }
  }
  handle_nr_uci_pucch_2_3_4(0, 0, frame, slot, &uci);
}

int main(int argc, char **argv)
{
  setbuf(stdout, NULL);
  /* the bench exists to run the checker: on unless the caller sets it */
  setenv("OAI_SCHED_CHECK", "1", 0);

  int N_RB = 106;
  int mu = 1;
  int n_slots = 20000;
  int num_ues = 1;
  int loglvl = OAILOG_WARNING;
  double dl_bler = 0.0, ul_bler = 0.0;
  int dl_bytes = 10000, ul_bytes = 1000;
  int max_mcs = 28;
  bool fail_on_findings = false;
  int trace_rnti = -1;
  uint64_t ssb_positions = 0;
  double mcs_lo = -1, mcs_hi = -1;
  double ph_lo = NAN, ph_hi = NAN;
  int ra_ues = 0;
  bool two_step = false;
  bool sched_3gpp = false, sched_3gpp_ul = false;
  int reconfig_bytes = 0;
  int ul_ues = -1;
  int max_frame_inactivity = 10;

  if ((uniqCfg = load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY)) == 0) {
    printf("Error: configuration module init failed\n");
    return 1;
  }

  int c;
  int do_csirs = 0;
  while ((c = getopt(argc, argv, "--:O:23a:A:P:B:b:c:C:dD:e:FI:KL:m:n:R:S:T:U:u:Y:h")) != -1) {
    if (c == 1 || c == '-' || c == 'O')
      continue;
    switch (c) {
      case '2':
        two_step = true;
        break;
      case '3':
        sched_3gpp = sched_3gpp_ul = true;
        break;
      case 'd':
        sched_3gpp = true;
        break;
      case 'a':
        ra_ues = atoi(optarg);
        break;
      case 'A':
        bench_ra.per_occasion = atoi(optarg);
        break;
      case 'B':
        dl_bler = atof(optarg);
        break;
      case 'b':
        ul_bler = atof(optarg);
        break;
      case 'c':
        if (sscanf(optarg, "%lf:%lf", &mcs_lo, &mcs_hi) == 1)
          mcs_hi = mcs_lo;
        break;
      case 'P':
        if (sscanf(optarg, "%lf:%lf", &ph_lo, &ph_hi) == 1)
          ph_hi = ph_lo;
        break;
      case 'C':
        reconfig_bytes = atoi(optarg);
        break;
      case 'D':
        dl_bytes = atoi(optarg);
        break;
      case 'e':
        max_mcs = atoi(optarg);
        break;
      case 'F':
        fail_on_findings = true;
        break;
      case 'L':
        loglvl = atoi(optarg);
        break;
      case 'm':
        mu = atoi(optarg);
        bench_mu = mu;
        break;
      case 'n':
        n_slots = atoi(optarg);
        break;
      case 'R':
        N_RB = atoi(optarg);
        break;
      case 'K':
        do_csirs = 1;
        break;
      case 'S':
        ssb_positions = strtoull(optarg, NULL, 0);
        break;
      case 'T':
        trace_rnti = strtol(optarg, NULL, 16);
        break;
      case 'U':
        ul_bytes = atoi(optarg);
        break;
      case 'u':
        num_ues = atoi(optarg);
        break;
      case 'I':
        max_frame_inactivity = atoi(optarg);
        break;
      case 'Y':
        ul_ues = atoi(optarg);
        break;
      case 'h':
      default:
        printf("nr_macbench - gNB MAC scheduler without PHY, FAPI loopback, conformance checker on\n\n");
        printf("  -a <n>           UEs joining by CBRA, one per PRACH occasion (default 0)\n");
        printf("  -A <n>           with -a, n preambles in each PRACH occasion (distinct RAPIDs, default 1)\n");
        printf("  -3               3gpp scheduler for DL and UL, legacy otherwise\n");
        printf("  -d               3gpp scheduler for DL only\n");
        printf("  -2               2-step RA (msgA of RA_2-Step.conf): every -a UE sends a MsgA\n");
        printf("  -B <bler>        DL BLER, drawn per HARQ bit (default 0)\n");
        printf("  -b <bler>        UL BLER, drawn per PUSCH (default 0)\n");
        printf("  -P <lo[:hi]>     Power headroom in dB of a 1-RB PUSCH, spread from lo to hi over the UEs (with -c):\n"
               "                   the UEs send a PHR, a PUSCH beyond it loses the missing dB of SNR (default off)\n");
        printf("  -c <lo[:hi]>     Link model: MCS index with 10 %% BLER, spread from lo to hi over the UEs;\n"
               "                   -B/-b become a residual BLER on top (default off: flat -B/-b)\n");
        printf("  -C <bytes>       Every UE gets an RRCReconfiguration of <bytes> on SRB1 with a new CellGroup; counts the\n"
               "                   slots where the CellGroup switch rides a PDSCH that does not end the message\n");
        printf("  -I <frames>      UL inactivity after which a UE gets a grant to send a BSR (default 10, as MACRLCs)\n");
        printf("  -Y <n>           Only the last n UEs have UL data (-U), the others none: they get the grants of\n"
               "                   long UL inactivity; counts the UEs whose SRs reach sr_TransMax without a grant\n");
        printf("  -D <bytes>       DL bytes pushed in RLC per UE and DL slot (default 10000, 0 = none)\n");
        printf("  -e <mcs>         DL max MCS (default 28)\n");
        printf("  -F               Exit 2 when the checker reports a finding or drops a PDU\n");
        printf("  -L <log_level>   Log level (default warning)\n");
        printf("  -m <numerology>  Numerology (default 1)\n");
        printf("  -n <num_slots>   Slots to run (default 20000)\n");
        printf("  -R <N_RB>        Resource blocks (default 106)\n");
        printf("  -K               CSI-RS and periodic CSI reports on PUCCH (do_CSIRS 1, as the rfsim gNB configuration)\n");
        printf("  -S <bitmap>      ssb-PositionsInBurst as in the gNB configuration file, e.g. 1 for one SSB (default:\n"
               "                   the 8 SSBs of fill_scc_sim); the SSB RSRP report size sets the PUCCH plan, so how\n"
               "                   many UEs the cell serves\n");
        printf("  -T <rnti>        Print the PDSCH, PUSCH and PUCCH of this RNTI (hex)\n");
        printf("  -U <bytes>       UL bytes arriving per UE and slot (default 1000, 0 = none)\n");
        printf("  -u <num_ues>     Connected UEs (default 1, max %d)\n", MAX_MOBILES_PER_GNB);
        return 0;
    }
  }
  if (num_ues < 0 || ra_ues < 0 || num_ues + ra_ues < 1 || num_ues + ra_ues > MAX_MOBILES_PER_GNB) {
    printf("Error: %d + %d UEs, the total range is 1 to %d\n", num_ues, ra_ues, MAX_MOBILES_PER_GNB);
    return 1;
  }

  logInit();
  set_glog(loglvl);
  randominit(); /* set OAI_RNGSEED to reproduce a run */
  /* taus() draws the TC-RNTIs: same seed as uniformrandom(), random without OAI_RNGSEED as in nr-softmodem */
  const char *seed = getenv("OAI_RNGSEED");
  set_taus_seed(seed ? strtoul(seed, NULL, 0) + 1 : 0);

  printf("=== nr_macbench (%s DL, %s UL): %d UEs, %d RBs, mu %d, DL BLER %.3f, UL BLER %.3f, DL %d B, UL %d B per slot, %d slots ===\n",
         sched_3gpp ? "3gpp" : "legacy", sched_3gpp_ul ? "3gpp" : "legacy", num_ues, N_RB, mu, dl_bler, ul_bler, dl_bytes, ul_bytes, n_slots);

  /* nr_mac_add_test_ue needs phy_test, which turns off PRACH and RA. With random access, the cell is
   * configured as standalone (CORESET 0, SIB1) and phy_test is on only while the test UEs are added */
  get_softmodem_params()->phy_test = ra_ues == 0;
  get_softmodem_params()->do_ra = 0;
  /* mac_top_init_gNB reads RC.gNB for the cell: keep an empty PHY instance */
  RC.gNB = calloc(1, sizeof(PHY_VARS_gNB *));
  RC.gNB[0] = calloc(1, sizeof(PHY_VARS_gNB));
  gNB = RC.gNB[0];
  AssertFatal((gNB->if_inst = NR_IF_Module_init(0)) != NULL, "Cannot register interface");
  gNB->if_inst->NR_PHY_config_req = phy_config_none;

  NR_ServingCellConfigCommon_t *scc = calloc(1, sizeof(*scc));
  prepare_scc(scc);
  uint64_t ssb_bitmap = 1;
  fill_scc_sim(scc, &ssb_bitmap, N_RB, N_RB, mu, mu);
  if (ssb_positions != 0)
    ssb_bitmap = ssb_positions;
  fix_scc(scc, ssb_bitmap);
  /* random access needs the common search spaces and SIB1 of a standalone cell */
  if (ra_ues > 0)
    add_common_search_spaces(scc);
  if (two_step)
    add_msga_config(scc);

  const nr_mac_config_t conf = {
      .pdsch_AntennaPorts = {.N1 = 1, .N2 = 1, .XP = 1},
      .pusch_AntennaPorts = 1,
      .minRXTXTIME = 6,
      .do_CSIRS = do_csirs,
      .do_SRS = 0,
      .num_dlharq = 16,
      .num_ulharq = 16,
      .maxMIMO_layers = 1,
      .force_256qam_off = false,
      .timer_config.sr_ProhibitTimer = 0,
      .timer_config.sr_TransMax = 64,
      .timer_config.sr_ProhibitTimer_v1700 = 0,
      .timer_config.t300 = 400,
      .timer_config.t301 = 400,
      .timer_config.t310 = 2000,
      .timer_config.n310 = 10,
      .timer_config.t311 = 3000,
      .timer_config.n311 = 1,
      .timer_config.t319 = 400,
      .num_agg_level_candidates = {0, 4, 4, 2, 0},
      .beam_config.beams_per_period = 1,
      /* MACRLC defaults (MACRLC_nr_paramdef.h): the initial SNR of the power control loops, the base of the
       * UL link adaptation until enough PUSCHs were measured, and the DTX counts before an UL failure (checked
       * outside phy-test, so with -a: at 0 the first PUSCH in error stops the UE) */
      .pusch.target_snrx10 = 200,
      .pucch.target_snrx10 = 150,
      .pusch.failure_thres = 10,
      .pucch.failure_thres = 10,
  };
  const nr_rlc_configuration_t rlc_config = {
      .srb = {.t_poll_retransmit = 45, .t_reassembly = 35, .t_status_prohibit = 0,
              .poll_pdu = -1, .poll_byte = -1, .max_retx_threshold = 8, .sn_field_length = 12},
      .drb_am = {.t_poll_retransmit = 45, .t_reassembly = 15, .t_status_prohibit = 0,
                 .poll_pdu = 64, .poll_byte = 1024 * 500, .max_retx_threshold = 32, .sn_field_length = 18},
      .drb_um = {.t_reassembly = 15, .sn_field_length = 12},
  };

  RC.nb_nr_macrlc_inst = 1;
  nr_cell_sched_t *cell;
  gNB_MAC_INST *mac = mac_top_init_gNB(ngran_gNB, &rlc_config, &cell);
  nr_mac_cell_config_scc(cell, scc, &conf);
  /* the gNB rejects a UE beyond what the periodic PUCCH plan serves (verify_radio_configuration); the
   * bench adds its UEs without that check, so it stops here instead */
  const int served = cell->pucch_res_bwp[conf.first_active_bwp].max_served_ues;
  if (num_ues + ra_ues > served) {
    printf("Error: the periodic PUCCH plan of this cell (%d RBs) serves %d UEs, %d + %d asked (see -S)\n", N_RB, served, num_ues, ra_ues);
    return 1;
  }
  /* with -a the cell is standalone, so nr_mac_cell_config_scc plans the PUCCH of the initial BWP only, while the
   * UEs added below get the dedicated BWP 1 of get_default_secondaryCellGroup: the whole carrier, as the initial
   * BWP of fill_scc_sim, hence the same plan */
  if (ra_ues > 0)
    cell->pucch_res_bwp[1] = cell->pucch_res_bwp[0];
  nr_mac_send_config_req(mac, &cell->config);
  /* the defaults of the MACRLCs section (MACRLC_nr_paramdef.h) */
  cell->ulsch_max_frame_inactivity = max_frame_inactivity;
  cell->dl_bler = (NR_bler_options_t){.upper = 0.15, .lower = 0.05, .min_mcs = 0, .max_mcs = max_mcs, .harq_round_max = 4};
  cell->ul_bler = (NR_bler_options_t){.upper = 0.15, .lower = 0.05, .min_mcs = 0, .max_mcs = 28, .harq_round_max = 4};
  cell->min_grant_prb = 5;
  /* the real schedulers in place of the phy_test ones */
  mac->pre_processor_dl = sched_3gpp ? sched3gpp_dl_preprocessor : nr_dlsch_preprocessor;
  mac->pre_processor_ul = sched_3gpp_ul ? sched3gpp_ul_preprocessor : nr_ulsch_preprocessor;
  mac->dl_ri_pmi_select = nr_dl_ri_pmi_select_default;
  mac->dl_mcs_select = nr_dl_mcs_select_default;
  mac->dl_beam_select = nr_dl_beam_select_default;
  mac->dl_tda_select = nr_dl_tda_select_default;
  mac->dl_rb_alloc = nr_dl_proportional_fair;
  mac->dl_lcid_alloc = nr_dl_lcid_alloc_default;
  if (ra_ues > 0) {
    const plmn_id_t plmn = {.mcc = 1, .mnc = 1, .mnc_digit_length = 2};
    nr_mac_configure_sib1(cell, &plmn, 12345678, 1);
    /* the served cell of the F1 Setup, copied into every Initial UL RRC Message Transfer */
    static f1ap_setup_req_t setup_req = {.num_cells_available = 1};
    setup_req.cell = calloc_or_fail(1, sizeof(*setup_req.cell));
    setup_req.cell[0].info.plmn = plmn;
    setup_req.cell[0].info.nr_cellid = 12345678;
    mac->f1_config.setup_req = &setup_req;
    mac->mac_rrc.initial_ul_rrc_message_transfer = cu_initial_ul_rrc;
    mac->mac_rrc.ue_context_release_request = cu_ue_context_release_request;
  }

  get_softmodem_params()->phy_test = 1;
  for (int u = 0; u < num_ues; u++) {
    rnti_t rnti = BENCH_FIRST_RNTI + u;
    NR_feature_set_ids_t ids = {0};
    NR_CellGroupConfig_t *cg = get_default_secondaryCellGroup(scc, NULL, &ids, 0, 1, &conf, cell, u, 0);
    cg->spCellConfig->reconfigurationWithSync = get_reconfiguration_with_sync(rnti, u, scc, 0);
    /* RLC UM: no UE to acknowledge an AM transmission */
    NR_RLC_BearerConfig_t *drb = get_DRB_RLC_BearerConfig(bench_drb_lcid, bench_drb_id, NR_RLC_Config_PR_um_Bi_Directional, 13, &rlc_config);
    cg->rlc_BearerToAddModList = calloc(1, sizeof(*cg->rlc_BearerToAddModList));
    asn1cSeqAdd(&cg->rlc_BearerToAddModList->list, drb);
    NR_RLC_BearerConfig_t *srb1 = reconfig_bytes > 0 ? get_SRB_RLC_BearerConfig(bench_srb1, 1, NR_LogicalChannelConfig__ul_SpecificParameters__bucketSizeDuration_ms5, &rlc_config) : NULL;
    if (srb1)
      asn1cSeqAdd(&cg->rlc_BearerToAddModList->list, srb1);
    AssertFatal(nr_mac_add_test_ue(mac, cell, rnti, cg), "Failed to add test UE %d\n", u);
    NR_UE_info_t *UE = find_connected_nr_ue(&mac->UE_info, rnti);
    AssertFatal(UE, "UE %d not connected\n", u);
    configure_UE_BWP(cell, scc, UE, false, NR_SearchSpace__searchSpaceType_PR_ue_Specific, -1, -1);
    nr_rlc_add_drb(rnti, bench_drb_id, drb);
    if (srb1)
      nr_rlc_add_srb(rnti, bench_srb1, srb1);
    UE->UE_sched_ctrl.dl_max_mcs = max_mcs;
  }

  /* preconfigured UEs first, then the UEs of random access as they connect */
  const int total = num_ues + ra_ues;
  bench_ue_t *ues = calloc(total, sizeof(*ues));
  AssertFatal(ues, "alloc failed\n");
  for (int u = 0; u < total; u++) {
    ues[u].mcs_target = mcs_lo < 0 ? -1 : total == 1 ? mcs_lo : mcs_lo + (mcs_hi - mcs_lo) * u / (total - 1);
    ues[u].ph_1rb = isnan(ph_lo) ? NAN : total == 1 ? ph_lo : ph_lo + (ph_hi - ph_lo) * u / (total - 1);
    if (u < num_ues) {
      ues[u].rnti = BENCH_FIRST_RNTI + u;
      ues[u].has_drb = true;
    }
  }
  int registered = num_ues;
  get_softmodem_params()->phy_test = ra_ues == 0;
  if (ra_ues > 0) {
    bench_ra.to_start = ra_ues;
  }
  NR_Sched_Rsp_t *rsp = calloc(1, sizeof(*rsp));
  double *t_sched = calloc(n_slots, sizeof(double));
  AssertFatal(ues && rsp && t_sched, "alloc failed\n");
  bench_stats_t st = {0};
  uint64_t reconfig_early = 0, reconfig_early_seen = 0;
  /* DL slots, once every UE is connected with -D traffic, and those without any PDSCH of a UE */
  uint64_t dl_slots_loaded = 0, dl_slots_empty = 0;
  int sr_failures = 0;
  if (reconfig_bytes > 0)
    start_reconfigs(mac, ues, num_ues, reconfig_bytes);
  const int slots_per_frame = cell->frame_structure.numb_slots_frame;

  int frame = 0, slot = 0;
  for (int n = 0; n < n_slots; n++) {
    const bool is_dl = is_dl_slot(slot, &cell->frame_structure);
    if (registered < total && get_num_connected_ue(&mac->UE_info) > registered) {
      FOR_EACH_CONNECTED_UE(&mac->UE_info, UE) {
        if (registered < total && bench_ue(ues, total, UE->rnti) == NULL)
          ues[registered++].rnti = UE->rnti;
      }
    }
    if (is_dl && dl_bytes > 0)
      refill_rlc_buffers(mac, ues, total, dl_bytes);
    for (int u = 0; u < registered; u++)
      if (ul_ues < 0 || u >= registered - ul_ues)
        ues[u].ul_backlog += ul_bytes;

    reset_sched_response(rsp, frame, slot, 0, 0);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    gNB_dlsch_ulsch_scheduler(0, 0, frame, slot, rsp);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    t_sched[n] = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
    if (reconfig_bytes > 0)
      reconfig_early += check_reconfig_action(mac, frame, &reconfig_early_seen);
    /* a UE whose SRs reach sr_TransMax without a grant falls back to random access (TS 38.321 5.4.4) */
    FOR_EACH_CONNECTED_UE(&mac->UE_info, UE) {
      const int trans_max = cell->radio_config.timer_config.sr_TransMax;
      if (trans_max > 0 && UE->UE_sched_ctrl.sr_cnt >= trans_max) {
        bench_ue_t *bu = bench_ue(ues, total, UE->rnti);
        if (bu && !bu->sr_failed) {
          bu->sr_failed = true;
          sr_failures++;
        }
      }
    }

    const nfapi_nr_dl_tti_request_body_t *dl = &rsp->DL_req.dl_tti_request_body;
    int ue_pdsch = 0;
    for (int i = 0; i < dl->nPDUs; i++)
      ue_pdsch += dl->dl_tti_pdu_list[i].PDUType == NFAPI_NR_DL_TTI_PDSCH_PDU_TYPE
                  && bench_ue(ues, total, dl->dl_tti_pdu_list[i].pdsch_pdu.pdsch_pdu_rel15.rnti) != NULL;
    if (is_dl && dl_bytes > 0 && registered == total && total > 0) {
      dl_slots_loaded++;
      dl_slots_empty += ue_pdsch == 0;
    }
    for (int i = 0; i < dl->nPDUs; i++) {
      const nfapi_nr_dl_tti_request_pdu_t *pdu = &dl->dl_tti_pdu_list[i];
      if (pdu->PDUType != NFAPI_NR_DL_TTI_PDSCH_PDU_TYPE)
        continue;
      const nfapi_nr_dl_tti_pdsch_pdu_rel15_t *p = &pdu->pdsch_pdu.pdsch_pdu_rel15;
      if (p->rnti == 0xffff)
        bench_ra.sib1_seen = true;
      if (p->rnti == trace_rnti)
        printf("%4d.%2d PDSCH mcs %d rv %d sym %d+%d rbs %d\n",
               frame, slot, p->mcsIndex[0], p->rvIndex[0], p->StartSymbolIndex, p->NrOfSymbols, p->rbSize);
      bench_ue_t *bu = bench_ue(ues, total, p->rnti);
      if (!bu)
        continue;
      bu->dl_mcs += p->mcsIndex[0];
      bu->dl_n++;
      st.dl_tbs++;
      /* new data only: a retransmission (rv != 0) repeats bytes already counted */
      if (p->rvIndex[0] == 0) {
        st.dl_bytes += p->TBSize[0];
        bu->dl_new_bytes += p->TBSize[0];
      }
      st.dl_mcs += p->mcsIndex[0];
      st.dl_rbs += p->rbSize;
    }

    NR_UE_info_t *traced = trace_rnti >= 0 ? find_connected_nr_ue(&mac->UE_info, trace_rnti) : NULL;
    if (traced) {
      const NR_UE_sched_ctrl_t *sc = &traced->UE_sched_ctrl;
      printf("%4d.%2d feedback queue:", frame, slot);
      for (int pid = sc->feedback_dl_harq.head; pid >= 0; pid = sc->feedback_dl_harq.next[pid])
        printf(" %d(r%d)@%d.%d", pid, sc->harq_processes[pid].round, sc->harq_processes[pid].feedback_frame, sc->harq_processes[pid].feedback_slot);
      printf("\n");
    }

    /* the PHY of this slot: everything the MAC asked to receive comes back now */
    const nfapi_nr_ul_tti_request_t *ul = &rsp->UL_tti_req;
    /* PUSCH really transmitted in this slot: a known UE, a Msg3, or a MsgA occasion used by a UE; two of
     * them on the same PRBs and symbols are both lost */
    bool sent[NFAPI_MAX_NUM_UL_PDU] = {0}, collided[NFAPI_MAX_NUM_UL_PDU] = {0};
    for (int i = 0; i < ul->n_pdus; i++) {
      const nfapi_nr_ul_tti_request_number_of_pdus_t *pdu = &ul->pdus_list[i];
      if (pdu->pdu_type != NFAPI_NR_UL_CONFIG_PUSCH_PDU_TYPE)
        continue;
      const rnti_t r = pdu->pusch_pdu.rnti;
      const NR_UE_info_t *ra_ue = find_ra_ue(&mac->UE_info, r);
      sent[i] = bench_ue(ues, total, r) != NULL || msga_sender(mac, r) != NULL || (ra_ue && ra_ue->rnti == r);
    }
    for (int i = 0; i < ul->n_pdus; i++)
      for (int j = i + 1; j < ul->n_pdus; j++)
        if (ul->pdus_list[i].pdu_type == NFAPI_NR_UL_CONFIG_PUSCH_PDU_TYPE && ul->pdus_list[j].pdu_type == NFAPI_NR_UL_CONFIG_PUSCH_PDU_TYPE
            && ul->pdus_list[i].pusch_pdu.rnti == ul->pdus_list[j].pusch_pdu.rnti)
          st.dup_pusch++;
    for (int i = 0; i < ul->n_pdus; i++)
      for (int j = i + 1; j < ul->n_pdus; j++)
        if (sent[i] && sent[j] && pusch_overlap(&ul->pdus_list[i].pusch_pdu, &ul->pdus_list[j].pusch_pdu)) {
          collided[i] = collided[j] = true;
          bench_ra.collisions++;
        }
    for (int i = 0; i < ul->n_pdus; i++) {
      const nfapi_nr_ul_tti_request_number_of_pdus_t *pdu = &ul->pdus_list[i];
      if (pdu->pdu_type == NFAPI_NR_UL_CONFIG_PUSCH_PDU_TYPE) {
        bench_ue_t *ue = bench_ue(ues, total, pdu->pusch_pdu.rnti);
        NR_UE_info_t *msga = ue ? NULL : msga_sender(mac, pdu->pusch_pdu.rnti);
        if (ue) {
          bench_ra.collided_ue += collided[i];
          loopback_pusch(&pdu->pusch_pdu, frame, slot, ul_bler, collided[i], ue, &st);
        } else if (msga) {
          /* MsgA PUSCH: the same CCCH as Msg3, sent once at the occasion of the preamble */
          bench_ra.msga_sent[bench_ra.num_msga_sent++ % MAX_MOBILES_PER_GNB] = msga->rnti;
          bench_ra.collided_msga += collided[i];
          loopback_msg3(&pdu->pusch_pdu, frame, slot, ul_bler, collided[i]);
        } else if (sent[i]) {
          loopback_msg3(&pdu->pusch_pdu, frame, slot, ul_bler, collided[i]);
        }
        if (pdu->pusch_pdu.rnti == trace_rnti)
          printf("%4d.%2d PUSCH mcs %d rv %d pid %d sym %d+%d\n", frame, slot, pdu->pusch_pdu.mcs_index, pdu->pusch_pdu.pusch_data.rv_index,
                 pdu->pusch_pdu.pusch_data.harq_process_id, pdu->pusch_pdu.start_symbol_index, pdu->pusch_pdu.nr_of_symbols);
      } else if (pdu->pdu_type == NFAPI_NR_UL_CONFIG_PRACH_PDU_TYPE) {
        loopback_prach(&pdu->prach_pdu, frame, slot);
      } else if (pdu->pdu_type == NFAPI_NR_UL_CONFIG_PUCCH_PDU_TYPE) {
        if (pdu->pucch_pdu.rnti == trace_rnti)
          printf("%4d.%2d PUCCH format %d harq bits %d sr %d sym %d+%d\n", frame, slot, pdu->pucch_pdu.format_type, pdu->pucch_pdu.bit_len_harq,
                 pdu->pucch_pdu.sr_flag, pdu->pucch_pdu.start_symbol_index, pdu->pucch_pdu.nr_of_symbols);
        loopback_pucch(&pdu->pucch_pdu, frame, slot, dl_bler, bench_ue(ues, total, pdu->pucch_pdu.rnti), &st);
      }
    }

    cu_send_rrc_setups();

    if (++slot == slots_per_frame) {
      slot = 0;
      frame = (frame + 1) % 1024;
    }
  }

  const double sim_s = n_slots * 1e-3 / (1 << mu);
  qsort(t_sched, n_slots, sizeof(double), cmp_double);
  double mean = 0;
  for (int n = 0; n < n_slots; n++)
    mean += t_sched[n];
  mean /= n_slots;
#define P(pct) (t_sched[(int)((pct) / 100.0 * (n_slots - 1))])
  printf("\n=== Results: %d UEs, %d slots, %.2f s simulated ===\n", num_ues, n_slots, sim_s);
  const uint64_t dl_n = st.dl_tbs ? st.dl_tbs : 1, ul_n = st.ul_tbs ? st.ul_tbs : 1;
  printf("  DL: %lu TB, %.2f Mb/s new data, mean MCS %.1f RBs %.1f, ACK %lu NACK %lu\n",
         st.dl_tbs, st.dl_bytes * 8e-6 / sim_s, (double)st.dl_mcs / dl_n, (double)st.dl_rbs / dl_n, st.dl_ack, st.dl_nack);
  printf("  UL: %lu TB, %.2f Mb/s delivered (offered %.2f), mean MCS %.1f RBs %.1f, CRC errors %lu, SR %lu\n",
         st.ul_tbs, st.ul_bytes * 8e-6 / sim_s, (double)ul_bytes * registered * 8e-6 * (1 << mu) * 1e3,
         (double)st.ul_mcs / ul_n, (double)st.ul_rbs / ul_n, st.ul_crc_err, st.sr);
  if (mcs_lo >= 0) {
    /* how close link adaptation gets to the MCS of 10 % BLER, averaged over the UEs */
    double dl_gap = 0, ul_gap = 0;
    int dl_ues = 0, ul_ues = 0;
    for (int u = 0; u < registered; u++) {
      if (ues[u].dl_n) {
        dl_gap += (double)ues[u].dl_mcs / ues[u].dl_n - ues[u].mcs_target;
        dl_ues++;
      }
      if (ues[u].ul_n) {
        ul_gap += (double)ues[u].ul_mcs / ues[u].ul_n - ues[u].mcs_target;
        ul_ues++;
      }
    }
    printf("  link model: mean MCS minus target, DL %+.1f over %d UEs, UL %+.1f over %d UEs\n",
           dl_ues ? dl_gap / dl_ues : 0, dl_ues, ul_ues ? ul_gap / ul_ues : 0, ul_ues);
  }
  if (registered > 1) {
    /* per UE DL new data: Jain's index and the 5th percentile */
    double sum = 0, sum2 = 0;
    double *thr = calloc(registered, sizeof(double));
    for (int u = 0; u < registered; u++) {
      thr[u] = ues[u].dl_new_bytes * 8e-6 / sim_s;
      sum += thr[u];
      sum2 += thr[u] * thr[u];
    }
    qsort(thr, registered, sizeof(double), cmp_double);
    printf("  DL per UE: Jain %.3f, 5th percentile %.2f Mb/s, min %.2f max %.2f\n",
           sum2 > 0 ? sum * sum / (registered * sum2) : 0, thr[(int)(0.05 * (registered - 1))], thr[0], thr[registered - 1]);
    free(thr);
  }
  if (ra_ues > 0)
    printf("  RA: %lu preambles, %lu Msg3, %lu RRCSetup, %d of %d UEs connected, %lu release requests\n",
           bench_ra.preambles, bench_ra.msg3s, bench_ra.setups, (int)get_num_connected_ue(&mac->UE_info) - num_ues, ra_ues, bench_ra.released);
  if (bench_ra.collisions)
    printf("  PUSCH collisions: %lu, lost MsgA %lu, lost PUSCH of connected UEs %lu\n", bench_ra.collisions, bench_ra.collided_msga, bench_ra.collided_ue);
  if (st.pucch_csi)
    printf("  PUCCH with CSI bits, CSI not reported: %lu\n", st.pucch_csi);
  printf("  PUSCH PDUs of a UE already sending in the slot: %lu\n", st.dup_pusch);
  printf("  UEs whose SRs reached sr_TransMax %d without a UL grant: %d\n", cell->radio_config.timer_config.sr_TransMax, sr_failures);
  {
    int with_data = 0, never = 0;
    for (int u = 0; u < registered; u++) {
      if (ul_bytes == 0 || (ul_ues >= 0 && u < registered - ul_ues))
        continue;
      with_data++;
      never += ues[u].ul_n == 0;
    }
    printf("  UEs with UL data never granted a PUSCH: %d of %d\n", never, with_data);
    sr_failures += never;
  }
  if (reconfig_bytes > 0) {
    int pending = 0;
    FOR_EACH_CONNECTED_UE(&mac->UE_info, UE)
      pending += UE->reconfigCellGroup != NULL;
    printf("  RRCReconfiguration (-C %d): CellGroup switch before the end of the message %lu (UE slots), switch still pending %d UEs\n",
           reconfig_bytes, reconfig_early, pending);
  }
  if (!isnan(ph_lo))
    printf("  PUSCH needing more power than the UE has (-P): %lu of %lu\n", st.ul_power_limited, st.ul_tbs);
  {
    /* DL allocations dropped: no PDCCH candidate, no PUCCH for the HARQ-ACK (the "CCE fail" and "PUCCH fail"
     * of nrMAC_stats) */
    uint64_t cce_fail = 0, pucch_fail = 0;
    FOR_EACH_CONNECTED_UE(&mac->UE_info, UE) {
      cce_fail += UE->UE_sched_ctrl.dl_cce_fail;
      pucch_fail += UE->UE_sched_ctrl.dl_pucch_fail;
    }
    printf("  DL allocations dropped: CCE fail %lu, PUCCH fail %lu\n", cce_fail, pucch_fail);
  }
  /* with a full DL buffer every DL slot should carry a PDSCH: more than 10 % empty is a failure. Not with -C: after
     the bench's CellGroup switch both schedulers leave DL slots empty (bench model, open) */
  const bool dl_starved = reconfig_bytes == 0 && dl_slots_loaded > 0 && dl_slots_empty * 10 > dl_slots_loaded;
  if (dl_slots_loaded > 0)
    printf("  DL slots with DL data waiting and no PDSCH: %lu of %lu%s\n", dl_slots_empty, dl_slots_loaded, dl_starved ? " (FAIL: > 10 %)" : "");
  printf("  scheduler per slot (us): mean %.1f p50 %.1f p99 %.1f max %.1f\n", mean, P(50), P(99), t_sched[n_slots - 1]);
#undef P

  /* the summary is logged at analysis level */
  set_log(NR_MAC, OAILOG_INFO);
  const uint64_t findings = nr_sched_check_summary();
  printf("  checker findings: %lu (rules in the log above)\n", findings);
  const uint64_t dropped = nr_sched_check_dropped();
  if (dropped)
    printf("  checker dropped %lu PDUs: the findings are incomplete\n", dropped);

  free(t_sched);
  free(rsp);
  free(ues);
  return (fail_on_findings && (findings || dropped)) || reconfig_early || sr_failures || dl_starved ? 2 : 0;
}
