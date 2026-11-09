/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "gtest/gtest.h"
extern "C" {
#include "common/platform_types.h"
#include "openair2/LAYER2/NR_MAC_UE/mac_proto.h"
#include "executables/softmodem-common.h"
#include "openair2/LAYER2/nr_rlc/nr_rlc_oai_api.h"
#include "common/utils/ocp_itti/intertask_interface.h"

static softmodem_params_t softmodem_params;
softmodem_params_t *get_softmodem_params(void)
{
  return &softmodem_params;
}
void nr_mac_rrc_ra_ind(const module_id_t mod_id, bool success)
{
  UNUSED(mod_id);
  UNUSED(success);
}
void nr_mac_rrc_msg3_ind(const module_id_t mod_id, const int rnti, bool prepare_payload)
{
  UNUSED(mod_id);
  UNUSED(rnti);
  UNUSED(prepare_payload);
}
void nr_mac_rlc_status_ind(uint16_t ue_id, frame_t frame, int n_ch, const logical_chan_id_t *ch, mac_rlc_status_resp_t *ret)
{
  UNUSED(ue_id);
  UNUSED(frame);
  UNUSED(n_ch);
  UNUSED(ch);
  UNUSED(ret);
}
void nr_mac_rrc_inactivity_timer_ind(const module_id_t mod_id)
{
  UNUSED(mod_id);
}
tbs_size_t nr_mac_rlc_data_req(const module_id_t module_idP,
                               const uint16_t ue_id,
                               const bool gnb_flagP,
                               const logical_chan_id_t channel_idP,
                               const tb_size_t tb_sizeP,
                               char *buffer_pP)
{
  UNUSED(module_idP);
  UNUSED(ue_id);
  UNUSED(gnb_flagP);
  UNUSED(channel_idP);
  UNUSED(tb_sizeP);
  UNUSED(buffer_pP);
  return 0;
}
void nr_mac_rlc_data_ind(const module_id_t module_idP,
                         const uint16_t ue_id,
                         const bool gnb_flagP,
                         const nr_rlc_data_ind_t *data,
                         int num_data)
{
  UNUSED(module_idP);
  UNUSED(ue_id);
  UNUSED(gnb_flagP);
  UNUSED(data);
  UNUSED(num_data);
}
void nr_mac_rrc_verification_failed(const module_id_t mod_id)
{
  UNUSED(mod_id);
}
bool nr_rlc_activate_srb0(int ue_id,
                          void *data,
                          void (*send_initial_ul_rrc_message)(int ue_id, const uint8_t *sdu, sdu_size_t sdu_len, void *data))
{
  UNUSED(ue_id);
  UNUSED(data);
  UNUSED(send_initial_ul_rrc_message);
  return true;
}
int nr_rlc_module_init(nr_rlc_op_mode_t mode)
{
  UNUSED(mode);
  return 0;
}
MessageDef *itti_alloc_new_message(task_id_t origin_task_id, instance_t originInstance, MessagesIds message_id)
{
  UNUSED(origin_task_id);
  UNUSED(originInstance);
  UNUSED(message_id);
  return NULL;
}
int itti_send_msg_to_task(task_id_t task_id, instance_t instance, MessageDef *message)
{
  UNUSED(task_id);
  UNUSED(instance);
  UNUSED(message);
  return 0;
}
typedef uint32_t channel_t;
int8_t nr_mac_rrc_data_ind_ue(const module_id_t module_id,
                              const int CC_id,
                              const uint8_t gNB_index,
                              const int hfn,
                              const frame_t frame,
                              const int slot,
                              const rnti_t rnti,
                              const uint32_t cellid,
                              const long arfcn,
                              const channel_t channel,
                              const uint8_t *pduP,
                              const sdu_size_t pdu_len)
{
  UNUSED(module_id);
  UNUSED(CC_id);
  UNUSED(gNB_index);
  UNUSED(hfn);
  UNUSED(frame);
  UNUSED(slot);
  UNUSED(rnti);
  UNUSED(cellid);
  UNUSED(arfcn);
  UNUSED(channel);
  UNUSED(pduP);
  UNUSED(pdu_len);
  return 0;
}
bool check_csi_report_consistency(const NR_CSI_MeasConfig_t *meas)
{
  UNUSED(meas);
  return true;
}
void nr_mac_rrc_meas_ind_ue(module_id_t module_id,
                            uint32_t gNB_index,
                            uint16_t Nid_cell,
                            bool csi_meas,
                            bool is_neighboring_cell,
                            int rsrp_dBm)
{
  UNUSED(module_id);
  UNUSED(gNB_index);
  UNUSED(Nid_cell);
  UNUSED(csi_meas);
  UNUSED(is_neighboring_cell);
  UNUSED(rsrp_dBm);
}
}
#include "common/utils/LOG/log.h"

/* PUCCH format 2 resource with 4 PRBs, 1 symbol and maxCodeRate 0.15: 4 * 8 * 1 * 2 * 0.15 = 9.6 UCI bits (no CRC below 12) */
class pucch_f2_overflow : public ::testing::Test {
 protected:
  NR_UE_MAC_INST_t mac = {};
  NR_UE_UL_BWP_t ul_bwp = {};
  NR_PUCCH_Config_t pucch_config = {};
  NR_PUCCH_ConfigCommon_t pucch_config_common = {};
  struct NR_SetupRelease_PUCCH_FormatConfig format2 = {};
  NR_PUCCH_FormatConfig_t format2_config = {};
  NR_PUCCH_MaxCodeRate_t max_code_rate = NR_PUCCH_MaxCodeRate_zeroDot15;
  NR_PUCCH_Resource_t res = {};
  NR_PUCCH_format2_t f2 = {};
  void SetUp() override
  {
    mac.current_UL_BWP = &ul_bwp;
    mac.pdsch_HARQ_ACK_Codebook = NR_PhysicalCellGroupConfig__pdsch_HARQ_ACK_Codebook_dynamic;
    mac.frame_structure.numb_slots_frame = 20;
    mac.frame_structure.frame_type = TDD;
    mac.nr_band = 78;
    mac.frequency_range = FR1;
    mac.p_Max = 23;
    ul_bwp.scs = 1;
    ul_bwp.BWPSize = 106;
    ul_bwp.pucch_Config = &pucch_config;
    ul_bwp.pucch_ConfigCommon = &pucch_config_common;
    format2_config.maxCodeRate = &max_code_rate;
    format2.present = NR_SetupRelease_PUCCH_FormatConfig_PR_setup;
    format2.choice.setup = &format2_config;
    pucch_config.format2 = &format2;
    f2.nrofPRBs = 4;
    f2.nrofSymbols = 1;
    f2.startingSymbolIndex = 13;
    res.pucch_ResourceId = 5;
    res.startingPRB = 16;
    res.format.present = NR_PUCCH_Resource__format_PR_format2;
    res.format.choice.format2 = &f2;
  }
  PUCCH_sched_t uci(int n_harq, int n_sr, int n_csi)
  {
    PUCCH_sched_t p = {};
    p.pucch_resource = &res;
    p.initial_pucch_id = -1;
    p.n_harq = n_harq;
    p.ack_payload = (1 << n_harq) - 1;
    p.n_sr = n_sr;
    p.sr_payload = n_sr;
    p.csi_payload.p1_bits = n_csi;
    p.csi_payload.part1_payload = 0x55;
    return p;
  }
};

/* 2 HARQ-ACK + 1 SR + 7 CSI = 10 bits > 9.6: the UE drops the CSI (38.213 9.2.5.2) and sends HARQ-ACK and SR */
TEST_F(pucch_f2_overflow, csi_dropped_when_harq_sr_csi_exceed_resource)
{
  PUCCH_sched_t p = uci(2, 1, 7);
  fapi_nr_ul_config_pucch_pdu pdu = {};
  EXPECT_EQ(nr_ue_configure_pucch(&mac, 8, 100, 0x2ce8, &p, &pdu), 0);
  EXPECT_EQ(pdu.format_type, 2);
  EXPECT_EQ(pdu.n_bit, 3);
  EXPECT_EQ(pdu.payload, (1ULL << 2) | 0x3);
  EXPECT_EQ(p.csi_payload.p1_bits, 0);
}

/* 1 HARQ-ACK + 1 SR + 7 CSI = 9 bits <= 9.6: everything is sent */
TEST_F(pucch_f2_overflow, csi_kept_when_it_fits)
{
  PUCCH_sched_t p = uci(1, 1, 7);
  fapi_nr_ul_config_pucch_pdu pdu = {};
  EXPECT_EQ(nr_ue_configure_pucch(&mac, 8, 100, 0x2ce8, &p, &pdu), 0);
  EXPECT_EQ(pdu.format_type, 2);
  EXPECT_EQ(pdu.n_bit, 9);
  EXPECT_EQ(pdu.payload, (0x55ULL << 2) | (1ULL << 1) | 0x1);
  EXPECT_EQ(pdu.prb_size, 4);
}

int main(int argc, char **argv)
{
  logInit();
  configmodule_interface_t *uniqCfg = load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY);
  g_log->log_component[NR_MAC].level = OAILOG_INFO;
  testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  end_configmodule(uniqCfg);
  return ret;
}
