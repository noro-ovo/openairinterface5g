/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* UE NAS security mode control against an AMF that retransmits the SECURITY MODE COMMAND (T3560, 24.501 5.4.2.7):
 * each retransmission is a new NAS message, protected with the next DL NAS COUNT. The static functions of
 * nr_nas_msg.c are tested through the same calls as the DL NAS dispatch (nas_security_rx_process() then the
 * message handler). */

#include "../../nr_nas_msg.c"

configmodule_interface_t *uniqCfg = NULL;

/* UE symbols outside the NAS, on paths these tests do not run */
int NB_UE_INST = 1;
softmodem_params_t *get_softmodem_params(void)
{
  static softmodem_params_t p;
  return &p;
}
void f2345(u8 k[16], u8 rand[16], u8 res[8], u8 ck[16], u8 ik[16], u8 ak[6], const u8 op_c[16])
{
  abort();
}
void nr_sdap_qos_rule_add(ue_id_t ue_id,
                          int pdusession_id,
                          uint8_t rule_id,
                          uint8_t qfi,
                          uint8_t precedence,
                          bool is_default,
                          const packet_filter_decoded_t *pf_list,
                          int num_pf)
{
  abort();
}
void nr_sdap_qos_rule_remove(ue_id_t ue_id, int pdusession_id, uint8_t rule_id)
{
  abort();
}
void nr_sdap_qos_rule_update(ue_id_t ue_id,
                             int pdusession_id,
                             uint8_t rule_id,
                             uint8_t qfi,
                             uint8_t precedence,
                             bool is_default,
                             const packet_filter_decoded_t *pf_list,
                             int num_pf,
                             bool replace)
{
  abort();
}
void nr_sdap_qos_rule_delete_pf(ue_id_t ue_id, int pdusession_id, uint8_t rule_id, const uint8_t *pf_ids, int num_ids)
{
  abort();
}

void exit_function(const char *file, const char *function, const int line, const char *s, const int assert)
{
  UNUSED(assert);
  printf("detected error at %s:%d:%s: %s\n", file, line, function, s);
  abort();
}

static int failures;
#define CHECK(c)                                          \
  do {                                                    \
    if (!(c)) {                                           \
      printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
      failures++;                                         \
    }                                                     \
  } while (0)

/* the AMF side: keys of the new context, NIA2 and NEA0, as derived by the AMF from KAMF */
static uint8_t amf_knas_enc[16], amf_knas_int[16];

/* protect a plain 5GMM message: [EPD][SHT][MAC0..3][SN][plain], MAC over [SN][plain] at DL NAS COUNT = count */
static int amf_protect(int sht, uint32_t count, const uint8_t *plain, int len, uint8_t *out)
{
  out[0] = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  out[1] = sht;
  out[6] = count & 0xff;
  memcpy(out + 7, plain, len);
  stream_security_container_t *c = stream_security_container_init(0, 2, amf_knas_enc, amf_knas_int);
  nas_stream_cipher_t s =
      {.context = c->integrity_context, .count = count, .bearer = 1, .direction = 1, .message = out + 6, .blength = (len + 1) << 3};
  stream_compute_integrity(c->integrity_algorithm, &s, out + 2);
  stream_security_container_delete(c);
  return len + 7;
}

static int amf_smc(uint32_t count, uint8_t *out)
{
  /* Security Mode Command: header, selected algorithms NEA0/NIA2, ngKSI 0, replayed UE security capabilities */
  const uint8_t plain[] = {FGS_MOBILITY_MANAGEMENT_MESSAGE, PLAIN_5GS_MSG, FGS_SECURITY_MODE_COMMAND, 0x02, 0x00, 0x02, 0xe0, 0xe0};
  return amf_protect(INTEGRITY_PROTECTED_WITH_NEW_SECU_CTX, count, plain, sizeof(plain), out);
}

/* a DL message as the UE dispatch handles it: returns the security state, runs the SMC handler for an SMC */
static security_state_t ue_rx(nr_ue_nas_t *nas, uint8_t *buf, int len, as_nas_info_t *reply)
{
  byte_array_t ba = {.buf = buf, .len = len};
  security_state_t st = nas_security_rx_process(nas, ba);
  *reply = (as_nas_info_t){0};
  if (st <= NAS_SECURITY_INTEGRITY_PASSED && get_msg_type(buf, len) == FGS_SECURITY_MODE_COMMAND)
    handle_security_mode_command(nas, reply, buf, len);
  return st;
}

static bool is_smc_complete(const as_nas_info_t *m)
{
  return m->length > 10 && m->nas_data[1] == INTEGRITY_PROTECTED_AND_CIPHERED_WITH_NEW_SECU_CTX;
}

static bool is_smc_reject(const as_nas_info_t *m)
{
  /* plain without a context, protected with the current context otherwise (NEA0 here: the payload is readable) */
  if (m->length >= 3 && m->nas_data[1] == PLAIN_5GS_MSG)
    return m->nas_data[2] == FGS_SECURITY_MODE_REJECT;
  return m->length >= 10 && m->nas_data[1] == INTEGRITY_PROTECTED_AND_CIPHERED && m->nas_data[9] == FGS_SECURITY_MODE_REJECT;
}

static nr_ue_nas_t *new_ue(void)
{
  static uicc_t uicc = {.imsiStr = "001010000000001",
                        .imeisvStr = "6754567890123413",
                        .routing_indicatorStr = "0000",
                        .nmc_size = 2};
  nr_ue_nas_t *nas = calloc_or_fail(1, sizeof(*nas));
  nas->uicc = &uicc;
  for (int i = 0; i < 32; i++)
    nas->security.kamf[i] = i * 7 + 1;
  return nas;
}

/* the AMF does not get the SECURITY MODE COMPLETE in time and retransmits the SMC with the next DL NAS COUNT */
static void test_smc_retransmission(void)
{
  nr_ue_nas_t *nas = new_ue();
  uint8_t m[64];
  as_nas_info_t r;

  int l = amf_smc(0, m);
  CHECK(ue_rx(nas, m, l, &r) <= NAS_SECURITY_INTEGRITY_PASSED);
  CHECK(is_smc_complete(&r));
  free(r.nas_data);

  l = amf_smc(1, m);
  CHECK(ue_rx(nas, m, l, &r) <= NAS_SECURITY_INTEGRITY_PASSED);
  CHECK(is_smc_complete(&r));
  CHECK(nas->security_container != NULL);
  free(r.nas_data);

  /* the Registration Accept that follows, integrity protected and ciphered (NEA0) at the next count */
  const uint8_t ra[] = {FGS_MOBILITY_MANAGEMENT_MESSAGE, PLAIN_5GS_MSG, FGS_REGISTRATION_ACCEPT, 0x01, 0x01};
  l = amf_protect(INTEGRITY_PROTECTED_AND_CIPHERED, 2, ra, sizeof(ra), m);
  CHECK(ue_rx(nas, m, l, &r) == NAS_SECURITY_INTEGRITY_PASSED);
}

/* an SMC that fails its MAC check is rejected and the current context stays in use (24.501 5.4.2.5) */
static void test_smc_bad_mac_keeps_context(void)
{
  nr_ue_nas_t *nas = new_ue();
  uint8_t m[64];
  as_nas_info_t r;

  int l = amf_smc(0, m);
  ue_rx(nas, m, l, &r);
  CHECK(is_smc_complete(&r));
  free(r.nas_data);

  l = amf_smc(1, m);
  m[2] ^= 0xff;
  ue_rx(nas, m, l, &r);
  CHECK(is_smc_reject(&r));
  CHECK(nas->security_container != NULL);
  free(r.nas_data);

  const uint8_t ra[] = {FGS_MOBILITY_MANAGEMENT_MESSAGE, PLAIN_5GS_MSG, FGS_REGISTRATION_ACCEPT, 0x01, 0x01};
  l = amf_protect(INTEGRITY_PROTECTED_AND_CIPHERED, 1, ra, sizeof(ra), m);
  CHECK(ue_rx(nas, m, l, &r) == NAS_SECURITY_INTEGRITY_PASSED);
}

/* without a security context, a protected message cannot be verified: it is not handed to the message handlers */
static void test_protected_without_context(void)
{
  nr_ue_nas_t *nas = new_ue();
  uint8_t m[64];
  as_nas_info_t r;
  const uint8_t ra[] = {FGS_MOBILITY_MANAGEMENT_MESSAGE, PLAIN_5GS_MSG, FGS_REGISTRATION_ACCEPT, 0x01, 0x01};
  int l = amf_protect(INTEGRITY_PROTECTED_AND_CIPHERED, 0, ra, sizeof(ra), m);
  CHECK(ue_rx(nas, m, l, &r) > NAS_SECURITY_INTEGRITY_PASSED);
}

int main(void)
{
  logInit();
  itti_init(TASK_MAX, tasks_info);
  derive_knas(0x01, 0, new_ue()->security.kamf, amf_knas_enc);
  derive_knas(0x02, 2, new_ue()->security.kamf, amf_knas_int);
  test_smc_retransmission();
  test_smc_bad_mac_keeps_context();
  test_protected_without_context();
  printf("%s: %d failures\n", failures ? "FAILED" : "PASSED", failures);
  return failures != 0;
}
