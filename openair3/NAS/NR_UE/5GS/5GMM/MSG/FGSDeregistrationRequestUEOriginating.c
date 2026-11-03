/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "FGSDeregistrationRequestUEOriginating.h"
#include <stdint.h>

#define IEI_NULL 0x00

int encode_fgs_deregistration_request_ue_originating(const fgs_deregistration_request_ue_originating_msg *drr,
                                                     uint8_t *buffer,
                                                     uint32_t len)
{
  int encoded = 0;
  const FGSDeregistrationType *dt = &drr->deregistrationtype;
  // ngKSI in bits 5 to 8, de-registration type in bits 1 to 4 (TS 24.501 8.2.12.1)
  *(buffer + encoded) = ((encode_nas_key_set_identifier(&drr->naskeysetidentifier, IEI_NULL) & 0x0f) << 4)
                        | ((dt->switchoff & 0x1) << 3) | ((dt->reregistration_required & 0x1) << 2) | (dt->access_type & 0x3);
  encoded++;

  int encode_result;

  if ((encode_result = encode_5gs_mobile_identity(&drr->fgsmobileidentity, 0, buffer + encoded, len - encoded)) < 0)
    return encode_result;
  else
    encoded += encode_result;

  return encoded;
}
