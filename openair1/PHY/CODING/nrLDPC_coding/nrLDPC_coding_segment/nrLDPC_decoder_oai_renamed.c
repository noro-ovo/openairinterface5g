/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*! \file PHY/CODING/nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_decoder_oai_renamed.c
 * \brief OAI's original LDPC decoder (nrLDPC_decoder/nrLDPC_decoder.c, unchanged) under hidden, renamed entry points,
 * so that libldpc.so can contain it next to the fast decoder; LDPCdecoder() of libldpc.so selects one of them
 * (nrLDPC_decoder_select.c).
 */

#include "PHY/CODING/nrLDPC_defs.h"

#define LDPCdecoder nrLDPC_oai_LDPCdecoder
#define LDPCinit nrLDPC_oai_LDPCinit
#define LDPCshutdown nrLDPC_oai_LDPCshutdown
/* hidden: the definitions below inherit the visibility of these first declarations */
__attribute__((visibility("hidden"))) LDPC_initfunc_t LDPCinit;
__attribute__((visibility("hidden"))) LDPC_shutdownfunc_t LDPCshutdown;
__attribute__((visibility("hidden"))) LDPC_decoderfunc_t LDPCdecoder;
#include "PHY/CODING/nrLDPC_decoder/nrLDPC_decoder.c"
