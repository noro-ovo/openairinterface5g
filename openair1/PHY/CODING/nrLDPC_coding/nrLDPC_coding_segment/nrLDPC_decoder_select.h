/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*! \file PHY/CODING/nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_decoder_select.h
 * \brief Choice of the LDPC decoder behind LDPCdecoder() in the segment coding library.
 *
 * libldpc.so contains two decoders with the same interface:
 *   fast : the vendored layered min-sum decoder (openair1/PHY/CODING/nrLDPC_decoder_fast), Qm-aware AGC, CRC-mode early
 *          stop, run-time dispatched AVX2 / AVX-512 / NEON / SVE2 kernels -- the default;
 *   oai  : OAI's original flooding decoder (openair1/PHY/CODING/nrLDPC_decoder/nrLDPC_decoder.c, unchanged).
 * The choice is made once per process, highest priority first:
 *   1. env OAI_LDPC_DECODER=fast|oai (experiments),
 *   2. config parameter nrLDPC_coding_segment.decoder (section nrLDPC_coding_segment of the gNB/UE config file, or
 *      --nrLDPC_coding_segment.decoder <fast|oai> on the command line, also in nr_ulsim / nr_dlsim / ldpctest),
 *   3. the build default (CMake NR_LDPC_DECODER_DEFAULT, "fast").
 * libldpc_oaidec.so (and libldpc_orig.so) contain OAI's decoder only; there the choice is always "oai".
 * The LLR pre-scale rule "default" of the segment decoder (nrLDPC_llr_prescale.h) depends on the selected decoder.
 *
 * Diagnostics (libldpc.so only, off unless set): env OAI_LDPC_DECODER_STATS=<file> appends, at exit, per (decoder,
 * Qm, layers, BG, R): calls, decoded, iterations, decode time (mean, percentiles) and an order-independent digest of
 * every call (inputs, outputs, return value); OAI_LDPC_DECODER_TRACE=<file> writes one byte per call ('1' decoded,
 * '0' failed) in call order.
 */

#ifndef __NRLDPC_DECODER_SELECT__H__
#define __NRLDPC_DECODER_SELECT__H__

typedef enum { NR_LDPC_DECODER_OAI = 0, NR_LDPC_DECODER_FAST = 1 } nrLDPC_decoder_id_t;

#define NR_LDPC_DEC_HIDDEN __attribute__((visibility("hidden")))

/// the decoder that LDPCdecoder() of this library runs (selected on the first call, thread-safe)
NR_LDPC_DEC_HIDDEN nrLDPC_decoder_id_t nrLDPC_decoder_selected(void);
/// "fast" / "oai"
NR_LDPC_DEC_HIDDEN const char *nrLDPC_decoder_name(nrLDPC_decoder_id_t id);
/// rule string used by the LLR pre-scale "default" for the selected decoder (see nrLDPC_llr_prescale.h)
NR_LDPC_DEC_HIDDEN const char *nrLDPC_decoder_llr_prescale_default(void);

#endif
