# Fast NR LDPC decoder (vendored)

Layered min-sum LDPC decoder for both NR base graphs (3GPP TS 38.212 §5.3.2), all 51 lifting sizes and OAI's six
(BG, R) graph configurations, int8 messages, with one kernel template instantiated for **AVX2**, **AVX-512BW**,
**NEON** (ARMv8.0 baseline), **SVE2** (vector-length agnostic) and portable C, selected at run time. It is the
default decoder of OAI's segment coding library `libldpc.so`; OAI's original decoder stays in the same library and
is selected with `nrLDPC_coding_segment.decoder = "oai"`.

| | |
|---|---|
| `upstream/` | the decoder sources, **do not edit here**: refreshed by `sync_from_upstream.sh` (provenance and file hashes in `UPSTREAM.txt`, design and validation in `upstream/README.md`) |
| `patches/` | OAI-side changes applied on top of `upstream/` by the sync script (`0001`: no unused `env_qm()` with `-DLDPC_DEC_OAI_HAS_QM`) |
| `CMakeLists.txt` | object library `ldpc_decoder_fast` (glob over `upstream/src/*.c`, hidden symbols, entry points renamed `nrLDPC_fast_LDPC*`), unit test `test_ldpc_decoder_fast` |
| `LICENSE.md` | licence note |

## How OAI uses it

* `libldpc.so` = segment layer (`nrLDPC_coding_segment`) + OAI's encoder + **both** decoders. `LDPCdecoder()` of the
  library is the selector `nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_decoder_select.c`; the choice is made once per
  process: env `OAI_LDPC_DECODER=fast|oai` > config `nrLDPC_coding_segment.decoder` (config file section or
  `--nrLDPC_coding_segment.decoder oai` on the command line of nr-softmodem, nr-uesoftmodem, nr_ulsim, nr_dlsim,
  ldpctest) > CMake `NR_LDPC_DECODER_DEFAULT` (`fast`). The active decoder and its implementation are logged at init
  (`LDPC decoder: fast (build default; implementation avx512)`).
* `libldpc_oaidec.so`: the same library with OAI's decoder only (`--loader.ldpc.shlibversion _oaidec`,
  `ldpctest -v _oaidec`); `libldpc_orig.so` unchanged (OAI's decoder + reference encoder).
* CMake `-DLDPC_DECODER_FAST=OFF` builds `libldpc.so` without the fast decoder (OAI's decoder only).
* The decoder reads `t_nrLDPC_dec_params.Qm` (`-DLDPC_DEC_OAI_HAS_QM`): its input AGC target depends on the modulation
  order; with OAI's CRC callback it uses NMS 7/8 + offset 1 and stops when the core rows are satisfied and the CRC passes.
* The segment decoder's LLR pre-scale (`nrLDPC_llr_prescale.h`) has a separate default rule for this decoder,
  `NR_LDPC_LLR_PRESCALE_DEFAULT_FAST`.
* Conventions are OAI's (input layout, fillers +127, packed output, return value `< numMaxIter` on success, abort
  flag). `numMaxIter` counts **layered** iterations (about two flooding iterations each).
* Diagnostics: `OAI_LDPC_DECODER_STATS=<file>` (per-key calls, iterations, decode time, call digest),
  `OAI_LDPC_DECODER_TRACE=<file>`; `LDPC_DEC_IMPL=scalar|avx2|avx512|neon|sve2|ref` forces a kernel. Other `LDPC_DEC_*`
  variables of `upstream/README.md` are experiment switches.

## Refreshing the sources (e.g. to v3)

```bash
cd openair1/PHY/CODING/nrLDPC_decoder_fast
./sync_from_upstream.sh --check <dec_std dir>   # what would change
./sync_from_upstream.sh <dec_std dir>           # copy include/ src/ tables/ test/ README.md, re-apply patches/
git add -A . && git commit                      # UPSTREAM.txt records the source and the file hashes
```

New or removed kernel files need no CMake change (glob). A patch that no longer applies stops the sync; a patch already
contained upstream is skipped. Then rebuild and run `test_ldpc_decoder_fast` and ldpctest with both decoders
(`ldpctest ... --nrLDPC_coding_segment.decoder fast|oai`).
