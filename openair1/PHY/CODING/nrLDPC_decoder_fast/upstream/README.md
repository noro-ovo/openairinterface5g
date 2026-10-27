# dec_std — 5G NR LDPC decoder (3GPP TS 38.212 §5.3.2), layered min-sum, NEON / SVE2 / AVX2 / AVX-512

A from-scratch LDPC decoder for both NR base graphs, all 51 lifting sizes and OAI's six (BG, R) rate
configurations, with one algorithm and one data layout shared by every SIMD target:

* **layered offset min-sum** (row-by-row; per-(BG, R) offset), int8 messages, saturating arithmetic;
* **one pass per row and superchunk**: all Q = APP - R of a superchunk stay in registers (row functions
  specialised per row shape), no Q store/reload; each column alternates between 2 (even degree) or 3 (odd degree)
  buffers, so reads (unaligned runs at a precomputed offset) and aligned writes never overlap; only the first
  B bytes are stored a second time (the wrap pad) instead of a doubled column;
* degree-1 (extension parity) columns: no APP stored, recomputed lazily from the row state for the syndrome/output;
* **AGC**: per-block input gain to a target mean |LLR| (statistic over the **non-zero** values below the int8
  rails), fused into the input copy; the target depends on (BG, R), on the **code rate actually transmitted** and
  optionally on the modulation order Qm: robust from 0.25x to 4x the ldpctest LLR scale and on OAI PHY LLRs;
* **only the rows whose degree-1 column received LLRs are decoded** (OAI selects R1/3 for every rate < 2/3 and
  zero-fills the tail; those rows are exact no-ops): -24 % .. -49 % instructions per iteration at code rates
  0.45 .. 0.65;
* early termination on the full syndrome; **with a CRC callback (OAI's PHY): core-row parity + CRC**, also after
  the core rows of an iteration; in that mode the default correction is NMS 7/8 + offset 1 (scale-tolerant on
  demapper LLRs);
* kernels from one template for **NEON** (ARMv8.0 baseline), **SVE2** (vector-length agnostic, run-time
  selected; default on aarch64 whenever present, also at VL 128), **AVX2**, **AVX-512BW**, plus a portable C
  instantiation and an independent scalar reference;
* an OAI drop-in entry point `LDPCdecoder(t_nrLDPC_dec_params*, int8_t*, uint8_t*, t_nrLDPC_time_stats*,
  decode_abort_t*)` with OAI's conventions (input layout, filler +127, output packing, return code, abort flag).

This is the speed-up version of the decoder delivered in the ldpc_orion package (2026-10-23). What changed, why,
and all before/after numbers: `/home/user/ldpc_speedup/README.md` (in the orion package: `../README.md`).

## 0. Results at a glance
**Correctness** (`results/`): bit-exact against the scalar reference for every implementation and unroll factor
(x86: scalar-C, AVX2, AVX-512, 6000 random blocks; aarch64 under qemu: scalar-C, NEON, SVE2 at VL 128/256/512,
3000 blocks each, Cortex-A53 model without SVE), incl. all 51 Z, 6 (BG, R), fillers (also not at +127), pure
noise with -128, LLR scales x1/16 .. x8, 1..20 iterations, AGC on/off/default, CRC mode with an oracle and a
hash CRC, 3 output modes, other corrections; OAI entry == own API (1500 calls); spec/OAI tables and encoder.

**BLER** (Eb/N0 at BLER 1e-2, Z 384, 4000 blocks/point; old = previous version): BG1 R1/3 1.21 dB (old 1.21),
BG2 R1/5 1.12 dB (1.11), BG1 R8/9 3.95 dB (3.94); CRC mode identical. With the AGC the 1e-2 point moves by at
most 0.16 dB (all six (BG, R)) between 0.25x and 4x the ldpctest scale, where the previous version lost up to 5 dB; on replayed OAI PHY LLRs (nr_ulsim, 17 operating points) TB BLER geo-mean 0.022
(previous 0.074, OAI 0.357).

**Speed**: per BG1 R1/3 iteration 148 k aarch64 instructions for SVE2 (previous 206 k) and 150 k for NEON (198 k);
estimated A720 time per block in scenario A (BG1 R1/3 7.8 dB) 45-47 us for SVE2 (measured previous: 61.4 us) and
27-30 us in CRC mode (estimates, see the top-level README).

## 1. Layout of this directory
| path | content |
|---|---|
| `include/ldpc_dec.h` | own API (`ldpc_decode`, implementation selection, parameters, AGC switch) |
| `include/ldpc_dec_oai.h`, `src/ldpc_oai_entry.c` | OAI-signature entry points `LDPCdecoder`, `LDPCinit`, `LDPCshutdown` (unchanged) |
| `include/ldpc_tables.h`, `tables/ldpc_tables.c` | base-graph tables (generated) |
| `tables/gen_tables.py` | table generator + cross-checks (spec CSVs, srsRAN, OAI) |
| `src/ldpc_dec.c` | plan (schedule, buffers, offset tables, core-row view), dispatch, workspace, AGC + input conditioning, output, driver (incl. CRC-mode stop) |
| `src/kern_tmpl.h` | the kernel: one row function per (row shape, vectors per step, keep-|x-R|, first iteration) + syndrome, on a small vector-op vocabulary |
| `src/kern_{neon,sve2,avx2,avx512,scalar}.c` | instantiations of the template |
| `src/ldpc_ref.c` | scalar reference of the algorithm, natural indexing, independent of the SIMD layout |
| `test/test_dec.c`, `test/tutil.[ch]` | correctness tests; spec encoder, independent syndrome, AWGN channel (also scaled) |
| `test/oai/` | OAI comparison harness (`cmp_oai`: tables/encoder, OAI entry, BLER, all-Z sweep, bench incl. `--crc`, `--scan` LLR-scale sweeps) + build script |
| `bench/` | own-API benchmark and scripts of the previous version |
| `results/` | raw outputs of the runs quoted in the top-level README |
| `Makefile` | `make` (x86-64 native) / `make ARCH=aarch64` (cross, static, for qemu) |

Build and run: as before (`make`, `make ARCH=aarch64`, `test/oai/build.sh`); `cmp_oai --scan` and `--bench ... --crc`
are new. Environment switches for experiments: `LDPC_DEC_IMPL`, `LDPC_DEC_U` (as before), `LDPC_DEC_ALG=ns,beta,rmax,insh`
(replaces the per-(BG, R) defaults), `LDPC_DEC_AGC=0|1`, `LDPC_DEC_AGCT=<percent of the targets>`,
`LDPC_DEC_AGCSTAT=0|1` (mean of all non-zero |x| / of 0 < |x| < 127), `LDPC_DEC_QMT=u,q2,q4,q6,q8` and
`LDPC_DEC_QMTN=...` (target multipliers per Qm in percent, offset / CRC-mode NMS), `LDPC_DEC_RSLOPE=a,b,c,d,e,f`
(rate slopes per class), `LDPC_DEC_CRCALG=0` (offset defaults also in CRC mode), `LDPC_DEC_QM=<qm>` (OAI entry:
modulation order when the caller passes none), `LDPC_DEC_ALLROWS=1` (decode every row; measurements only),
`LDPC_DEC_AGCZERO=1` (v1 statistic with zeros counted; experiments only).

## 2. Base-graph tables and how they were validated

`tables/gen_tables.py` reads the **spec** as primary source: `spec/bg1_Vij.csv`, `spec/bg2_Vij.csv` (Tables
5.3.2-2/-3 extracted from the official 38.212 .docx) and Table 5.3.2-1, and emits `tables/ldpc_tables.c`: the edge
list (row, column) of each base graph in row-major order with the raw V_ij for the 8 set indices (316 + 197
entries, P_ij = V_ij mod Z is computed when a plan is built), the 51 lifting sizes and the set index of each Z.
The generator refuses to write the file unless all of the following agree:

| check | result |
|---|---|
| raw table extracts of five releases (38212-fd0, gf0, hd0, i80, j40) byte-identical | identical |
| raw Table 5.3.2-2/-3 extracts == flattened `bg*_Vij.csv` | identical |
| srsRAN_Project (git clone, main 4bf15439, `ldpc_luts_impl.cpp`, raw V_ij per iLS): same edge sets, same V_ij | 316 x 8 and 197 x 8 identical |
| OAI 9313cbea decoder LUTs (`nrLDPC_lut.h`: `circShift_BG*_Z*_CNG*` + `posBnInCnProcBuf_*`): every OAI check node matched to a spec row, shift == V_ij mod Z | 16 116 (BG1) + 10 047 (BG2) (edge, Z) values identical, all 51 Z |

Encoder-level checks (`test/tutil.c` encodes from the spec: GF(2) Gauss-Jordan inverse of the 4Z x 4Z core
parity block, then the extension parity row by row; `t_syndrome` evaluates H c directly from the table):

* `test_dec` [1]: every Z x both BG, random information bits: spec-encoder codeword satisfies H c = 0 (all 46/42 rows).
* `cmp_oai --tables`: OAI's encoder (`LDPCencoder`, `ldpc_encoder_optim8segmulti.c`) output for all 51 Z x 2 BG,
  with and without filler bits (408 codewords): **bit-identical to the spec encoder and H c = 0** with the
  spec-table syndrome. No discrepancy between OAI and the spec was found for the tables or the encoder.

## 3. Conventions (OAI-compatible)

* Input: `int8` LLRs, positive = bit 0, `llr[c*Z + j]` for column c = 0 .. ncol-1 with ncol = 68/35/27 (BG1 R 1/3,
  2/3, 8/9 = OAI R codes 13/23/89) and 52/32/17 (BG2 R 1/5, 1/3, 2/3 = 15/13/23); the first 2Z LLRs (punctured
  systematic columns) are 0, filler positions K'..K-1 are +127 (OAI's `nr_process_decode_segment` convention).
  The input buffer is not modified.
* Rows used: rows 0 .. ncol-Kb-1 of the base graph (46/13/5 for BG1, 42/22/7 for BG2), i.e. the parity-check
  matrix of the transmitted part, as in OAI.
* Output (`LDPC_OUT_BITS`, OAI `nrLDPC_outMode_BIT`): the K = Kb*Z systematic bits (Kb = 22 / 10, fillers
  included, they decode to 0), packed MSB-first: byte b bit 7 = bit 8b; ceil(K/8) bytes are written (OAI's caller
  copies K>>3 bytes). `BITINT8`: K bytes 0/1. `LLRINT8`: the ncol*Z a-posteriori LLRs in input order.
* OAI entry `LDPCdecoder`: `numMaxIter` = maximum number of **layered** iterations; returns iterations-1 on success
  and `numMaxIter` on failure, setting the abort flag (as OAI does); returns `numMaxIter` at once if the abort flag
  is already set (polled before each iteration). With `check_crc` the success criterion is OAI's: CRC of the
  K' decoded bits. For BG1 R1/3, R2/3 and BG2 R1/5, R1/3 the decoder stops when the core rows 0-3 are satisfied and
  the CRC passes (checked after every iteration and after the core rows of iterations >= 2, section 4); for the other
  rates it stops on a zero syndrome and evaluates the CRC then or after the last iteration. Kprime is also used by the
  AGC (filler range).
* Compiled inside OAI with `-DLDPC_DEC_OAI_HEADERS` the entry uses OAI's own `nrLDPC_types.h` / `coding_defs.h`;
  standalone it uses layout-identical mirrors (`include/ldpc_dec_oai.h`). The harness builds it against OAI's
  headers and checks it against the own API (`cmp_oai --entry`).

## 4. Algorithm

**Layered (row-serial) offset min-sum.** Layers are the base-graph rows 0..M-1 in natural order; one layer
processes its Z checks in parallel (a row touches every column at most once, so the Z checks of a row are
independent). For check t of row i and every edge (i, c) with shift P (H block = identity shifted right by P):

```
Q_e   = sat8(APP_c[(t+P) mod Z] - R_e[t])        (degree-1 columns: Q_e = channel LLR, exactly)
m1,m2 = two smallest |Q_e| (|-128| = 128),  s = parity of (Q_e < 0)
R_e[t] = (-1)^(s xor [Q_e<0]) * corr(|Q_e| == m1 ? m2 : m1)
APP_c[(t+P) mod Z] = sat8(Q_e + R_e[t])
corr(m) = clamp(m - beta, 0, rmax)                (offset min-sum, |messages| <= rmax)
```

`|Q_e| == m1 ? m2 : m1` needs no index of the minimum: with ties m2 = m1, so the result is exact.
Layered scheduling uses each new APP in the very next layer and converges in about half the iterations of
flooding (OAI). After each iteration the full syndrome of the hard decision is evaluated (row by row, leaving at
the first unsatisfied row, so it is cheap while the decoder is still far from a codeword).

**Fixed-point design (why rmax and the input scaling matter).** With int8 a-posteriori values the classic
layered-decoder failure is APP saturation: a saturated APP (+127) has lost part of its sum; if the new message
of a layer is smaller than the old one, the APP drops by the difference and never recovers what was cut off.
Unprotected, this made the decoder *leave* valid codewords after convergence (block error rate ~10 % at high
SNR in the first version). The fix used here is the usual "two bits of headroom": messages are clamped to
|R| <= rmax = 47, so for a saturated APP Q = 127 - R_old >= 80 and the new message (>= min(80, others) - beta)
does not fall below rmax because of the saturation itself (condition 127 - rmax - beta >= rmax). Channel LLRs
are optionally halved (in_shift 1, `(llr+1)>>1`) to leave more room for the messages.

**Parameters** (offset beta, input scaling; normalised min-sum `m - (m >> n)` is also implemented) were tuned on
ldpctest's LLR quantisation (step sigma/16) at BLER ~1e-2 for each (BG, R), Z = 384, 8 iterations
(raw sweeps in `results/param_sweep.txt`). Normalised min-sum with truncating integer scaling barely changes small
magnitudes and behaved like plain min-sum (much worse); the best offset decreases with the code rate:

| (BG, R) | BG1 1/3 | BG1 2/3 | BG1 8/9 | BG2 1/5 | BG2 1/3 | BG2 2/3 |
|---|---|---|---|---|---|---|
| channel scaling | /2 | 1 | 1 | /2 | /2 | 1 |
| beta (in input LLR units) | 2 (=4) | 3 | 1 | 2 (=4) | 2 (=4) | 2 |

The defaults can be overridden per call (`nshift`, `beta`, `rmax`, `in_shift` in `ldpc_dec_params_t`). They depend
on the LLR scale of the demodulator; with OAI's PHY scaling they should be re-checked (section 9).

**Changes in this version (algorithm level).**

* |Q| is saturated to 127 in the minimum search (SQABS semantics). Outputs differ from |-128| = 128 only if
  corr(127) != corr(128), i.e. nshift = 0, beta >= 1 and rmax > 127 - beta: never with the default parameters
  (rmax 47), so the default decoder is bit-identical to the previous one apart from the AGC below.
* **AGC** (from the dec_std follow-up, `/home/user/ldpc_scratch/dec_std_followup`): per block, gain
  g = T / mean|x| with the per-(BG, R) base targets T = 18 / 25 / 34 (BG1 R1/3, 2/3, 8/9), 16 / 18 / 25 (BG2 R1/5,
  1/3, 2/3) of the follow-up, clamped to [1/8, 8], represented as G / 2^sh (G in [64, 127]); chosen with integer
  arithmetic only (identical on every platform). The mean is taken over the transmitted, non-filler positions,
  **only over the non-zero values below the int8 rails (0 < |x| < 127)** and, for blocks of >= 8 KB, over every
  other 16-byte group (cost). Zeros carry no scale information: v2 fixes v1, which counted the zero-filled tail
  that OAI leaves when it selects R1/3 for rates up to 2/3 (30-55 % of the positions), so the gain depended on the
  rate-matching fill. **Target (v2)** = T x m_rate x m_Qm: m_rate = 1 + s (r - r_class), r = K' / (non-zero span
  beyond 2Z minus fillers), r_class = Kb / (ncol - 2) of the R code, slopes s = 3.0 / 4.3 / 0 (BG1 R1/3, 2/3, 8/9),
  1.5 / 4.5 / 2.5 (BG2 R1/5, 1/3, 2/3), fitted on ldpctest with zero tails (a block that leaves part of its
  graph untransmitted runs at a higher SNR, where the offset needs a hotter input); m_Qm from `qm` (own API) or
  `LDPCdecoder_set_qm()` / `-DLDPC_DEC_OAI_HAS_QM` (OAI entry): offset defaults 1.25 / 1 / 1 / 1.5 for Qm 2 / 4 / 6 / 8,
  CRC-mode NMS 0.8 / 1 / 1 / 2, unknown or BPSK-like (qm 0, 1): 1, fitted on 44 replayed nr_ulsim operating points. Applied with in_shift folded in (one rounding): L = clamp(sat8((x G + 2^(t-1)) >> t), +-cmax),
  t = sh + in_shift. Fillers [kprime, K) bypass the gain (+127). Default on; `agc` = -1 or `LDPC_DEC_AGC=0` gives
  the previous input path. Why the unsaturated mean: OAI's PHY LLRs of 16/64/256QAM have 13-49 % of the values at the
  rails; the plain mean then scales them down too much (16QAM wants x0.4, 256QAM x1.5 at similar mean |x|); on
  unsaturated (ldpctest) input both statistics are identical.
* **Active rows (v2)**: rows beyond the last non-zero input LLR (their degree-1 column received nothing) are not
  processed and not checked. Such a row is an exact no-op: its degree-1 Q is 0, so m1 = 0 and every message to the
  other columns is 0; its check can always be satisfied by its own parity bit. Every column of rows 0..3 has at
  least two edges in rows 0..3, so column kinds and clamps are those of the full graph; the plan is built for the
  active rows (cached per thread, 8 plans). The APP of a skipped degree-1 column is reported as its channel value 0
  (LLR output). Compared with v1, decisions are identical; the full-syndrome stop can only come earlier (v1 also
  checked the skipped rows, whose parity bit can be 0 with a zero message).
* **CRC-mode algorithm (v2)**: with a CRC callback and default parameters, NMS 7/8 + offset 1 (nshift 3, beta 1,
  rmax 47, input halved) instead of the per-(BG, R) offset defaults: on 44 replayed OAI PHY operating points TB BLER
  geo-mean 0.0121 (Qm passed) / 0.0140 (unknown) vs 0.0130 / 0.0177 for the offset defaults, same iterations; it is
  0.1-0.2 dB worse at low rates on ldpctest's ideal LLRs, which never use CRC mode, and costs +4.7 % vector operations
per iteration (the 7/8 scaling of m1 and m2 per row and superchunk). `LDPC_DEC_CRCALG=0` reverts.
* **CRC mode** (a CRC callback is given, early termination on, and the core rows 0-3 hold at most 60 % of the edges:
  BG1 R1/3, R2/3, BG2 R1/5, R1/3): stop when the core rows are satisfied and the CRC of the systematic bits passes,
  checked after every iteration and after the core rows of iterations >= 2 (counted as an iteration). The core rows
  contain every systematic and core-parity column, so the decoded bits are a codeword of the core code and pass the
  CRC; the extension parity is not needed for the output. Other codes keep the full-syndrome rule. OAI's own decoder
  checks the CRC alone after every iteration; this rule is stricter. `stats.core_stop` reports such a stop.

## 5. Data layout and kernel
**Column buffers (ping-pong).** A column c of degree >= 2 owns 2 (even degree) or 3 (odd degree) buffers of
Z + 2B bytes (B = U*W, the plan's largest superchunk). A buffer holds the column in the frame of the row that wrote
it, buf[j] = APP_c[(j + s) mod Z], plus a copy of its first min(B, Z) bytes at buf[Z..] (the wrap pad). Visit v of
the column (in row order) reads buffer rb(v) and writes rb(v+1) != rb(v) (odd degrees use the third buffer for the
last two visits), so every iteration starts and ends in buffer 0 (frame of the column's last row). A row with shift s
reads superchunk k of an edge as the contiguous run buf[(d + k B_r) mod Z ..], d = (s - s_prev) mod Z, taken from a
precomputed table (`rdtab`, one int32 per row, superchunk and edge), and writes it aligned at buf[k B_r]; the first
min(B, Z) bytes are copied once more to buf[Z..] (descending k, so this copy is the last write of a column and
overwrites the padding spill). Compared with the doubled column of the previous version this removes the second
store of every vector (now B bytes per column update instead of Z) and halves the column memory per buffer.

**Degree-1 columns** keep a read-only rotated channel copy; their APP is not stored. Rows with a degree-1 column store
their state (m1c, c1, c2) per superchunk; the syndrome and the LLR output recompute APP = Q + R from it (lazy).

**Messages.** Full int8 messages R per multi edge and check, per row [k][e][U_r W] (BG1 R1/3 Z 384: 105 KB).
Compressed messages (min1/min2/index/sign) were evaluated and not used: rebuilding R_old costs 4-5 vector ALU
operations per edge, more than the load + store it saves on a core whose 2 vector pipes are the bottleneck
(top-level README, section 3).

**Row kernel (src/kern_tmpl.h), per row and superchunk k (descending), U_r vectors:**

* P1, multi edge e: x = APP run, Q = sqsub(x, R), ab = |x - R| (unsaturated SABD; exact, see below), kept in
  registers; first two edges: m1 = min, m2 = max (2 ops), then m2 = max(min(m2, ab), m1), m1 = min(m1, ab);
  sg ^= Q (SVE2: EOR3). Degree-1 edge: Q = channel, read only.
* state: m1c = min(m1, lim), c1 = +-corr(m1c), c2 = +-corr(min(m2, lim)); lim <= 127 is the smallest magnitude from
  which corr() is constant, so corr(m) = max(0, m - (m >> nshift) - beta) needs no rmax clamp.
* P2, multi edge: R = (|Q| == m1c ? c2 : c1) negated where Q < 0 (SVE2: range test -m1c <= Q <= m1c with two chained
  compares instead of ABS + compare; rows where registers allow keep ab and compare ab == m1c), stored in place;
  APP = sqadd(Q, R) stored aligned into the write buffer.
* Exactness of the shortcuts: with m1 <= lim the edge(s) at the minimum compare equal in every form; if m1 > lim (or
  m1 = 127 with saturated Q) no or every edge compares equal, but then m2 >= m1 and c1 == c2. Bit-exactness against
  the reference is tested for every variant.

Row shapes (nn multi + n1 degree-1 edges) of the six rates: BG1 2..9+1, 19+0; BG2 2..5+1, 8+0, 10+0. U_r per row
from the register budget (32 registers: U = 4 for nn = 2, U = 2 with ab kept for nn <= 4, U = 1 with ab kept for
nn <= 10, U = 1 for nn = 19; AVX2 (16 registers): U = 1, Q of large rows in a stack buffer). Operations per multi
edge-vector, SVE2: P1 5.5 vector ALU (with EOR3), P2 5 (ab kept) or 6, 2 stores; NEON: 6 + 6/7 + 2 stores.
SVE2-specific fixed sequences (single asm blocks) avoid the MOV / MOVPRFX copies GCC otherwise inserts.

**Syndrome** in superchunks of the plan's B (table `sytab`), degree-1 contributions recomputed from the row state,
lane mask only in the padded superchunk; a second table (`sytabm`) addresses the core rows in the buffers/frames
after the core rows of an iteration (CRC-mode check).

**Z not a multiple of the vector width**: Z is padded to a multiple of B; padding lanes compute garbage that never
reaches a valid lane; descending superchunks + the final wrap-pad copy keep every valid run correct.

**SIMD mapping and dispatch.**

| target | vector | how selected | notes |
|---|---|---|---|
| NEON | 16 B | aarch64 baseline (armv8-a) | default on aarch64 without SVE2 |
| SVE2 | VL (VLA) | `getauxval(AT_HWCAP2) & HWCAP2_SVE2` (+ HWCAP_SVE); functions compiled with `#pragma GCC target("arch=armv8.5-a+sve2")` | **default whenever available, also at VL 128** (measured 3-6 % faster than NEON on the A720 with the previous kernel; now also fewer instructions); Z <= 16: NEON |
| AVX2 | 32 B | `__builtin_cpu_supports("avx2")`, `#pragma GCC target` | also used for Z <= 32 on AVX-512 machines |
| AVX-512 | 64 B | `__builtin_cpu_supports` avx512f+bw+vl | compares/selects/negation through k-masks |
| scalar | 8 B (C) | always | portable fallback |

`LDPC_DEC_IMPL=neon|sve2|avx2|avx512|scalar|ref` forces an implementation; `LDPC_DEC_U` the maximum vectors per step.
Plans (offsets, schedule, offset tables of a few KB on the heap) are cached per thread (4 entries).

## 6. Validation methodology (what each test does)
* `test/test_dec.c` (natively and under qemu): [1] spec encoder + independent syndrome for every Z and BG; [2] random
  blocks decoded by the reference (`ldpc_ref.c`) and by every implementation with U = 1, 2, 4, comparing output
  buffer (pre-filled 0xA5), iteration count, syndrome flag and CRC result; the first 306 blocks cover all 51 Z x 6
  (BG, R); random LLR scales (x1/16 .. x8), fillers not at +127, AGC default/on/off, kprime passed or not, CRC mode
  with an oracle CRC and with a hash CRC that fails 2 of 3 bit patterns (exercises "core rows satisfied, CRC failed,
  continue"), other corrections; [3] degenerate inputs; [4] the vector input-conditioning kernel (AGC gain, in_shift,
  clamp) for every int8 value and every gain, and the AGC statistic pass, against their scalar definitions.
* `test/oai/cmp_oai.c`: `--tables`, `--entry` (OAI entry == own API; kprime is now always passed since the AGC uses
  the filler range), `--bler` (optionally `--crc`), `--scan` (Eb/N0 at 1e-2 for several settings on a scaled channel:
  `--f` scale, `--mode 1` OCUDU-like 5L, `--sat`), `--bench` (`--crc`: harness CRC-24A, as OAI's PHY), `--sweepz`.
* Instruction counts and the dynamic instruction mix: `/home/user/ldpc_speedup/tools/` (qemu insn / hotblocks plugins).

## 7. OAI observations (anomalies found while comparing)

1. **Parity-check success in the last iteration is reported as failure.** In PC mode (`check_crc == NULL`)
   `nrLDPC_decoder_core` increments `numIter` after a successful parity check; if this happens in the last loop
   pass, it returns `numMaxIter` and `LDPCdecoder` sets the abort flag although the output is a valid codeword.
   `cmp_oai --oaidiag 1,384,13,25`: at 2.5 dB 80 of 200 blocks are flagged as failed, all with correct
   information bits and zero syndrome. Together with the slow convergence of flooding min-sum this makes
   OAI's PC-mode failure flag unusable near its waterfall: at its BLER-1e-2 point 96 % of the blocks are
   flagged (`OAI_flagfail` column in `results/bler_*.txt`). OAI's PHY uses CRC mode, where success breaks the
   loop before the increment, so production is not affected; ldpctest and any PC-mode user are.
2. **`nrLDPC_outMode_LLRINT8` returns hard bits.** After `nrLDPC_llrRes2llrOut` writes the LLRs into `p_out`,
   the `else` branch (meant for BITINT8) runs `nrLDPC_llr2bit` on the same buffer, so the caller receives 0/1
   bytes instead of LLRs (`nrLDPC_decoder.c`, both the PC and the CRC path). dec_std returns LLRs in this mode.
3. Iteration count convention: OAI runs up to `numMaxIter + 1` flooding iterations (one before the loop) and
   returns the loop count; dec_std's OAI entry treats `numMaxIter` as layered iterations and returns
   iterations - 1 on success (so `< numMaxIter` still means success for OAI's callers).
4. No table or encoder discrepancy between OAI and 38.212 (section 2); no lifting size where OAI behaves out of
   line with its neighbours (`results/sweepz.txt`; short codes simply need more SNR).

## 8. Design decisions and their justification
**Changes in this version and why** (details and numbers: top-level README):

* Fused single-pass rows with Q in registers instead of two passes over the edges with Q stored in the message slot:
  removes one vector store and one load per edge-vector and the row-state store/load round trip.
* Ping-pong column buffers + wrap pad instead of the doubled column: removes the second store of every vector.
* Lazy degree-1 APP: removes the APP computation and store of the 42 degree-1 edges per iteration (BG1 R1/3).
* Unsaturated |x - R| (SABD), magnitude clamp lim, first-two-edges peel, SVE2 range-test equality, keep-ab variant:
  fewer vector ALU operations per edge; all exact.
* SVE2 default at VL 128 (measured faster on the A720 already with the previous kernel).
* AGC and CRC-mode stop: robustness on real PHY LLRs and fewer iterations in OAI's CRC mode.

The previous section 8 text about the doubled layout and the VL-128 dispatch policy describes the old version.



* **Layered instead of flooding**: about half the iterations for the same BLER and a lower waterfall at a fixed
  iteration budget (1 dB at 8 iterations here); per iteration it does the same work (each edge visited once).
* **Offset min-sum, per-rate offset, int8 with message clamp**: see section 4. The clamp (rmax = 47) and the
  per-column channel clamp (degree-1 columns: |L| <= rmax - 1, others 2 rmax - 1, so the messages can always
  overrule the channel value; without it about 0.2 % of correctly decoded BG2 R2/3 blocks never satisfied the
  syndrome at 3.25 dB because a strongly wrong parity bit of degree 1 could not be flipped) cost no measurable
  BLER and make the int8 decoder stable at high SNR.
* **Full messages instead of compressed (min1/min2/index/signs)**: per edge-vector the full form needs 1 load
  + 1 store; the compressed form would need ~4-5 ALU operations to rebuild R_old. The workspace (161 KB for
  BG1 Z=384) lives in L2; on x86 the message stream (~105 KB read + written per iteration) is a visible part
  of the 8 us per iteration, a compressed format would be the next step there.
* **Rotated frame + double store** (section 5): circular shifts cost nothing beyond an unaligned load and one
  extra store per vector; alternatives (in-register rotation with TBL/VPERMB, or a separate wrap fix-up copy)
  cost more instructions per edge.
* **U vectors per step**: amortises the per-edge offset load and loop control (NEON Z = 384: U = 4 executes
  27 % fewer instructions than U = 1).
* **NEON vs SVE2 at VL 128**: both process 16 checks per instruction; SVE2 needs `movprfx` for the non-destructive
  `abs` and saves one instruction in the conditional negate: net +3-4 % instructions. The dispatcher therefore
  takes NEON when SVE2 runs at 128 bits and SVE2 when the vector is longer (VL 256: -45 % instructions). Forcing
  SVE2 (`LDPC_DEC_IMPL=sve2`) is supported and tested. GCC needed an empty `asm` register pin after the SVE edge
  loops to stop it from copying all accumulators every iteration (8 `mov`s per edge).
* **AVX-512**: compares, selects and the sign fix-up use k-masks (`vpcmpeqb`->k, `vpblendmb`, `vpmovb2m`, masked
  `vpsubb`); padding lanes are handled by the uniform padding/descending-store scheme rather than masked stores,
  which would add a mask computation per tail vector and gain nothing (pad lanes are never read back as valid).
  For Z <= 32 the AVX2 kernel is used.

## 9. Limitations, open issues, what to measure on the A720
* **Estimates, not measurements**: A720 times in the top-level README are estimates from qemu instruction counts
  (constant IPC) and from a 2-vector-pipe model calibrated on the measured previous version; the orion package runs
  both versions in the same harness to measure them.
* The Qm factors need Qm from the caller: OAI's `t_nrLDPC_dec_params` has no Qm field yet (use
  `LDPCdecoder_set_qm()` or build with `-DLDPC_DEC_OAI_HAS_QM`). Without it 256QAM keeps a residual loss
  (geo-mean 0.033 vs 0.024 with Qm). All PHY data are BG1 except 1L QPSK MCS 2 (BG2 R1/5), 1-2 layers, AWGN and
  TDL-A; other channels, layers and MCS are untested.
* CRC mode is only as safe as the CRC: the core-row parity is checked before every CRC evaluation, so it is stricter
  than OAI's CRC-only stop; undetected-error behaviour with a real CRC-24 was not simulated beyond the oracle.
* `stats.syndrome_ok` in CRC mode reports the core-row parity at the stop.
* No error-floor study below 1e-4 (20 000-block runs at 2-6 dB above the waterfall: 0 errors).

## 10. Reproducing

This version: see `/home/user/ldpc_speedup/README.md` section 8 (scripts in `/home/user/ldpc_speedup/tools/`). The
commands below are those of the previous version (still valid for tests and BLER).


```
python3 -I tables/gen_tables.py /home/user/ldpc_scratch/spec \
    ext/srsRAN_Project/lib/phy/upper/channel_coding/ldpc/ldpc_luts_impl.cpp \
    /home/user/downloads/oai_ref/openair1/PHY/CODING/nrLDPC_decoder/nrLDPC_lut.h tables/ldpc_tables.c
make && make ARCH=aarch64
./build/x86_64/test_dec --blocks 6000
for vl in 16 32 64; do qemu-aarch64-static -cpu max,sve-default-vector-length=$vl build/aarch64/test_dec; done
test/oai/build.sh && OAI_AVX512=0 OUT=build/x86_64/cmp_oai_avx2 test/oai/build.sh && ARCH=aarch64 test/oai/build.sh
build/x86_64/cmp_oai --tables; build/x86_64/cmp_oai --entry
build/x86_64/cmp_oai --bler 1,384,13 --snr 0.875:2.5:0.125 --blocks 4000 --threads 2   # etc., bench/bler_summary.py
bench/run_x86.sh; bench/run_aarch64.sh
```
(srsRAN is fetched with `git clone --depth 1 --filter=blob:none --sparse https://github.com/srsran/srsRAN_Project`
+ `git fetch --depth 1 origin main` into `ext/`; the arm-neon-ldpc decoder files are extracted read-only with
`git show arm-neon-ldpc:<path>` into `ext/oai_neon_dec/`. Neither is part of the tarball.)
