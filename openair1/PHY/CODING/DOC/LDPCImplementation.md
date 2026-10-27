<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# LDPC coding implementation
This document gives an overview of the different LDPC coding implementations (functional or not) available with Open Air Interface.

[[_TOC_]]

The LDPC encoder and decoder are implemented in a shared library, dynamically loaded at run-time using the [oai shared library loader](file://../../../../common/utils/DOC/loader.md).
Two types of library are available with two different interfaces. There are libraries implementing the encoder and decoder of slots and libraries implementing the encoder and decoder of code segments.

## LDPC slot coding
The interface of the library is defined in [nrLDPC_coding_interface.h](file://../nrLDPC_coding/nrLDPC_coding_interface.h).
The code loading the LDPC library is in [nrLDPC_coding_interface_load.c](file://../nrLDPC_coding/nrLDPC_coding_interface_load.c), in function `load_nrLDPC_coding_interface`, which must be called at init time.

### Selecting the LDPC library at run time

By default the function `int load_nrLDPC_coding_interface(void)` looks for `libldpc.so`.\
This default behavior can be changed using the oai loader configuration options in the configuration file or from the command line as shown below:

#### Examples of ldpc shared lib selection when running nr softmodem's:

loading default `libldpc.so`:

```
./nr-softmodem -O libconfig:gnb.band78.tm1.106PRB.usrpx300.conf:dbgl5
```

`libldpc.so` has its decoder implemented in [nrLDPC_coding_segment_decoder.c](file://../nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_coding_segment_decoder.c).\
Its encoder is implemented in [nrLDPC_coding_segment_encoder.c](file://../nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_coding_segment_encoder.c).

loading `libldpc_aal.so` instead of `libldpc.so`:

`make ldpc_aal`

This command creates the `libldpc_aal.so` shared library.

```
Building C object CMakeFiles/ldpc_aal.dir/openair1/PHY/CODING/nrLDPC_coding/nrLDPC_coding_aal/nrLDPC_coding_aal.c.o
Linking C shared module libldpc_aal.so
```

At runtime, to successfully use LDPC accelerators (e.g., Xilinx T2/Intel ACCs), you will need to install the corresponding drivers and tools.
Please refer to the dedicated documentation at [LDPC_OFFLOAD_SETUP.md](file://../../../../doc/LDPC_OFFLOAD_SETUP.md).

```
./nr-softmodem -O  libconfig:gnb.band78.sa.fr1.106PRB.usrpb210.conf:dbgl5 --rfsim --rfsimulator.[0].serveraddr server  --log_config.gtpu_log_level info  --loader.ldpc.shlibversion _aal --nrLDPC_coding_aal.dpdk_dev 01:00.0 --nrLDPC_coding_aal.dpdk_core_list 0-1
```

`libldpc_aal.so` has its decoder and its encoder implemented in [nrLDPC_coding_aal.c](file://../nrLDPC_coding/nrLDPC_coding_aal/nrLDPC_coding_aal.c).

#### Examples of ldpc shared lib selection when running ldpctest:

Slot coding libraries cannot be used yet within ldpctest.

But they can be used within nr_ulsim, nr_dlsim, nr_ulschsim and nr_dlschsim.\
In these PHY simulators, using the slot coding libraries is enabled in the exact same way as in nr-softmodem.

### LDPC libraries
Libraries implementing the slotwise LDPC coding must be named `libldpc<_version>.so`. They must implement four functions: `nrLDPC_coding_init`, `nrLDPC_coding_shutdown`, `nrLDPC_coding_decoder` and `nrLDPC_coding_encoder`. The prototypes for these functions is defined in [nrLDPC_coding_interface.h](file://../nrLDPC_coding/nrLDPC_coding_interface.h).

`libldpc.so` is completed.

`libldpc_aal.so` is completed.

### Decoder input LLR scale (`libldpc.so`, `libldpc_orig.so`)

The segment layer of `libldpc.so` / `libldpc_orig.so` ([nrLDPC_coding_segment_decoder.c](file://../nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_coding_segment_decoder.c))
converts the int16 channel LLRs of each code block to the int8 decoder input with a saturating pack. The int16 LLRs
of the PHY are not on a common scale: 16QAM and 64QAM LLRs are several times larger than QPSK LLRs, the multi-layer
receivers produce even larger ones, and up to 97 % of them saturate at +-127 after the pack. Before the pack, the
segment layer therefore applies a gain that depends on the modulation order Qm and on the number of layers
([nrLDPC_llr_prescale.h](file://../nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_llr_prescale.h)):

| layers | QPSK | 16QAM | 64QAM | 256QAM |
|---|---|---|---|---|
| 1 | x1 | x1/4 | x1 | x1 |
| 2 or more | x1/4 | x1/16 | x1/2 | x1 |

This default was measured with OAI's decoder and the receivers of this code base, in `nr_ulsim` and `nr_dlsim`. For
each entry it is the largest tested gain that was not worse than the plain pack at any tested MCS. The gain is
applied in int16 with symmetric rounding (round half away from zero), so it also removes most of the saturation.
Filler bits stay at +127 and punctured/untransmitted positions stay 0. The rule is selected (highest priority first) by:

* the environment variable `OAI_LDPC_LLR_PRESCALE` (for experiments),
* the configuration parameter `llr_prescale` of the section `nrLDPC_coding_segment`, in the configuration file or on
  the command line, e.g. `--nrLDPC_coding_segment.llr_prescale off` (also in `nr_ulsim`/`nr_dlsim`),
* the built-in default.

Values:
* `default`;
* `off`: legacy conversion, bit-exact;
* `g2,g4,g6,g8`: the gains for QPSK, 16QAM, 64QAM and 256QAM, for any number of layers;
* `g2,g4,g6,g8/h2,h4,h6,h8`: `g` for 1 layer, `h` for 2 or more layers.

Each gain must be in [1/16, 1) or one of 1, 2, 4, 8; for example `1,0.25,1,1/0.25,0.0625,0.5,1` is `default`. The
active rule is logged at init. A gain of 1 uses the unmodified conversion, so those cases stay bit-exact with the
legacy code. A receiver with a different LLR scale (e.g. a new multi-layer detector) needs its own rule.

The decoder is told what it receives: `t_nrLDPC_dec_params` carries `Qm` (0 = unknown, e.g. `ldpctest`), `nb_layers`
and `llr_prescale_q8` (the gain above in Q8, 256 = x1). OAI's decoders ignore these fields. They are appended at the
end of the structure, so decoders built against the previous layout keep working.

## LDPC segment coding
The interface of the library is defined in [nrLDPC_defs.h](file://../nrLDPC_defs.h) as typedefs of the functions of the interface.
The name of the functions implementing these typedefs can be found in [nrLDPC_extern.h](file://../nrLDPC_extern.h).

LDPC segment coding libraries are not loaded directly by OAI but are statically linked in libraries implementing the LDPC slot coding interface using an adaptation layer between the slot coding interface and the segment coding interface.  
This way, these libraries can be loaded to implement the slot coding interface as well as the segment coding interface in `ldpctest`.  
`ldpctest` therefore loads directly the segment coding interface with the loader implemented in [nrLDPC_load.c](file://../nrLDPC_load.c) as function `load_nrLDPClib`.

### Selecting the LDPC library at run time

By default the function `int load_nrLDPClib(void)` looks for `libldpc.so`, this default behavior can be changed using the oai loader configuration options in the configuration file or from the command line as shown below:

#### Examples of ldpc shared lib selection when running nr softmodem's:

loading `libldpc_orig.so` instead of `libldpc.so`:

```
./nr-softmodem -O libconfig:gnb.band78.tm1.106PRB.usrpx300.conf:dbgl5  --loader.ldpc.shlibversion _orig
```

A mechanism to select ldpc implementation is also available in the `ldpctest` phy simulator via the `-v` option, which can be used to specify the version of the ldpc shared library to be used.

#### Examples of ldpc shared lib selection when running ldpctest:

Loading libldpc_orig.so, the legacy implementation of the ldpc decoder:

```
$ ./ldpctest -v _orig
```

### LDPC libraries
The prototypes for the functions of the interface are defined in [nrLDPC_defs.h](file://nrLDPC_defs.h) as typedefs.
Libraries implementing the LDPC algorithms must be named `libldpc<_version>.so`, they must implement four functions with the names and types:
* `LDPCinit` implementing type `LDPC_initfunc_t *`
* `LDPCshutdown` implementing type `LDPC_shutdownfunc_t *`
* `LDPCdecoder` implementing type `LDPC_decoderfunc_t *`
* `LDPCencoder` implementing type `LDPC_encoderfunc_t *`
* `LDPCencoder32` implementing type `LDPC_encoderfunc32_t *`, specifically for
  CUDA offload. This is not used by the LDPC slot coding interface, but must be
  defined in a shared library to load successfully.
