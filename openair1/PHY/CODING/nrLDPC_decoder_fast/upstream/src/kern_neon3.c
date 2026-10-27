/* NEON kernel compiled for ARMv8.2-A + SHA3 (EOR3): selected at run time by ldpc_kern_neon() when HWCAP_SHA3 is
   set; the armv8-a build of everything else is unchanged. */
#if defined(__aarch64__)
#if !defined(__ARM_FEATURE_SHA3) /* baseline without SHA3 (e.g. armv8.2-a): enable it for this file only */
#pragma GCC target("arch=armv8.2-a+sha3")
#endif
#define LDPC_NEON_SHA3 1
#include "kern_neon.c"
#endif
