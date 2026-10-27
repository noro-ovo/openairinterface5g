/* NEON kernel compiled for ARMv8.2-A + SHA3 (EOR3): selected at run time by ldpc_kern_neon() when HWCAP_SHA3 is
   set; the armv8-a build of everything else is unchanged. */
#if defined(__aarch64__)
#pragma GCC target("+sha3") /* adds SHA3 (EOR3) to the baseline instead of setting the architecture */
#define LDPC_NEON_SHA3 1
#include "kern_neon.c"
#endif
