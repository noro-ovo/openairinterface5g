/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "openair2/LAYER2/NR_MAC_COMMON/nr_mac_common.h"
#include "executables/softmodem-common.h"
#include "common/config/config_paramdesc.h"
#include "common/utils/LOG/log.h"

/* LOG, pulled in by nr_compute_tbs */
configmodule_interface_t *uniqCfg = NULL;
void exit_function(const char *file, const char *function, const int line, const char *s, const int assert)
{
  abort();
}

softmodem_params_t *get_softmodem_params(void)
{
  return NULL;
}

int main(void)
{
  logInit();
  /* 38.214 5.1.3.2, MCS 2 of Table 5.1.3.1-1 (R x 10240 = 1930, Qm 2), 89 PRBs, 10 symbols, 6 DM-RS REs:
   * N_RE = 114 x 89 = 10146 and N_info = 10146 x 193 / 1024 x 2 = 3824.6 > 3824, so step 4 applies:
   * n = 6, N'_info = max(3840, 64 x round(3800.6 / 64)) = 3840, R <= 1/4 gives C = 2 and
   * TBS = 16 x ceil(3864 / 16) - 24 = 3848. Truncating N_info to 3824 took step 3 and gave 3824. */
  assert(nr_compute_tbs(2, 1930, 89, 10, 6, 0, 0, 1) == 3848);
  /* just below: MCS 3 (R x 10240 = 2510), 260 PRBs, 3 symbols, 6 DM-RS REs: N_info = 30 x 260 x 251 / 1024 x 2
   * = 3823.8, step 3, n = 5, N'_info = 32 x 119 = 3808, TBS 3824 */
  assert(nr_compute_tbs(2, 2510, 260, 3, 6, 0, 0, 1) == 3824);
  printf("nr_compute_tbs tests passed\n");
  return 0;
}
