/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* A measurement gap starts in the frame with SFN mod T = FLOOR(gapOffset / 10), T = MGRP / 10, at subframe
 * gapOffset mod 10 (38.331 5.5.2.9), mgta earlier, and lasts mgl. nr_measgap_scheduling() stops scheduling the UE
 * through the transm_interrupt timer; the UE must not be scheduled in any slot of its gap. */

#include <stdio.h>
#include <stdlib.h>

#include "common/utils/assertions.h"
#include "LAYER2/NR_MAC_gNB/mac_proto.h"
#include "LAYER2/NR_MAC_gNB/nr_mac_gNB.h"
#include "LAYER2/NR_MAC_gNB/nr_radio_config.h"

static NR_MeasurementTimingConfiguration_t *mtc_sf160(long offset)
{
  NR_MeasurementTimingConfiguration_t *mtc = calloc_or_fail(1, sizeof(*mtc));
  mtc->criticalExtensions.present = NR_MeasurementTimingConfiguration__criticalExtensions_PR_c1;
  mtc->criticalExtensions.choice.c1 = calloc_or_fail(1, sizeof(*mtc->criticalExtensions.choice.c1));
  mtc->criticalExtensions.choice.c1->present = NR_MeasurementTimingConfiguration__criticalExtensions__c1_PR_measTimingConf;
  NR_MeasurementTimingConfiguration_IEs_t *ies = calloc_or_fail(1, sizeof(*ies));
  mtc->criticalExtensions.choice.c1->choice.measTimingConf = ies;
  ies->measTiming = calloc_or_fail(1, sizeof(*ies->measTiming));
  NR_MeasTiming_t *mt = calloc_or_fail(1, sizeof(*mt));
  mt->frequencyAndTiming = calloc_or_fail(1, sizeof(*mt->frequencyAndTiming));
  mt->frequencyAndTiming->ssb_MeasurementTimingConfiguration.periodicityAndOffset.present = NR_SSB_MTC__periodicityAndOffset_PR_sf160;
  mt->frequencyAndTiming->ssb_MeasurementTimingConfiguration.periodicityAndOffset.choice.sf160 = offset;
  ASN_SEQUENCE_ADD(&ies->measTiming->list, mt);
  return mtc;
}

// returns the number of gap slots in which the UE can be scheduled, for every gapOffset of an MGRP of 160 ms
int test_measgap(int mu, int *checked)
{
  const int slots_frame = 10 << mu;
  const int slots_sf = 1 << mu;
  int bad = 0, bad_offsets = 0;
  *checked = 0;
  for (long offset = 0; offset < 160; offset++) {
    NR_MeasurementTimingConfiguration_t *mtc = mtc_sf160(offset);
    gNB_MAC_INST mac = {0};
    nr_cell_sched_t cell = {0};
    cell.frame_structure.numb_slots_frame = slots_frame;
    NR_UE_info_t ue = {0};
    ue.pcell = &cell;
    ue.measgap_config = create_measgap_config(mtc, mu, 2);
    ASN_STRUCT_FREE(asn_DEF_NR_MeasurementTimingConfiguration, mtc);
    const measgap_config_t *mgc = &ue.measgap_config;
    AssertFatal(mgc->enable && mgc->mgrp_ms == 160, "no measurement gap\n");
    mac.UE_info.connected_ue_list[0] = &ue; // NULL-terminated

    // gap in absolute slots: subframe gapOffset mod 10 of frame FLOOR(gapOffset / 10), mgta earlier, mgl long
    const int period = 16 * slots_frame;
    const int gap_start = (int)(offset / 10) * slots_frame + (int)(offset % 10) * slots_sf - mgc->n_slots_mgta;
    const int gap_len = (int)(mgc->mgl_ms * slots_sf);
    // two periods, as in nr_mac_schedule_slot(): the timer update (nr_mac_update_timers()) then nr_measgap_scheduling()
    int open = 0;
    for (int abs_slot = 0; abs_slot < 2 * period; abs_slot++) {
      const frame_t frame = abs_slot / slots_frame;
      const slot_t slot = abs_slot % slots_frame;
      if (nr_timer_tick(&ue.UE_sched_ctrl.transm_interrupt))
        nr_timer_stop(&ue.UE_sched_ctrl.transm_interrupt);
      nr_measgap_scheduling(&mac, &cell, frame, slot);
      const bool blocked = nr_timer_is_active(&ue.UE_sched_ctrl.transm_interrupt);
      const int in_period = ((abs_slot - gap_start) % period + period) % period;
      if (abs_slot >= period && in_period < gap_len) {
        (*checked)++;
        open += !blocked;
      }
    }
    if (open > 0) {
      if (bad_offsets < 5)
        printf("mu %d gapOffset %ld: UE schedulable in %d of %d gap slots\n", mu, offset, open, gap_len);
      bad_offsets++;
    }
    bad += open;
  }
  printf("mu %d: %d gap slots checked, %d with the UE schedulable (%d gapOffsets)\n", mu, *checked, bad, bad_offsets);
  return bad;
}
