/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "nr_ue_tun.h"
#include "common/utils/assertions.h"
#include "common/utils/tuntap_if.h"
#include "common/utils/LOG/log.h"
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/socket.h>

void nr_ue_tun_store_qfi(nas_ue_pdu_tun_t *t, uint8_t qfi)
{
  DevAssert(t != NULL);
  t->qfi = qfi;
}

void nr_ue_tun_create_ip_if(nas_ue_pdu_tun_t *t, const char *ipv4, const char *ipv6, int ue_id, int pdu_session_id)
{
  DevAssert(t != NULL);
  /* the interface survives a re-registration: only the address may have changed */
  if (t->sock < 0)
    t->sock = tuntap_alloc(IFF_TUN, t->ifname);
  tun_config(t->ifname, ipv4, ipv6);
  if (ipv4)
    setup_ue_ipv4_route(t->ifname, ue_id, pdu_session_id, ipv4);
}

void nr_ue_tun_create_eth_if(nas_ue_pdu_tun_t *t)
{
  DevAssert(t != NULL);
  /* the interface of an earlier PDU session with this ID is reused (it was set down at its release) */
  if (t->sock < 0)
    t->sock = tuntap_alloc(IFF_TAP, t->ifname);
  tap_config(t->ifname);
}

/** @brief PDU session released: the interface goes down, its fd and name stay for a new PDU session with this ID
 * (nr_ue_tun_create_ip_if()/nr_ue_tun_create_eth_if() bring it up again on the same fd) */
void nr_ue_tun_set_down(const nas_ue_pdu_tun_t *t)
{
  DevAssert(t != NULL);
  if (t->sock < 0)
    return;
  int fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0) {
    LOG_E(NAS, "%s: socket() failed: %d, %s\n", t->ifname, errno, strerror(errno));
    return;
  }
  struct ifreq ifr = {0};
  strncpy(ifr.ifr_name, t->ifname, sizeof(ifr.ifr_name) - 1);
  if (ioctl(fd, SIOCGIFFLAGS, &ifr) == 0) {
    ifr.ifr_flags &= ~IFF_UP;
    if (ioctl(fd, SIOCSIFFLAGS, &ifr) == 0)
      LOG_I(NAS, "interface %s down (fd %d kept)\n", t->ifname, t->sock);
    else
      LOG_E(NAS, "%s: SIOCSIFFLAGS failed: %d, %s\n", t->ifname, errno, strerror(errno));
  } else {
    LOG_E(NAS, "%s: SIOCGIFFLAGS failed: %d, %s\n", t->ifname, errno, strerror(errno));
  }
  close(fd);
}

/** @brief Drop the packets queued on the TUN fd (none must reach a later PDU session with this ID)
 * @note only with no reader running on the fd */
void nr_ue_tun_flush(const nas_ue_pdu_tun_t *t)
{
  DevAssert(t != NULL);
  if (t->sock < 0)
    return;
  int f = fcntl(t->sock, F_GETFL, 0);
  if (f == -1 || fcntl(t->sock, F_SETFL, f | O_NONBLOCK) == -1) {
    LOG_E(NAS, "%s: fcntl() failed: %d, %s\n", t->ifname, errno, strerror(errno));
    return;
  }
  char buf[2048];
  int n = 0;
  while (read(t->sock, buf, sizeof(buf)) > 0)
    n++;
  fcntl(t->sock, F_SETFL, f);
  if (n > 0)
    LOG_I(NAS, "%s: %d queued packets dropped\n", t->ifname, n);
}
