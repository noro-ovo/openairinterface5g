/*
 * OAI device for an OC1 board running our own FPGA image, through libocsdr
 * (https://github.com/arash9638444/ocsdr). Same behaviour as OCUDU's ocsdr radio backend.
 *
 * Select it with --device.name oai_ocsdr (or device = { name = "oai_ocsdr"; } in the config).
 * Options go in the RU's sdr_addrs, comma separated key=value:
 *   board=N       PCIe board index (default 0)
 *   host=ADDR     board over 10G Ethernet instead of PCIe (FPGA release v13+), e.g. 10.0.0.2
 *   tx_shift=N    TX samples are shifted left by N bits, saturating (default 2)
 *   ring=N        DMA ring blocks per direction, power of two 2..256 (default 256)
 *   skip_init=1   keep the clocks, JESD links and AD9371 as they are
 *   frontend=0/1  RF front end on (TX to the connectors, LNAs on), default 1
 *   rx_atten=dB   front-end RX attenuators 0..31 (default 0)
 *   rx1_atten=dB  RX1 attenuator only, overrides rx_atten (RX2 has about 20 dB less front-end gain; this balances them)
 *   tx_atten=dB   AD9371 TX attenuation 0..41.95 (default 1.95)
 *   rx_gain=dB    AD9371 RX gain 0..30 above index 195 (default 20, index 235)
 *   rx2_gain=dB   AD9371 RX2 gain only, overrides rx_gain
 *   tx_advance=N  samples TX is sent ahead of RX time (default 174, measured with "ocsdr stream latency")
 *   tx_gate=0/1   FPGA TX gating (image v16s+ over Ethernet; over PCIe, an image that gates on the PCIe TX stream):
 *                 TX PAs off between the bursts that the write flags end (TX_BURST_END...), i.e. in TDD uplink
 *                 slots; run without --continuous-tx
 *   tx_gate_lead=N, tx_gate_lag=N  samples the gate opens before / stays open after a burst (128 / 256)
 *   tx_gate_mode=M  0 PAs off between bursts, 1 PAs off + TX switches to the 50 ohm loads, 2 switches only,
 *                   PAs stay on (default; per-burst PA shutdown leaves RX1 ~6 dB noisier for >100 us)
 *   tx_flush_us=N  send a partly filled TX block N us before its start time if no more samples came (default 0:
 *                  off). A block only leaves when full, so the tail of a slot waits for the next slot's write; with
 *                  this a late next write loses its first samples (at most one block) instead of the earlier
 *                  slot losing its tail, which holds the PUCCH.
 *   rx_only=1     receive only: TX amplifiers shut down, TX switches off the connectors, TX samples dropped in the
 *                 driver (nothing is transmitted, e.g. to scan public networks)
 *   tx_lead_stats=1 log once per second how early TX blocks are handed over (block start - board time at commit)
 *   fd=1          front end: OFDM demodulation in the FPGA (register set v11). The FPGA FFT windows are placed where
 *                 OAI takes them (slot grid on the device timestamp, minus fd_nta and fd_adv); nr_fep() then takes
 *                 the symbols from the driver and demodulates only the ones that did not arrive. Needs 273 PRB at
 *                 30 kHz (4096-point FFT)
 *   fd_nta=N      OAI N_TA_offset in samples (default 1600), fd_adv=N  OAI nb_prefix_samples / ofdm_offset_divisor
 *                 (default 36 = 288 / 8)
 *   fd_tx=1       front end TX: OFDM modulation in the FPGA (register set v13). OAI hands the frequency-domain slot
 *                 (trx_write_fd_func) instead of samples; fd_tx_scale=N inverse FFT schedule (default 0x055 = 1/16,
 *                 OAI's idft scaling followed by tx_shift 2)
 *   fd_wait_us=N  when OAI asks for a symbol that is due from the FPGA within N us, wait for it instead of letting
 *                 OAI demodulate it (default 0: never wait; the last symbols of a slot come ~30-60 us after the slot)
 *   rx_sleep_us=N when the next RX block is not there yet, sleep until N us before the samples the read still needs
 *                 are due, then poll (default 60; 0: poll all the time, one core busy)
 */

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <sstream>

extern "C" {
#include "common_lib.h"
#include "common/utils/LOG/log.h"
#include "ocsdr/ocsdr.h"
}
#include "ocsdr_conv.h"

static constexpr unsigned MAX_CHANNELS    = 2;
static constexpr int      RING_TIMEOUT_US = 100000;
static constexpr double   SAMPLE_RATE_HZ  = 122.88e6;
static constexpr double   TX_ATTEN_MAX_DB = 41.95;
static constexpr int      RX_GAIN_INDEX_0 = 195;

struct ocsdr_args {
  int         board      = 0;
  std::string host;
  unsigned    tx_shift   = 2;
  unsigned    ring       = 256;
  bool        skip_init  = false;
  bool        frontend   = true;
  double      rx_atten   = 0;
  double      rx1_atten  = -1; // < 0: same as rx_atten
  double      tx_atten   = 1.95;
  double      rx_gain    = 20;
  double      rx2_gain   = -1; // < 0: same as rx_gain
  int         tx_advance = 174;
  bool        tx_gate      = false;
  unsigned    tx_gate_lead = 128;
  unsigned    tx_gate_lag  = 256;
  unsigned    tx_gate_mode = 2;
  unsigned    tx_flush_us  = 0;
  bool        tx_lead_stats = false;
  bool        rx_only      = false;
  unsigned    rx_sleep_us  = 60;
  bool        fd           = false; // front end: OFDM demodulation in the FPGA (register set v11)
  unsigned    fd_nta       = 1600;  // OAI N_TA_offset in samples (FR1 25600 Tc at 122.88 MS/s)
  unsigned    fd_adv       = 36;    // OAI nb_prefix_samples / ofdm_offset_divisor (288 / 8)
  unsigned    fd_wait_us   = 0;     // wait for a symbol due within this time instead of letting OAI demodulate it
  bool        fd_tx        = false; // front end TX: OFDM modulation in the FPGA (register set v13)
  unsigned    fd_tx_scale  = 0x055; // inverse FFT scaling schedule (1/16: OAI's idft 1/64 then << 2)
};
struct fd_cache;

struct ocsdr_state {
  ocsdr_args    args;
  ocsdr_dev    *dev    = nullptr;
  ocsdr_stream *stream = nullptr;
  unsigned      hw_ch  = 1;
  unsigned      bs     = 0;  // sample times per block
  std::atomic<bool> running{false};

  // receive
  const int16_t *rx_block = nullptr;
  uint64_t       rx_block_ts = 0;
  uint64_t       rx_next_ts  = 0;
  bool           rx_first    = true;
  uint64_t       rx_dropped  = 0;
  uint64_t       rx_sleeps   = 0;
  double         rx_slept_us = 0;
  fd_cache      *fd = nullptr;  // front end symbols (fd=1)
  bool           fdtx = false;  // front end TX on (fd_tx=1 on a v13+ image)
  uint32_t       fdtx_late = 0, fdtx_err = 0;

  // transmit
  int16_t *tx_block     = nullptr;
  uint64_t tx_block_ts  = 0;
  unsigned tx_fill      = 0;
  bool     tx_closed    = true;
  unsigned tx_since_chk = 0;
  uint32_t last_late = 0, last_underflow = 0;
  std::mutex  tx_mtx;            // tx_* state, shared by ocsdr_write and the flusher
  std::thread tx_flusher;
  uint64_t    tx_sent_until = 0; // samples before this time already went out in an early-flushed block
  uint64_t    tx_early = 0, tx_trimmed = 0;
  // tx_lead_stats: per second, lead time of the committed blocks in 50 us bins (last bin: >= 750 us)
  int64_t     lead_min = INT64_MAX;
  uint64_t    lead_n = 0, lead_neg = 0, lead_hist[16] = {0}, lead_next_log = 0;
  // and the age of the RX samples when ocsdr_read returns them
  int64_t     rx_age_max = 0, rx_age_sum = 0;
  uint64_t    rx_age_n = 0, rx_age_over200 = 0;

  // signal levels, from one sample in LEVEL_STEP, logged about every 4 s
  double   tx_sq = 0, rx_sq = 0;
  uint64_t tx_n = 0, rx_n = 0, tx_clip = 0;
  unsigned tx_blocks = 0, rx_blocks = 0;
};

static constexpr unsigned LEVEL_STEP   = 16;
static constexpr unsigned LEVEL_BLOCKS = 1u << 18;

static double dbfs(double sq, uint64_t n)
{
  return n == 0 || sq == 0 ? -INFINITY : 10 * log10(sq / n / (32768.0 * 32768.0));
}

static bool parse_args(const char *s, ocsdr_args &out)
{
  if (s == nullptr)
    return true;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty())
      continue;
    auto eq = item.find('=');
    if (eq == std::string::npos) {
      LOG_E(HW, "ocsdr: sdr_addrs item '%s' is not key=value\n", item.c_str());
      return false;
    }
    std::string key = item.substr(0, eq), value = item.substr(eq + 1);
    if (key == "host") {
      out.host = value;
      continue;
    }
    char *end = nullptr;
    double v = strtod(value.c_str(), &end);
    if (end == value.c_str() || *end != '\0') {
      LOG_E(HW, "ocsdr: sdr_addrs item '%s' needs a number\n", item.c_str());
      return false;
    }
    if (key == "board" && v >= 0)
      out.board = (int)v;
    else if (key == "rx_only")
      out.rx_only = v != 0;
    else if (key == "rx_sleep_us" && v >= 0 && v <= 10000)
      out.rx_sleep_us = (unsigned)v;
    else if (key == "fd")
      out.fd = v != 0;
    else if (key == "fd_nta" && v >= 0 && v < 61440)
      out.fd_nta = (unsigned)v;
    else if (key == "fd_adv" && v >= 0 && v < 352)
      out.fd_adv = (unsigned)v;
    else if (key == "fd_wait_us" && v >= 0 && v <= 500)
      out.fd_wait_us = (unsigned)v;
    else if (key == "fd_tx")
      out.fd_tx = v != 0;
    else if (key == "fd_tx_scale" && v >= 0 && v <= 0xfff)
      out.fd_tx_scale = (unsigned)v;
    else if (key == "tx_lead_stats")
      out.tx_lead_stats = v != 0;
    else if (key == "tx_flush_us" && v >= 0 && v <= 10000)
      out.tx_flush_us = (unsigned)v;
    else if (key == "tx_shift" && v >= 0 && v <= 15)
      out.tx_shift = (unsigned)v;
    else if (key == "ring" && v >= 2 && v <= 256 && (((unsigned)v) & ((unsigned)v - 1)) == 0)
      out.ring = (unsigned)v;
    else if (key == "skip_init")
      out.skip_init = v != 0;
    else if (key == "frontend")
      out.frontend = v != 0;
    else if (key == "rx_atten" && v >= 0 && v <= 31.5)
      out.rx_atten = v;
    else if (key == "rx1_atten" && v >= 0 && v <= 31.5)
      out.rx1_atten = v;
    else if (key == "tx_atten" && v >= 0 && v <= TX_ATTEN_MAX_DB)
      out.tx_atten = v;
    else if (key == "rx_gain" && v >= 0 && v <= 30)
      out.rx_gain = v;
    else if (key == "rx2_gain" && v >= 0 && v <= 30)
      out.rx2_gain = v;
    else if (key == "tx_advance" && v >= 0 && v < 10000)
      out.tx_advance = (int)v;
    else if (key == "tx_gate")
      out.tx_gate = v != 0;
    else if (key == "tx_gate_lead" && v >= 0 && v <= 65535)
      out.tx_gate_lead = (unsigned)v;
    else if (key == "tx_gate_lag" && v >= 0 && v <= 16383)
      out.tx_gate_lag = (unsigned)v;
    else if (key == "tx_gate_mode" && v >= 0 && v <= 2)
      out.tx_gate_mode = (unsigned)v;
    else {
      LOG_E(HW,
            "ocsdr: unknown or out-of-range sdr_addrs item '%s' (keys: board, host, tx_shift, ring, skip_init, frontend, "
            "rx_atten, rx1_atten, tx_atten, rx_gain, rx2_gain, tx_advance, tx_gate, tx_gate_lead, tx_gate_lag, "
            "tx_gate_mode, tx_flush_us, tx_lead_stats, rx_only, rx_sleep_us, fd, fd_nta, fd_adv, fd_wait_us, fd_tx, fd_tx_scale)\n",
            item.c_str());
      return false;
    }
  }
  return true;
}

// ---- receive ----

// Sleeps until rx_sleep_us before the board clock reaches end_ts (the end of the samples the read still needs),
// so the reader does not poll the ring for most of a slot. The last rx_sleep_us are polled as before: the time
// from the last sample's arrival to the return of the read does not change.
static void rx_sleep(ocsdr_state *s, uint64_t end_ts)
{
  const uint64_t now = ocsdr_stream_time(s->stream);
  if (end_ts <= now)
    return;
  const double wait_us = (end_ts - now) * 1e6 / SAMPLE_RATE_HZ - s->args.rx_sleep_us;
  if (wait_us < 20) // not worth a sleep
    return;
  struct timespec t = {0, (long)(wait_us * 1e3)};
  clock_nanosleep(CLOCK_MONOTONIC, 0, &t, nullptr);
  s->rx_sleeps++;
  s->rx_slept_us += wait_us;
}

// ---- front end (FPGA register set v11+, sdr_addrs fd=1): frequency-domain symbols ----
// The FPGA sends two blocks per symbol (bins 0..1637 and 2458..4095, both channels interleaved) in the RX ring.
// ocsdr_read() keeps them here, keyed by the timestamp of the FFT window; OAI's nr_fep() asks for each symbol with
// the window it would use itself (ocsdr_get_rxdataF) and demodulates the samples when the symbol is not here. A
// symbol still in the ring (the last ones of a slot arrive while the RU thread waits for nr_fep) is taken from the
// ring without dequeuing it (look-ahead). Register set v12+: the FPGA only sends the symbols OAI asks for, a mask
// learnt from OAI's requests (one frame, 20 slots x 14 symbols).
// Writers (ocsdr_read, the look-ahead) hold fd_mtx; readers (the RU thread pool) use each entry's seqlock.
static constexpr unsigned FD_ENTRIES = 64; // ~2.3 slots of symbols
static constexpr unsigned FD_HALF = OCSDR_RX_FD_BINS * 2; // words per half block (2 channels)
static constexpr unsigned FD_LOOKAHEAD = 64;              // ring blocks the look-ahead examines
static constexpr uint64_t FD_SLOT = 61440;                // samples per slot (30 kHz, 122.88 MS/s)
// window start -> both halves of the symbol in host memory: FFT output starts ~8350 samples after the window's first
// sample (xsim, fft4096_rx), half B ends 4096 later, plus FIFO, arbiter and DMA (~600)
static constexpr uint64_t FD_ARRIVAL_SAMPLES = 8350 + 4096 + 600;

struct fd_entry {
  std::atomic<uint32_t> seq{0}; // odd while being written
  std::atomic<uint64_t> ts{UINT64_MAX};
  std::atomic<uint8_t>  halves{0}; // bit 0 half A, bit 1 half B
  uint32_t w[2][FD_HALF];          // [half][bin * 2 + channel]
};

struct fd_cache {
  fd_entry e[FD_ENTRIES];
  unsigned wr = 0;           // entry of the latest symbol (under mtx)
  std::mutex mtx;            // cache writers
  std::atomic<bool> reader_active{false}; // ocsdr_read is using the ring: no look-ahead
  std::atomic<uint64_t> hits{0}, misses{0}, waits{0}, ahead{0};
  uint64_t blocks = 0, dropped = 0, overflow = 0;
  // symbol mask (v12+): learnt from OAI's requests, written to the FPGA by ocsdr_read
  bool mask_hw = false;                  // the image has the mask
  std::atomic<uint16_t> learnt[20];      // per slot of the frame: symbols OAI asked for
  uint16_t programmed[20];               // mask in the FPGA
  std::atomic<bool> mask_dirty{false};
  uint64_t start_ts = 0;                 // grid start
  fd_cache()
  {
    for (int i = 0; i < 20; i++) {
      learnt[i] = 0;
      programmed[i] = 0x3fff;
    }
  }
};

// slot of the frame (0..19) and symbol of an FFT window start, on OAI's grid; false if it is not a window start
static bool fd_slot_symbol(const ocsdr_state *s, uint64_t win_ts, unsigned *slot, unsigned *sym)
{
  const uint64_t rel = win_ts + s->args.fd_adv + s->args.fd_nta, within = rel % FD_SLOT;
  *slot = (unsigned)((rel / FD_SLOT) % 20);
  if (within == 352) {
    *sym = 0;
    return true;
  }
  if (within < 4448 + 288 || (within - 4448 - 288) % 4384 != 0 || (within - 4448 - 288) / 4384 > 12)
    return false;
  *sym = (unsigned)((within - 4448 - 288) / 4384 + 1);
  return true;
}

// keeps one frequency-domain block in the cache (caller holds c->mtx); a half already there is not copied again
static void fd_store(fd_cache *c, uint64_t ts, uint16_t flags, const int16_t *iq)
{
  const unsigned half = (flags & OCSDR_RX_FLAG_FD_HALF_B) ? 1 : 0;
  fd_entry *e = &c->e[c->wr];
  if (e->ts.load(std::memory_order_relaxed) != ts) { // a new symbol: next entry
    c->wr = (c->wr + 1) % FD_ENTRIES;
    e = &c->e[c->wr];
    e->seq.fetch_add(1, std::memory_order_acq_rel); // odd: readers retry or give up
    e->halves.store(0, std::memory_order_relaxed);
    e->ts.store(ts, std::memory_order_relaxed);
  } else if (e->halves.load(std::memory_order_relaxed) & (1u << half)) {
    return; // taken earlier by the look-ahead
  } else {
    e->seq.fetch_add(1, std::memory_order_acq_rel);
  }
  memcpy(e->w[half], iq, sizeof(e->w[half]));
  e->halves.store(e->halves.load(std::memory_order_relaxed) | (1u << half), std::memory_order_relaxed);
  e->seq.fetch_add(1, std::memory_order_release); // even again: consistent
}

// takes the held block if it is a frequency-domain one (then releases it): true, else false (time-domain samples)
static bool fd_take(ocsdr_state *s, uint32_t dropped)
{
  const uint16_t flags = ocsdr_stream_rx_flags(s->stream);
  if (!(flags & OCSDR_RX_FLAG_FD))
    return false;
  fd_cache *c = s->fd;
  if (c) {
    std::lock_guard<std::mutex> lock(c->mtx);
    c->blocks++;
    c->dropped += dropped;
    c->overflow += (flags & OCSDR_RX_FLAG_FD_OVERFLOW) != 0;
    fd_store(c, s->rx_block_ts, flags, s->rx_block);
  }
  ocsdr_stream_rx_release(s->stream);
  return true;
}

// copies the symbol for antenna aid if the cache has it (both halves, not being overwritten): 0, else -1. Newest
// entries first: OAI asks for symbols soon after they arrive.
static int fd_lookup(fd_cache *c, openair0_timestamp_t win_ts, int aid, uint32_t *d)
{
  const unsigned newest = c->wr; // racy read: only a starting point
  for (unsigned n = 0; n < FD_ENTRIES; n++) {
    fd_entry *e = &c->e[(newest + FD_ENTRIES - n) % FD_ENTRIES];
    if (e->ts.load(std::memory_order_relaxed) != (uint64_t)win_ts)
      continue;
    const uint32_t s0 = e->seq.load(std::memory_order_acquire);
    if ((s0 & 1) || e->halves.load(std::memory_order_relaxed) != 3 || e->ts.load(std::memory_order_relaxed) != (uint64_t)win_ts)
      return -1;
    ocsdr_fd_extract(d, e->w[1], aid, OCSDR_RX_FD_BINS);                     // negative frequencies first
    ocsdr_fd_extract(d + OCSDR_RX_FD_BINS, e->w[0], aid, OCSDR_RX_FD_BINS);
    std::atomic_thread_fence(std::memory_order_acquire);
    return e->seq.load(std::memory_order_relaxed) == s0 ? 0 : -1; // overwritten while copying: OAI demodulates
  }
  return -1;
}

// look-ahead: frequency-domain blocks that arrived but are still in the ring go into the cache now. Only while
// ocsdr_read is not using the ring (in OAI the RU thread waits for nr_fep, so the ring is idle then).
static void fd_lookahead(ocsdr_state *s)
{
  fd_cache *c = s->fd;
  if (c->reader_active.load(std::memory_order_acquire))
    return;
  std::lock_guard<std::mutex> lock(c->mtx);
  if (c->reader_active.load(std::memory_order_acquire))
    return;
  for (unsigned k = 0; k < FD_LOOKAHEAD; k++) {
    const int16_t *iq;
    uint64_t ts;
    uint16_t flags;
    if (ocsdr_stream_rx_peek(s->stream, k, &iq, &ts, &flags) != 0)
      break;
    if (flags & OCSDR_RX_FLAG_FD) {
      fd_store(c, ts, flags, iq);
      c->ahead.fetch_add(1, std::memory_order_relaxed);
    }
  }
}

// OAI's trx_get_rxdataF_func: the symbol whose FFT window starts at win_ts, antenna aid, in the layout of
// nr_symbol_fep_ul_shifted (bins 2458..4095 then 0..1637). 0 if it was here, -1 otherwise.
static int ocsdr_get_rxdataF(openair0_device_t *device, openair0_timestamp_t win_ts, int aid, void *dst, int nbins)
{
  ocsdr_state *s = (ocsdr_state *)device->priv;
  fd_cache *c = s->fd;
  if (!c || nbins != (int)(2 * OCSDR_RX_FD_BINS) || aid < 0 || aid > 1)
    return -1;
  unsigned slot, sym;
  if (c->mask_hw && fd_slot_symbol(s, win_ts, &slot, &sym)) { // learn the symbols OAI uses
    const uint16_t bit = 1u << sym;
    if (!(c->learnt[slot].fetch_or(bit, std::memory_order_relaxed) & bit) && !(c->programmed[slot] & bit))
      c->mask_dirty.store(true, std::memory_order_release);
  }
  for (int attempt = 0; attempt < 3; attempt++) {
    if (attempt == 1) {
      fd_lookahead(s); // still in the ring?
    } else if (attempt == 2) {
      // not arrived yet: wait for it when it is due within fd_wait_us
      if (!s->args.fd_wait_us)
        break;
      const uint64_t due = (uint64_t)win_ts + FD_ARRIVAL_SAMPLES, now = ocsdr_stream_time(s->stream);
      const double wait_us = due > now ? (due - now) * 1e6 / SAMPLE_RATE_HZ : 0;
      if (wait_us > s->args.fd_wait_us)
        break;
      if (wait_us > 0) {
        struct timespec t = {0, (long)(wait_us * 1e3)};
        clock_nanosleep(CLOCK_MONOTONIC, 0, &t, nullptr);
      }
      c->waits.fetch_add(1, std::memory_order_relaxed);
      fd_lookahead(s);
    }
    if (fd_lookup(c, win_ts, aid, (uint32_t *)dst) == 0) {
      c->hits.fetch_add(1, std::memory_order_relaxed);
      return 0;
    }
  }
  c->misses.fetch_add(1, std::memory_order_relaxed);
  return -1;
}

// writes the learnt symbol mask to the FPGA (from ocsdr_read, the only thread writing stream registers while running),
// once two frames of requests were seen, and again when OAI asks for a symbol outside it
static void fd_mask_update(ocsdr_state *s)
{
  fd_cache *c = s->fd;
  if (!c->mask_hw || !c->mask_dirty.load(std::memory_order_acquire))
    return;
  if (ocsdr_stream_time(s->stream) < c->start_ts + 40 * FD_SLOT)
    return; // learning: the first two frames go out complete
  c->mask_dirty.store(false, std::memory_order_relaxed);
  uint32_t w[10] = {0};
  unsigned n = 0;
  for (int i = 0; i < 20; i++) {
    c->programmed[i] = c->learnt[i].load(std::memory_order_relaxed);
    for (int b = 0; b < 14; b++)
      if (c->programmed[i] & (1u << b)) {
        const unsigned bit = 14 * i + b;
        w[bit / 32] |= 1u << (bit % 32);
        n++;
      }
  }
  for (int i = 0; i < 10; i++)
    ocsdr_wr32(s->dev, OCSDR_ST_FD_MASK + 4 * i, w[i]);
  ocsdr_wr32(s->dev, OCSDR_ST_FD_CTRL, 1 | 4); // keep enabled, load the mask
  LOG_I(HW, "ocsdr: front end: the FPGA now sends the %u symbols per frame OAI asks for\n", n);
}

static int ocsdr_read(openair0_device_t *device, openair0_timestamp_t *ptimestamp, void **buff, int nsamps, int nb_ant)
{
  ocsdr_state *s = (ocsdr_state *)device->priv;
  const unsigned hw = s->hw_ch, bs = s->bs;
  struct reader_flag { // no look-ahead into the ring while this read uses it
    fd_cache *c;
    reader_flag(fd_cache *c_) : c(c_) { if (c) { std::lock_guard<std::mutex> l(c->mtx); c->reader_active.store(true, std::memory_order_release); } }
    ~reader_flag() { if (c) c->reader_active.store(false, std::memory_order_release); }
  } busy(s->fd);
  if (s->fd)
    fd_mask_update(s);

  auto zero = [&](unsigned from, unsigned n) {
    for (int a = 0; a < nb_ant; a++)
      memset((uint32_t *)buff[a] + from, 0, n * sizeof(uint32_t));
  };

  // The first block fixes the timeline; afterwards every sample is copied from the block that holds its timestamp,
  // or zero when the FPGA dropped that block, so the returned timestamps stay contiguous.
  if (s->rx_first) {
    uint32_t dropped = 0;
    do {
      if (ocsdr_stream_rx_acquire(s->stream, &s->rx_block, &s->rx_block_ts, &dropped, RING_TIMEOUT_US) != 0) {
        LOG_E(HW, "ocsdr: no RX block for %d ms\n", RING_TIMEOUT_US / 1000);
        return -1;
      }
    } while (fd_take(s, dropped));
    s->rx_next_ts = s->rx_block_ts;
    s->rx_first   = false;
  }
  *ptimestamp = s->rx_next_ts;

  unsigned pos = 0;
  while (pos < (unsigned)nsamps) {
    if (!s->running.load(std::memory_order_relaxed)) {
      zero(pos, nsamps - pos);
      break;
    }
    if (s->rx_block == nullptr) {
      uint32_t dropped = 0;
      int r = -ETIMEDOUT;
      if (s->args.rx_sleep_us) { // block not there yet: sleep until rx_sleep_us before the end of this read is due
        r = ocsdr_stream_rx_acquire(s->stream, &s->rx_block, &s->rx_block_ts, &dropped, 0);
        if (r != 0)
          rx_sleep(s, s->rx_next_ts + (nsamps - pos));
      }
      if (r != 0)
        r = ocsdr_stream_rx_acquire(s->stream, &s->rx_block, &s->rx_block_ts, &dropped, RING_TIMEOUT_US);
      if (r != 0) {
        s->rx_block = nullptr;
        LOG_W(HW, "ocsdr: no RX block for %d ms\n", RING_TIMEOUT_US / 1000);
        continue;
      }
      if (fd_take(s, dropped)) { // a frequency-domain block (front end): cached for ocsdr_get_rxdataF, not samples
        s->rx_block = nullptr;
        continue;
      }
      if (dropped != 0) {
        s->rx_dropped += dropped;
        LOG_W(HW, "ocsdr: the FPGA dropped %u RX block(s) before ts=%lu\n", dropped, s->rx_block_ts);
      }
    }
    if (s->rx_block_ts + bs <= s->rx_next_ts) {
      ocsdr_stream_rx_release(s->stream);
      s->rx_block = nullptr;
      continue;
    }
    if (s->rx_block_ts > s->rx_next_ts) {
      unsigned n = (unsigned)std::min<uint64_t>(s->rx_block_ts - s->rx_next_ts, nsamps - pos);
      zero(pos, n);
      pos += n;
      s->rx_next_ts += n;
      continue;
    }
    unsigned off = (unsigned)(s->rx_next_ts - s->rx_block_ts);
    unsigned n   = std::min(bs - off, nsamps - pos);
    uint32_t *out[MAX_CHANNELS] = {};
    const unsigned nout = std::min((unsigned)nb_ant, hw);
    for (unsigned a = 0; a < nout; a++)
      out[a] = (uint32_t *)buff[a] + pos;
    ocsdr_rx_unpack(out, nout, (const uint32_t *)s->rx_block + (size_t)off * hw, hw, n);
    for (int a = nout; a < nb_ant; a++)
      memset((uint32_t *)buff[a] + pos, 0, n * sizeof(uint32_t));
    for (unsigned j = 0; j < n; j += LEVEL_STEP) {
      const int16_t *v = (const int16_t *)buff[0] + 2 * (pos + j);
      s->rx_sq += (double)v[0] * v[0] + (double)v[1] * v[1];
      s->rx_n++;
    }
    pos += n;
    s->rx_next_ts += n;
    if (off + n == bs) {
      if (++s->rx_blocks == LEVEL_BLOCKS) {
        LOG_I(HW, "ocsdr: RX1 %.1f dBFS, %lu sleep(s) for %.0f ms\n", dbfs(s->rx_sq, s->rx_n), s->rx_sleeps, s->rx_slept_us / 1e3);
        if (s->fd) {
          fd_cache *c = s->fd;
          LOG_I(HW, "ocsdr: front end: %lu FD blocks (%lu dropped by the FPGA, %lu with FFT overflow), symbols taken by OAI %lu (%lu after a wait, %lu FD blocks taken ahead from the ring), demodulated by OAI %lu\n",
                c->blocks, c->dropped, c->overflow, c->hits.exchange(0), c->waits.exchange(0), c->ahead.exchange(0), c->misses.exchange(0));
          c->blocks = c->dropped = c->overflow = 0;
        }
        s->rx_sleeps   = 0;
        s->rx_slept_us = 0;
        s->rx_sq = 0;
        s->rx_n = s->rx_blocks = 0;
      }
      ocsdr_stream_rx_release(s->stream);
      s->rx_block = nullptr;
    }
  }
  if (s->args.tx_lead_stats) { // age of the samples handed over: how far behind real time the reader runs
    const int64_t age_us = ((int64_t)ocsdr_stream_time(s->stream) - (int64_t)s->rx_next_ts) * 1e6 / SAMPLE_RATE_HZ;
    s->rx_age_max = std::max(s->rx_age_max, age_us);
    s->rx_age_sum += age_us;
    s->rx_age_n++;
    if (age_us > 200)
      s->rx_age_over200++;
  }
  return nsamps;
}

// ---- transmit ----

static void tx_check_counters(ocsdr_state *s)
{
  ocsdr_stream_counters c;
  if (ocsdr_stream_counters_read(s->stream, &c) != 0)
    return;
  if (c.tx_late != s->last_late)
    LOG_W(HW, "ocsdr: %u TX block(s) arrived late and were dropped\n", c.tx_late - s->last_late);
  if (c.tx_underflow != s->last_underflow)
    LOG_W(HW, "ocsdr: %u TX underflow(s)\n", c.tx_underflow - s->last_underflow);
  s->last_late      = c.tx_late;
  s->last_underflow = c.tx_underflow;
}

static void tx_lead_record(ocsdr_state *s)
{
  const uint64_t now = ocsdr_stream_time(s->stream);
  const int64_t lead_us = ((int64_t)s->tx_block_ts - (int64_t)now) * 1e6 / SAMPLE_RATE_HZ;
  s->lead_min = std::min(s->lead_min, lead_us);
  s->lead_n++;
  if (lead_us < 0)
    s->lead_neg++;
  else
    s->lead_hist[std::min<int64_t>(lead_us / 50, 15)]++;
  if (now < s->lead_next_log)
    return;
  if (s->lead_next_log) {
    char h[16 * 8], *p = h;
    for (int i = 0; i < 16; i++)
      p += sprintf(p, " %lu", s->lead_hist[i]);
    LOG_I(HW, "ocsdr: TX lead: %lu blocks, min %ld us, %lu behind time, hist(50 us bins)%s\n", s->lead_n, s->lead_min, s->lead_neg, h);
    LOG_I(HW, "ocsdr: RX age at read: %lu reads, mean %ld us, max %ld us, %lu over 200 us\n", s->rx_age_n,
          s->rx_age_n ? s->rx_age_sum / (int64_t)s->rx_age_n : 0, s->rx_age_max, s->rx_age_over200);
  }
  s->rx_age_max = s->rx_age_sum = 0;
  s->rx_age_n = s->rx_age_over200 = 0;
  s->lead_min = INT64_MAX;
  s->lead_n = s->lead_neg = 0;
  memset(s->lead_hist, 0, sizeof(s->lead_hist));
  s->lead_next_log = now + (uint64_t)SAMPLE_RATE_HZ;
}

static void tx_flush(ocsdr_state *s, bool end_of_burst)
{
  if (s->tx_block == nullptr) {
    if (end_of_burst && !s->tx_closed) {
      ocsdr_stream_tx_end_burst(s->stream);
      s->tx_closed = true;
    }
    return;
  }
  memset(s->tx_block + 2 * s->hw_ch * s->tx_fill, 0, (size_t)(s->bs - s->tx_fill) * 2 * s->hw_ch * sizeof(int16_t));
  if (s->args.tx_lead_stats)
    tx_lead_record(s);
  ocsdr_stream_tx_commit(s->stream, s->tx_block_ts, OCSDR_TX_TIMED | (end_of_burst ? OCSDR_TX_EOB : 0));
  s->tx_block  = nullptr;
  s->tx_closed = end_of_burst;
  if ((++s->tx_since_chk & 1023) == 0)
    tx_check_counters(s);
  if (++s->tx_blocks == LEVEL_BLOCKS) {
    LOG_I(HW, "ocsdr: TX1 %.1f dBFS (all samples, idle included), %lu of %lu checked samples clipped\n",
          dbfs(s->tx_sq, s->tx_n), s->tx_clip, s->tx_n);
    if (s->args.tx_flush_us)
      LOG_I(HW, "ocsdr: %lu TX block(s) sent early by the flusher, %lu late sample(s) dropped\n", s->tx_early, s->tx_trimmed);
    s->tx_sq = 0;
    s->tx_n = s->tx_clip = s->tx_blocks = 0;
  }
}

static int ocsdr_write(openair0_device_t *device, openair0_timestamp_t timestamp, void **buff, int nsamps, int nb_ant, int flags)
{
  ocsdr_state *s = (ocsdr_state *)device->priv;
  if (!s->running.load(std::memory_order_relaxed) || s->args.rx_only)
    return nsamps;
  timestamp -= device->openair0_cfg->command_line_sample_advance + device->openair0_cfg->tx_sample_advance;

  const unsigned hw = s->hw_ch, bs = s->bs;
  const unsigned nch = std::min((unsigned)nb_ant, hw);
  const int burst = flags & 0xf;
  uint64_t ts = timestamp;
  std::lock_guard<std::mutex> lock(s->tx_mtx);

  // A block holds consecutive samples. A short forward jump is zero-filled inside the open block; anything else
  // closes the block as the end of a burst and opens a new one at the new timestamp.
  if (s->tx_block != nullptr && ts != s->tx_block_ts + s->tx_fill) {
    if (ts > s->tx_block_ts + s->tx_fill && ts < s->tx_block_ts + bs) {
      unsigned gap = (unsigned)(ts - (s->tx_block_ts + s->tx_fill));
      memset(s->tx_block + 2 * hw * s->tx_fill, 0, (size_t)gap * 2 * hw * sizeof(int16_t));
      s->tx_fill += gap;
    } else {
      tx_flush(s, true);
    }
  }

  const int32_t lim = INT16_MAX;
  const unsigned sh = s->args.tx_shift;
  unsigned pos = 0;
  // the start of this write may belong to a block the flusher already sent (zero-filled): drop those samples
  if (s->tx_block == nullptr && ts < s->tx_sent_until) {
    pos = (unsigned)std::min<uint64_t>(s->tx_sent_until - ts, (uint64_t)nsamps);
    s->tx_trimmed += pos;
  }
  while (pos < (unsigned)nsamps) {
    if (s->tx_block == nullptr) {
      if (ocsdr_stream_tx_acquire(s->stream, &s->tx_block, RING_TIMEOUT_US) != 0) {
        s->tx_block = nullptr;
        LOG_W(HW, "ocsdr: TX ring full for %d ms, dropping %d samples\n", RING_TIMEOUT_US / 1000, nsamps - pos);
        return nsamps;
      }
      s->tx_block_ts = ts + pos;
      s->tx_fill     = 0;
    }
    unsigned n = std::min(bs - s->tx_fill, nsamps - pos);
    const int16_t *in[MAX_CHANNELS] = {};
    for (unsigned c = 0; c < nch; c++)
      in[c] = (const int16_t *)buff[c] + 2 * pos;
    ocsdr_tx_pack(s->tx_block + 2 * (size_t)s->tx_fill * hw, in, nch, hw, n, sh);
    for (unsigned i = 0; i < n; i += LEVEL_STEP) { // level and clipping from one channel-0 sample in LEVEL_STEP
      const int32_t re = (int32_t)in[0][2 * i] * (1 << sh), im = (int32_t)in[0][2 * i + 1] * (1 << sh);
      s->tx_sq += (double)re * re + (double)im * im;
      s->tx_n++;
      s->tx_clip += re > lim || re < -lim || im > lim || im < -lim;
    }
    s->tx_fill += n;
    pos += n;
    if (s->tx_fill == bs)
      tx_flush(s, false);
  }
  if (burst == TX_BURST_END || burst == TX_BURST_START_AND_END || burst == TX_BURST_END_NO_TIME_SPEC)
    tx_flush(s, true);
  return nsamps;
}

// front end TX: the symbols of a slot from OAI's txdataF, two blocks per symbol (bins 0..1637 = txdataF[1638..3275],
// bins 2458..4095 = txdataF[0..1637]), both antennas interleaved, header = start of the symbol's cyclic prefix
static int ocsdr_write_fd(openair0_device_t *device, openair0_timestamp_t timestamp, void **txdataF, int first_symbol,
                          int num_symbols, int nb_ant, int flags)
{
  ocsdr_state *s = (ocsdr_state *)device->priv;
  if (!s->running.load(std::memory_order_relaxed) || s->args.rx_only)
    return num_symbols;
  const uint64_t slot_ts = timestamp - device->openair0_cfg->command_line_sample_advance - device->openair0_cfg->tx_sample_advance;
  const unsigned nch = std::min((unsigned)nb_ant, s->hw_ch);
  std::lock_guard<std::mutex> lock(s->tx_mtx);
  for (int l = first_symbol; l < first_symbol + num_symbols; l++) {
    // numerology 1 at 122.88 MS/s: symbol 0 4448 samples (CP 352), the others 4384 (CP 288)
    const uint64_t ts = slot_ts + (l == 0 ? 0 : 4448 + (uint64_t)(l - 1) * 4384);
    const int16_t *sym = nullptr;
    for (int half = 0; half < 2; half++) {
      int16_t *blk;
      if (ocsdr_stream_tx_acquire(s->stream, &blk, RING_TIMEOUT_US) != 0) {
        LOG_W(HW, "ocsdr: TX ring full for %d ms, front end symbols dropped\n", RING_TIMEOUT_US / 1000);
        return num_symbols;
      }
      const int16_t *in[MAX_CHANNELS] = {};
      for (unsigned c = 0; c < nch; c++) {
        sym = (const int16_t *)txdataF[c] + 2 * 4096 * (l - first_symbol);
        in[c] = sym + 2 * (half == 0 ? OCSDR_RX_FD_BINS : 0);
      }
      ocsdr_tx_pack(blk, in, nch, s->hw_ch, OCSDR_RX_FD_BINS, 0);
      ocsdr_stream_tx_commit(s->stream, ts, OCSDR_TX_FD | (half ? OCSDR_TX_FD_HALF_B : 0) | OCSDR_TX_FD_SYMBOL(l));
    }
  }
  (void)flags;
  if ((++s->tx_since_chk & 255) == 0) {
    const uint32_t late = ocsdr_rd32(s->dev, OCSDR_ST_FDTX_LATE), err = ocsdr_rd32(s->dev, OCSDR_ST_FDTX_ERR);
    if (late != s->fdtx_late || err != s->fdtx_err)
      LOG_W(HW, "ocsdr: front end TX: %u symbol(s) late, %u half mismatch(es)\n", late - s->fdtx_late, err - s->fdtx_err);
    s->fdtx_late = late;
    s->fdtx_err  = err;
  }
  return num_symbols;
}

// ---- control ----

static int set_gains(ocsdr_state *s, unsigned nch)
{
  int index = RX_GAIN_INDEX_0 + (int)lround(2 * s->args.rx_gain);
  for (unsigned c = 1; c <= nch; c++) {
    const double g = (c == 2 && s->args.rx2_gain >= 0) ? s->args.rx2_gain : s->args.rx_gain;
    const int idx = RX_GAIN_INDEX_0 + (int)lround(2 * g);
    if (ocsdr_rf_tx_atten(s->dev, c, s->args.tx_atten) != 0 || ocsdr_rf_rx_gain(s->dev, c, idx) != 0)
      return -1;
  }
  LOG_I(HW, "ocsdr: TX attenuation %.2f dB, RX gain index %d (%.1f dB) on %u channel(s)\n", s->args.tx_atten, index, s->args.rx_gain, nch);
  return 0;
}

// Sends the open TX block, zero-filled, when its start time is less than tx_flush_us away
static void tx_flusher_loop(ocsdr_state *s)
{
  const uint64_t margin = (uint64_t)(s->args.tx_flush_us * 1e-6 * SAMPLE_RATE_HZ);
  // above OAI's real-time threads (they use max - 2): a starved flusher is no flusher
  struct sched_param sp = {.sched_priority = sched_get_priority_max(SCHED_RR) - 1};
  if (pthread_setschedparam(pthread_self(), SCHED_RR, &sp))
    LOG_W(HW, "ocsdr: TX flusher runs without real-time priority\n");
  while (s->running.load(std::memory_order_relaxed)) {
    usleep(20);
    std::lock_guard<std::mutex> lock(s->tx_mtx);
    if (s->tx_block == nullptr || ocsdr_stream_time(s->stream) + margin < s->tx_block_ts)
      continue;
    s->tx_sent_until = s->tx_block_ts + s->bs;
    s->tx_early++;
    tx_flush(s, false);
  }
}

static int ocsdr_start(openair0_device_t *device)
{
  ocsdr_state *s = (ocsdr_state *)device->priv;
  s->rx_first = true;
  s->rx_block = nullptr;
  s->tx_block = nullptr;
  s->running  = true;
  int ret = ocsdr_stream_rx_start(s->stream);
  if (ret == 0)
    ret = ocsdr_stream_tx_start(s->stream);
  if (ret != 0) {
    LOG_E(HW, "ocsdr: cannot start streaming (%s)\n", strerror(-ret));
    return -1;
  }
  LOG_I(HW, "ocsdr: streaming started at sample clock %lu\n", ocsdr_stream_time(s->stream));
  if (s->fd) {
    // OAI's slot grid is the device timestamp (ts_offset 0): slot n starts at n * 61440. The FPGA grid starts at a
    // slot boundary about 2 ms ahead, moved back by OAI's window advance (nb_prefix / divisor) and N_TA_offset.
    const uint64_t slot = 61440, now = ocsdr_stream_time(s->stream);
    const uint64_t start = (now / slot + 4) * slot - s->args.fd_adv - s->args.fd_nta;
    ocsdr_wr32(s->dev, OCSDR_ST_FD_SCALE, 0x555);
    s->fd->start_ts = start;
    if (s->fd->mask_hw) { // all symbols until the mask is learnt
      s->fd->mask_dirty.store(true, std::memory_order_release); // write the learnt mask once learning ends
      for (int i = 0; i < 10; i++)
        ocsdr_wr32(s->dev, OCSDR_ST_FD_MASK + 4 * i, 0xffffffffu);
      ocsdr_wr32(s->dev, OCSDR_ST_FD_START_SLOT, (uint32_t)(((start + s->args.fd_adv + s->args.fd_nta) / FD_SLOT) % 20));
    }
    ocsdr_wr32(s->dev, OCSDR_ST_FD_START_LO, (uint32_t)start);
    ocsdr_wr32(s->dev, OCSDR_ST_FD_START_HI, (uint32_t)(start >> 32));
    ocsdr_wr32(s->dev, OCSDR_ST_FD_CTRL, 1);
    LOG_I(HW, "ocsdr: front end on: FPGA OFDM demodulation from sample %lu (advance %u, N_TA_offset %u)\n", start,
          s->args.fd_adv, s->args.fd_nta);
  }
  if (s->fdtx) {
    ocsdr_wr32(s->dev, OCSDR_ST_FDTX_SCALE, s->args.fd_tx_scale);
    ocsdr_wr32(s->dev, OCSDR_ST_FDTX_CTRL, 1);
    s->fdtx_late = ocsdr_rd32(s->dev, OCSDR_ST_FDTX_LATE);
    s->fdtx_err  = ocsdr_rd32(s->dev, OCSDR_ST_FDTX_ERR);
    LOG_I(HW, "ocsdr: front end TX on: FPGA OFDM modulation, schedule 0x%03x\n", s->args.fd_tx_scale);
  }
  if (s->args.tx_flush_us) {
    s->tx_sent_until = 0;
    s->tx_flusher = std::thread(tx_flusher_loop, s);
    LOG_I(HW, "ocsdr: TX blocks still open %u us before their start are sent zero-filled\n", s->args.tx_flush_us);
  }
  return 0;
}

static int ocsdr_stop(openair0_device_t *device)
{
  ocsdr_state *s = (ocsdr_state *)device->priv;
  bool expected = true;
  if (!s->running.compare_exchange_strong(expected, false))
    return 0;
  if (s->tx_flusher.joinable())
    s->tx_flusher.join();
  tx_flush(s, true);
  if (s->fd)
    ocsdr_wr32(s->dev, OCSDR_ST_FD_CTRL, 0);
  if (s->fdtx) {
    LOG_I(HW, "ocsdr: front end TX: %u symbols played, %u late, %u inverse FFT overflow(s)\n",
          ocsdr_rd32(s->dev, OCSDR_ST_FDTX_PLAYED), ocsdr_rd32(s->dev, OCSDR_ST_FDTX_LATE), ocsdr_rd32(s->dev, OCSDR_ST_FDTX_OVF));
    ocsdr_wr32(s->dev, OCSDR_ST_FDTX_CTRL, 0);
  }
  ocsdr_stream_tx_stop(s->stream);
  ocsdr_stream_rx_stop(s->stream);
  tx_check_counters(s);
  LOG_I(HW, "ocsdr: stopped, %lu RX block(s) dropped by the FPGA\n", s->rx_dropped);
  return 0;
}

static void ocsdr_end(openair0_device_t *device)
{
  ocsdr_state *s = (ocsdr_state *)device->priv;
  if (s == nullptr)
    return;
  ocsdr_stop(device);
  if (s->stream != nullptr)
    ocsdr_stream_close(s->stream);
  if (s->dev != nullptr) {
    ocsdr_tx_gate(s->dev, false, 0, 0, 0);
    ocsdr_rf_frontend(s->dev, false, 0.0);
    ocsdr_close(s->dev);
  }
  delete s->fd;
  delete s;
  device->priv = nullptr;
}

static int ocsdr_set_freq(openair0_device_t *device, openair0_config_t *cfg)
{
  ocsdr_state *s = (ocsdr_state *)device->priv;
  LOG_I(HW, "ocsdr: TX %.3f MHz, RX %.3f MHz\n", cfg[0].tx_freq[0] / 1e6, cfg[0].rx_freq[0] / 1e6);
  return ocsdr_rf_set_freq(s->dev, (uint64_t)llround(cfg[0].tx_freq[0]), (uint64_t)llround(cfg[0].rx_freq[0])) == 0 ? 0 : -1;
}

static int ocsdr_set_gains(openair0_device_t *device, openair0_config_t *cfg)
{
  // gains come from sdr_addrs (tx_atten, rx_gain), as in the OCUDU backend
  return 0;
}

static int ocsdr_get_stats(openair0_device_t *device)
{
  return 0;
}

static int ocsdr_reset_stats(openair0_device_t *device)
{
  return 0;
}

extern "C" int device_init(openair0_device_t *device, openair0_config_t *openair0_cfg)
{
  if (device->priv != nullptr) {
    LOG_E(HW, "ocsdr: device already initialised\n");
    return 0;
  }
  ocsdr_state *s = new ocsdr_state;
  if (!parse_args(openair0_cfg[0].sdr_addrs, s->args)) {
    delete s;
    return -1;
  }
  if (fabs(openair0_cfg[0].sample_rate - SAMPLE_RATE_HZ) >= 1.0) {
    LOG_E(HW, "ocsdr: the sampling rate must be 122.88 MHz (got %.0f Hz; do not use -E)\n", openair0_cfg[0].sample_rate);
    delete s;
    return -1;
  }
  const unsigned nch = std::max(openair0_cfg[0].tx_num_channels, openair0_cfg[0].rx_num_channels);
  if (openair0_cfg[0].tx_num_channels != openair0_cfg[0].rx_num_channels || nch < 1 || nch > MAX_CHANNELS) {
    LOG_E(HW, "ocsdr: nb_tx and nb_rx must be equal, 1 or 2 (got %d and %d)\n", openair0_cfg[0].tx_num_channels, openair0_cfg[0].rx_num_channels);
    delete s;
    return -1;
  }

  s->dev = s->args.host.empty() ? ocsdr_open(s->args.board) : ocsdr_open_net(s->args.host.c_str());
  if (s->dev == nullptr) {
    LOG_E(HW, "ocsdr: cannot open board %s (our FPGA image)\n", s->args.host.empty() ? std::to_string(s->args.board).c_str() : s->args.host.c_str());
    delete s;
    return -1;
  }

  const uint64_t tx_hz = (uint64_t)llround(openair0_cfg[0].tx_freq[0]), rx_hz = (uint64_t)llround(openair0_cfg[0].rx_freq[0]);
  if (!s->args.skip_init) {
    LOG_I(HW, "ocsdr: bringing up clocks, JESD204 links and AD9371 (about 5 s)\n");
    if (ocsdr_clock_init(s->dev) != 0) {
      LOG_E(HW, "ocsdr: HMC7044 clock init failed\n");
      ocsdr_close(s->dev);
      delete s;
      return -1;
    }
    ocsdr_rf_config rf = {};
    rf.tx_hz = tx_hz;
    rf.rx_hz = rx_hz;
    ocsdr_rf_report rep = {};
    if (ocsdr_rf_init(s->dev, &rf, &rep) != 0) {
      LOG_E(HW,
            "ocsdr: RF bring-up failed: MCS 0x%02x, PLLs 0x%02x, deframer 0x%02x, FPGA links TX %s RX %s\n",
            rep.mcs_status, rep.pll_lock, rep.deframer,
            ocsdr_jesd_tx_state_name(rep.fpga_tx_state), ocsdr_jesd_rx_state_name(rep.fpga_rx_state));
      ocsdr_close(s->dev);
      delete s;
      return -1;
    }
    LOG_I(HW, "ocsdr: RF up, TX %.3f MHz, RX %.3f MHz\n", tx_hz / 1e6, rx_hz / 1e6);
  } else if (ocsdr_rf_set_freq(s->dev, tx_hz, rx_hz) != 0) {
    LOG_E(HW, "ocsdr: cannot tune the LOs\n");
  }

  int ret = ocsdr_stream_open(s->dev, s->args.ring, &s->stream);
  if (ret != 0) {
    LOG_E(HW, "ocsdr: cannot open the streaming rings (%s): is the ocsdr_dma driver loaded?\n", strerror(-ret));
    ocsdr_close(s->dev);
    delete s;
    return -1;
  }
  s->hw_ch = ocsdr_stream_channels(s->stream);
  s->bs    = ocsdr_stream_block_samples(s->stream);
  if (nch > s->hw_ch) {
    LOG_E(HW, "ocsdr: %u antennas configured, but this FPGA image carries %u channel(s)\n", nch, s->hw_ch);
    ocsdr_stream_close(s->stream);
    ocsdr_close(s->dev);
    delete s;
    return -1;
  }
  if (s->args.frontend) {
    if (ocsdr_rf_frontend(s->dev, true, s->args.rx_atten) != 0)
      LOG_E(HW, "ocsdr: cannot set the RF front end\n");
    else
      LOG_I(HW, "ocsdr: RF front end on, RX attenuators %.1f dB\n", s->args.rx_atten);
    if (s->args.rx1_atten >= 0 && ocsdr_rx_atten_set(s->dev, 1, s->args.rx1_atten) == 0)
      LOG_I(HW, "ocsdr: RX1 attenuator %.1f dB\n", s->args.rx1_atten);
  } else {
    LOG_W(HW, "ocsdr: RF front end left in its safe state (TX off): leakage only\n");
  }
  if (s->args.rx_only) {
    // keep the RX path as set above, shut both transmitters and take their switches off the connectors
    uint32_t v = ocsdr_rfctrl(s->dev);
    v &= ~(OCSDR_RF_TX1_SPDT1 | OCSDR_RF_TX1_SPDT2 | OCSDR_RF_TX2_SPDT1 | OCSDR_RF_TX2_SPDT2 | OCSDR_RF_ORX1_SPDT
           | OCSDR_RF_ORX2_SPDT);
    v |= OCSDR_RF_TX1_SHUTDOWN | OCSDR_RF_TX2_SHUTDOWN;
    ocsdr_rfctrl_set(s->dev, v);
    s->args.tx_atten = TX_ATTEN_MAX_DB;
    LOG_I(HW, "ocsdr: receive only: TX shut down (RF control 0x%08x), TX samples dropped\n", ocsdr_rfctrl(s->dev));
  }
  if (s->args.tx_gate) {
    if (ocsdr_tx_gate(s->dev,
                      true,
                      s->args.tx_gate_lead,
                      s->args.tx_gate_lag,
                      s->args.tx_gate_mode == 0   ? 0u
                      : s->args.tx_gate_mode == 1 ? OCSDR_TXGATE_LOAD
                                                  : OCSDR_TXGATE_LOAD | OCSDR_TXGATE_PA_ON)
        != 0) {
      LOG_E(HW, "ocsdr: TX gating needs FPGA image v16s or later\n");
      return -1;
    }
    if (s->args.host.empty())
      LOG_I(HW, "ocsdr: TX gating over PCIe needs an image that gates on the PCIe TX stream, else the gate stays open\n");
    LOG_I(HW,
          "ocsdr: TX gating on between bursts, mode %u (lead %u, lag %u samples)\n",
          s->args.tx_gate_mode,
          s->args.tx_gate_lead,
          s->args.tx_gate_lag);
  }
  set_gains(s, s->hw_ch);

  openair0_cfg[0].tx_sample_advance = s->args.tx_advance;
  openair0_cfg[0].tx_bw             = 100e6;
  openair0_cfg[0].rx_bw             = 100e6;
  LOG_I(HW, "ocsdr: %uT%uR on a %u-channel image, %u samples per block, TX advance %d samples, TX << %u\n",
        nch, nch, s->hw_ch, s->bs, s->args.tx_advance, s->args.tx_shift);
  if (s->args.fd_tx) {
    const uint32_t regset = ocsdr_rd32(s->dev, 0x3000) & 0xff;
    if (!s->args.host.empty() || s->hw_ch != 2 || regset < 0x0d)
      LOG_E(HW, "ocsdr: fd_tx=1 needs our image with the front end TX (register set v13+, PCIe, 2 channels; this one: v%u): OAI modulates\n", regset);
    else
      s->fdtx = true;
  }
  if (s->args.fd) {
    const uint32_t regset = ocsdr_rd32(s->dev, 0x3000) & 0xff; // OCSDR_OWN_ID: 0x0C1F00vv
    if (!s->args.host.empty() || s->hw_ch != 2 || regset < 0x0b)
      LOG_E(HW, "ocsdr: fd=1 needs our image with the front end (register set v11+, PCIe, 2 channels; this one: v%u, %u channel(s)%s): OAI demodulates\n",
            regset, s->hw_ch, s->args.host.empty() ? "" : ", Ethernet");
    else
    {
      s->fd = new fd_cache; // once, at start-up
      s->fd->mask_hw = regset >= 0x0c;
      if (!s->fd->mask_hw)
        LOG_W(HW, "ocsdr: register set v%u: the FPGA sends every symbol (the symbol mask needs v12)\n", regset);
    }
  }

  device->priv                 = s;
  device->type                 = OCSDR_DEV;
  device->host_type            = RAU_HOST;
  device->openair0_cfg         = openair0_cfg;
  device->trx_start_func       = ocsdr_start;
  device->trx_stop_func        = ocsdr_stop;
  device->trx_end_func         = ocsdr_end;
  device->trx_read_func        = ocsdr_read;
  device->trx_write_func       = ocsdr_write;
  device->trx_get_stats_func   = ocsdr_get_stats;
  device->trx_reset_stats_func = ocsdr_reset_stats;
  device->trx_set_freq_func    = ocsdr_set_freq;
  device->trx_set_gains_func   = ocsdr_set_gains;
  device->trx_get_rxdataF_func = s->fd ? ocsdr_get_rxdataF : nullptr;
  device->trx_write_fd_func    = s->fdtx ? ocsdr_write_fd : nullptr;
  return 0;
}
