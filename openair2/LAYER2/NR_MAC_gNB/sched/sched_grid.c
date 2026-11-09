/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "sched_grid.h"

#include <stdlib.h>
#include <string.h>

#define CCE_WORDS ((GRID_MAX_CCES + 63) / 64)
#define POOL_WORDS ((GRID_POOL_SIZE + 63) / 64)

enum { UNDO_ALLOC, UNDO_CCES, UNDO_POOL };

typedef struct {
  int64_t slot; ///< absolute slot held, -1 if never used
  uint64_t rb[GRID_SYMBOLS][GRID_RB_WORDS];
  int8_t dir[GRID_SYMBOLS]; ///< -1 unused, 0 DL, 1 UL: a flexible symbol takes the direction of its first use
  int16_t beam[GRID_SYMBOLS]; ///< -1 unused
  uint16_t users[GRID_SYMBOLS]; ///< allocations on the symbol, to release the direction and beam
  uint64_t cce[GRID_MAX_CORESETS][CCE_WORDS];
  uint64_t pool[GRID_MAX_POOLS][POOL_WORDS];
  int n_alloc;
  grid_alloc_t alloc[GRID_MAX_ALLOCS];
} slot_state_t;

struct sched_grid {
  grid_cfg_t cfg;
  int64_t current;
  slot_state_t *slots;
};

static void slot_reset(slot_state_t *s, int64_t slot)
{
  memset(s, 0, sizeof(*s));
  s->slot = slot;
  memset(s->dir, -1, sizeof(s->dir));
  for (int i = 0; i < GRID_SYMBOLS; i++)
    s->beam[i] = -1;
}

sched_grid_t *grid_create(const grid_cfg_t *cfg)
{
  if (cfg->n_rb < 1 || cfg->n_rb > GRID_MAX_RB || cfg->horizon_slots < 1 || cfg->period_slots < 1 || cfg->period_slots > GRID_MAX_PERIOD)
    return NULL;
  sched_grid_t *g = calloc(1, sizeof(*g));
  if (g == NULL)
    return NULL;
  g->cfg = *cfg;
  g->slots = calloc(cfg->horizon_slots, sizeof(*g->slots));
  if (g->slots == NULL) {
    free(g);
    return NULL;
  }
  for (int i = 0; i < cfg->horizon_slots; i++)
    slot_reset(&g->slots[i], -1);
  return g;
}

void grid_free(sched_grid_t *grid)
{
  if (grid == NULL)
    return;
  free(grid->slots);
  free(grid);
}

void grid_set_current(sched_grid_t *grid, int64_t slot)
{
  grid->current = slot;
}

static grid_err_t slot_index(const sched_grid_t *g, int64_t slot)
{
  if (slot < g->current)
    return GRID_ERR_PAST;
  if (slot >= g->current + g->cfg.horizon_slots)
    return GRID_ERR_HORIZON;
  return GRID_OK;
}

/* the state of an absolute slot inside the horizon, cleared when the ring entry held an older slot */
static slot_state_t *slot_get(sched_grid_t *g, int64_t slot)
{
  slot_state_t *s = &g->slots[slot % g->cfg.horizon_slots];
  if (s->slot != slot)
    slot_reset(s, slot);
  return s;
}

/* read-only view: NULL means an empty slot */
static const slot_state_t *slot_peek(const sched_grid_t *g, int64_t slot)
{
  const slot_state_t *s = &g->slots[slot % g->cfg.horizon_slots];
  return s->slot == slot ? s : NULL;
}

grid_sym_t grid_symbol_type(const sched_grid_t *grid, int64_t slot, int symbol)
{
  return grid->cfg.sym[slot % grid->cfg.period_slots][symbol];
}

void grid_txn_begin(grid_txn_t *txn)
{
  txn->len = 0;
}

void grid_txn_commit(grid_txn_t *txn)
{
  txn->len = 0;
}

/* bits [start, start + n) of word w, as a mask of that word */
static uint64_t word_mask(int w, int start, int n)
{
  const int lo = start - 64 * w, hi = start + n - 64 * w; /* bit range in this word, may exceed [0, 64) */
  if (hi <= 0 || lo >= 64)
    return 0;
  const uint64_t up = hi >= 64 ? ~0ULL : (1ULL << hi) - 1;
  const uint64_t down = lo <= 0 ? ~0ULL : ~((1ULL << lo) - 1);
  return up & down;
}

static bool rbs_free(const uint64_t *bits, int start, int n)
{
  for (int w = start / 64; w <= (start + n - 1) / 64; w++)
    if (bits[w] & word_mask(w, start, n))
      return false;
  return true;
}

static void rbs_set(uint64_t *bits, int start, int n, bool on)
{
  for (int w = start / 64; w <= (start + n - 1) / 64; w++) {
    const uint64_t m = word_mask(w, start, n);
    bits[w] = on ? bits[w] | m : bits[w] & ~m;
  }
}

/* direction and beam of one symbol for a use */
/* direction and beam of one symbol for a use; types: the symbol directions of the slot in the period */
static grid_err_t symbol_allows(const sched_grid_t *g, const slot_state_t *s, const grid_sym_t *types, int sym, bool ul, int beam)
{
  const grid_sym_t type = types[sym];
  if ((type == GRID_SYM_DL && ul) || (type == GRID_SYM_UL && !ul))
    return GRID_ERR_DIRECTION;
  if (s != NULL && s->dir[sym] >= 0 && s->dir[sym] != ul)
    return GRID_ERR_DIRECTION;
  if (g->cfg.analog_beams && s != NULL && s->beam[sym] >= 0 && s->beam[sym] != beam)
    return GRID_ERR_BEAM;
  return GRID_OK;
}

static const grid_sym_t *slot_types(const sched_grid_t *g, int64_t slot)
{
  return g->cfg.sym[slot % g->cfg.period_slots];
}

grid_err_t grid_check(const sched_grid_t *grid, int64_t slot, const grid_alloc_t *a)
{
  const grid_err_t e = slot_index(grid, slot);
  if (e != GRID_OK)
    return e;
  if (a->n_rb < 1 || a->rb_start < 0 || a->rb_start + a->n_rb > grid->cfg.n_rb || a->n_sym < 1 || a->sym_start < 0
      || a->sym_start + a->n_sym > GRID_SYMBOLS)
    return GRID_ERR_INVALID;
  const slot_state_t *s = slot_peek(grid, slot);
  const grid_sym_t *types = slot_types(grid, slot);
  for (int sym = a->sym_start; sym < a->sym_start + a->n_sym; sym++) {
    const grid_err_t d = symbol_allows(grid, s, types, sym, a->ul, a->beam);
    if (d != GRID_OK)
      return d;
  }
  for (int sym = a->sym_start; s != NULL && sym < a->sym_start + a->n_sym; sym++)
    if (!rbs_free(s->rb[sym], a->rb_start, a->n_rb))
      return GRID_ERR_CONFLICT;
  if (s != NULL && s->n_alloc == GRID_MAX_ALLOCS)
    return GRID_ERR_FULL;
  return GRID_OK;
}

static grid_err_t undo_push(grid_txn_t *txn, int type, int64_t slot, int index, int first, int count)
{
  if (txn == NULL)
    return GRID_OK;
  if (txn->len == GRID_TXN_MAX)
    return GRID_ERR_FULL;
  txn->undo[txn->len].type = type;
  txn->undo[txn->len].slot = slot;
  txn->undo[txn->len].index = index;
  txn->undo[txn->len].first = first;
  txn->undo[txn->len].count = count;
  txn->len++;
  return GRID_OK;
}

static void use_symbols(slot_state_t *s, const grid_alloc_t *a, bool on)
{
  for (int sym = a->sym_start; sym < a->sym_start + a->n_sym; sym++) {
    rbs_set(s->rb[sym], a->rb_start, a->n_rb, on);
    if (on) {
      s->users[sym]++;
      s->dir[sym] = a->ul;
      s->beam[sym] = a->beam;
    } else if (--s->users[sym] == 0) {
      s->dir[sym] = -1;
      s->beam[sym] = -1;
    }
  }
}

grid_err_t grid_try_reserve(sched_grid_t *grid, grid_txn_t *txn, int64_t slot, const grid_alloc_t *a)
{
  grid_err_t e = grid_check(grid, slot, a);
  if (e != GRID_OK)
    return e;
  if (txn != NULL && txn->len == GRID_TXN_MAX)
    return GRID_ERR_FULL;
  slot_state_t *s = slot_get(grid, slot);
  undo_push(txn, UNDO_ALLOC, slot, s->n_alloc, 0, 0);
  s->alloc[s->n_alloc++] = *a;
  use_symbols(s, a, true);
  return GRID_OK;
}

static uint64_t *bits_of(slot_state_t *s, int type, int index)
{
  return type == UNDO_CCES ? s->cce[index] : s->pool[index];
}

static grid_err_t reserve_bits(sched_grid_t *grid, grid_txn_t *txn, int64_t slot, int type, int index, int first, int count, int size)
{
  const grid_err_t e = slot_index(grid, slot);
  if (e != GRID_OK)
    return e;
  const int max_index = type == UNDO_CCES ? GRID_MAX_CORESETS : GRID_MAX_POOLS;
  if (index < 0 || index >= max_index || first < 0 || count < 1 || first + count > size)
    return GRID_ERR_INVALID;
  const slot_state_t *peek = slot_peek(grid, slot);
  if (peek != NULL && !rbs_free(type == UNDO_CCES ? peek->cce[index] : peek->pool[index], first, count))
    return GRID_ERR_CONFLICT;
  if (txn != NULL && txn->len == GRID_TXN_MAX)
    return GRID_ERR_FULL;
  slot_state_t *s = slot_get(grid, slot);
  undo_push(txn, type, slot, index, first, count);
  rbs_set(bits_of(s, type, index), first, count, true);
  return GRID_OK;
}

grid_err_t grid_try_reserve_cces(sched_grid_t *grid, grid_txn_t *txn, int64_t slot, int coreset, int first, int count)
{
  return reserve_bits(grid, txn, slot, UNDO_CCES, coreset, first, count, GRID_MAX_CCES);
}

grid_err_t grid_try_reserve_pool(sched_grid_t *grid, grid_txn_t *txn, int64_t slot, int pool, int index)
{
  return reserve_bits(grid, txn, slot, UNDO_POOL, pool, index, 1, GRID_POOL_SIZE);
}

void grid_txn_abort(sched_grid_t *grid, grid_txn_t *txn)
{
  for (int i = txn->len - 1; i >= 0; i--) {
    const int64_t slot = txn->undo[i].slot;
    /* a slot that left the horizon has nothing to undo */
    if (slot_index(grid, slot) != GRID_OK || slot_peek(grid, slot) == NULL)
      continue;
    slot_state_t *s = slot_get(grid, slot);
    if (txn->undo[i].type == UNDO_ALLOC) {
      const int idx = txn->undo[i].index;
      use_symbols(s, &s->alloc[idx], false);
      memmove(&s->alloc[idx], &s->alloc[idx + 1], (s->n_alloc - idx - 1) * sizeof(s->alloc[0]));
      s->n_alloc--;
    } else {
      rbs_set(bits_of(s, txn->undo[i].type, txn->undo[i].index), txn->undo[i].first, txn->undo[i].count, false);
    }
  }
  txn->len = 0;
}

int grid_free_run(const sched_grid_t *grid, int64_t slot, bool ul, int beam, int sym_start, int n_sym, int rb_lo, int rb_hi, int *start)
{
  *start = 0;
  if (slot_index(grid, slot) != GRID_OK || n_sym < 1 || sym_start < 0 || sym_start + n_sym > GRID_SYMBOLS)
    return 0;
  if (rb_lo < 0)
    rb_lo = 0;
  if (rb_hi > grid->cfg.n_rb)
    rb_hi = grid->cfg.n_rb;
  const slot_state_t *s = slot_peek(grid, slot);
  const grid_sym_t *types = slot_types(grid, slot);
  uint64_t used[GRID_RB_WORDS] = {0};
  for (int sym = sym_start; sym < sym_start + n_sym; sym++) {
    if (symbol_allows(grid, s, types, sym, ul, beam) != GRID_OK)
      return 0;
    for (int w = 0; s != NULL && w < GRID_RB_WORDS; w++)
      used[w] |= s->rb[sym][w];
  }
  /* longest run of free RBs in [rb_lo, rb_hi): walk the runs word by word; RBs outside the range are
   * masked as used */
  int best = 0, run = 0, run_start = rb_lo;
  for (int w = rb_lo / 64; w * 64 < rb_hi; w++) {
    const uint64_t free_bits = ~used[w] & word_mask(w, rb_lo, rb_hi - rb_lo);
    int pos = 0;
    while (pos < 64) {
      const uint64_t rest = free_bits >> pos;
      if ((rest & 1) == 0) { /* used RB: the run ends, jump to the next free one */
        run = 0;
        if (rest == 0)
          break;
        pos += __builtin_ctzll(rest);
        continue;
      }
      const uint64_t ones = ~rest; /* free from pos up to the next used bit or the end of the word */
      const int len = ones == 0 ? 64 - pos : __builtin_ctzll(ones);
      if (run == 0)
        run_start = w * 64 + pos;
      run += len;
      if (run > best) {
        best = run;
        *start = run_start;
      }
      pos += len;
    }
  }
  return best;
}

int grid_allocs(const sched_grid_t *grid, int64_t slot, const grid_alloc_t **allocs)
{
  const slot_state_t *s = slot_index(grid, slot) == GRID_OK ? slot_peek(grid, slot) : NULL;
  *allocs = s != NULL ? s->alloc : NULL;
  return s != NULL ? s->n_alloc : 0;
}

grid_err_t grid_used(const sched_grid_t *grid, int64_t slot, bool ul, int symbol, uint64_t rbs[GRID_RB_WORDS])
{
  memset(rbs, 0, sizeof(uint64_t) * GRID_RB_WORDS);
  const grid_err_t e = slot_index(grid, slot);
  if (e != GRID_OK)
    return e;
  if (symbol < 0 || symbol >= GRID_SYMBOLS)
    return GRID_ERR_INVALID;
  const slot_state_t *s = slot_peek(grid, slot);
  if (s == NULL || s->dir[symbol] != (int8_t)ul)
    return GRID_OK;
  memcpy(rbs, s->rb[symbol], sizeof(uint64_t) * GRID_RB_WORDS);
  return GRID_OK;
}

grid_err_t grid_mark(sched_grid_t *grid, int64_t slot, bool ul, int beam, int symbol, const uint64_t rbs[GRID_RB_WORDS])
{
  const grid_err_t e = slot_index(grid, slot);
  if (e != GRID_OK)
    return e;
  if (symbol < 0 || symbol >= GRID_SYMBOLS)
    return GRID_ERR_INVALID;
  bool any = false;
  for (int w = 0; w < GRID_RB_WORDS; w++)
    any |= rbs[w] != 0;
  if (!any)
    return GRID_OK;
  slot_state_t *s = slot_get(grid, slot);
  const grid_err_t d = symbol_allows(grid, s, slot_types(grid, slot), symbol, ul, beam);
  if (d != GRID_OK)
    return d;
  for (int w = 0; w < GRID_RB_WORDS; w++)
    s->rb[symbol][w] |= rbs[w];
  /* kept for the life of the slot */
  s->users[symbol]++;
  s->dir[symbol] = ul;
  s->beam[symbol] = beam;
  return GRID_OK;
}
