/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include <stdio.h>
#include <stdlib.h>

#include "../sched_grid.h"

static int failures;

#define EXPECT_EQ(value, expected)                                                         \
  do {                                                                                     \
    const long long v_ = (value), e_ = (expected);                                         \
    if (v_ != e_) {                                                                        \
      printf("FAIL %s:%d %s = %lld, expected %lld\n", __func__, __LINE__, #value, v_, e_); \
      failures++;                                                                          \
    }                                                                                      \
  } while (0)

#define HORIZON 64

/* 7D-S-2U with 10 slots, the mixed slot 7 has 6 DL, 4 flexible and 4 UL symbols */
static sched_grid_t *tdd_grid(bool analog)
{
  static grid_cfg_t cfg;
  cfg = (grid_cfg_t){.n_rb = 106, .horizon_slots = HORIZON, .period_slots = 10, .analog_beams = analog};
  for (int s = 0; s < 10; s++)
    for (int i = 0; i < GRID_SYMBOLS; i++)
      cfg.sym[s][i] = s < 7 ? GRID_SYM_DL : s > 7 ? GRID_SYM_UL : i < 6 ? GRID_SYM_DL : i < 10 ? GRID_SYM_FLEX : GRID_SYM_UL;
  return grid_create(&cfg);
}

static grid_alloc_t dl(int rb, int n_rb, int sym, int n_sym)
{
  return (grid_alloc_t){.kind = GRID_PDSCH, .owner = 0x4601, .rb_start = rb, .n_rb = n_rb, .sym_start = sym, .n_sym = n_sym};
}

static grid_alloc_t ul(int rb, int n_rb, int sym, int n_sym)
{
  return (grid_alloc_t){.kind = GRID_PUSCH, .ul = true, .owner = 0x4601, .rb_start = rb, .n_rb = n_rb, .sym_start = sym, .n_sym = n_sym};
}

static void test_reserve(void)
{
  sched_grid_t *g = tdd_grid(false);
  grid_set_current(g, 100);
  grid_alloc_t a = dl(0, 10, 1, 13);
  EXPECT_EQ(grid_try_reserve(g, NULL, 100, &a), GRID_OK);
  a = dl(5, 1, 13, 1);
  EXPECT_EQ(grid_try_reserve(g, NULL, 100, &a), GRID_ERR_CONFLICT);
  a = dl(10, 10, 1, 13);
  EXPECT_EQ(grid_try_reserve(g, NULL, 100, &a), GRID_OK);
  a = dl(0, 20, 0, 1); /* same RBs, the free symbol 0 */
  EXPECT_EQ(grid_try_reserve(g, NULL, 100, &a), GRID_OK);
  const grid_alloc_t *list;
  EXPECT_EQ(grid_allocs(g, 100, &list), 3);
  EXPECT_EQ(list[1].rb_start, 10);

  /* directions: UL in a DL slot, DL in an UL slot */
  a = ul(0, 10, 0, 14);
  EXPECT_EQ(grid_try_reserve(g, NULL, 101, &a), GRID_ERR_DIRECTION);
  a = dl(0, 10, 0, 14);
  EXPECT_EQ(grid_try_reserve(g, NULL, 108, &a), GRID_ERR_DIRECTION);
  a = ul(0, 10, 0, 14);
  EXPECT_EQ(grid_try_reserve(g, NULL, 108, &a), GRID_OK);
  /* the mixed slot: DL on DL and flexible symbols takes the flexible symbols 6..7 for DL */
  a = dl(0, 10, 1, 7);
  EXPECT_EQ(grid_try_reserve(g, NULL, 107, &a), GRID_OK);
  a = ul(50, 10, 7, 3); /* symbol 7 is now DL, on other RBs too */
  EXPECT_EQ(grid_try_reserve(g, NULL, 107, &a), GRID_ERR_DIRECTION);
  a = ul(50, 10, 8, 6);
  EXPECT_EQ(grid_try_reserve(g, NULL, 107, &a), GRID_OK);
  a = dl(60, 10, 9, 1); /* symbol 9 is UL now */
  EXPECT_EQ(grid_try_reserve(g, NULL, 107, &a), GRID_ERR_DIRECTION);
  a = dl(0, 10, 10, 2); /* UL symbols */
  EXPECT_EQ(grid_try_reserve(g, NULL, 107, &a), GRID_ERR_DIRECTION);

  /* range and horizon */
  a = dl(100, 7, 1, 13);
  EXPECT_EQ(grid_try_reserve(g, NULL, 102, &a), GRID_ERR_INVALID);
  a = dl(0, 1, 13, 2);
  EXPECT_EQ(grid_try_reserve(g, NULL, 102, &a), GRID_ERR_INVALID);
  a = dl(0, 0, 1, 1);
  EXPECT_EQ(grid_try_reserve(g, NULL, 102, &a), GRID_ERR_INVALID);
  a = dl(0, 10, 1, 13);
  EXPECT_EQ(grid_try_reserve(g, NULL, 99, &a), GRID_ERR_PAST);
  EXPECT_EQ(grid_try_reserve(g, NULL, 100 + HORIZON, &a), GRID_ERR_HORIZON);
  EXPECT_EQ(grid_try_reserve(g, NULL, 100 + HORIZON - 1, &a), GRID_OK);
  /* grid_check reserves nothing */
  a = dl(30, 10, 1, 13);
  EXPECT_EQ(grid_check(g, 103, &a), GRID_OK);
  EXPECT_EQ(grid_check(g, 103, &a), GRID_OK);
  EXPECT_EQ(grid_allocs(g, 103, &list), 0);
  grid_free(g);
}

static void test_transaction(void)
{
  sched_grid_t *g = tdd_grid(false);
  grid_set_current(g, 0);
  grid_txn_t txn;
  grid_txn_begin(&txn);
  grid_alloc_t a = dl(0, 50, 1, 13);
  EXPECT_EQ(grid_try_reserve(g, &txn, 1, &a), GRID_OK);
  EXPECT_EQ(grid_try_reserve_cces(g, &txn, 1, 1, 0, 4), GRID_OK);
  a = ul(0, 20, 8, 6);
  EXPECT_EQ(grid_try_reserve(g, &txn, 7, &a), GRID_OK);
  EXPECT_EQ(grid_try_reserve_pool(g, &txn, 7, 0, 3), GRID_OK);
  grid_txn_abort(g, &txn);
  /* everything is free again, flexible symbols included */
  int start;
  EXPECT_EQ(grid_free_run(g, 1, false, 0, 1, 13, 0, 106, &start), 106);
  EXPECT_EQ(grid_try_reserve_cces(g, NULL, 1, 1, 0, 4), GRID_OK);
  EXPECT_EQ(grid_try_reserve_pool(g, NULL, 7, 0, 3), GRID_OK);
  a = dl(0, 10, 6, 4); /* flexible 6..9 as DL: allowed after the abort released them */
  EXPECT_EQ(grid_try_reserve(g, NULL, 7, &a), GRID_OK);
  const grid_alloc_t *list;
  EXPECT_EQ(grid_allocs(g, 1, &list), 0);
  EXPECT_EQ(grid_allocs(g, 7, &list), 1);

  /* a committed transaction stays; aborting it afterwards does nothing */
  grid_txn_begin(&txn);
  a = dl(0, 50, 1, 13);
  EXPECT_EQ(grid_try_reserve(g, &txn, 2, &a), GRID_OK);
  grid_txn_commit(&txn);
  grid_txn_abort(g, &txn);
  EXPECT_EQ(grid_allocs(g, 2, &list), 1);

  /* two reservations of one transaction in the same slot, undone in reverse order */
  grid_txn_begin(&txn);
  a = dl(60, 5, 1, 13);
  EXPECT_EQ(grid_try_reserve(g, &txn, 2, &a), GRID_OK);
  a = dl(70, 5, 1, 13);
  EXPECT_EQ(grid_try_reserve(g, &txn, 2, &a), GRID_OK);
  grid_txn_abort(g, &txn);
  EXPECT_EQ(grid_allocs(g, 2, &list), 1);
  EXPECT_EQ(grid_free_run(g, 2, false, 0, 1, 13, 50, 106, &start), 56);
  EXPECT_EQ(start, 50);

  /* the undo log is bounded */
  grid_txn_begin(&txn);
  int ok = 0;
  for (int i = 0; i < GRID_TXN_MAX + 1; i++)
    ok += grid_try_reserve_pool(g, &txn, 3, 1, i) == GRID_OK;
  EXPECT_EQ(ok, GRID_TXN_MAX);
  EXPECT_EQ(grid_try_reserve_pool(g, &txn, 3, 1, 100), GRID_ERR_FULL);
  grid_txn_abort(g, &txn);
  EXPECT_EQ(grid_try_reserve_pool(g, NULL, 3, 1, 0), GRID_OK);
  grid_free(g);
}

static void test_ring_and_capacity(void)
{
  sched_grid_t *g = tdd_grid(false);
  grid_set_current(g, 0);
  grid_alloc_t a = dl(0, 106, 0, 14);
  EXPECT_EQ(grid_try_reserve(g, NULL, 0, &a), GRID_OK);
  /* slot HORIZON (a DL slot of the period) uses the same ring entry: it starts empty */
  grid_set_current(g, 1);
  const grid_alloc_t *list;
  EXPECT_EQ(grid_allocs(g, HORIZON, &list), 0);
  EXPECT_EQ(grid_try_reserve(g, NULL, HORIZON, &a), GRID_OK);
  EXPECT_EQ(grid_allocs(g, 0, &list), 0); /* the past is gone */

  /* at most GRID_MAX_ALLOCS allocations per slot */
  int ok = 0;
  for (int i = 0; i < GRID_MAX_ALLOCS + 1; i++) {
    a = dl(i, 1, 1, 13);
    ok += grid_try_reserve(g, NULL, 10, &a) == GRID_OK;
  }
  EXPECT_EQ(ok, GRID_MAX_ALLOCS);
  a = dl(105, 1, 1, 13);
  EXPECT_EQ(grid_check(g, 10, &a), GRID_ERR_FULL);
  grid_free(g);
}

static void test_beams(void)
{
  sched_grid_t *g = tdd_grid(true);
  grid_set_current(g, 0);
  grid_alloc_t a = dl(0, 10, 0, 2);
  a.beam = 0;
  EXPECT_EQ(grid_try_reserve(g, NULL, 1, &a), GRID_OK);
  a = dl(50, 10, 1, 2);
  a.beam = 1; /* symbol 1 is on beam 0 */
  EXPECT_EQ(grid_try_reserve(g, NULL, 1, &a), GRID_ERR_BEAM);
  a.sym_start = 2;
  EXPECT_EQ(grid_try_reserve(g, NULL, 1, &a), GRID_OK);
  int start;
  EXPECT_EQ(grid_free_run(g, 1, false, 1, 0, 2, 0, 106, &start), 0);
  EXPECT_EQ(grid_free_run(g, 1, false, 0, 0, 2, 0, 106, &start), 96);
  grid_free(g);
  /* without analog beams the beam does not lock symbols */
  g = tdd_grid(false);
  grid_set_current(g, 0);
  a = dl(0, 10, 0, 2);
  EXPECT_EQ(grid_try_reserve(g, NULL, 1, &a), GRID_OK);
  a = dl(50, 10, 1, 2);
  a.beam = 1;
  EXPECT_EQ(grid_try_reserve(g, NULL, 1, &a), GRID_OK);
  grid_free(g);
}

static void test_free_run_and_bits(void)
{
  sched_grid_t *g = tdd_grid(false);
  grid_set_current(g, 0);
  grid_alloc_t a = dl(0, 10, 1, 13);
  grid_try_reserve(g, NULL, 0, &a);
  a = dl(20, 10, 5, 2);
  grid_try_reserve(g, NULL, 0, &a);
  int start;
  EXPECT_EQ(grid_free_run(g, 0, false, 0, 1, 13, 0, 106, &start), 76);
  EXPECT_EQ(start, 30);
  EXPECT_EQ(grid_free_run(g, 0, false, 0, 1, 13, 0, 25, &start), 10);
  EXPECT_EQ(start, 10);
  /* symbols 7..13 only: RBs 20..29 are free there */
  EXPECT_EQ(grid_free_run(g, 0, false, 0, 7, 7, 0, 106, &start), 96);
  EXPECT_EQ(grid_free_run(g, 0, true, 0, 1, 13, 0, 106, &start), 0); /* UL in a DL slot */
  EXPECT_EQ(grid_free_run(g, 0, false, 0, 10, 5, 0, 106, &start), 0); /* symbols beyond the slot */

  EXPECT_EQ(grid_try_reserve_cces(g, NULL, 0, 0, 0, 4), GRID_OK);
  EXPECT_EQ(grid_try_reserve_cces(g, NULL, 0, 0, 2, 4), GRID_ERR_CONFLICT);
  EXPECT_EQ(grid_try_reserve_cces(g, NULL, 0, 0, 4, 4), GRID_OK);
  EXPECT_EQ(grid_try_reserve_cces(g, NULL, 0, 1, 0, 4), GRID_OK);
  EXPECT_EQ(grid_try_reserve_cces(g, NULL, 0, 0, GRID_MAX_CCES - 1, 2), GRID_ERR_INVALID);
  EXPECT_EQ(grid_try_reserve_cces(g, NULL, 0, GRID_MAX_CORESETS, 0, 1), GRID_ERR_INVALID);
  EXPECT_EQ(grid_try_reserve_pool(g, NULL, 0, 0, 5), GRID_OK);
  EXPECT_EQ(grid_try_reserve_pool(g, NULL, 0, 0, 5), GRID_ERR_CONFLICT);
  EXPECT_EQ(grid_try_reserve_pool(g, NULL, 0, 0, GRID_POOL_SIZE), GRID_ERR_INVALID);
  EXPECT_EQ(grid_try_reserve_pool(g, NULL, -1, 0, 1), GRID_ERR_PAST);
  grid_free(g);
}

static void test_mark(void)
{
  sched_grid_t *g = tdd_grid(false);
  grid_set_current(g, 0);
  uint64_t rbs[GRID_RB_WORDS] = {0};
  rbs[0] = 0xff00; /* RBs 8..15 */
  rbs[1] = 1; /* RB 64 */
  EXPECT_EQ(grid_mark(g, 0, false, 0, 3, rbs), GRID_OK);
  int start;
  /* free: 0..7, 16..63, 65..105 */
  EXPECT_EQ(grid_free_run(g, 0, false, 0, 3, 1, 0, 106, &start), 48);
  EXPECT_EQ(start, 16);
  EXPECT_EQ(grid_free_run(g, 0, false, 0, 3, 1, 64, 106, &start), 41);
  EXPECT_EQ(start, 65);
  grid_alloc_t a = dl(10, 2, 2, 3);
  EXPECT_EQ(grid_try_reserve(g, NULL, 0, &a), GRID_ERR_CONFLICT);
  const grid_alloc_t *list;
  EXPECT_EQ(grid_allocs(g, 0, &list), 0);
  /* a marked flexible symbol keeps the direction of the mark */
  EXPECT_EQ(grid_mark(g, 7, true, 0, 8, rbs), GRID_OK);
  a = dl(30, 2, 8, 1);
  EXPECT_EQ(grid_try_reserve(g, NULL, 7, &a), GRID_ERR_DIRECTION);
  EXPECT_EQ(grid_mark(g, 0, true, 0, 3, rbs), GRID_ERR_DIRECTION);
  uint64_t none[GRID_RB_WORDS] = {0};
  EXPECT_EQ(grid_mark(g, 8, false, 0, 0, none), GRID_OK); /* nothing to mark in an UL slot */
  grid_free(g);
}

/* grid_free_run against a scan RB by RB, on random marks, ranges and symbol spans (word boundaries at 64, 128, 192) */
static void test_free_run_random(void)
{
  static grid_cfg_t cfg;
  cfg = (grid_cfg_t){.n_rb = 273, .horizon_slots = 8, .period_slots = 1};
  for (int i = 0; i < GRID_SYMBOLS; i++)
    cfg.sym[0][i] = GRID_SYM_DL;
  sched_grid_t *g = grid_create(&cfg);
  unsigned seed = 12345;
  for (int iter = 0; iter < 20000; iter++) {
    const int64_t slot = iter;
    grid_set_current(g, slot);
    const int density = rand_r(&seed) % 4; /* 0: empty .. 3: dense */
    for (int sym = 0; sym < GRID_SYMBOLS; sym++) {
      uint64_t rbs[GRID_RB_WORDS] = {0};
      for (int rb = 0; rb < 273; rb++)
        if (density && rand_r(&seed) % (2 + 8 * (3 - density)) == 0)
          rbs[rb / 64] |= 1ULL << (rb % 64);
      grid_mark(g, slot, false, 0, sym, rbs);
    }
    const int s0 = rand_r(&seed) % GRID_SYMBOLS, n_sym = 1 + rand_r(&seed) % (GRID_SYMBOLS - s0);
    int lo = rand_r(&seed) % 273, hi = rand_r(&seed) % 274;
    if (lo > hi) {
      const int t = lo;
      lo = hi;
      hi = t;
    }
    uint64_t used[GRID_RB_WORDS] = {0};
    for (int sym = s0; sym < s0 + n_sym; sym++) {
      uint64_t u[GRID_RB_WORDS];
      grid_used(g, slot, false, sym, u);
      for (int w = 0; w < GRID_RB_WORDS; w++)
        used[w] |= u[w];
    }
    int best = 0, run = 0, best_start = 0;
    for (int rb = lo; rb < hi; rb++) {
      run = used[rb / 64] >> (rb % 64) & 1 ? 0 : run + 1;
      if (run > best) {
        best = run;
        best_start = rb - run + 1;
      }
    }
    int start;
    const int got = grid_free_run(g, slot, false, 0, s0, n_sym, lo, hi, &start);
    EXPECT_EQ(got, best);
    if (best > 0)
      EXPECT_EQ(start, best_start);
    if (got != best || (best > 0 && start != best_start))
      break;
  }
  grid_free(g);
}

int main(void)
{
  test_reserve();
  test_transaction();
  test_ring_and_capacity();
  test_beams();
  test_free_run_and_bits();
  test_mark();
  test_free_run_random();
  if (failures) {
    printf("%d failures\n", failures);
    return 1;
  }
  printf("all sched_grid tests passed\n");
  return 0;
}
