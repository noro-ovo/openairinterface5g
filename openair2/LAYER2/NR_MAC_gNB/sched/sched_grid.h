/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* Resource grid of the scheduler: the only record of what is used, per future slot.
 * Every reservation goes through grid_try_reserve*() and fails with a reason instead of
 * overwriting. Reservations are grouped in a transaction that is committed or undone as a whole.
 * Slots are absolute (SFN x slots per frame + slot, counted without wrap); the grid keeps
 * horizon_slots slots from the current one. All memory is allocated in grid_create(). */

#ifndef SCHED_GRID_H_
#define SCHED_GRID_H_

#include <stdbool.h>
#include <stdint.h>

#define GRID_SYMBOLS 14
#define GRID_MAX_RB 275
#define GRID_RB_WORDS ((GRID_MAX_RB + 63) / 64)
#define GRID_MAX_ALLOCS 96
#define GRID_MAX_CORESETS 4
#define GRID_MAX_CCES 192
#define GRID_MAX_POOLS 4
#define GRID_POOL_SIZE 128
#define GRID_MAX_PERIOD 320
#define GRID_TXN_MAX 64

typedef enum { GRID_SYM_DL, GRID_SYM_UL, GRID_SYM_FLEX } grid_sym_t;

typedef enum {
  GRID_SSB,
  GRID_PDCCH,
  GRID_PDSCH,
  GRID_CSIRS,
  GRID_PRACH,
  GRID_MSGA_PUSCH,
  GRID_PUSCH,
  GRID_PUCCH,
  GRID_SRS,
  GRID_RESERVED,
} grid_kind_t;

typedef enum {
  GRID_OK = 0,
  GRID_ERR_PAST = -1, ///< the slot is before the current one
  GRID_ERR_HORIZON = -2, ///< the slot is beyond the horizon
  GRID_ERR_INVALID = -3, ///< RBs or symbols out of range
  GRID_ERR_DIRECTION = -4, ///< a symbol does not allow this direction
  GRID_ERR_CONFLICT = -5, ///< an RB of a symbol is already used
  GRID_ERR_BEAM = -6, ///< a symbol is already used by another beam
  GRID_ERR_FULL = -7, ///< no room for one more allocation or undo record
} grid_err_t;

typedef struct {
  int n_rb; ///< carrier RBs, at most GRID_MAX_RB
  int horizon_slots; ///< slots kept ahead, more than the largest K0/K1/K2 and the PRACH period
  int period_slots; ///< TDD period, 1 for FDD
  grid_sym_t sym[GRID_MAX_PERIOD][GRID_SYMBOLS]; ///< direction of each symbol of the period
  bool analog_beams; ///< one beam per symbol
} grid_cfg_t;

typedef struct {
  grid_kind_t kind;
  bool ul; ///< direction it needs
  int owner; ///< RNTI, or -1 for cell-wide signals
  int beam;
  int rb_start, n_rb;
  int sym_start, n_sym;
} grid_alloc_t;

typedef struct sched_grid sched_grid_t;

typedef struct {
  int len;
  struct {
    uint8_t type;
    int64_t slot;
    int16_t index, first, count;
  } undo[GRID_TXN_MAX];
} grid_txn_t;

sched_grid_t *grid_create(const grid_cfg_t *cfg);
void grid_free(sched_grid_t *grid);
/// make slot the current one; slots before it can no longer be reserved
void grid_set_current(sched_grid_t *grid, int64_t slot);

grid_sym_t grid_symbol_type(const sched_grid_t *grid, int64_t slot, int symbol);

void grid_txn_begin(grid_txn_t *txn);
/// undo every reservation of the transaction, in reverse order
void grid_txn_abort(sched_grid_t *grid, grid_txn_t *txn);
void grid_txn_commit(grid_txn_t *txn);

/// reserve RBs x symbols; on success the reservation is recorded in txn (may be NULL: no undo)
grid_err_t grid_try_reserve(sched_grid_t *grid, grid_txn_t *txn, int64_t slot, const grid_alloc_t *a);
/// what grid_try_reserve would return, without reserving
grid_err_t grid_check(const sched_grid_t *grid, int64_t slot, const grid_alloc_t *a);
/// reserve CCEs first..first+count-1 of a CORESET
grid_err_t grid_try_reserve_cces(sched_grid_t *grid, grid_txn_t *txn, int64_t slot, int coreset, int first, int count);
/// reserve index of a resource pool (PUCCH resources)
grid_err_t grid_try_reserve_pool(sched_grid_t *grid, grid_txn_t *txn, int64_t slot, int pool, int index);

/// the longest run of free RBs in [rb_lo, rb_hi) on all symbols sym_start..sym_start+n_sym-1, for
/// direction ul and beam; returns its length and its start in *start (0 if none)
int grid_free_run(const sched_grid_t *grid, int64_t slot, bool ul, int beam, int sym_start, int n_sym, int rb_lo, int rb_hi, int *start);

/// mark RBs of one symbol as used by something outside the grid (transition: what the legacy code
/// reserved in its VRB maps), without an allocation record and without undo
grid_err_t grid_mark(sched_grid_t *grid, int64_t slot, bool ul, int beam, int symbol, const uint64_t rbs[GRID_RB_WORDS]);

/// RBs taken on one symbol of a slot in direction ul (none when the symbol is used by the other direction)
grid_err_t grid_used(const sched_grid_t *grid, int64_t slot, bool ul, int symbol, uint64_t rbs[GRID_RB_WORDS]);

/// allocations of a slot, in reservation order
int grid_allocs(const sched_grid_t *grid, int64_t slot, const grid_alloc_t **allocs);

#endif /* SCHED_GRID_H_ */
