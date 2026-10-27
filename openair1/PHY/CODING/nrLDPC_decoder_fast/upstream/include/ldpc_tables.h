/* Base-graph tables of 3GPP TS 38.212 Sec. 5.3.2 (generated data in tables/ldpc_tables.c). */
#ifndef LDPC_TABLES_H
#define LDPC_TABLES_H
#include <stdint.h>

typedef struct {
  uint8_t row, col;
  uint16_t v[8]; /* V_ij for set index iLS = 0..7 (Table 5.3.2-2 / 5.3.2-3); P_ij = V_ij mod Z */
} ldpc_bg_edge_t;

#define LDPC_BG1_EDGES 316
#define LDPC_BG2_EDGES 197
extern const ldpc_bg_edge_t ldpc_bg1_edges[LDPC_BG1_EDGES];
extern const ldpc_bg_edge_t ldpc_bg2_edges[LDPC_BG2_EDGES];
extern const uint16_t ldpc_lifting_sizes[51]; /* Table 5.3.2-1, ascending */
extern const uint8_t ldpc_ils_of_z[385];      /* set index of Z, 255 if Z is not a lifting size */

#endif
