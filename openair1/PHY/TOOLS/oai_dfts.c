/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#if defined(__x86_64__) || defined(__i386__)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include <execinfo.h>
#include <stddef.h>
#include <complex.h>
#include <immintrin.h>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define Q15_INV_SQRT2                       ((int16_t)23170)
#define Q15_INV_SQRT3                       ((int16_t)18919)  
#define Q15_HALF_SQRT3                      ((int16_t)28378)
#define Q15_SCALE_1_OVER_SQRT64             ((int16_t)4096)


#define SR_MAX_LOG2 17
#define MAX_N 65536
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define OAIDFTS_MAIN
//#ifndef MR_MAIN
//#include "PHY/defs_common.h"
//#include "PHY/impl_defs_top.h"
//#else
#include "time_meas.h"
#include "LOG/log.h"
#define debug_msg
#define ONE_OVER_SQRT2_Q15 23170

//int oai_exit=0;
//#endif

#define ONE_OVER_SQRT3_Q15 18919

#include "../sse_intrin.h"

#include "assertions.h"

#include "tools_defs.h"

#define print_shorts(s,x) printf("%s %d,%d,%d,%d,%d,%d,%d,%d\n",s,(x)[0],(x)[1],(x)[2],(x)[3],(x)[4],(x)[5],(x)[6],(x)[7])
#define print_shorts256(s,x) printf("%s %d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",s,(x)[0],(x)[1],(x)[2],(x)[3],(x)[4],(x)[5],(x)[6],(x)[7],(x)[8],(x)[9],(x)[10],(x)[11],(x)[12],(x)[13],(x)[14],(x)[15])

#define print_ints(s,x) printf("%s %d %d %d %d\n",s,(x)[0],(x)[1],(x)[2],(x)[3])


const static int16_t conjugatedft[32] __attribute__((aligned(32))) = {-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1};


const static int16_t reflip[32]  __attribute__((aligned(32))) = {1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1};





//============================================================================
//HELPERS
//============================================================================

static inline int16_t sat_i16(long v)
{
    if (v > 32767) return 32767;
    if (v < -32768) return -32767;
    return (int16_t)v;
}

static inline int log2_int(unsigned int N)
{
    return __builtin_ctz(N);
}
static void *aligned_malloc64(size_t size)
{
    void *ptr = NULL;

    if (posix_memalign(&ptr, 64, size) != 0) {
        return NULL;
    }

    return ptr;
}

static inline __m128i scale_half_q15_128(__m128i x)
{
    return _mm_srai_epi16(x, 1);   // divide by 2
}
static inline __m256i scale_q15_1_over_sqrt64_256(__m256i x)
{
    const __m256i s = _mm256_set1_epi16(Q15_SCALE_1_OVER_SQRT64);
    return _mm256_mulhrs_epi16(x, s);
}


static inline int is_power_of_two_int(int x)
{
    return x > 0 && ((x & (x - 1)) == 0);
}

static inline simde__m256i c16_mul_q15_simd256(simde__m256i x,
                                                 simde__m256i w_re_negim,
                                                 simde__m256i w_im_re)
{
    //simde__m256i zero  = simde_mm256_setzero_si256();
    simde__m256i round = simde_mm256_set1_epi32(1 << 14);

    simde__m256i re32 = simde_mm256_madd_epi16(x, w_re_negim);
    simde__m256i im32 = simde_mm256_madd_epi16(x, w_im_re);

    re32 = simde_mm256_srai_epi32(simde_mm256_add_epi32(re32, round), 15);
    im32 = simde_mm256_srai_epi32(simde_mm256_add_epi32(im32, round), 15);
/*
    simde__m256i re16 = simde_mm256_packs_epi32(re32, zero);
    simde__m256i im16 = simde_mm256_packs_epi32(im32, zero);

    return simde_mm256_unpacklo_epi16(re16, im16);
*/

    simde__m256i packed = simde_mm256_packs_epi32(re32, im32);

    const simde__m256i mask = simde_mm256_set_epi8(
        15,14, 7, 6, 13,12, 5, 4, 11,10, 3, 2, 9, 8, 1, 0,
        15,14, 7, 6, 13,12, 5, 4, 11,10, 3, 2, 9, 8, 1, 0
    );

    return simde_mm256_shuffle_epi8(packed, mask);
}



static inline int16_t sat16_i32(int32_t x)
{
    if (x > 32767)  return 32767;
    if (x < -32767) return -32767;
    return (int16_t)x;
}

static inline int16_t q15_from_float(float x)
{
    return sat16_i32((int32_t)lrintf(32767.0f * x));
}


static inline __m128i swap_complex_pairs_i16_128(__m128i a)
{
    const __m128i shuf = _mm_set_epi8(
        13,12, 15,14,
         9, 8, 11,10,
         5, 4,  7, 6,
         1, 0,  3, 2
    );

    return _mm_shuffle_epi8(a, shuf);
}

static inline __m128i complex_mul4_prepack_q15_128(
    __m128i a,
    __m128i w_re_re,
    __m128i w_im_signed
)
{
    const __m128i a_swapped = swap_complex_pairs_i16_128(a);

    const __m128i prod_re = _mm_mulhrs_epi16(a,         w_re_re);
    const __m128i prod_im = _mm_mulhrs_epi16(a_swapped, w_im_signed);

    return _mm_adds_epi16(prod_re, prod_im);
}

static inline __m128i mullts_q15_128(__m128i z)
{
    /*
     * j * (r + ji) = -i + jr
     *
     * [r i] -> [-i r]
     */
    const __m128i swapped = swap_complex_pairs_i16_128(z);

    const __m128i sign = _mm_setr_epi16(
        -1, 1,
        -1, 1,
        -1, 1,
        -1, 1
    );

    return _mm_sign_epi16(swapped, sign);
}

static inline __m128i mul_minuslts_q15_128(__m128i z)
{
    /*
     * -j * (r + ji) = i - jr
     *
     * [r i] -> [i -r]
     */
    const __m128i swapped = swap_complex_pairs_i16_128(z);

    const __m128i sign = _mm_setr_epi16(
        1, -1,
        1, -1,
        1, -1,
        1, -1
    );

    return _mm_sign_epi16(swapped, sign);
}

static inline __m128i q15_mul_i16_128(__m128i x, int16_t q15)
{
    return _mm_mulhrs_epi16(x, _mm_set1_epi16(q15));
}


static inline __m128i add3s_epi16(__m128i a, __m128i b, __m128i c)
{
    return _mm_adds_epi16(_mm_adds_epi16(a, b), c);
}

typedef enum {
    DFT_DIR_FORWARD = -1,
    DFT_DIR_INVERSE = 1
} dft_dir_t;
#define ALIGNMENT 32

static void *aligned_malloc(size_t size)
{
    void *ptr = NULL;

    if (posix_memalign(&ptr, ALIGNMENT, size) != 0)
        return NULL;

    return ptr;
}
//=====================================================================================
//TWIDDLES START
//=====================================================================================

typedef struct {

    int N;
    int initialized;

    float complex *forward;
    float complex *inverse;

    

    int r3_q15_blocks;
    __m128i *r3_q15_w1_re;
    __m128i *r3_q15_w1_im;
    __m128i *r3_q15_w2_re;
    __m128i *r3_q15_w2_im;
    __m128i *r3_q15_w1_re_inv;
    __m128i *r3_q15_w1_im_inv;
    __m128i *r3_q15_w2_re_inv;
    __m128i *r3_q15_w2_im_inv;

    __m256i C64_RE_RE_q15_256[8]           __attribute__((aligned(64)));
    __m256i C64_IM_SIGNED_q15_256[8]       __attribute__((aligned(64)));

    __m256i W128_RE_RE_q15_256[8]      __attribute__((aligned(64)));
    __m256i W128_IM_SIGNED_q15_256[8]  __attribute__((aligned(64)));

    __m256i C64_RE_RE_q15_256_inverse[8]     __attribute__((aligned(64)));
    __m256i C64_IM_SIGNED_q15_256_inverse[8] __attribute__((aligned(64)));

    __m256i W128_RE_RE_q15_256_inverse[8]     __attribute__((aligned(64)));
    __m256i W128_IM_SIGNED_q15_256_inverse[8] __attribute__((aligned(64)));


    int init_done;
} TwiddleTable;

static TwiddleTable g_tables[MAX_N + 1];




static inline int16_t f32_to_q15(float x)
{
    int v = (int)lrintf(x * 32768.0f);

    if (v > 32767)
        v = 32767;
    if (v < -32767)
        v = -32767;

    return (int16_t)v;
}



static inline __m128i pack4_twiddle_q15_re_re(const float complex *W,
                                              int k0,
                                              int mul,
                                              int N)
{
    int16_t v[8];

    for (int j = 0; j < 4; j++) {
        const int k = (mul * (k0 + j)) % N;
        const int16_t wr = q15_from_float(crealf(W[k])) / sqrtf(3);

        v[2 * j + 0] = wr;
        v[2 * j + 1] = wr;
    }

    return _mm_loadu_si128((const __m128i *)v);
}

static inline __m128i pack4_twiddle_q15_im_im(const float complex *W,
                                              int k0,
                                              int mul,
                                              int N)
{
    int16_t v[8];

    for (int j = 0; j < 4; j++) {
        const int k = (mul * (k0 + j)) % N;
        const int16_t wi = q15_from_float(cimagf(W[k])) / sqrtf(3);

        v[2 * j + 0] = -wi;
        v[2 * j + 1] = wi;
    }

    return _mm_loadu_si128((const __m128i *)v);
}


static inline __m256i pack8_twiddle_q15_re_re256i(const float complex *W,
                                                  int k0,
                                                  int mul,
                                                  int N)
{
    int16_t v[16] __attribute__((aligned(32)));

    for (int j = 0; j < 8; j++) {
        const int k = (mul * (k0 + j)) % N;
        const int16_t wr = (N==64) ? q15_from_float(crealf(W[k]))/8 : q15_from_float(crealf(W[k])) / sqrtf(2.0f);;

        v[2 * j + 0] = wr;
        v[2 * j + 1] = wr;
    }

    return _mm256_load_si256((const __m256i *)(const void *)v);
}

static inline __m256i pack8_twiddle_q15_im_signed256i(const float complex *W,
                                                      int k0,
                                                      int mul,
                                                      int N)
{
    int16_t v[16] __attribute__((aligned(32)));

    for (int j = 0; j < 8; j++) {
        const int k = (mul * (k0 + j)) % N;
        const int16_t wi = (N==64) ? q15_from_float(cimagf(W[k]))/8 : q15_from_float(cimagf(W[k])) / sqrtf(2.0f);

        v[2 * j + 0] = -wi;
        v[2 * j + 1] = wi;
    }

    return _mm256_load_si256((const __m256i *)(const void *)v);
}



static int twiddle_table_64_128_create_q15_simd(TwiddleTable *table)
{
    const int N = table->N;

    for (int m = 0; m < 8; m++) {
        if (N==64){
            table->C64_RE_RE_q15_256[m] =
                pack8_twiddle_q15_re_re256i(table->forward, 0, m, N);

            table->C64_IM_SIGNED_q15_256[m] =
                pack8_twiddle_q15_im_signed256i(table->forward, 0, m, N);

            table->C64_RE_RE_q15_256_inverse[m] =
                pack8_twiddle_q15_re_re256i(table->inverse, 0, m, N);

            table->C64_IM_SIGNED_q15_256_inverse[m] =
                pack8_twiddle_q15_im_signed256i(table->inverse, 0, m, N);
        }
        else {
            table->W128_RE_RE_q15_256[m] =
                pack8_twiddle_q15_re_re256i(table->forward, 8 * m, 1, N);

            table->W128_IM_SIGNED_q15_256[m] =
                pack8_twiddle_q15_im_signed256i(table->forward, 8 * m, 1, N);

            table->W128_RE_RE_q15_256_inverse[m] =
                pack8_twiddle_q15_re_re256i(table->inverse, 8 * m, 1, N);

            table->W128_IM_SIGNED_q15_256_inverse[m] =
                pack8_twiddle_q15_im_signed256i(table->inverse, 8 * m, 1, N);
        }
    }
    return 1;
}
static int twiddle_table_create_radix3_q15_simd(TwiddleTable *table)
{
    const int N = table->N;

    if (N <= 0) {
        return 0;
    }

    if (N % 3 != 0) {
        return 1;
    }

    const int size   = N / 3;
    const int blocks = (size + 3) / 4;

    table->r3_q15_blocks = blocks;

    table->r3_q15_w1_re     = aligned_malloc(sizeof(__m128i) * blocks);
    table->r3_q15_w1_im     = aligned_malloc(sizeof(__m128i) * blocks);
    table->r3_q15_w2_re     = aligned_malloc(sizeof(__m128i) * blocks);
    table->r3_q15_w2_im     = aligned_malloc(sizeof(__m128i) * blocks);

    table->r3_q15_w1_re_inv = aligned_malloc(sizeof(__m128i) * blocks);
    table->r3_q15_w1_im_inv = aligned_malloc(sizeof(__m128i) * blocks);
    table->r3_q15_w2_re_inv = aligned_malloc(sizeof(__m128i) * blocks);
    table->r3_q15_w2_im_inv = aligned_malloc(sizeof(__m128i) * blocks);

    if (!table->r3_q15_w1_re     || !table->r3_q15_w1_im ||
        !table->r3_q15_w2_re     || !table->r3_q15_w2_im ||
        !table->r3_q15_w1_re_inv || !table->r3_q15_w1_im_inv ||
        !table->r3_q15_w2_re_inv || !table->r3_q15_w2_im_inv) {
        return 0;
    }


    for (int b = 0; b < blocks; b++) {
        const int k0 = 4 * b;

        table->r3_q15_w1_re[b] =
            pack4_twiddle_q15_re_re(table->forward, k0, 1, N);
        table->r3_q15_w1_im[b] =
            pack4_twiddle_q15_im_im(table->forward, k0, 1, N);

        table->r3_q15_w2_re[b] =
            pack4_twiddle_q15_re_re(table->forward, k0, 2, N);
        table->r3_q15_w2_im[b] =
            pack4_twiddle_q15_im_im(table->forward, k0, 2, N);

        table->r3_q15_w1_re_inv[b] =
            pack4_twiddle_q15_re_re(table->inverse, k0, 1, N);
        table->r3_q15_w1_im_inv[b] =
            pack4_twiddle_q15_im_im(table->inverse, k0, 1, N);

        table->r3_q15_w2_re_inv[b] =
            pack4_twiddle_q15_re_re(table->inverse, k0, 2, N);
        table->r3_q15_w2_im_inv[b] =
            pack4_twiddle_q15_im_im(table->inverse, k0, 2, N);
    }

    return 1;
}


static TwiddleTable *twiddle_table_create(int N)
{

    TwiddleTable *table = &g_tables[N];

    memset(table, 0, sizeof(*table));

    table->N = N;

    table->forward = aligned_malloc((size_t)N * sizeof(*table->forward));
    table->inverse = aligned_malloc((size_t)N * sizeof(*table->inverse));

    if (!table->forward || !table->inverse) {
        fprintf(stderr, "twiddle_table_create: allocation failed for N=%d\n", N);
        return NULL;
    }

    for (int k = 0; k < N; k++) {
        float theta = 2.0f * (float)M_PI * (float)k / (float)N;

        float c = cosf(theta);
        float s = sinf(theta);

        table->forward[k] = c - I * s;
        table->inverse[k] = c + I * s;
    }

    if (N % 3 == 0) {
        if (!twiddle_table_create_radix3_q15_simd(table)) {
            return NULL;
        }
    }


    if (N % 4 == 0) {
        if (!twiddle_table_64_128_create_q15_simd(table)) {
            return NULL;
        }
    }

    table->initialized = 1;
    return table;
}
const TwiddleTable *twiddle_table_get(int N)
{
    if (N <= 0 || N > MAX_N) {
        fprintf(stderr, "twiddle_table_get: invalid N=%d, MAX_N=%d\n", N, MAX_N);
        abort();
    }

    if (!g_tables[N].initialized) {
        if (!twiddle_table_create(N)) {
            return NULL;
        }
    }

    return &g_tables[N];
}


typedef struct {
    int N;
    int blocks;
    int initialized;
    simde__m256i *W1_RE_NEGIM;
    simde__m256i *W1_IM_RE;
    simde__m256i *W3_RE_NEGIM;
    simde__m256i *W3_IM_RE;
} sr_twiddle_simd_t;

static sr_twiddle_simd_t sr_twiddles_fwd[SR_MAX_LOG2 + 1];
static sr_twiddle_simd_t sr_twiddles_bwd[SR_MAX_LOG2 + 1];

static int init_sr_twiddle_simd(sr_twiddle_simd_t *tw, int N, dft_dir_t dir)
{
    int quarter = N / 4;
    int blocks = quarter / 8;

    tw->N = N;
    tw->blocks = blocks;

    tw->W1_RE_NEGIM = aligned_alloc(32, sizeof(simde__m256i) * blocks);
    tw->W1_IM_RE    = aligned_alloc(32, sizeof(simde__m256i) * blocks);
    tw->W3_RE_NEGIM = aligned_alloc(32, sizeof(simde__m256i) * blocks);
    tw->W3_IM_RE    = aligned_alloc(32, sizeof(simde__m256i) * blocks);

    for (int b = 0; b < blocks; b++) {
        int16_t w1_re_negim[16] __attribute__((aligned(64)));
        int16_t w1_im_re[16]    __attribute__((aligned(64)));
        int16_t w3_re_negim[16] __attribute__((aligned(64)));
        int16_t w3_im_re[16]    __attribute__((aligned(64)));

        for (int j = 0; j < 8; j++) {
            int k = 8 * b + j;

            float theta1 = (float)dir * 2.0f * (float)M_PI * k / (float)N;
            float theta3 = (float)dir * 6.0f * (float)M_PI * k / (float)N;

            int16_t w1r = sat_i16(lrintf(32767.0f * cosf(theta1))) / 2;
            int16_t w1i = sat_i16(lrintf(32767.0f * sinf(theta1))) / 2;

            int16_t w3r = sat_i16(lrintf(32767.0f * cosf(theta3))) / 2;
            int16_t w3i = sat_i16(lrintf(32767.0f * sinf(theta3))) / 2;

            w1_re_negim[2*j]     = w1r;
            w1_re_negim[2*j + 1] = sat_i16(-(long)w1i);
            w1_im_re[2*j]        = w1i;
            w1_im_re[2*j + 1]    = w1r;

            w3_re_negim[2*j]     = w3r;
            w3_re_negim[2*j + 1] = sat_i16(-(long)w3i);
            w3_im_re[2*j]        = w3i;
            w3_im_re[2*j + 1]    = w3r;
        }

        tw->W1_RE_NEGIM[b] = simde_mm256_load_si256((simde__m256i *)w1_re_negim);
        tw->W1_IM_RE[b]    = simde_mm256_load_si256((simde__m256i *)w1_im_re);
        tw->W3_RE_NEGIM[b] = simde_mm256_load_si256((simde__m256i *)w3_re_negim);
        tw->W3_IM_RE[b]    = simde_mm256_load_si256((simde__m256i *)w3_im_re);
    }
    return 1;
}


static sr_twiddle_simd_t *sr_twiddle_table_create(int N, dft_dir_t dir)
{
    const int idx = log2_int((unsigned int)N);

    sr_twiddle_simd_t *tw =
        (dir == DFT_DIR_FORWARD)
            ? &sr_twiddles_fwd[idx]
            : &sr_twiddles_bwd[idx];

    memset(tw, 0, sizeof(*tw));

    if (!init_sr_twiddle_simd(tw, N, dir)) {
        return NULL;
    }

    tw->initialized = 1;
    return tw;
}

const sr_twiddle_simd_t *sr_twiddle_table_get(int N, dft_dir_t dir)
{

    const int idx = log2_int((unsigned int)N);

    if (idx > SR_MAX_LOG2) {
        fprintf(stderr, "sr_twiddle_table_get: log2(N)=%d exceeds SR_MAX_LOG2=%d\n",
                idx, SR_MAX_LOG2);
        abort();
    }

    sr_twiddle_simd_t *tw =
        (dir == DFT_DIR_FORWARD)
            ? &sr_twiddles_fwd[idx]
            : &sr_twiddles_bwd[idx];

    if (!tw->initialized) {
        if (!sr_twiddle_table_create(N, dir)) {
            return NULL;
        }
    }

    return tw;
}
//=====================================================================================
//TWIDDLES FIN
//=====================================================================================

//===================================================================
// DFT64 8x8 int
//===================================================================

static inline __m256i swap_complex_pairs_i16_256(__m256i a)
{
    const __m256i shuf = _mm256_setr_epi8(
         2, 3,  0, 1,
         6, 7,  4, 5,
        10,11,  8, 9,
        14,15, 12,13,

         2, 3,  0, 1,
         6, 7,  4, 5,
        10,11,  8, 9,
        14,15, 12,13
    );

    return _mm256_shuffle_epi8(a, shuf);
}

static inline __m256i mul_j_i16_256(__m256i z)
{
    const __m256i swapped = swap_complex_pairs_i16_256(z);

    const __m256i sign = _mm256_setr_epi16(
        -1, +1,
        -1, +1,
        -1, +1,
        -1, +1,
        -1, +1,
        -1, +1,
        -1, +1,
        -1, +1
    );

    return _mm256_sign_epi16(swapped, sign);
}

static inline __m256i mul_minus_j_i16_256(__m256i z)
{
    const __m256i swapped = swap_complex_pairs_i16_256(z);

    const __m256i sign = _mm256_setr_epi16(
        +1, -1,
        +1, -1,
        +1, -1,
        +1, -1,
        +1, -1,
        +1, -1,
        +1, -1,
        +1, -1
    );

    return _mm256_sign_epi16(swapped, sign);
}

static inline __m256i mul_minus_j_dir_i16_256(__m256i z, dft_dir_t dir)
{
    return (dir == DFT_DIR_FORWARD)
        ? mul_minus_j_i16_256(z)
        : mul_j_i16_256(z);
}

static inline __m256i mul_plus_j_dir_i16_256(__m256i z, dft_dir_t dir)
{
    return (dir == DFT_DIR_FORWARD)
        ? mul_j_i16_256(z)
        : mul_minus_j_i16_256(z);
}

static inline void transpose8_complex_i16_256(
    __m256i *r0,
    __m256i *r1,
    __m256i *r2,
    __m256i *r3,
    __m256i *r4,
    __m256i *r5,
    __m256i *r6,
    __m256i *r7)
{

    const __m256i a = *r0;
    const __m256i b = *r1;
    const __m256i c = *r2;
    const __m256i d = *r3;
    const __m256i e = *r4;
    const __m256i f = *r5;
    const __m256i g = *r6;
    const __m256i h = *r7;


    const __m256i t0 = _mm256_unpacklo_epi32(a, b);
    const __m256i t1 = _mm256_unpackhi_epi32(a, b);
    const __m256i t2 = _mm256_unpacklo_epi32(c, d);
    const __m256i t3 = _mm256_unpackhi_epi32(c, d);
    const __m256i t4 = _mm256_unpacklo_epi32(e, f);
    const __m256i t5 = _mm256_unpackhi_epi32(e, f);
    const __m256i t6 = _mm256_unpacklo_epi32(g, h);
    const __m256i t7 = _mm256_unpackhi_epi32(g, h);


    const __m256i s0 = _mm256_unpacklo_epi64(t0, t2);
    const __m256i s1 = _mm256_unpackhi_epi64(t0, t2);
    const __m256i s2 = _mm256_unpacklo_epi64(t1, t3);
    const __m256i s3 = _mm256_unpackhi_epi64(t1, t3);

    const __m256i s4 = _mm256_unpacklo_epi64(t4, t6);
    const __m256i s5 = _mm256_unpackhi_epi64(t4, t6);
    const __m256i s6 = _mm256_unpacklo_epi64(t5, t7);
    const __m256i s7 = _mm256_unpackhi_epi64(t5, t7);

 
    *r0 = _mm256_permute2x128_si256(s0, s4, 0x20);
    *r1 = _mm256_permute2x128_si256(s1, s5, 0x20);
    *r2 = _mm256_permute2x128_si256(s2, s6, 0x20);
    *r3 = _mm256_permute2x128_si256(s3, s7, 0x20);

    *r4 = _mm256_permute2x128_si256(s0, s4, 0x31);
    *r5 = _mm256_permute2x128_si256(s1, s5, 0x31);
    *r6 = _mm256_permute2x128_si256(s2, s6, 0x31);
    *r7 = _mm256_permute2x128_si256(s3, s7, 0x31);
}

static inline __m256i complex_mul8_prepack_q15_256(
    __m256i a,
    __m256i w_re_re,
    __m256i w_im_signed)
{
    const __m256i a_swapped = swap_complex_pairs_i16_256(a);

    const __m256i prod_re = _mm256_mulhrs_epi16(a,         w_re_re);
    const __m256i prod_im = _mm256_mulhrs_epi16(a_swapped, w_im_signed);

    return _mm256_adds_epi16(prod_re, prod_im);
}



static inline void dft8x8lts_q15_256_dir(
    const __m256i x0,
    const __m256i x1,
    const __m256i x2,
    const __m256i x3,
    const __m256i x4,
    const __m256i x5,
    const __m256i x6,
    const __m256i x7,
    __m256i *Y0,
    __m256i *Y1,
    __m256i *Y2,
    __m256i *Y3,
    __m256i *Y4,
    __m256i *Y5,
    __m256i *Y6,
    __m256i *Y7,
    dft_dir_t dir)
{
    const __m256i c = _mm256_set1_epi16(Q15_INV_SQRT2);

    const __m256i s04 = _mm256_adds_epi16(x0, x4);
    const __m256i d04 = _mm256_subs_epi16(x0, x4);

    const __m256i s15 = _mm256_adds_epi16(x1, x5);
    const __m256i d15 = _mm256_subs_epi16(x1, x5);

    const __m256i s26 = _mm256_adds_epi16(x2, x6);
    const __m256i d26 = _mm256_subs_epi16(x2, x6);

    const __m256i s37 = _mm256_adds_epi16(x3, x7);
    const __m256i d37 = _mm256_subs_epi16(x3, x7);

    const __m256i s02 = _mm256_adds_epi16(s04, s26);
    const __m256i d02 = _mm256_subs_epi16(s04, s26);

    const __m256i s13 = _mm256_adds_epi16(s15, s37);
    const __m256i d13 = _mm256_subs_epi16(s15, s37);

    *Y0 = _mm256_adds_epi16(s02, s13);
    *Y4 = _mm256_subs_epi16(s02, s13);

    *Y2 = _mm256_adds_epi16(d02, mul_minus_j_dir_i16_256(d13, dir));
    *Y6 = _mm256_adds_epi16(d02, mul_plus_j_dir_i16_256(d13, dir));

    const __m256i p = _mm256_adds_epi16(d15, d37);
    const __m256i q = _mm256_subs_epi16(d15, d37);

    const __m256i d26_mj = mul_minus_j_dir_i16_256(d26, dir);
    const __m256i d26_pj = mul_plus_j_dir_i16_256(d26, dir);

    const __m256i base_mj = _mm256_adds_epi16(d04, d26_mj);
    const __m256i base_pj = _mm256_adds_epi16(d04, d26_pj);

    const __m256i t1_arg = _mm256_adds_epi16(q, mul_minus_j_dir_i16_256(p, dir));
    const __m256i t3_arg = _mm256_adds_epi16(q, mul_plus_j_dir_i16_256(p, dir));

    const __m256i t1 = _mm256_mulhrs_epi16(c, t1_arg);
    const __m256i t3 = _mm256_mulhrs_epi16(c, t3_arg);

    *Y1 = _mm256_adds_epi16(base_mj, t1);
    *Y5 = _mm256_subs_epi16(base_mj, t1);

    *Y7 = _mm256_adds_epi16(base_pj, t3);
    *Y3 = _mm256_subs_epi16(base_pj, t3);
}

static inline void dft64ltslts(
    const c16_t *src,
    c16_t *dst,
    dft_dir_t dir)
{
    const __m256i x0 = _mm256_loadu_si256((const __m256i *)(const void *)(src + 0));
    const __m256i x1 = _mm256_loadu_si256((const __m256i *)(const void *)(src + 8));
    const __m256i x2 = _mm256_loadu_si256((const __m256i *)(const void *)(src + 16));
    const __m256i x3 = _mm256_loadu_si256((const __m256i *)(const void *)(src + 24));
    const __m256i x4 = _mm256_loadu_si256((const __m256i *)(const void *)(src + 32));
    const __m256i x5 = _mm256_loadu_si256((const __m256i *)(const void *)(src + 40));
    const __m256i x6 = _mm256_loadu_si256((const __m256i *)(const void *)(src + 48));
    const __m256i x7 = _mm256_loadu_si256((const __m256i *)(const void *)(src + 56));

    __m256i H0, H1, H2, H3;
    __m256i H4, H5, H6, H7;

    dft8x8lts_q15_256_dir(
        x0, x1, x2, x3,
        x4, x5, x6, x7,
        &H0, &H1, &H2, &H3,
        &H4, &H5, &H6, &H7,
        dir
    );
    const TwiddleTable *tw = twiddle_table_get(64);
    const __m256i *C64_RE =
        (dir == DFT_DIR_FORWARD)
            ? tw->C64_RE_RE_q15_256
            : tw->C64_RE_RE_q15_256_inverse;

    const __m256i *C64_IM =
        (dir == DFT_DIR_FORWARD)
            ? tw->C64_IM_SIGNED_q15_256
            : tw->C64_IM_SIGNED_q15_256_inverse;
    H0 = _mm256_srai_epi16(H0, 3);

    H1 = complex_mul8_prepack_q15_256(H1, C64_RE[1], C64_IM[1]);
    H2 = complex_mul8_prepack_q15_256(H2, C64_RE[2], C64_IM[2]);
    H3 = complex_mul8_prepack_q15_256(H3, C64_RE[3], C64_IM[3]);
    H4 = complex_mul8_prepack_q15_256(H4, C64_RE[4], C64_IM[4]);
    H5 = complex_mul8_prepack_q15_256(H5, C64_RE[5], C64_IM[5]);
    H6 = complex_mul8_prepack_q15_256(H6, C64_RE[6], C64_IM[6]);
    H7 = complex_mul8_prepack_q15_256(H7, C64_RE[7], C64_IM[7]);

    /*
     * Transpose complex 8x8.
     */
    transpose8_complex_i16_256(
        &H0, &H1, &H2, &H3,
        &H4, &H5, &H6, &H7
    );

    __m256i Y0, Y1, Y2, Y3;
    __m256i Y4, Y5, Y6, Y7;

    /*
     * Second stage.
     */
    dft8x8lts_q15_256_dir(
        H0, H1, H2, H3,
        H4, H5, H6, H7,
        &Y0, &Y1, &Y2, &Y3,
        &Y4, &Y5, &Y6, &Y7,
        dir
    );
    _mm256_storeu_si256((__m256i *)(void *)(dst + 0),  Y0);
    _mm256_storeu_si256((__m256i *)(void *)(dst + 8),  Y1);
    _mm256_storeu_si256((__m256i *)(void *)(dst + 16), Y2);
    _mm256_storeu_si256((__m256i *)(void *)(dst + 24), Y3);
    _mm256_storeu_si256((__m256i *)(void *)(dst + 32), Y4);
    _mm256_storeu_si256((__m256i *)(void *)(dst + 40), Y5);
    _mm256_storeu_si256((__m256i *)(void *)(dst + 48), Y6);
    _mm256_storeu_si256((__m256i *)(void *)(dst + 56), Y7);
}

//===================================================================
// DFT128 int
//===================================================================
static inline __m256i scale_q15_inv_sqrt2_256(__m256i x)
{
    const __m256i s = _mm256_set1_epi16(Q15_INV_SQRT2);
    return _mm256_mulhrs_epi16(x, s);
}

static inline void dft128_stage0_blk_q15_256_dir(
    const c16_t *src,
    c16_t *a,
    c16_t *b,
    int blk,
    dft_dir_t dir)
{
    const __m256i x0 = _mm256_loadu_si256(
        (const __m256i *)(const void *)(src + 8 * blk)
    );

    const __m256i x1 = _mm256_loadu_si256(
        (const __m256i *)(const void *)(src + 64 + 8 * blk)
    );

    __m256i sum  = _mm256_adds_epi16(x0, x1);
    __m256i diff = _mm256_subs_epi16(x0, x1);

    sum = scale_q15_inv_sqrt2_256(sum);
    const TwiddleTable *tw = twiddle_table_get(128);
    const __m256i *W128_RE =
        (dir == DFT_DIR_FORWARD)
            ? tw->W128_RE_RE_q15_256
            : tw->W128_RE_RE_q15_256_inverse;

    const __m256i *W128_IM =
        (dir == DFT_DIR_FORWARD)
            ? tw->W128_IM_SIGNED_q15_256
            : tw->W128_IM_SIGNED_q15_256_inverse;

    diff = complex_mul8_prepack_q15_256(
        diff,
        W128_RE[blk],
        W128_IM[blk]
    );

    _mm256_store_si256(
        (__m256i *)(void *)(a + 8 * blk),
        sum
    );

    _mm256_store_si256(
        (__m256i *)(void *)(b + 8 * blk),
        diff
    );
}

static inline void interleave64_complex_q15_256(
    const c16_t *A,
    const c16_t *B,
    c16_t *dst)
{
    for (int blk = 0; blk < 8; blk++) {
        const __m256i va = _mm256_load_si256(
            (const __m256i *)(const void *)(A + 8 * blk)
        );

        const __m256i vb = _mm256_load_si256(
            (const __m256i *)(const void *)(B + 8 * blk)
        );

        /*
         * va = [A0 A1 A2 A3 | A4 A5 A6 A7]
         * vb = [B0 B1 B2 B3 | B4 B5 B6 B7]
         *
         * Each A0/B0 is one c16_t = 32 bits.
         */
        const __m256i lo = _mm256_unpacklo_epi32(va, vb);
        const __m256i hi = _mm256_unpackhi_epi32(va, vb);

        /*
         * out0 = [A0 B0 A1 B1 A2 B2 A3 B3]
         * out1 = [A4 B4 A5 B5 A6 B6 A7 B7]
         */
        const __m256i out0 = _mm256_permute2x128_si256(lo, hi, 0x20);
        const __m256i out1 = _mm256_permute2x128_si256(lo, hi, 0x31);

        _mm256_storeu_si256(
            (__m256i *)(void *)(dst + 16 * blk),
            out0
        );

        _mm256_storeu_si256(
            (__m256i *)(void *)(dst + 16 * blk + 8),
            out1
        );
    }
}

static inline void dft128lts_dir(
    const c16_t *src,
    c16_t *dst,
    dft_dir_t dir)
{
    c16_t a[64] __attribute__((aligned(32)));
    c16_t b[64] __attribute__((aligned(32)));

    c16_t A[64] __attribute__((aligned(32)));
    c16_t B[64] __attribute__((aligned(32)));

    for (int blk = 0; blk < 8; blk++) {
        dft128_stage0_blk_q15_256_dir(src, a, b, blk, dir);
    }

    dft64ltslts(a, A, dir);
    dft64ltslts(b, B, dir);

    interleave64_complex_q15_256(A, B, dst);
}



//===================================================================
// DFT SPLIT
//===================================================================
static inline void sr_combine_simd(c16_t *E,
                                   c16_t *O1,
                                   c16_t *O3,
                                   c16_t *y,
                                   int N,
                                   const sr_twiddle_simd_t *tw,
                                   dft_dir_t dir)
{
    int half = N / 2;
    int quarter = N / 4;

    const simde__m256i swap_mask =
        simde_mm256_setr_epi8(
             2,  3,  0,  1,  6,  7,  4,  5,
            10, 11,  8,  9, 14, 15, 12, 13,
             2,  3,  0,  1,  6,  7,  4,  5,
            10, 11,  8,  9, 14, 15, 12, 13
        );

    simde__m256i sign_mask;

    if (dir == DFT_DIR_FORWARD) {

        sign_mask = simde_mm256_setr_epi16(
             1, -1,  1, -1,  1, -1,  1, -1,
             1, -1,  1, -1,  1, -1,  1, -1
        );
    } else {
        sign_mask = simde_mm256_setr_epi16(
            -1,  1, -1,  1, -1,  1, -1,  1,
            -1,  1, -1,  1, -1,  1, -1,  1
        );
    }

    const simde__m256i sqrt2_inv =
        simde_mm256_set1_epi16(Q15_INV_SQRT2);

    for (int b = 0; b < tw->blocks; b++) {
        int k = 8 * b;

        simde__m256i O1v = simde_mm256_load_si256((simde__m256i *)&O1[k]);
        simde__m256i O3v = simde_mm256_load_si256((simde__m256i *)&O3[k]);

        simde__m256i t1 = c16_mul_q15_simd256(O1v,
                                               tw->W1_RE_NEGIM[b],
                                               tw->W1_IM_RE[b]);

        simde__m256i t2 = c16_mul_q15_simd256(O3v,
                                               tw->W3_RE_NEGIM[b],
                                               tw->W3_IM_RE[b]);

        simde__m256i a = simde_mm256_add_epi16(t1, t2);
        simde__m256i d = simde_mm256_sub_epi16(t1, t2);

        simde__m256i bval = simde_mm256_shuffle_epi8(d, swap_mask);
        bval = simde_mm256_sign_epi16(bval, sign_mask);

        simde__m256i E0 = simde_mm256_load_si256((simde__m256i *)&E[k]);
        simde__m256i E1 = simde_mm256_load_si256((simde__m256i *)&E[k + quarter]);

        E0 = simde_mm256_mulhrs_epi16(E0, sqrt2_inv);
        E1 = simde_mm256_mulhrs_epi16(E1, sqrt2_inv);
        /*
        a    = simde_mm256_srai_epi16(a, 1);
        bval = simde_mm256_srai_epi16(bval, 1);
        */
        simde__m256i Y0 = simde_mm256_add_epi16(E0, a);
        simde__m256i Y2 = simde_mm256_sub_epi16(E0, a);
        simde__m256i Y1 = simde_mm256_add_epi16(E1, bval);
        simde__m256i Y3 = simde_mm256_sub_epi16(E1, bval);

        simde_mm256_store_si256((simde__m256i *)&y[k],               Y0);
        simde_mm256_store_si256((simde__m256i *)&y[k + quarter],     Y1);
        simde_mm256_store_si256((simde__m256i *)&y[k + half],        Y2);
        simde_mm256_store_si256((simde__m256i *)&y[k + 3 * quarter], Y3);
    }
}

static inline void pack_split_radix_input_avx2_fused(const c16_t *__restrict x,
                                                       c16_t *__restrict sub_in,
                                                       int N)
{
    _Static_assert(sizeof(c16_t) == 4, "c16_t must be 32-bit");

    const int half    = N >> 1;
    const int quarter = N >> 2;

    c16_t *__restrict E_in  = sub_in;
    c16_t *__restrict O1_in = sub_in + half;
    c16_t *__restrict O3_in = sub_in + half + quarter;

    const simde__m256i idx =
        simde_mm256_setr_epi32(0, 2, 4, 6, 1, 5, 3, 7);

    int in = 0;
    int e  = 0;
    int o  = 0;

    for (; in + 32 <= N; in += 32, e += 16, o += 8) {
        simde__m256i v0 =
            simde_mm256_loadu_si256((const simde__m256i *)&x[in + 0]);

        simde__m256i v1 =
            simde_mm256_loadu_si256((const simde__m256i *)&x[in + 8]);

        simde__m256i v2 =
            simde_mm256_loadu_si256((const simde__m256i *)&x[in + 16]);

        simde__m256i v3 =
            simde_mm256_loadu_si256((const simde__m256i *)&x[in + 24]);

        /*
         * p0 = [x0,  x2,  x4,  x6,  x1,  x5,  x3,  x7]
         * p1 = [x8,  x10, x12, x14, x9,  x13, x11, x15]
         * p2 = [x16, x18, x20, x22, x17, x21, x19, x23]
         * p3 = [x24, x26, x28, x30, x25, x29, x27, x31]
         */
        simde__m256i p0 = simde_mm256_permutevar8x32_epi32(v0, idx);
        simde__m256i p1 = simde_mm256_permutevar8x32_epi32(v1, idx);
        simde__m256i p2 = simde_mm256_permutevar8x32_epi32(v2, idx);
        simde__m256i p3 = simde_mm256_permutevar8x32_epi32(v3, idx);

        /*
         * E output:
         *
         * low128(p0) = x0,  x2,  x4,  x6
         * low128(p1) = x8,  x10, x12, x14
         * low128(p2) = x16, x18, x20, x22
         * low128(p3) = x24, x26, x28, x30
         *
         * On stocke en 128-bit pour éviter vinserti64x2 / vinserti128.
         */
        simde_mm_store_si128((simde__m128i *)&E_in[e + 0],
                             simde_mm256_castsi256_si128(p0));

        simde_mm_store_si128((simde__m128i *)&E_in[e + 4],
                             simde_mm256_castsi256_si128(p1));

        simde_mm_store_si128((simde__m128i *)&E_in[e + 8],
                             simde_mm256_castsi256_si128(p2));

        simde_mm_store_si128((simde__m128i *)&E_in[e + 12],
                             simde_mm256_castsi256_si128(p3));

        /*
         * high128(p0) = x1,  x5,  x3,  x7
         * high128(p1) = x9,  x13, x11, x15
         * high128(p2) = x17, x21, x19, x23
         * high128(p3) = x25, x29, x27, x31
         */
        simde__m128i h0 = simde_mm256_extracti128_si256(p0, 1);
        simde__m128i h1 = simde_mm256_extracti128_si256(p1, 1);
        simde__m128i h2 = simde_mm256_extracti128_si256(p2, 1);
        simde__m128i h3 = simde_mm256_extracti128_si256(p3, 1);

        /*
         * O1:
         * unpacklo_epi64(h0,h1) = x1,  x5,  x9,  x13
         * unpacklo_epi64(h2,h3) = x17, x21, x25, x29
         *
         * O3:
         * unpackhi_epi64(h0,h1) = x3,  x7,  x11, x15
         * unpackhi_epi64(h2,h3) = x19, x23, x27, x31
         */
        simde__m128i o1_0 = simde_mm_unpacklo_epi64(h0, h1);
        simde__m128i o3_0 = simde_mm_unpackhi_epi64(h0, h1);

        simde__m128i o1_1 = simde_mm_unpacklo_epi64(h2, h3);
        simde__m128i o3_1 = simde_mm_unpackhi_epi64(h2, h3);

        simde_mm_store_si128((simde__m128i *)&O1_in[o + 0], o1_0);
        simde_mm_store_si128((simde__m128i *)&O1_in[o + 4], o1_1);

        simde_mm_store_si128((simde__m128i *)&O3_in[o + 0], o3_0);
        simde_mm_store_si128((simde__m128i *)&O3_in[o + 4], o3_1);
    }
}



static void dft_split_radix_pure_simd_core(c16_t *__restrict x,
                                           c16_t *__restrict y,
                                           c16_t *__restrict work,
                                           int N,
                                           dft_dir_t dir)
{
    if (N == 64) {
        dft64ltslts(x, y, dir);
        return;
    }

    if (N == 128) {
        dft128lts_dir(x, y, dir);
        return;
    }
    

    const int half    = N >> 1;
    const int quarter = N >> 2;

    c16_t *sub_in  = work;
    c16_t *sub_out = work + N;

    c16_t *E  = sub_out;
    c16_t *O1 = sub_out + half;
    c16_t *O3 = sub_out + half + quarter;

    pack_split_radix_input_avx2_fused(x, sub_in, N);

    dft_split_radix_pure_simd_core(sub_in,
                                   E,
                                   work + 2 * N,
                                   half,
                                   dir);

    dft_split_radix_pure_simd_core(sub_in + half,
                                   O1,
                                   work + 2 * N,
                                   quarter,
                                   dir);

    dft_split_radix_pure_simd_core(sub_in + half + quarter,
                                   O3,
                                   work + 2 * N,
                                   quarter,
                                   dir);
    const sr_twiddle_simd_t *table = sr_twiddle_table_get(N, dir);

    if (!table) {
        return;
    }

    sr_combine_simd(E, O1, O3, y, N, table, dir);
}


static void dft_split_radix_pure_simd(c16_t *x, c16_t *y, int N, dft_dir_t dir)
{
    c16_t work[262144] __attribute__((aligned(64)));

    if (!work) {
        printf("work allocation failed\n");
        return;
    }

    dft_split_radix_pure_simd_core(x, y, work, N, dir);

}

//===================================================================
// RADIX_3
//===================================================================
static inline void radix3_combine4_q15_128_fast(__m128i A,
                                                __m128i X1,
                                                __m128i X2,
                                                __m128i w1_re_re,
                                                __m128i w1_im_im,
                                                __m128i w2_re_re,
                                                __m128i w2_im_im,
                                                __m128i *Y0,
                                                __m128i *Y1,
                                                __m128i *Y2,
                                                dft_dir_t dir)
{
    /*
     * B = W1 * X1
     * C = W2 * X2
     */
    const __m128i Bs = complex_mul4_prepack_q15_128(X1, w1_re_re, w1_im_im);
    const __m128i Cs = complex_mul4_prepack_q15_128(X2, w2_re_re, w2_im_im);

    /*
     * Correct scaling for N = 3 * size when sub-FFTs are already scaled:
     *
     * final scale = 1 / sqrt(3)
     *
     * Y0 = (A + B + C) / sqrt(3)
     */
    const __m128i As = q15_mul_i16_128(A, Q15_INV_SQRT3);
    //const __m128i Bs = q15_mul_i16_128(B, Q15_INV_SQRT3);
    //const __m128i Cs = q15_mul_i16_128(C, Q15_INV_SQRT3);

    *Y0 = add3s_epi16(As, Bs, Cs);

    /*
     * base = A/sqrt(3) - B/(2*sqrt(3)) - C/(2*sqrt(3))
     */
    const __m128i Bh = _mm_srai_epi16(Bs,1);
    const __m128i Ch = _mm_srai_epi16(Cs,1);

    const __m128i base = _mm_subs_epi16(_mm_subs_epi16(As, Bh), Ch);

    /*
     * Z = c3 * (B - C) / sqrt(3)
     *   = 0.5 * (B - C)
     */

    const __m128i Z = q15_mul_i16_128(_mm_subs_epi16(Bs, Cs),Q15_HALF_SQRT3);

    /*
     * Y1 = base - j*Z
     * Y2 = base + j*Z
     */
    *Y1 = _mm_adds_epi16(base, (dir == DFT_DIR_FORWARD) ? mul_minuslts_q15_128(Z) : mullts_q15_128(Z));
    *Y2 = _mm_adds_epi16(base, (dir == DFT_DIR_FORWARD) ? mullts_q15_128(Z) : mul_minuslts_q15_128(Z));
}


static void radix_3_fft_c16_scaled(const c16_t *src,
                                           c16_t *dst,
                                           int N,
                                           dft_dir_t dir)
{
    if ((N % 3) != 0) {
        printf("radix_3_fft_forward_c16_scaled: N must be divisible by 3\n");
        return;
    }

    const int size = N / 3;

    if (!is_power_of_two_int(size)) {
        printf("radix_3_fft_forward_c16_scaled: N/3 must be power of two\n");
        return;
    }

    const TwiddleTable *tw = twiddle_table_get(N);

    if (!tw ||
        !tw->r3_q15_w1_re ||
        !tw->r3_q15_w1_im ||
        !tw->r3_q15_w2_re ||
        !tw->r3_q15_w2_im ||
        !tw->r3_q15_w1_re_inv ||
        !tw->r3_q15_w1_im_inv ||
        !tw->r3_q15_w2_re_inv ||
        !tw->r3_q15_w2_im_inv) {
        printf("radix_3_fft_c16_scaled: missing radix-3 Q15 twiddles\n");
        return;
    }

    c16_t *work = aligned_malloc64(sizeof(c16_t) * 6 * (size_t)N);

    if (!work) {
        printf("radix_3_fft_forward_c16_scaled: work allocation failed\n");
        return;
    }

    c16_t *in       = work;
    c16_t *tmp      = work + N;
    c16_t *sub_work = work + 2 * N;

    for (int n = 0; n < size; n++) {
        in[0 * size + n] = src[3 * n + 0];
        in[1 * size + n] = src[3 * n + 1];
        in[2 * size + n] = src[3 * n + 2];
    }

    dft_split_radix_pure_simd_core(in + 0 * size,
                                   tmp + 0 * size,
                                   sub_work,
                                   size,
                                   dir);

    dft_split_radix_pure_simd_core(in + 1 * size,
                                   tmp + 1 * size,
                                   sub_work,
                                   size,
                                   dir);

    dft_split_radix_pure_simd_core(in + 2 * size,
                                   tmp + 2 * size,
                                   sub_work,
                                   size,
                                   dir);

    const __m128i *w1_re_tbl =
        (dir == DFT_DIR_FORWARD) ? tw->r3_q15_w1_re : tw->r3_q15_w1_re_inv;

    const __m128i *w1_im_tbl =
        (dir == DFT_DIR_FORWARD) ? tw->r3_q15_w1_im : tw->r3_q15_w1_im_inv;

    const __m128i *w2_re_tbl =
        (dir == DFT_DIR_FORWARD) ? tw->r3_q15_w2_re : tw->r3_q15_w2_re_inv;

    const __m128i *w2_im_tbl =
        (dir == DFT_DIR_FORWARD) ? tw->r3_q15_w2_im : tw->r3_q15_w2_im_inv;

    int k = 0;

    for (; k + 3 < size; k += 4) {
        const int b = k >> 2;

        const __m128i A  = _mm_loadu_si128((const __m128i *)(tmp + 0 * size + k));
        const __m128i X1 = _mm_loadu_si128((const __m128i *)(tmp + 1 * size + k));
        const __m128i X2 = _mm_loadu_si128((const __m128i *)(tmp + 2 * size + k));

        __m128i Y0;
        __m128i Y1;
        __m128i Y2;

        radix3_combine4_q15_128_fast(A,
                                      X1,
                                      X2,
                                      w1_re_tbl[b],
                                      w1_im_tbl[b],
                                      w2_re_tbl[b],
                                      w2_im_tbl[b],
                                      &Y0,
                                      &Y1,
                                      &Y2,
                                      dir);

        _mm_storeu_si128((__m128i *)(dst + 0 * size + k), Y0);
        _mm_storeu_si128((__m128i *)(dst + 1 * size + k), Y1);
        _mm_storeu_si128((__m128i *)(dst + 2 * size + k), Y2);
    }

    if (k != size) {
        printf("radix_3_fft_forward_c16_scaled: scalar tail not implemented, size=%d\n", size);
    }

    free(work);
}

//===================================================================
// DFT MIXED RADIX
//===================================================================
static void dft_mixed_radix_c16_scaled(c16_t *x, c16_t *y, int N, dft_dir_t dir)
{
    
    if ((N % 3) == 0 && is_power_of_two_int(N / 3)) {
        radix_3_fft_c16_scaled(x, y, N, dir);
        return;
    }
    if (is_power_of_two_int(N)) {
        dft_split_radix_pure_simd(x, y, N, dir);
        return;
    }

    printf("dft_mixed_radix_c16_scaled: unsupported N = %d\n", N);
}

#define DEFINE_MIXED_DFT_ONLY(N)                                      \
void dft##N(int16_t *input, int16_t *output, uint8_t scale_flag)       \
{                                                                     \
    (void)scale_flag;                                                 \
                                                                      \
                                                                      \
    dft_mixed_radix_c16_scaled((c16_t *)input,                        \
                               (c16_t *)output,                       \
                               N,                                     \
                               DFT_DIR_FORWARD);                      \
}

#define DEFINE_MIXED_IDFT_ONLY(N)                                     \
void idft##N(int16_t *input, int16_t *output, uint8_t scale_flag)      \
{                                                                     \
    (void)scale_flag;                                                 \
                                                                      \
                                                                      \
    dft_mixed_radix_c16_scaled((c16_t *)input,                        \
                               (c16_t *)output,                       \
                               N,                                     \
                               DFT_DIR_INVERSE);                      \
}


DEFINE_MIXED_DFT_ONLY(192)
DEFINE_MIXED_DFT_ONLY(384)
DEFINE_MIXED_DFT_ONLY(768)
DEFINE_MIXED_DFT_ONLY(1536)
DEFINE_MIXED_DFT_ONLY(3072)
DEFINE_MIXED_DFT_ONLY(6144)
DEFINE_MIXED_DFT_ONLY(12288)
DEFINE_MIXED_DFT_ONLY(64)
DEFINE_MIXED_DFT_ONLY(128)
DEFINE_MIXED_DFT_ONLY(256)
DEFINE_MIXED_DFT_ONLY(512)
DEFINE_MIXED_DFT_ONLY(1024)
DEFINE_MIXED_DFT_ONLY(2048)
DEFINE_MIXED_DFT_ONLY(4096)
DEFINE_MIXED_DFT_ONLY(8192)
DEFINE_MIXED_DFT_ONLY(16384)

DEFINE_MIXED_IDFT_ONLY(64)
DEFINE_MIXED_IDFT_ONLY(128)
DEFINE_MIXED_IDFT_ONLY(256)
DEFINE_MIXED_IDFT_ONLY(512)
DEFINE_MIXED_IDFT_ONLY(1024)
DEFINE_MIXED_IDFT_ONLY(2048)
DEFINE_MIXED_IDFT_ONLY(4096)
DEFINE_MIXED_IDFT_ONLY(8192)
DEFINE_MIXED_IDFT_ONLY(16384)
DEFINE_MIXED_IDFT_ONLY(192)
DEFINE_MIXED_IDFT_ONLY(384)
DEFINE_MIXED_IDFT_ONLY(768)
DEFINE_MIXED_IDFT_ONLY(1536)
DEFINE_MIXED_IDFT_ONLY(3072)
DEFINE_MIXED_IDFT_ONLY(6144)
DEFINE_MIXED_IDFT_ONLY(12288)

__attribute__((always_inline)) static inline void cmac(simde__m128i a, simde__m128i b, simde__m128i *re32, simde__m128i *im32)
{
  simde__m128i cmac_tmp, cmac_tmp_re32, cmac_tmp_im32;

  cmac_tmp = simde_mm_sign_epi16(b, *(simde__m128i *)reflip);
  cmac_tmp_re32 = simde_mm_madd_epi16(a, cmac_tmp);

  //  cmac_tmp    = simde_mm_shufflelo_epi16(b, SIMDE_MM_SHUFFLE(2,3,0,1));
  //  cmac_tmp    = simde_mm_shufflehi_epi16(cmac_tmp, SIMDE_MM_SHUFFLE(2,3,0,1));
  cmac_tmp = simde_mm_shuffle_epi8(b, simde_mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
  cmac_tmp_im32 = simde_mm_madd_epi16(cmac_tmp, a);

  *re32 = simde_mm_add_epi32(*re32, cmac_tmp_re32);
  *im32 = simde_mm_add_epi32(*im32, cmac_tmp_im32);
}

__attribute__((always_inline)) static inline void cmacc(simde__m128i a, simde__m128i b, simde__m128i *re32, simde__m128i *im32)
{
  simde__m128i cmac_tmp, cmac_tmp_re32, cmac_tmp_im32;

  cmac_tmp_re32 = simde_mm_madd_epi16(a, b);

  cmac_tmp = simde_mm_sign_epi16(b, *(simde__m128i *)reflip);
  //  cmac_tmp    = simde_mm_shufflelo_epi16(b, SIMDE_MM_SHUFFLE(2,3,0,1));
  //  cmac_tmp    = simde_mm_shufflehi_epi16(cmac_tmp, SIMDE_MM_SHUFFLE(2,3,0,1));
  cmac_tmp = simde_mm_shuffle_epi8(cmac_tmp, simde_mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
  cmac_tmp_im32 = simde_mm_madd_epi16(cmac_tmp, a);

  *re32 = simde_mm_add_epi32(*re32, cmac_tmp_re32);
  *im32 = simde_mm_add_epi32(*im32, cmac_tmp_im32);
}

__attribute__((always_inline)) static inline void cmac_256(simde__m256i a, simde__m256i b, simde__m256i *re32, simde__m256i *im32)
{
  simde__m256i cmac_tmp, cmac_tmp_re32, cmac_tmp_im32;
  simde__m256i imshuffle = simde_mm256_set_epi8(29,
                                                28,
                                                31,
                                                30,
                                                25,
                                                24,
                                                27,
                                                26,
                                                21,
                                                20,
                                                23,
                                                22,
                                                17,
                                                16,
                                                19,
                                                18,
                                                13,
                                                12,
                                                15,
                                                14,
                                                9,
                                                8,
                                                11,
                                                10,
                                                5,
                                                4,
                                                7,
                                                6,
                                                1,
                                                0,
                                                3,
                                                2);

  cmac_tmp = simde_mm256_sign_epi16(b, *(simde__m256i *)reflip);
  cmac_tmp_re32  = simde_mm256_madd_epi16(a,cmac_tmp);

  cmac_tmp       = simde_mm256_shuffle_epi8(b,imshuffle);
  cmac_tmp_im32  = simde_mm256_madd_epi16(cmac_tmp,a);

  *re32 = simde_mm256_add_epi32(*re32,cmac_tmp_re32);
  *im32 = simde_mm256_add_epi32(*im32,cmac_tmp_im32);
}

__attribute__((always_inline)) static inline void cmult(simde__m128i a, simde__m128i b, simde__m128i *re32, simde__m128i *im32)
{
  register simde__m128i mmtmpb;

  mmtmpb = simde_mm_sign_epi16(b, *(simde__m128i *)reflip);
  *re32 = simde_mm_madd_epi16(a, mmtmpb);
  //  mmtmpb    = simde_mm_shufflelo_epi16(b, SIMDE_MM_SHUFFLE(2,3,0,1));
  //  mmtmpb    = simde_mm_shufflehi_epi16(mmtmpb, SIMDE_MM_SHUFFLE(2,3,0,1));
  mmtmpb = simde_mm_shuffle_epi8(b, simde_mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
  *im32 = simde_mm_madd_epi16(a, mmtmpb);
}

__attribute__((always_inline)) static inline void cmult_256(simde__m256i a, simde__m256i b, simde__m256i *re32, simde__m256i *im32)
{
  register simde__m256i mmtmpb;
  simde__m256i const perm_mask = simde_mm256_set_epi8(29,
                                                      28,
                                                      31,
                                                      30,
                                                      25,
                                                      24,
                                                      27,
                                                      26,
                                                      21,
                                                      20,
                                                      23,
                                                      22,
                                                      17,
                                                      16,
                                                      19,
                                                      18,
                                                      13,
                                                      12,
                                                      15,
                                                      14,
                                                      9,
                                                      8,
                                                      11,
                                                      10,
                                                      5,
                                                      4,
                                                      7,
                                                      6,
                                                      1,
                                                      0,
                                                      3,
                                                      2);

  mmtmpb = simde_mm256_sign_epi16(b, *(simde__m256i *)reflip);
  *re32     = simde_mm256_madd_epi16(a,mmtmpb);
  mmtmpb    = simde_mm256_shuffle_epi8(b,perm_mask);
  *im32 = simde_mm256_madd_epi16(a, mmtmpb);
}

__attribute__((always_inline)) static inline void cmultc(simde__m128i a, simde__m128i b, simde__m128i *re32, simde__m128i *im32)
{
  register simde__m128i mmtmpb;

  *re32 = simde_mm_madd_epi16(a, b);
  mmtmpb = simde_mm_sign_epi16(b, *(simde__m128i *)reflip);
  mmtmpb = simde_mm_shuffle_epi8(mmtmpb, simde_mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
  *im32 = simde_mm_madd_epi16(a, mmtmpb);
}

__attribute__((always_inline)) static inline void cmultc_256(simde__m256i a, simde__m256i b, simde__m256i *re32, simde__m256i *im32)
{
  register simde__m256i mmtmpb;
  simde__m256i const perm_mask = simde_mm256_set_epi8(29,
                                                      28,
                                                      31,
                                                      30,
                                                      25,
                                                      24,
                                                      27,
                                                      26,
                                                      21,
                                                      20,
                                                      23,
                                                      22,
                                                      17,
                                                      16,
                                                      19,
                                                      18,
                                                      13,
                                                      12,
                                                      15,
                                                      14,
                                                      9,
                                                      8,
                                                      11,
                                                      10,
                                                      5,
                                                      4,
                                                      7,
                                                      6,
                                                      1,
                                                      0,
                                                      3,
                                                      2);

  *re32     = simde_mm256_madd_epi16(a,b);
  mmtmpb = simde_mm256_sign_epi16(b, *(simde__m256i *)reflip);
  mmtmpb    = simde_mm256_shuffle_epi8(mmtmpb,perm_mask);
  *im32 = simde_mm256_madd_epi16(a, mmtmpb);
}

__attribute__((always_inline)) static inline simde__m128i cpack(simde__m128i xre, simde__m128i xim)
{
  register simde__m128i cpack_tmp1, cpack_tmp2;

  cpack_tmp1 = simde_mm_unpacklo_epi32(xre, xim);
  cpack_tmp2 = simde_mm_unpackhi_epi32(xre, xim);
  return (simde_mm_packs_epi32(simde_mm_srai_epi32(cpack_tmp1, 15), simde_mm_srai_epi32(cpack_tmp2, 15)));
}

__attribute__((always_inline)) static inline simde__m256i cpack_256(simde__m256i xre, simde__m256i xim)
{
  register simde__m256i cpack_tmp1, cpack_tmp2;

  cpack_tmp1 = simde_mm256_unpacklo_epi32(xre,xim);
  cpack_tmp2 = simde_mm256_unpackhi_epi32(xre,xim);
  return(simde_mm256_packs_epi32(simde_mm256_srai_epi32(cpack_tmp1,15),simde_mm256_srai_epi32(cpack_tmp2,15)));

}

__attribute__((always_inline)) static inline void packed_cmult(simde__m128i a, simde__m128i b, simde__m128i *c)
{
  simde__m128i cre, cim;
  cmult(a,b,&cre,&cim);
  *c = cpack(cre,cim);

}

__attribute__((always_inline)) static inline void packed_cmult_256(simde__m256i a, simde__m256i b, simde__m256i *c)
{
  simde__m256i cre, cim;
  cmult_256(a,b,&cre,&cim);
  *c = cpack_256(cre,cim);

}

__attribute__((always_inline)) static inline void packed_cmultc(simde__m128i a, simde__m128i b, simde__m128i *c)
{
  simde__m128i cre, cim;

  cmultc(a,b,&cre,&cim);
  *c = cpack(cre,cim);

}

__attribute__((always_inline)) static inline simde__m128i packed_cmult2(simde__m128i a, simde__m128i b, simde__m128i b2);

static inline simde__m128i packed_cmult2(simde__m128i a, simde__m128i b, simde__m128i b2)
{
  register simde__m128i cre, cim;

  cre = simde_mm_madd_epi16(a, b);
  cim = simde_mm_madd_epi16(a, b2);

  return(cpack(cre,cim));

}

__attribute__((always_inline)) static inline simde__m256i packed_cmult2_256(simde__m256i a, simde__m256i b, simde__m256i b2)
{
  register simde__m256i cre, cim;

  cre       = simde_mm256_madd_epi16(a,b);
  cim       = simde_mm256_madd_epi16(a,b2);

  return(cpack_256(cre,cim));

}

const static int16_t W0s[16]__attribute__((aligned(32))) = {32767,0,32767,0,32767,0,32767,0,32767,0,32767,0,32767,0,32767,0};

const static int16_t W13s[16]__attribute__((aligned(32))) = {-16384,-28378,-16384,-28378,-16384,-28378,-16384,-28378,-16384,-28378,-16384,-28378,-16384,-28378,-16384,-28378};
const static int16_t W23s[16]__attribute__((aligned(32))) = {-16384,28378,-16384,28378,-16384,28378,-16384,28378,-16384,28378,-16384,28378,-16384,28378,-16384,28378};

const static int16_t W15s[16]__attribute__((aligned(32))) = {10126,-31163,10126,-31163,10126,-31163,10126,-31163,10126,-31163,10126,-31163,10126,-31163,10126,-31163};
const static int16_t W25s[16]__attribute__((aligned(32))) = {-26509,-19260,-26509,-19260,-26509,-19260,-26509,-19260,-26509,-19260,-26509,-19260,-26509,-19260,-26509,-19260};
const static int16_t W35s[16]__attribute__((aligned(32))) = {-26510,19260,-26510,19260,-26510,19260,-26510,19260,-26510,19260,-26510,19260,-26510,19260,-26510,19260};
const static int16_t W45s[16]__attribute__((aligned(32))) = {10126,31163,10126,31163,10126,31163,10126,31163,10126,31163,10126,31163,10126,31163,10126,31163};

const simde__m128i *W0 = (simde__m128i *)W0s;
const simde__m128i *W13 = (simde__m128i *)W13s;
const simde__m128i *W23 = (simde__m128i *)W23s;
const simde__m128i *W15 = (simde__m128i *)W15s;
const simde__m128i *W25 = (simde__m128i *)W25s;
const simde__m128i *W35 = (simde__m128i *)W35s;
const simde__m128i *W45 = (simde__m128i *)W45s;

const simde__m256i *W0_256 = (simde__m256i *)W0s;
const simde__m256i *W13_256 = (simde__m256i *)W13s;
const simde__m256i *W23_256 = (simde__m256i *)W23s;
const simde__m256i *W15_256 = (simde__m256i *)W15s;
const simde__m256i *W25_256 = (simde__m256i *)W25s;
const simde__m256i *W35_256 = (simde__m256i *)W35s;
const simde__m256i *W45_256 = (simde__m256i *)W45s;

const static int16_t dft_norm_table[16] = {9459,  //12
					   6689,//24
					   5461,//36
					   4729,//482
					   4230,//60
					   23170,//72
					   3344,//96
					   3153,//108
					   2991,//120
					   18918,//sqrt(3),//144
					   18918,//sqrt(3),//180
					   16384,//2, //192
					   18918,//sqrt(3), // 216
					   16384,//2, //240
					   18918,//sqrt(3), // 288
					   14654
}; //sqrt(5) //300

__attribute__((always_inline)) static inline void bfly2(simde__m128i *x0,
                                                        simde__m128i *x1,
                                                        simde__m128i *y0,
                                                        simde__m128i *y1,
                                                        simde__m128i *tw)
{
  simde__m128i x0r_2, x0i_2, x1r_2, x1i_2, dy0r, dy1r, dy0i, dy1i;
  simde__m128i bfly2_tmp1, bfly2_tmp2;

  cmult(*(x0),*(W0),&x0r_2,&x0i_2);
  cmult(*(x1),*(tw),&x1r_2,&x1i_2);

  dy0r = simde_mm_srai_epi32(simde_mm_add_epi32(x0r_2, x1r_2), 15);
  dy1r = simde_mm_srai_epi32(simde_mm_sub_epi32(x0r_2, x1r_2), 15);
  dy0i = simde_mm_srai_epi32(simde_mm_add_epi32(x0i_2, x1i_2), 15);
  //  printf("y0i %d\n",((int16_t *)y0i)[0]);
  dy1i = simde_mm_srai_epi32(simde_mm_sub_epi32(x0i_2, x1i_2), 15);

  bfly2_tmp1 = simde_mm_unpacklo_epi32(dy0r, dy0i);
  bfly2_tmp2 = simde_mm_unpackhi_epi32(dy0r, dy0i);
  *y0 = simde_mm_packs_epi32(bfly2_tmp1, bfly2_tmp2);

  bfly2_tmp1 = simde_mm_unpacklo_epi32(dy1r, dy1i);
  bfly2_tmp2 = simde_mm_unpackhi_epi32(dy1r, dy1i);
  *y1 = simde_mm_packs_epi32(bfly2_tmp1, bfly2_tmp2);
}

__attribute__((always_inline)) static inline void bfly2_256(simde__m256i *x0,
                                                            simde__m256i *x1,
                                                            simde__m256i *y0,
                                                            simde__m256i *y1,
                                                            simde__m256i *tw)
{
  simde__m256i x0r_2, x0i_2, x1r_2, x1i_2, dy0r, dy1r, dy0i, dy1i;
  simde__m256i bfly2_tmp1, bfly2_tmp2;

  cmult_256(*(x0),*(W0_256),&x0r_2,&x0i_2);
  cmult_256(*(x1),*(tw),&x1r_2,&x1i_2);

  dy0r = simde_mm256_srai_epi32(simde_mm256_add_epi32(x0r_2,x1r_2),15);
  dy1r = simde_mm256_srai_epi32(simde_mm256_sub_epi32(x0r_2,x1r_2),15);
  dy0i = simde_mm256_srai_epi32(simde_mm256_add_epi32(x0i_2,x1i_2),15);
  //  printf("y0i %d\n",((int16_t *)y0i)[0]);
  dy1i = simde_mm256_srai_epi32(simde_mm256_sub_epi32(x0i_2,x1i_2),15);

  bfly2_tmp1 = simde_mm256_unpacklo_epi32(dy0r,dy0i);
  bfly2_tmp2 = simde_mm256_unpackhi_epi32(dy0r,dy0i);
  *y0 = simde_mm256_packs_epi32(bfly2_tmp1,bfly2_tmp2);

  bfly2_tmp1 = simde_mm256_unpacklo_epi32(dy1r,dy1i);
  bfly2_tmp2 = simde_mm256_unpackhi_epi32(dy1r,dy1i);
  *y1 = simde_mm256_packs_epi32(bfly2_tmp1,bfly2_tmp2);
}

__attribute__((always_inline)) static inline void bfly2_tw1(simde__m128i *x0, simde__m128i *x1, simde__m128i *y0, simde__m128i *y1)
{
  *y0 = simde_mm_adds_epi16(*x0, *x1);
  *y1 = simde_mm_subs_epi16(*x0, *x1);
}

__attribute__((always_inline)) static inline void bfly2_16_256(simde__m256i *x0,
                                                               simde__m256i *x1,
                                                               simde__m256i *y0,
                                                               simde__m256i *y1,
                                                               simde__m256i *tw,
                                                               simde__m256i *twb)
{
  //  register simde__m256i x1t;
  simde__m256i x1t;

  x1t = packed_cmult2_256(*(x1),*(tw),*(twb));
  /*
  print_shorts256("x0",(int16_t*)x0);
  print_shorts256("x1",(int16_t*)x1);
  print_shorts256("tw",(int16_t*)tw);
  print_shorts256("twb",(int16_t*)twb);
  print_shorts256("x1t",(int16_t*)&x1t);*/
  *y0  = simde_mm256_adds_epi16(*x0,x1t);
  *y1  = simde_mm256_subs_epi16(*x0,x1t);
  
  /*print_shorts256("y0",(int16_t*)y0);
    print_shorts256("y1",(int16_t*)y1);*/
}

__attribute__((always_inline)) static inline void ibfly2_256(simde__m256i *x0,
                                                             simde__m256i *x1,
                                                             simde__m256i *y0,
                                                             simde__m256i *y1,
                                                             simde__m256i *tw)
{
  simde__m256i x0r_2, x0i_2, x1r_2, x1i_2, dy0r, dy1r, dy0i, dy1i;
  simde__m256i bfly2_tmp1, bfly2_tmp2;

  cmultc_256(*(x0),*(W0_256),&x0r_2,&x0i_2);
  cmultc_256(*(x1),*(tw),&x1r_2,&x1i_2);

  dy0r = simde_mm256_srai_epi32(simde_mm256_add_epi32(x0r_2,x1r_2),15);
  dy1r = simde_mm256_srai_epi32(simde_mm256_sub_epi32(x0r_2,x1r_2),15);
  dy0i = simde_mm256_srai_epi32(simde_mm256_add_epi32(x0i_2,x1i_2),15);
  //  printf("y0i %d\n",((int16_t *)y0i)[0]);
  dy1i = simde_mm256_srai_epi32(simde_mm256_sub_epi32(x0i_2,x1i_2),15);

  bfly2_tmp1 = simde_mm256_unpacklo_epi32(dy0r,dy0i);
  bfly2_tmp2 = simde_mm256_unpackhi_epi32(dy0r,dy0i);
  *y0 = simde_mm256_packs_epi32(bfly2_tmp1,bfly2_tmp2);

  bfly2_tmp1 = simde_mm256_unpacklo_epi32(dy1r,dy1i);
  bfly2_tmp2 = simde_mm256_unpackhi_epi32(dy1r,dy1i);
  *y1 = simde_mm256_packs_epi32(bfly2_tmp1,bfly2_tmp2);
}


// This is the radix-3 butterfly (fft)

__attribute__((always_inline)) static inline void bfly3(simde__m128i *x0,
                                                        simde__m128i *x1,
                                                        simde__m128i *x2,
                                                        simde__m128i *y0,
                                                        simde__m128i *y1,
                                                        simde__m128i *y2,
                                                        simde__m128i *tw1,
                                                        simde__m128i *tw2)
{
  simde__m128i tmpre, tmpim, x1_2, x2_2;

  packed_cmult(*(x1),*(tw1),&x1_2);
  packed_cmult(*(x2),*(tw2),&x2_2);
  *(y0) = simde_mm_adds_epi16(*(x0), simde_mm_adds_epi16(x1_2, x2_2));
  cmult(x1_2,*(W13),&tmpre,&tmpim);
  cmac(x2_2,*(W23),&tmpre,&tmpim);
  *(y1) = cpack(tmpre,tmpim);
  *(y1) = simde_mm_adds_epi16(*(x0), *(y1));
  cmult(x1_2,*(W23),&tmpre,&tmpim);
  cmac(x2_2,*(W13),&tmpre,&tmpim);
  *(y2) = cpack(tmpre,tmpim);
  *(y2) = simde_mm_adds_epi16(*(x0), *(y2));
}

__attribute__((always_inline)) static inline void bfly3_256(simde__m256i *x0,
                                                            simde__m256i *x1,
                                                            simde__m256i *x2,
                                                            simde__m256i *y0,
                                                            simde__m256i *y1,
                                                            simde__m256i *y2,
                                                            simde__m256i *tw1,
                                                            simde__m256i *tw2)
{
  simde__m256i tmpre, tmpim, x1_2, x2_2;

  packed_cmult_256(*(x1),*(tw1),&x1_2);
  packed_cmult_256(*(x2),*(tw2),&x2_2);
  *(y0)  = simde_mm256_adds_epi16(*(x0),simde_mm256_adds_epi16(x1_2,x2_2));
  cmult_256(x1_2,*(W13_256),&tmpre,&tmpim);
  cmac_256(x2_2,*(W23_256),&tmpre,&tmpim);
  *(y1) = cpack_256(tmpre,tmpim);
  *(y1) = simde_mm256_adds_epi16(*(x0),*(y1));
  cmult_256(x1_2,*(W23_256),&tmpre,&tmpim);
  cmac_256(x2_2,*(W13_256),&tmpre,&tmpim);
  *(y2) = cpack_256(tmpre,tmpim);
  *(y2) = simde_mm256_adds_epi16(*(x0),*(y2));
}

__attribute__((always_inline)) static inline void ibfly3(simde__m128i *x0,
                                                         simde__m128i *x1,
                                                         simde__m128i *x2,
                                                         simde__m128i *y0,
                                                         simde__m128i *y1,
                                                         simde__m128i *y2,
                                                         simde__m128i *tw1,
                                                         simde__m128i *tw2)
{
  simde__m128i tmpre, tmpim, x1_2, x2_2;

  packed_cmultc(*(x1),*(tw1),&x1_2);
  packed_cmultc(*(x2),*(tw2),&x2_2);
  *(y0) = simde_mm_adds_epi16(*(x0), simde_mm_adds_epi16(x1_2, x2_2));
  cmultc(x1_2,*(W13),&tmpre,&tmpim);
  cmacc(x2_2,*(W23),&tmpre,&tmpim);
  *(y1) = cpack(tmpre,tmpim);
  *(y1) = simde_mm_adds_epi16(*(x0), *(y1));
  cmultc(x1_2,*(W23),&tmpre,&tmpim);
  cmacc(x2_2,*(W13),&tmpre,&tmpim);
  *(y2) = cpack(tmpre,tmpim);
  *(y2) = simde_mm_adds_epi16(*(x0), *(y2));
}

__attribute__((always_inline)) static inline void bfly3_tw1(simde__m128i *x0,
                                                            simde__m128i *x1,
                                                            simde__m128i *x2,
                                                            simde__m128i *y0,
                                                            simde__m128i *y1,
                                                            simde__m128i *y2)
{
  simde__m128i tmpre, tmpim;

  *(y0) = simde_mm_adds_epi16(*(x0), simde_mm_adds_epi16(*(x1), *(x2)));
  cmult(*(x1),*(W13),&tmpre,&tmpim);
  cmac(*(x2),*(W23),&tmpre,&tmpim);
  *(y1) = cpack(tmpre,tmpim);
  *(y1) = simde_mm_adds_epi16(*(x0), *(y1));
  cmult(*(x1),*(W23),&tmpre,&tmpim);
  cmac(*(x2),*(W13),&tmpre,&tmpim);
  *(y2) = cpack(tmpre,tmpim);
  *(y2) = simde_mm_adds_epi16(*(x0), *(y2));
}

__attribute__((always_inline)) static inline void bfly3_tw1_256(simde__m256i *x0,
                                                                simde__m256i *x1,
                                                                simde__m256i *x2,
                                                                simde__m256i *y0,
                                                                simde__m256i *y1,
                                                                simde__m256i *y2)
{
  simde__m256i tmpre, tmpim;

  *(y0) = simde_mm256_adds_epi16(*(x0),simde_mm256_adds_epi16(*(x1),*(x2)));
  cmult_256(*(x1),*(W13_256),&tmpre,&tmpim);
  cmac_256(*(x2),*(W23_256),&tmpre,&tmpim);
  *(y1) = cpack_256(tmpre,tmpim);
  *(y1) = simde_mm256_adds_epi16(*(x0),*(y1));
  cmult_256(*(x1),*(W23_256),&tmpre,&tmpim);
  cmac_256(*(x2),*(W13_256),&tmpre,&tmpim);
  *(y2) = cpack_256(tmpre,tmpim);
  *(y2) = simde_mm256_adds_epi16(*(x0),*(y2));
}

__attribute__((always_inline)) static inline void bfly4(simde__m128i *x0,
                                                        simde__m128i *x1,
                                                        simde__m128i *x2,
                                                        simde__m128i *x3,
                                                        simde__m128i *y0,
                                                        simde__m128i *y1,
                                                        simde__m128i *y2,
                                                        simde__m128i *y3,
                                                        simde__m128i *tw1,
                                                        simde__m128i *tw2,
                                                        simde__m128i *tw3)
{
  simde__m128i x1r_2, x1i_2, x2r_2, x2i_2, x3r_2, x3i_2, dy0r, dy0i, dy1r, dy1i, dy2r, dy2i, dy3r, dy3i;

  //  cmult(*(x0),*(W0),&x0r_2,&x0i_2);
  cmult(*(x1),*(tw1),&x1r_2,&x1i_2);
  cmult(*(x2),*(tw2),&x2r_2,&x2i_2);
  cmult(*(x3),*(tw3),&x3r_2,&x3i_2);
  //  dy0r = simde_mm_add_epi32(x0r_2,simde_mm_add_epi32(x1r_2,simde_mm_add_epi32(x2r_2,x3r_2)));
  //  dy0i = simde_mm_add_epi32(x0i_2,simde_mm_add_epi32(x1i_2,simde_mm_add_epi32(x2i_2,x3i_2)));
  //  *(y0)  = cpack(dy0r,dy0i);
  dy0r = simde_mm_add_epi32(x1r_2, simde_mm_add_epi32(x2r_2, x3r_2));
  dy0i = simde_mm_add_epi32(x1i_2, simde_mm_add_epi32(x2i_2, x3i_2));
  *(y0) = simde_mm_add_epi16(*(x0), cpack(dy0r, dy0i));
  //  dy1r = simde_mm_add_epi32(x0r_2,simde_mm_sub_epi32(x1i_2,simde_mm_add_epi32(x2r_2,x3i_2)));
  //  dy1i = simde_mm_sub_epi32(x0i_2,simde_mm_add_epi32(x1r_2,simde_mm_sub_epi32(x2i_2,x3r_2)));
  //  *(y1)  = cpack(dy1r,dy1i);
  dy1r = simde_mm_sub_epi32(x1i_2, simde_mm_add_epi32(x2r_2, x3i_2));
  dy1i = simde_mm_sub_epi32(simde_mm_sub_epi32(x3r_2, x2i_2), x1r_2);
  *(y1) = simde_mm_add_epi16(*(x0), cpack(dy1r, dy1i));
  //  dy2r = simde_mm_sub_epi32(x0r_2,simde_mm_sub_epi32(x1r_2,simde_mm_sub_epi32(x2r_2,x3r_2)));
  //  dy2i = simde_mm_sub_epi32(x0i_2,simde_mm_sub_epi32(x1i_2,simde_mm_sub_epi32(x2i_2,x3i_2)));
  //  *(y2)  = cpack(dy2r,dy2i);
  dy2r = simde_mm_sub_epi32(simde_mm_sub_epi32(x2r_2, x3r_2), x1r_2);
  dy2i = simde_mm_sub_epi32(simde_mm_sub_epi32(x2i_2, x3i_2), x1i_2);
  *(y2) = simde_mm_add_epi16(*(x0), cpack(dy2r, dy2i));
  //  dy3r = simde_mm_sub_epi32(x0r_2,simde_mm_add_epi32(x1i_2,simde_mm_sub_epi32(x2r_2,x3i_2)));
  //  dy3i = simde_mm_add_epi32(x0i_2,simde_mm_sub_epi32(x1r_2,simde_mm_add_epi32(x2i_2,x3r_2)));
  //  *(y3) = cpack(dy3r,dy3i);
  dy3r = simde_mm_sub_epi32(simde_mm_sub_epi32(x3i_2, x2r_2), x1i_2);
  dy3i = simde_mm_sub_epi32(x1r_2, simde_mm_add_epi32(x2i_2, x3r_2));
  *(y3) = simde_mm_add_epi16(*(x0), cpack(dy3r, dy3i));
}

__attribute__((always_inline)) static inline void bfly4_256(simde__m256i *x0,
                                                            simde__m256i *x1,
                                                            simde__m256i *x2,
                                                            simde__m256i *x3,
                                                            simde__m256i *y0,
                                                            simde__m256i *y1,
                                                            simde__m256i *y2,
                                                            simde__m256i *y3,
                                                            simde__m256i *tw1,
                                                            simde__m256i *tw2,
                                                            simde__m256i *tw3)
{
  simde__m256i x1r_2, x1i_2, x2r_2, x2i_2, x3r_2, x3i_2, dy0r, dy0i, dy1r, dy1i, dy2r, dy2i, dy3r, dy3i;

  //  cmult(*(x0),*(W0),&x0r_2,&x0i_2);
  cmult_256(*(x1),*(tw1),&x1r_2,&x1i_2);
  cmult_256(*(x2),*(tw2),&x2r_2,&x2i_2);
  cmult_256(*(x3),*(tw3),&x3r_2,&x3i_2);
  //  dy0r = simde_mm_add_epi32(x0r_2,simde_mm_add_epi32(x1r_2,simde_mm_add_epi32(x2r_2,x3r_2)));
  //  dy0i = simde_mm_add_epi32(x0i_2,simde_mm_add_epi32(x1i_2,simde_mm_add_epi32(x2i_2,x3i_2)));
  //  *(y0)  = cpack(dy0r,dy0i);
  dy0r = simde_mm256_add_epi32(x1r_2,simde_mm256_add_epi32(x2r_2,x3r_2));
  dy0i = simde_mm256_add_epi32(x1i_2,simde_mm256_add_epi32(x2i_2,x3i_2));
  *(y0)  = simde_mm256_add_epi16(*(x0),cpack_256(dy0r,dy0i));
  //  dy1r = simde_mm_add_epi32(x0r_2,simde_mm_sub_epi32(x1i_2,simde_mm_add_epi32(x2r_2,x3i_2)));
  //  dy1i = simde_mm_sub_epi32(x0i_2,simde_mm_add_epi32(x1r_2,simde_mm_sub_epi32(x2i_2,x3r_2)));
  //  *(y1)  = cpack(dy1r,dy1i);
  dy1r = simde_mm256_sub_epi32(x1i_2,simde_mm256_add_epi32(x2r_2,x3i_2));
  dy1i = simde_mm256_sub_epi32(simde_mm256_sub_epi32(x3r_2,x2i_2),x1r_2);
  *(y1)  = simde_mm256_add_epi16(*(x0),cpack_256(dy1r,dy1i));
  //  dy2r = simde_mm_sub_epi32(x0r_2,simde_mm_sub_epi32(x1r_2,simde_mm_sub_epi32(x2r_2,x3r_2)));
  //  dy2i = simde_mm_sub_epi32(x0i_2,simde_mm_sub_epi32(x1i_2,simde_mm_sub_epi32(x2i_2,x3i_2)));
  //  *(y2)  = cpack(dy2r,dy2i);
  dy2r = simde_mm256_sub_epi32(simde_mm256_sub_epi32(x2r_2,x3r_2),x1r_2);
  dy2i = simde_mm256_sub_epi32(simde_mm256_sub_epi32(x2i_2,x3i_2),x1i_2);
  *(y2)  = simde_mm256_add_epi16(*(x0),cpack_256(dy2r,dy2i));
  //  dy3r = simde_mm_sub_epi32(x0r_2,simde_mm_add_epi32(x1i_2,simde_mm_sub_epi32(x2r_2,x3i_2)));
  //  dy3i = simde_mm_add_epi32(x0i_2,simde_mm_sub_epi32(x1r_2,simde_mm_add_epi32(x2i_2,x3r_2)));
  //  *(y3) = cpack(dy3r,dy3i);
  dy3r = simde_mm256_sub_epi32(simde_mm256_sub_epi32(x3i_2,x2r_2),x1i_2);
  dy3i = simde_mm256_sub_epi32(x1r_2,simde_mm256_add_epi32(x2i_2,x3r_2));
  *(y3) = simde_mm256_add_epi16(*(x0),cpack_256(dy3r,dy3i));
}

__attribute__((always_inline)) static inline void ibfly4_256(simde__m256i *x0,
                                                             simde__m256i *x1,
                                                             simde__m256i *x2,
                                                             simde__m256i *x3,
                                                             simde__m256i *y0,
                                                             simde__m256i *y1,
                                                             simde__m256i *y2,
                                                             simde__m256i *y3,
                                                             simde__m256i *tw1,
                                                             simde__m256i *tw2,
                                                             simde__m256i *tw3)
{
  simde__m256i x1r_2, x1i_2, x2r_2, x2i_2, x3r_2, x3i_2, dy0r, dy0i, dy1r, dy1i, dy2r, dy2i, dy3r, dy3i;

  cmultc_256(*(x1),*(tw1),&x1r_2,&x1i_2);
  cmultc_256(*(x2),*(tw2),&x2r_2,&x2i_2);
  cmultc_256(*(x3),*(tw3),&x3r_2,&x3i_2);

  dy0r = simde_mm256_add_epi32(x1r_2,simde_mm256_add_epi32(x2r_2,x3r_2));
  dy0i = simde_mm256_add_epi32(x1i_2,simde_mm256_add_epi32(x2i_2,x3i_2));
  *(y0)  = simde_mm256_add_epi16(*(x0),cpack_256(dy0r,dy0i));
  dy3r = simde_mm256_sub_epi32(x1i_2,simde_mm256_add_epi32(x2r_2,x3i_2));
  dy3i = simde_mm256_sub_epi32(simde_mm256_sub_epi32(x3r_2,x2i_2),x1r_2);
  *(y3)  = simde_mm256_add_epi16(*(x0),cpack_256(dy3r,dy3i));
  dy2r = simde_mm256_sub_epi32(simde_mm256_sub_epi32(x2r_2,x3r_2),x1r_2);
  dy2i = simde_mm256_sub_epi32(simde_mm256_sub_epi32(x2i_2,x3i_2),x1i_2);
  *(y2)  = simde_mm256_add_epi16(*(x0),cpack_256(dy2r,dy2i));
  dy1r = simde_mm256_sub_epi32(simde_mm256_sub_epi32(x3i_2,x2r_2),x1i_2);
  dy1i = simde_mm256_sub_epi32(x1r_2,simde_mm256_add_epi32(x2i_2,x3r_2));
  *(y1) = simde_mm256_add_epi16(*(x0),cpack_256(dy1r,dy1i));
}

__attribute__((always_inline)) static inline void bfly4_tw1(simde__m128i *x0,
                                                            simde__m128i *x1,
                                                            simde__m128i *x2,
                                                            simde__m128i *x3,
                                                            simde__m128i *y0,
                                                            simde__m128i *y1,
                                                            simde__m128i *y2,
                                                            simde__m128i *y3)
{
  register simde__m128i x1_flip, x3_flip, x02t, x13t;
  register simde__m128i complex_shuffle = simde_mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);

  x02t = simde_mm_adds_epi16(*(x0), *(x2));
  x13t = simde_mm_adds_epi16(*(x1), *(x3));
  *(y0) = simde_mm_adds_epi16(x02t, x13t);
  *(y2) = simde_mm_subs_epi16(x02t, x13t);
  x1_flip = simde_mm_sign_epi16(*(x1), *(simde__m128i *)conjugatedft);
  x1_flip = simde_mm_shuffle_epi8(x1_flip, complex_shuffle);
  x3_flip = simde_mm_sign_epi16(*(x3), *(simde__m128i *)conjugatedft);
  x3_flip = simde_mm_shuffle_epi8(x3_flip, complex_shuffle);
  x02t = simde_mm_subs_epi16(*(x0), *(x2));
  x13t = simde_mm_subs_epi16(x1_flip, x3_flip);
  *(y1) = simde_mm_adds_epi16(x02t, x13t); // x0 + x1f - x2 - x3f
  *(y3) = simde_mm_subs_epi16(x02t, x13t); // x0 - x1f - x2 + x3f

  /*
  *(y0) = simde_mm_adds_epi16(*(x0),simde_mm_adds_epi16(*(x1),simde_mm_adds_epi16(*(x2),*(x3))));
  x1_flip = simde_mm_sign_epi16(*(x1),*(simde__m128i*)conjugatedft);
  x1_flip = simde_mm_shuffle_epi8(x1_flip,simde_mm_set_epi8(13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2));
  x3_flip = simde_mm_sign_epi16(*(x3),*(simde__m128i*)conjugatedft);
  x3_flip = simde_mm_shuffle_epi8(x3_flip,simde_mm_set_epi8(13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2));
  *(y1)   = simde_mm_adds_epi16(*(x0),simde_mm_subs_epi16(x1_flip,simde_mm_adds_epi16(*(x2),x3_flip)));
  *(y2)   = simde_mm_subs_epi16(*(x0),simde_mm_subs_epi16(*(x1),simde_mm_subs_epi16(*(x2),*(x3))));
  *(y3)   = simde_mm_subs_epi16(*(x0),simde_mm_adds_epi16(x1_flip,simde_mm_subs_epi16(*(x2),x3_flip)));
  */
}

__attribute__((always_inline)) static inline void bfly4_tw1_256(simde__m256i *x0,
                                                                simde__m256i *x1,
                                                                simde__m256i *x2,
                                                                simde__m256i *x3,
                                                                simde__m256i *y0,
                                                                simde__m256i *y1,
                                                                simde__m256i *y2,
                                                                simde__m256i *y3)
{
  register simde__m256i x1_flip, x3_flip, x02t, x13t;
  register simde__m256i complex_shuffle = simde_mm256_set_epi8(29,
                                                               28,
                                                               31,
                                                               30,
                                                               25,
                                                               24,
                                                               27,
                                                               26,
                                                               21,
                                                               20,
                                                               23,
                                                               22,
                                                               17,
                                                               16,
                                                               19,
                                                               18,
                                                               13,
                                                               12,
                                                               15,
                                                               14,
                                                               9,
                                                               8,
                                                               11,
                                                               10,
                                                               5,
                                                               4,
                                                               7,
                                                               6,
                                                               1,
                                                               0,
                                                               3,
                                                               2);

  x02t    = simde_mm256_adds_epi16(*(x0),*(x2));
  x13t    = simde_mm256_adds_epi16(*(x1),*(x3));
  *(y0)   = simde_mm256_adds_epi16(x02t,x13t);
  *(y2)   = simde_mm256_subs_epi16(x02t,x13t);
  x1_flip = simde_mm256_sign_epi16(*(x1), *(simde__m256i *)conjugatedft);
  x1_flip = simde_mm256_shuffle_epi8(x1_flip,complex_shuffle);
  x3_flip = simde_mm256_sign_epi16(*(x3), *(simde__m256i *)conjugatedft);
  x3_flip = simde_mm256_shuffle_epi8(x3_flip,complex_shuffle);
  x02t    = simde_mm256_subs_epi16(*(x0),*(x2));
  x13t    = simde_mm256_subs_epi16(x1_flip,x3_flip);
  *(y1)   = simde_mm256_adds_epi16(x02t,x13t);  // x0 + x1f - x2 - x3f
  *(y3)   = simde_mm256_subs_epi16(x02t,x13t);  // x0 - x1f - x2 + x3f
}

__attribute__((always_inline)) static inline void bfly4_16_256(simde__m256i *x0,
                                                               simde__m256i *x1,
                                                               simde__m256i *x2,
                                                               simde__m256i *x3,
                                                               simde__m256i *y0,
                                                               simde__m256i *y1,
                                                               simde__m256i *y2,
                                                               simde__m256i *y3,
                                                               simde__m256i *tw1,
                                                               simde__m256i *tw2,
                                                               simde__m256i *tw3,
                                                               simde__m256i *tw1b,
                                                               simde__m256i *tw2b,
                                                               simde__m256i *tw3b)
{
  register simde__m256i x1t, x2t, x3t, x02t, x13t;
  register simde__m256i x1_flip, x3_flip;
  register simde__m256i complex_shuffle = simde_mm256_set_epi8(29,
                                                               28,
                                                               31,
                                                               30,
                                                               25,
                                                               24,
                                                               27,
                                                               26,
                                                               21,
                                                               20,
                                                               23,
                                                               22,
                                                               17,
                                                               16,
                                                               19,
                                                               18,
                                                               13,
                                                               12,
                                                               15,
                                                               14,
                                                               9,
                                                               8,
                                                               11,
                                                               10,
                                                               5,
                                                               4,
                                                               7,
                                                               6,
                                                               1,
                                                               0,
                                                               3,
                                                               2);

  // each input xi is assumed to be to consecutive vectors xi0 xi1 on which to perform the 8 butterflies
  // [xi00 xi01 xi02 xi03 xi10 xi20 xi30 xi40]
  // each output yi is the same

  x1t = packed_cmult2_256(*(x1),*(tw1),*(tw1b));
  x2t = packed_cmult2_256(*(x2),*(tw2),*(tw2b));
  x3t = packed_cmult2_256(*(x3),*(tw3),*(tw3b));

  x02t  = simde_mm256_adds_epi16(*(x0),x2t);
  x13t  = simde_mm256_adds_epi16(x1t,x3t);
  *(y0)   = simde_mm256_adds_epi16(x02t,x13t);
  *(y2)   = simde_mm256_subs_epi16(x02t,x13t);

  x1_flip = simde_mm256_sign_epi16(x1t, *(simde__m256i *)conjugatedft);
  x1_flip = simde_mm256_shuffle_epi8(x1_flip,complex_shuffle);
  x3_flip = simde_mm256_sign_epi16(x3t, *(simde__m256i *)conjugatedft);
  x3_flip = simde_mm256_shuffle_epi8(x3_flip,complex_shuffle);
  x02t  = simde_mm256_subs_epi16(*(x0),x2t);
  x13t  = simde_mm256_subs_epi16(x1_flip,x3_flip);
  *(y1)   = simde_mm256_adds_epi16(x02t,x13t);  // x0 + x1f - x2 - x3f
  *(y3) = simde_mm256_subs_epi16(x02t, x13t); // x0 - x1f - x2 + x3f
}

__attribute__((always_inline)) static inline void ibfly4_16_256(simde__m256i *x0,
                                                                simde__m256i *x1,
                                                                simde__m256i *x2,
                                                                simde__m256i *x3,
                                                                simde__m256i *y0,
                                                                simde__m256i *y1,
                                                                simde__m256i *y2,
                                                                simde__m256i *y3,
                                                                simde__m256i *tw1,
                                                                simde__m256i *tw2,
                                                                simde__m256i *tw3,
                                                                simde__m256i *tw1b,
                                                                simde__m256i *tw2b,
                                                                simde__m256i *tw3b)
{
  register simde__m256i x1t, x2t, x3t, x02t, x13t;
  register simde__m256i x1_flip, x3_flip;
  register simde__m256i complex_shuffle = simde_mm256_set_epi8(29,
                                                               28,
                                                               31,
                                                               30,
                                                               25,
                                                               24,
                                                               27,
                                                               26,
                                                               21,
                                                               20,
                                                               23,
                                                               22,
                                                               17,
                                                               16,
                                                               19,
                                                               18,
                                                               13,
                                                               12,
                                                               15,
                                                               14,
                                                               9,
                                                               8,
                                                               11,
                                                               10,
                                                               5,
                                                               4,
                                                               7,
                                                               6,
                                                               1,
                                                               0,
                                                               3,
                                                               2);

  // each input xi is assumed to be to consecutive vectors xi0 xi1 on which to perform the 8 butterflies
  // [xi00 xi01 xi02 xi03 xi10 xi20 xi30 xi40]
  // each output yi is the same

  x1t = packed_cmult2_256(*(x1),*(tw1),*(tw1b));
  x2t = packed_cmult2_256(*(x2),*(tw2),*(tw2b));
  x3t = packed_cmult2_256(*(x3),*(tw3),*(tw3b));

  x02t  = simde_mm256_adds_epi16(*(x0),x2t);
  x13t  = simde_mm256_adds_epi16(x1t,x3t);
  *(y0)   = simde_mm256_adds_epi16(x02t,x13t);
  *(y2)   = simde_mm256_subs_epi16(x02t,x13t);

  x1_flip = simde_mm256_sign_epi16(x1t, *(simde__m256i *)conjugatedft);
  x1_flip = simde_mm256_shuffle_epi8(x1_flip,complex_shuffle);
  x3_flip = simde_mm256_sign_epi16(x3t, *(simde__m256i *)conjugatedft);
  x3_flip = simde_mm256_shuffle_epi8(x3_flip,complex_shuffle);
  x02t  = simde_mm256_subs_epi16(*(x0),x2t);
  x13t  = simde_mm256_subs_epi16(x1_flip,x3_flip);
  *(y3)   = simde_mm256_adds_epi16(x02t,x13t);  // x0 + x1f - x2 - x3f
  *(y1) = simde_mm256_subs_epi16(x02t, x13t); // x0 - x1f - x2 + x3f
}

__attribute__((always_inline)) static inline void bfly5(simde__m128i *x0,
                                                        simde__m128i *x1,
                                                        simde__m128i *x2,
                                                        simde__m128i *x3,
                                                        simde__m128i *x4,
                                                        simde__m128i *y0,
                                                        simde__m128i *y1,
                                                        simde__m128i *y2,
                                                        simde__m128i *y3,
                                                        simde__m128i *y4,
                                                        simde__m128i *tw1,
                                                        simde__m128i *tw2,
                                                        simde__m128i *tw3,
                                                        simde__m128i *tw4)
{
  simde__m128i x1_2, x2_2, x3_2, x4_2, tmpre, tmpim;

  packed_cmult(*(x1),*(tw1),&x1_2);
  packed_cmult(*(x2),*(tw2),&x2_2);
  packed_cmult(*(x3),*(tw3),&x3_2);
  packed_cmult(*(x4),*(tw4),&x4_2);

  *(y0) = simde_mm_adds_epi16(*(x0), simde_mm_adds_epi16(x1_2, simde_mm_adds_epi16(x2_2, simde_mm_adds_epi16(x3_2, x4_2))));
  cmult(x1_2,*(W15),&tmpre,&tmpim);
  cmac(x2_2,*(W25),&tmpre,&tmpim);
  cmac(x3_2,*(W35),&tmpre,&tmpim);
  cmac(x4_2,*(W45),&tmpre,&tmpim);
  *(y1) = cpack(tmpre,tmpim);
  *(y1) = simde_mm_adds_epi16(*(x0), *(y1));

  cmult(x1_2,*(W25),&tmpre,&tmpim);
  cmac(x2_2,*(W45),&tmpre,&tmpim);
  cmac(x3_2,*(W15),&tmpre,&tmpim);
  cmac(x4_2,*(W35),&tmpre,&tmpim);
  *(y2) = cpack(tmpre,tmpim);
  *(y2) = simde_mm_adds_epi16(*(x0), *(y2));

  cmult(x1_2,*(W35),&tmpre,&tmpim);
  cmac(x2_2,*(W15),&tmpre,&tmpim);
  cmac(x3_2,*(W45),&tmpre,&tmpim);
  cmac(x4_2,*(W25),&tmpre,&tmpim);
  *(y3) = cpack(tmpre,tmpim);
  *(y3) = simde_mm_adds_epi16(*(x0), *(y3));

  cmult(x1_2,*(W45),&tmpre,&tmpim);
  cmac(x2_2,*(W35),&tmpre,&tmpim);
  cmac(x3_2,*(W25),&tmpre,&tmpim);
  cmac(x4_2,*(W15),&tmpre,&tmpim);
  *(y4) = cpack(tmpre,tmpim);
  *(y4) = simde_mm_adds_epi16(*(x0), *(y4));
}

__attribute__((always_inline)) static inline void bfly5_tw1(simde__m128i *x0,
                                                            simde__m128i *x1,
                                                            simde__m128i *x2,
                                                            simde__m128i *x3,
                                                            simde__m128i *x4,
                                                            simde__m128i *y0,
                                                            simde__m128i *y1,
                                                            simde__m128i *y2,
                                                            simde__m128i *y3,
                                                            simde__m128i *y4)
{
  simde__m128i tmpre, tmpim;

  *(y0) = simde_mm_adds_epi16(*(x0), simde_mm_adds_epi16(*(x1), simde_mm_adds_epi16(*(x2), simde_mm_adds_epi16(*(x3), *(x4)))));
  cmult(*(x1),*(W15),&tmpre,&tmpim);
  cmac(*(x2),*(W25),&tmpre,&tmpim);
  cmac(*(x3),*(W35),&tmpre,&tmpim);
  cmac(*(x4),*(W45),&tmpre,&tmpim);
  *(y1) = cpack(tmpre,tmpim);
  *(y1) = simde_mm_adds_epi16(*(x0), *(y1));
  cmult(*(x1),*(W25),&tmpre,&tmpim);
  cmac(*(x2),*(W45),&tmpre,&tmpim);
  cmac(*(x3),*(W15),&tmpre,&tmpim);
  cmac(*(x4),*(W35),&tmpre,&tmpim);
  *(y2) = cpack(tmpre,tmpim);
  *(y2) = simde_mm_adds_epi16(*(x0), *(y2));
  cmult(*(x1),*(W35),&tmpre,&tmpim);
  cmac(*(x2),*(W15),&tmpre,&tmpim);
  cmac(*(x3),*(W45),&tmpre,&tmpim);
  cmac(*(x4),*(W25),&tmpre,&tmpim);
  *(y3) = cpack(tmpre,tmpim);
  *(y3) = simde_mm_adds_epi16(*(x0), *(y3));
  cmult(*(x1),*(W45),&tmpre,&tmpim);
  cmac(*(x2),*(W35),&tmpre,&tmpim);
  cmac(*(x3),*(W25),&tmpre,&tmpim);
  cmac(*(x4),*(W15),&tmpre,&tmpim);
  *(y4) = cpack(tmpre,tmpim);
  *(y4) = simde_mm_adds_epi16(*(x0), *(y4));
}

// performs 4x4 transpose of input x (complex interleaved) using 128bit SIMD intrinsics
// i.e. x = [x0r x0i x1r x1i ... x15r x15i], y = [x0r x0i x4r x4i x8r x8i x12r x12i x1r x1i x5r x5i x9r x9i x13r x13i x2r x2i ... x15r x15i]
__attribute__((always_inline)) static inline void transpose16_ooff_simd256(simde__m256i *x, simde__m256i *y, int off)
{
  register simde__m256i ytmp0, ytmp1, ytmp2, ytmp3, ytmp4, ytmp5, ytmp6, ytmp7;
  simde__m256i *y2 = y;
  simde__m256i const perm_mask = simde_mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0);

  ytmp0 = simde_mm256_permutevar8x32_epi32(x[0],perm_mask);  // x00 x10 x01 x11 x02 x12 x03 x13
  ytmp1 = simde_mm256_permutevar8x32_epi32(x[1],perm_mask);  // x20 x30 x21 x31 x22 x32 x23 x33
  ytmp2 = simde_mm256_permutevar8x32_epi32(x[2],perm_mask);  // x40 x50 x41 x51 x42 x52 x43 x53
  ytmp3 = simde_mm256_permutevar8x32_epi32(x[3],perm_mask);  // x60 x70 x61 x71 x62 x72 x63 x73
  ytmp4 = simde_mm256_unpacklo_epi64(ytmp0,ytmp1);           // x00 x10 x20 x30 x01 x11 x21 x31
  ytmp5 = simde_mm256_unpackhi_epi64(ytmp0,ytmp1);           // x02 x12 x22 x32 x03 x13 x23 x33
  ytmp6 = simde_mm256_unpacklo_epi64(ytmp2,ytmp3);           // x40 x50 x60 x70 x41 x51 x61 x71
  ytmp7 = simde_mm256_unpackhi_epi64(ytmp2,ytmp3);           // x42 x52 x62 x72 x43 x53 x63 x73

  *y2    = simde_mm256_insertf128_si256(ytmp4,simde_mm256_extracti128_si256(ytmp6,0),1);  //x00 x10 x20 x30 x40 x50 x60 x70
  y2+=off;  
  *y2    = simde_mm256_insertf128_si256(ytmp6,simde_mm256_extracti128_si256(ytmp4,1),0);  //x01 x11 x21 x31 x41 x51 x61 x71
  y2+=off;  
  *y2    = simde_mm256_insertf128_si256(ytmp5,simde_mm256_extracti128_si256(ytmp7,0),1);  //x00 x10 x20 x30 x40 x50 x60 x70
  y2+=off;  
  *y2    = simde_mm256_insertf128_si256(ytmp7,simde_mm256_extracti128_si256(ytmp5,1),0);  //x01 x11 x21 x31 x41 x51 x61 x71
}

__attribute__((always_inline)) static inline void transpose4_ooff_simd256(simde__m256i *x, simde__m256i *y, int off)
{
  simde__m256i const perm_mask = simde_mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
  simde__m256i perm_tmp0, perm_tmp1;

  // x[0] = [x0 x1 x2 x3 x4 x5 x6 x7]
  // x[1] = [x8 x9 x10 x11 x12 x13 x14]
  // y[0] = [x0 x2 x4 x6 x8 x10 x12 x14]
  // y[off] = [x1 x3 x5 x7 x9 x11 x13 x15]
  perm_tmp0 = simde_mm256_permutevar8x32_epi32(x[0],perm_mask);
  perm_tmp1 = simde_mm256_permutevar8x32_epi32(x[1],perm_mask);
  y[0]   = simde_mm256_insertf128_si256(perm_tmp0,simde_mm256_extracti128_si256(perm_tmp1,0),1);
  y[off] = simde_mm256_insertf128_si256(perm_tmp1,simde_mm256_extracti128_si256(perm_tmp0,1),0);
}

// 16-point optimized DFT kernel

const static int16_t tw16[24] __attribute__((aligned(32))) = { 32767,0,30272,-12540,23169 ,-23170,12539 ,-30273,
                                                  32767,0,23169,-23170,0     ,-32767,-23170,-23170,
                                                  32767,0,12539,-30273,-23170,-23170,-30273,12539
                                                };

const static int16_t tw16c[24] __attribute__((aligned(32))) = { 0,32767,12540,30272,23170,23169 ,30273 ,12539,
                                                   0,32767,23170,23169,32767,0     ,23170 ,-23170,
                                                   0,32767,30273,12539,23170,-23170,-12539,-30273
                                                 };

const static int16_t tw16rep[48] __attribute__((aligned(32))) = { 32767,0,30272,-12540,23169 ,-23170,12539 ,-30273,32767,0,30272,-12540,23169 ,-23170,12539 ,-30273,
						     32767,0,23169,-23170,0     ,-32767,-23170,-23170,32767,0,23169,-23170,0     ,-32767,-23170,-23170,
						     32767,0,12539,-30273,-23170,-23170,-30273,12539,32767,0,12539,-30273,-23170,-23170,-30273,12539
                                                   };

const static int16_t tw16arep[48] __attribute__((aligned(32))) = {32767,0,30272,12540,23169 ,23170,12539 ,30273,32767,0,30272,12540,23169 ,23170,12539 ,30273,
						     32767,0,23169,23170,0     ,32767,-23170,23170,32767,0,23169,23170,0     ,32767,-23170,23170,
						     32767,0,12539,30273,-23170,23170,-30273,-12539,32767,0,12539,30273,-23170,23170,-30273,-12539
                                                    }; 

const static int16_t tw16brep[48] __attribute__((aligned(32))) = { 0,32767,-12540,30272,-23170,23169 ,-30273,12539,0,32767,-12540,30272,-23170,23169 ,-30273,12539,
                                                      0,32767,-23170,23169,-32767,0     ,-23170,-23170,0,32767,-23170,23169,-32767,0     ,-23170,-23170,
                                                      0,32767,-30273,12539,-23170,-23170,12539 ,-30273,0,32767,-30273,12539,-23170,-23170,12539 ,-30273
                                                    };

const static int16_t tw16crep[48] __attribute__((aligned(32))) = { 0,32767,12540,30272,23170,23169 ,30273 ,12539,0,32767,12540,30272,23170,23169 ,30273 ,12539,
						      0,32767,23170,23169,32767,0     ,23170 ,-23170,0,32767,23170,23169,32767,0     ,23170 ,-23170,
						      0,32767,30273,12539,23170,-23170,-12539,-30273,0,32767,30273,12539,23170,-23170,-12539,-30273
                                                    };
#if 0
const static int16_t tw16a[24] __attribute__((aligned(32))) = {32767,0,30272,12540,23169 ,23170,12539 ,30273,
                                                  32767,0,23169,23170,0     ,32767,-23170,23170,
                                                  32767,0,12539,30273,-23170,23170,-30273,-12539
                                                 };

const static int16_t tw16b[24] __attribute__((aligned(32))) = { 0,32767,-12540,30272,-23170,23169 ,-30273,12539,
                                                   0,32767,-23170,23169,-32767,0     ,-23170,-23170,
                                                   0,32767,-30273,12539,-23170,-23170,12539 ,-30273
                                                 };

static inline void dft16(int16_t *x,int16_t *y) __attribute__((always_inline)
{
  simde__m128i *tw16a_128 = (simde__m128i *)tw16a, *tw16b_128 = (simde__m128i *)tw16b, *x128 = (simde__m128i *)x,
               *y128 = (simde__m128i *)y;

  /*  This is the original version before unrolling

  bfly4_tw1(x128,x128+1,x128+2,x128+3,
      y128,y128+1,y128+2,y128+3);

  transpose16(y128,ytmp);

  bfly4_16(ytmp,ytmp+1,ytmp+2,ytmp+3,
     y128,y128+1,y128+2,y128+3,
     tw16_128,tw16_128+1,tw16_128+2);
  */

  register simde__m128i x1_flip, x3_flip, x02t, x13t;
  register simde__m128i ytmp0, ytmp1, ytmp2, ytmp3, xtmp0, xtmp1, xtmp2, xtmp3;
  register simde__m128i complex_shuffle = simde_mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);

  // First stage : 4 Radix-4 butterflies without input twiddles

  x02t = simde_mm_adds_epi16(x128[0], x128[2]);
  x13t = simde_mm_adds_epi16(x128[1], x128[3]);
  xtmp0 = simde_mm_adds_epi16(x02t, x13t);
  xtmp2 = simde_mm_subs_epi16(x02t, x13t);
  x1_flip = simde_mm_sign_epi16(x128[1], *(simde__m128i *)conjugatedft);
  x1_flip = simde_mm_shuffle_epi8(x1_flip, complex_shuffle);
  x3_flip = simde_mm_sign_epi16(x128[3], *(simde__m128i *)conjugatedft);
  x3_flip = simde_mm_shuffle_epi8(x3_flip, complex_shuffle);
  x02t = simde_mm_subs_epi16(x128[0], x128[2]);
  x13t = simde_mm_subs_epi16(x1_flip, x3_flip);
  xtmp1 = simde_mm_adds_epi16(x02t, x13t); // x0 + x1f - x2 - x3f
  xtmp3 = simde_mm_subs_epi16(x02t, x13t); // x0 - x1f - x2 + x3f

  ytmp0 = simde_mm_unpacklo_epi32(xtmp0, xtmp1);
  ytmp1 = simde_mm_unpackhi_epi32(xtmp0, xtmp1);
  ytmp2 = simde_mm_unpacklo_epi32(xtmp2, xtmp3);
  ytmp3 = simde_mm_unpackhi_epi32(xtmp2, xtmp3);
  xtmp0 = simde_mm_unpacklo_epi64(ytmp0, ytmp2);
  xtmp1 = simde_mm_unpackhi_epi64(ytmp0, ytmp2);
  xtmp2 = simde_mm_unpacklo_epi64(ytmp1, ytmp3);
  xtmp3 = simde_mm_unpackhi_epi64(ytmp1, ytmp3);

  // Second stage : 4 Radix-4 butterflies with input twiddles
  xtmp1 = packed_cmult2(xtmp1,tw16a_128[0],tw16b_128[0]);
  xtmp2 = packed_cmult2(xtmp2,tw16a_128[1],tw16b_128[1]);
  xtmp3 = packed_cmult2(xtmp3,tw16a_128[2],tw16b_128[2]);

  x02t = simde_mm_adds_epi16(xtmp0, xtmp2);
  x13t = simde_mm_adds_epi16(xtmp1, xtmp3);
  y128[0] = simde_mm_adds_epi16(x02t, x13t);
  y128[2] = simde_mm_subs_epi16(x02t, x13t);
  x1_flip = simde_mm_sign_epi16(xtmp1, *(simde__m128i *)conjugatedft);
  x1_flip = simde_mm_shuffle_epi8(x1_flip, simde_mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
  x3_flip = simde_mm_sign_epi16(xtmp3, *(simde__m128i *)conjugatedft);
  x3_flip = simde_mm_shuffle_epi8(x3_flip, simde_mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
  x02t = simde_mm_subs_epi16(xtmp0, xtmp2);
  x13t = simde_mm_subs_epi16(x1_flip, x3_flip);
  y128[1] = simde_mm_adds_epi16(x02t, x13t); // x0 + x1f - x2 - x3f
  y128[3] = simde_mm_subs_epi16(x02t, x13t); // x0 - x1f - x2 + x3f
}
#endif

// Does two 16-point DFTS (x[0 .. 15] is 128 LSBs of input vector, x[16..31] is in 128 MSBs)
__attribute__((always_inline)) static inline void dft16_simd256(int16_t *x, int16_t *y)
{
  simde__m256i *tw16a_256 = (simde__m256i *)tw16arep, *tw16b_256 = (simde__m256i *)tw16brep, *x256 = (simde__m256i *)x,
               *y256 = (simde__m256i *)y;

  simde__m256i x1_flip, x3_flip, x02t, x13t;
  simde__m256i ytmp0, ytmp1, ytmp2, ytmp3, xtmp0, xtmp1, xtmp2, xtmp3;
  const simde__m256i complex_shuffle = simde_mm256_set_epi8(29,
                                                            28,
                                                            31,
                                                            30,
                                                            25,
                                                            24,
                                                            27,
                                                            26,
                                                            21,
                                                            20,
                                                            23,
                                                            22,
                                                            17,
                                                            16,
                                                            19,
                                                            18,
                                                            13,
                                                            12,
                                                            15,
                                                            14,
                                                            9,
                                                            8,
                                                            11,
                                                            10,
                                                            5,
                                                            4,
                                                            7,
                                                            6,
                                                            1,
                                                            0,
                                                            3,
                                                            2);

  // First stage : 4 Radix-4 butterflies without input twiddles

  x02t    = simde_mm256_adds_epi16(x256[0],x256[2]);
  x13t    = simde_mm256_adds_epi16(x256[1],x256[3]);
  xtmp0   = simde_mm256_adds_epi16(x02t,x13t);
  xtmp2   = simde_mm256_subs_epi16(x02t,x13t);
  x1_flip = simde_mm256_sign_epi16(x256[1], *(simde__m256i *)conjugatedft);
  x1_flip = simde_mm256_shuffle_epi8(x1_flip,complex_shuffle);
  x3_flip = simde_mm256_sign_epi16(x256[3], *(simde__m256i *)conjugatedft);
  x3_flip = simde_mm256_shuffle_epi8(x3_flip,complex_shuffle);
  x02t    = simde_mm256_subs_epi16(x256[0],x256[2]);
  x13t    = simde_mm256_subs_epi16(x1_flip,x3_flip);
  xtmp1   = simde_mm256_adds_epi16(x02t,x13t);  // x0 + x1f - x2 - x3f
  xtmp3   = simde_mm256_subs_epi16(x02t,x13t);  // x0 - x1f - x2 + x3f

  /*  print_shorts256("xtmp0",(int16_t*)&xtmp0);
      print_shorts256("xtmp1",(int16_t*)&xtmp1);
  print_shorts256("xtmp2",(int16_t*)&xtmp2);
  print_shorts256("xtmp3",(int16_t*)&xtmp3);*/

  ytmp0   = simde_mm256_unpacklo_epi32(xtmp0,xtmp1);  
  ytmp1   = simde_mm256_unpackhi_epi32(xtmp0,xtmp1);
  ytmp2   = simde_mm256_unpacklo_epi32(xtmp2,xtmp3);
  ytmp3   = simde_mm256_unpackhi_epi32(xtmp2,xtmp3);
  xtmp0   = simde_mm256_unpacklo_epi64(ytmp0,ytmp2);
  xtmp1   = simde_mm256_unpackhi_epi64(ytmp0,ytmp2);
  xtmp2   = simde_mm256_unpacklo_epi64(ytmp1,ytmp3);
  xtmp3   = simde_mm256_unpackhi_epi64(ytmp1,ytmp3);

  // Second stage : 4 Radix-4 butterflies with input twiddles
  xtmp1 = packed_cmult2_256(xtmp1,tw16a_256[0],tw16b_256[0]);
  xtmp2 = packed_cmult2_256(xtmp2,tw16a_256[1],tw16b_256[1]);
  xtmp3 = packed_cmult2_256(xtmp3,tw16a_256[2],tw16b_256[2]);

  /*  print_shorts256("xtmp0",(int16_t*)&xtmp0);
  print_shorts256("xtmp1",(int16_t*)&xtmp1);
  print_shorts256("xtmp2",(int16_t*)&xtmp2);
  print_shorts256("xtmp3",(int16_t*)&xtmp3);*/

  x02t    = simde_mm256_adds_epi16(xtmp0,xtmp2);
  x13t    = simde_mm256_adds_epi16(xtmp1,xtmp3);
  ytmp0 = simde_mm256_srai_epi16(simde_mm256_adds_epi16(x02t, x13t), 2);
  ytmp2 = simde_mm256_srai_epi16(simde_mm256_subs_epi16(x02t, x13t), 2);
  x1_flip = simde_mm256_sign_epi16(xtmp1, *(simde__m256i *)conjugatedft);
  x1_flip = simde_mm256_shuffle_epi8(x1_flip,complex_shuffle);
  x3_flip = simde_mm256_sign_epi16(xtmp3, *(simde__m256i *)conjugatedft);
  x3_flip = simde_mm256_shuffle_epi8(x3_flip,complex_shuffle);
  x02t    = simde_mm256_subs_epi16(xtmp0,xtmp2);
  x13t    = simde_mm256_subs_epi16(x1_flip,x3_flip);
  ytmp1 = simde_mm256_srai_epi16(simde_mm256_adds_epi16(x02t, x13t), 2); // x0 + x1f - x2 - x3f
  ytmp3 = simde_mm256_srai_epi16(simde_mm256_subs_epi16(x02t, x13t), 2); // x0 - x1f - x2 + x3f

  // [y0  y1  y2  y3  y16 y17 y18 y19]
  // [y4  y5  y6  y7  y20 y21 y22 y23]
  // [y8  y9  y10 y11 y24 y25 y26 y27]
  // [y12 y13 y14 y15 y28 y29 y30 y31]

  y256[0] = simde_mm256_insertf128_si256(ytmp0,simde_mm256_extracti128_si256(ytmp1,0),1);
  y256[1] = simde_mm256_insertf128_si256(ytmp2,simde_mm256_extracti128_si256(ytmp3,0),1);
  y256[2] = simde_mm256_insertf128_si256(ytmp1,simde_mm256_extracti128_si256(ytmp0,1),0);
  y256[3] = simde_mm256_insertf128_si256(ytmp3,simde_mm256_extracti128_si256(ytmp2,1),0);

  // [y0  y1  y2  y3  y4  y5  y6  y7]
  // [y8  y9  y10 y11 y12 y13 y14 y15]
  // [y16 y17 y18 y19 y20 y21 y22 y23]
  // [y24 y25 y26 y27 y28 y29 y30 y31]
}

__attribute__((always_inline)) static inline void idft16(int16_t *x, int16_t *y)
{
  simde__m128i *tw16a_128 = (simde__m128i *)tw16, *tw16b_128 = (simde__m128i *)tw16c, *x128 = (simde__m128i *)x,
               *y128 = (simde__m128i *)y;

  /*
  bfly4_tw1(x128,x128+1,x128+2,x128+3,
      y128,y128+1,y128+2,y128+3);

  transpose16(y128,ytmp);

  bfly4_16(ytmp,ytmp+1,ytmp+2,ytmp+3,
     y128,y128+1,y128+2,y128+3,
     tw16_128,tw16_128+1,tw16_128+2);
  */

  register simde__m128i x1_flip, x3_flip, x02t, x13t;
  register simde__m128i ytmp0, ytmp1, ytmp2, ytmp3, xtmp0, xtmp1, xtmp2, xtmp3;

  // First stage : 4 Radix-4 butterflies without input twiddles

  x02t = simde_mm_adds_epi16(x128[0], x128[2]);
  x13t = simde_mm_adds_epi16(x128[1], x128[3]);
  xtmp0 = simde_mm_adds_epi16(x02t, x13t);
  xtmp2 = simde_mm_subs_epi16(x02t, x13t);
  x1_flip = simde_mm_sign_epi16(x128[1], *(simde__m128i *)conjugatedft);
  x1_flip = simde_mm_shuffle_epi8(x1_flip, simde_mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
  x3_flip = simde_mm_sign_epi16(x128[3], *(simde__m128i *)conjugatedft);
  x3_flip = simde_mm_shuffle_epi8(x3_flip, simde_mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
  x02t = simde_mm_subs_epi16(x128[0], x128[2]);
  x13t = simde_mm_subs_epi16(x1_flip, x3_flip);
  xtmp3 = simde_mm_adds_epi16(x02t, x13t); // x0 + x1f - x2 - x3f
  xtmp1 = simde_mm_subs_epi16(x02t, x13t); // x0 - x1f - x2 + x3f

  ytmp0 = simde_mm_unpacklo_epi32(xtmp0, xtmp1);
  ytmp1 = simde_mm_unpackhi_epi32(xtmp0, xtmp1);
  ytmp2 = simde_mm_unpacklo_epi32(xtmp2, xtmp3);
  ytmp3 = simde_mm_unpackhi_epi32(xtmp2, xtmp3);
  xtmp0 = simde_mm_unpacklo_epi64(ytmp0, ytmp2);
  xtmp1 = simde_mm_unpackhi_epi64(ytmp0, ytmp2);
  xtmp2 = simde_mm_unpacklo_epi64(ytmp1, ytmp3);
  xtmp3 = simde_mm_unpackhi_epi64(ytmp1, ytmp3);

  // Second stage : 4 Radix-4 butterflies with input twiddles
  xtmp1 = packed_cmult2(xtmp1,tw16a_128[0],tw16b_128[0]);
  xtmp2 = packed_cmult2(xtmp2,tw16a_128[1],tw16b_128[1]);
  xtmp3 = packed_cmult2(xtmp3,tw16a_128[2],tw16b_128[2]);

  x02t = simde_mm_adds_epi16(xtmp0, xtmp2);
  x13t = simde_mm_adds_epi16(xtmp1, xtmp3);
  y128[0] = simde_mm_adds_epi16(x02t, x13t);
  y128[2] = simde_mm_subs_epi16(x02t, x13t);
  x1_flip = simde_mm_sign_epi16(xtmp1, *(simde__m128i *)conjugatedft);
  x1_flip = simde_mm_shuffle_epi8(x1_flip, simde_mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
  x3_flip = simde_mm_sign_epi16(xtmp3, *(simde__m128i *)conjugatedft);
  x3_flip = simde_mm_shuffle_epi8(x3_flip, simde_mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
  x02t = simde_mm_subs_epi16(xtmp0, xtmp2);
  x13t = simde_mm_subs_epi16(x1_flip, x3_flip);
  y128[3] = simde_mm_adds_epi16(x02t, x13t); // x0 + x1f - x2 - x3f
  y128[1] = simde_mm_subs_epi16(x02t, x13t); // x0 - x1f - x2 + x3f
}

void idft16f(int16_t *x,int16_t *y) {
  idft16(x,y);
}

// Does two 16-point IDFTS (x[0 .. 15] is 128 LSBs of input vector, x[16..31] is in 128 MSBs)
__attribute__((always_inline)) static inline void idft16_simd256(int16_t *x, int16_t *y)
{
  simde__m256i *tw16a_256 = (simde__m256i *)tw16rep, *tw16b_256 = (simde__m256i *)tw16crep, *x256 = (simde__m256i *)x,
               *y256 = (simde__m256i *)y;
  register simde__m256i x1_flip, x3_flip, x02t, x13t;
  register simde__m256i ytmp0, ytmp1, ytmp2, ytmp3, xtmp0, xtmp1, xtmp2, xtmp3;
  const simde__m256i complex_shuffle = simde_mm256_set_epi8(29,
                                                            28,
                                                            31,
                                                            30,
                                                            25,
                                                            24,
                                                            27,
                                                            26,
                                                            21,
                                                            20,
                                                            23,
                                                            22,
                                                            17,
                                                            16,
                                                            19,
                                                            18,
                                                            13,
                                                            12,
                                                            15,
                                                            14,
                                                            9,
                                                            8,
                                                            11,
                                                            10,
                                                            5,
                                                            4,
                                                            7,
                                                            6,
                                                            1,
                                                            0,
                                                            3,
                                                            2);

  // First stage : 4 Radix-4 butterflies without input twiddles

  x02t    = simde_mm256_adds_epi16(x256[0],x256[2]);
  x13t    = simde_mm256_adds_epi16(x256[1],x256[3]);
  xtmp0   = simde_mm256_adds_epi16(x02t,x13t);
  xtmp2   = simde_mm256_subs_epi16(x02t,x13t);
  x1_flip = simde_mm256_sign_epi16(x256[1], *(simde__m256i *)conjugatedft);
  x1_flip = simde_mm256_shuffle_epi8(x1_flip,complex_shuffle);
  x3_flip = simde_mm256_sign_epi16(x256[3], *(simde__m256i *)conjugatedft);
  x3_flip = simde_mm256_shuffle_epi8(x3_flip,complex_shuffle);
  x02t    = simde_mm256_subs_epi16(x256[0],x256[2]);
  x13t    = simde_mm256_subs_epi16(x1_flip,x3_flip);
  xtmp3   = simde_mm256_adds_epi16(x02t,x13t);  // x0 + x1f - x2 - x3f
  xtmp1   = simde_mm256_subs_epi16(x02t,x13t);  // x0 - x1f - x2 + x3f

  ytmp0   = simde_mm256_unpacklo_epi32(xtmp0,xtmp1);  
  ytmp1   = simde_mm256_unpackhi_epi32(xtmp0,xtmp1);
  ytmp2   = simde_mm256_unpacklo_epi32(xtmp2,xtmp3);
  ytmp3   = simde_mm256_unpackhi_epi32(xtmp2,xtmp3);
  xtmp0   = simde_mm256_unpacklo_epi64(ytmp0,ytmp2);
  xtmp1   = simde_mm256_unpackhi_epi64(ytmp0,ytmp2);
  xtmp2   = simde_mm256_unpacklo_epi64(ytmp1,ytmp3);
  xtmp3   = simde_mm256_unpackhi_epi64(ytmp1,ytmp3);

  // Second stage : 4 Radix-4 butterflies with input twiddles
  xtmp1 = packed_cmult2_256(xtmp1,tw16a_256[0],tw16b_256[0]);
  xtmp2 = packed_cmult2_256(xtmp2,tw16a_256[1],tw16b_256[1]);
  xtmp3 = packed_cmult2_256(xtmp3,tw16a_256[2],tw16b_256[2]);

  x02t    = simde_mm256_adds_epi16(xtmp0,xtmp2);
  x13t    = simde_mm256_adds_epi16(xtmp1,xtmp3);
  ytmp0   = simde_mm256_adds_epi16(x02t,x13t);
  ytmp2   = simde_mm256_subs_epi16(x02t,x13t);
  x1_flip = simde_mm256_sign_epi16(xtmp1, *(simde__m256i *)conjugatedft);
  x1_flip = simde_mm256_shuffle_epi8(x1_flip,complex_shuffle);
  x3_flip = simde_mm256_sign_epi16(xtmp3, *(simde__m256i *)conjugatedft);
  x3_flip = simde_mm256_shuffle_epi8(x3_flip,complex_shuffle);
  x02t    = simde_mm256_subs_epi16(xtmp0,xtmp2);
  x13t    = simde_mm256_subs_epi16(x1_flip,x3_flip);
  ytmp3   = simde_mm256_adds_epi16(x02t,x13t);  // x0 + x1f - x2 - x3f
  ytmp1   = simde_mm256_subs_epi16(x02t,x13t);  // x0 - x1f - x2 + x3f

  // [y0  y1  y2  y3  y16 y17 y18 y19]
  // [y4  y5  y6  y7  y20 y21 y22 y23]
  // [y8  y9  y10 y11 y24 y25 y26 y27]
  // [y12 y13 y14 y15 y28 y29 y30 y31]

  y256[0] = simde_mm256_insertf128_si256(ytmp0,simde_mm256_extracti128_si256(ytmp1,0),1);
  y256[1] = simde_mm256_insertf128_si256(ytmp2,simde_mm256_extracti128_si256(ytmp3,0),1);
  y256[2] = simde_mm256_insertf128_si256(ytmp1,simde_mm256_extracti128_si256(ytmp0,1),0);
  y256[3] = simde_mm256_insertf128_si256(ytmp3,simde_mm256_extracti128_si256(ytmp2,1),0);

}

#define simd256_q15_t simde__m256i
#define shiftright_int16_simd256(a,shift) simde_mm256_srai_epi16(a,shift)
#define set1_int16_simd256(a) simde_mm256_set1_epi16(a);
#define mulhi_int16_simd256(a,b) simde_mm256_mulhrs_epi16(a,b);


int16_t tw32768[2*16384] __attribute__((aligned(32)));

void dft32768(int16_t *x,int16_t *y,unsigned char scale)
{

  simd256_q15_t xtmp[4096],*xtmpp,*x256 = (simd256_q15_t *)x;
  simd256_q15_t ytmp[4096],*tw32768_256p=(simd256_q15_t *)tw32768,*y256=(simd256_q15_t *)y,*y256p=(simd256_q15_t *)y;

  simd256_q15_t *ytmpp = &ytmp[0];
  int i;
  simd256_q15_t ONE_OVER_SQRT2_Q15_128 = set1_int16_simd256(ONE_OVER_SQRT2_Q15);
  
  xtmpp = xtmp;

  for (i=0; i<256; i++) {
    transpose4_ooff_simd256(x256  ,xtmpp,2048);
    transpose4_ooff_simd256(x256+2,xtmpp+1,2048);
    transpose4_ooff_simd256(x256+4,xtmpp+2,2048);
    transpose4_ooff_simd256(x256+6,xtmpp+3,2048);
    transpose4_ooff_simd256(x256+8,xtmpp+4,2048);
    transpose4_ooff_simd256(x256+10,xtmpp+5,2048);
    transpose4_ooff_simd256(x256+12,xtmpp+6,2048);
    transpose4_ooff_simd256(x256+14,xtmpp+7,2048);
    transpose4_ooff_simd256(x256+16,xtmpp+8,2048);
    transpose4_ooff_simd256(x256+18,xtmpp+9,2048);
    transpose4_ooff_simd256(x256+20,xtmpp+10,2048);
    transpose4_ooff_simd256(x256+22,xtmpp+11,2048);
    transpose4_ooff_simd256(x256+24,xtmpp+12,2048);
    transpose4_ooff_simd256(x256+26,xtmpp+13,2048);
    transpose4_ooff_simd256(x256+28,xtmpp+14,2048);
    transpose4_ooff_simd256(x256+30,xtmpp+15,2048);
    transpose4_ooff_simd256(x256+32,xtmpp+16,2048);
    transpose4_ooff_simd256(x256+34,xtmpp+17,2048);
    transpose4_ooff_simd256(x256+36,xtmpp+18,2048);
    transpose4_ooff_simd256(x256+38,xtmpp+19,2048);
    transpose4_ooff_simd256(x256+40,xtmpp+20,2048);
    transpose4_ooff_simd256(x256+42,xtmpp+21,2048);
    transpose4_ooff_simd256(x256+44,xtmpp+22,2048);
    transpose4_ooff_simd256(x256+46,xtmpp+23,2048);
    transpose4_ooff_simd256(x256+48,xtmpp+24,2048);
    transpose4_ooff_simd256(x256+50,xtmpp+25,2048);
    transpose4_ooff_simd256(x256+52,xtmpp+26,2048);
    transpose4_ooff_simd256(x256+54,xtmpp+27,2048);
    transpose4_ooff_simd256(x256+56,xtmpp+28,2048);
    transpose4_ooff_simd256(x256+58,xtmpp+29,2048);
    transpose4_ooff_simd256(x256+60,xtmpp+30,2048);
    transpose4_ooff_simd256(x256+62,xtmpp+31,2048);
    x256+=64;
    xtmpp+=32;
  }

  dft16384((int16_t*)(xtmp),(int16_t*)ytmp,1);
  dft16384((int16_t*)(xtmp+2048),(int16_t*)(ytmp+2048),1);


  for (i=0; i<2048; i++) {
    bfly2_256(ytmpp,ytmpp+2048,
	      y256p,y256p+2048,
	      tw32768_256p);
    tw32768_256p++;
    y256p++;
    ytmpp++;
  }

  if (scale>0) {
    y256p = y256;

    for (i=0; i<64; i++) {
      y256p[0]  = mulhi_int16_simd256(y256p[0],ONE_OVER_SQRT2_Q15_128);
      y256p[1]  = mulhi_int16_simd256(y256p[1],ONE_OVER_SQRT2_Q15_128);
      y256p[2]  = mulhi_int16_simd256(y256p[2],ONE_OVER_SQRT2_Q15_128);
      y256p[3]  = mulhi_int16_simd256(y256p[3],ONE_OVER_SQRT2_Q15_128);
      y256p[4]  = mulhi_int16_simd256(y256p[4],ONE_OVER_SQRT2_Q15_128);
      y256p[5]  = mulhi_int16_simd256(y256p[5],ONE_OVER_SQRT2_Q15_128);
      y256p[6]  = mulhi_int16_simd256(y256p[6],ONE_OVER_SQRT2_Q15_128);
      y256p[7]  = mulhi_int16_simd256(y256p[7],ONE_OVER_SQRT2_Q15_128);
      y256p[8]  = mulhi_int16_simd256(y256p[8],ONE_OVER_SQRT2_Q15_128);
      y256p[9]  = mulhi_int16_simd256(y256p[9],ONE_OVER_SQRT2_Q15_128);
      y256p[10] = mulhi_int16_simd256(y256p[10],ONE_OVER_SQRT2_Q15_128);
      y256p[11] = mulhi_int16_simd256(y256p[11],ONE_OVER_SQRT2_Q15_128);
      y256p[12] = mulhi_int16_simd256(y256p[12],ONE_OVER_SQRT2_Q15_128);
      y256p[13] = mulhi_int16_simd256(y256p[13],ONE_OVER_SQRT2_Q15_128);
      y256p[14] = mulhi_int16_simd256(y256p[14],ONE_OVER_SQRT2_Q15_128);
      y256p[15] = mulhi_int16_simd256(y256p[15],ONE_OVER_SQRT2_Q15_128);
      y256p+=16;
    }
  }

}

void idft32768(int16_t *x,int16_t *y,unsigned char scale)
{

  simd256_q15_t xtmp[4096],*xtmpp,*x256 = (simd256_q15_t *)x;
  simd256_q15_t ytmp[4096],*tw32768_256p=(simd256_q15_t *)tw32768,*y256=(simd256_q15_t *)y,*y256p=(simd256_q15_t *)y;
  simd256_q15_t *ytmpp = &ytmp[0];
  int i;
  simd256_q15_t ONE_OVER_SQRT2_Q15_128 = set1_int16_simd256(ONE_OVER_SQRT2_Q15);
  
  xtmpp = xtmp;

  for (i=0; i<64; i++) {
    transpose4_ooff_simd256(x256  ,xtmpp,2048);
    transpose4_ooff_simd256(x256+2,xtmpp+1,2048);
    transpose4_ooff_simd256(x256+4,xtmpp+2,2048);
    transpose4_ooff_simd256(x256+6,xtmpp+3,2048);
    transpose4_ooff_simd256(x256+8,xtmpp+4,2048);
    transpose4_ooff_simd256(x256+10,xtmpp+5,2048);
    transpose4_ooff_simd256(x256+12,xtmpp+6,2048);
    transpose4_ooff_simd256(x256+14,xtmpp+7,2048);
    transpose4_ooff_simd256(x256+16,xtmpp+8,2048);
    transpose4_ooff_simd256(x256+18,xtmpp+9,2048);
    transpose4_ooff_simd256(x256+20,xtmpp+10,2048);
    transpose4_ooff_simd256(x256+22,xtmpp+11,2048);
    transpose4_ooff_simd256(x256+24,xtmpp+12,2048);
    transpose4_ooff_simd256(x256+26,xtmpp+13,2048);
    transpose4_ooff_simd256(x256+28,xtmpp+14,2048);
    transpose4_ooff_simd256(x256+30,xtmpp+15,2048);
    transpose4_ooff_simd256(x256+32,xtmpp+16,2048);
    transpose4_ooff_simd256(x256+34,xtmpp+17,2048);
    transpose4_ooff_simd256(x256+36,xtmpp+18,2048);
    transpose4_ooff_simd256(x256+38,xtmpp+19,2048);
    transpose4_ooff_simd256(x256+40,xtmpp+20,2048);
    transpose4_ooff_simd256(x256+42,xtmpp+21,2048);
    transpose4_ooff_simd256(x256+44,xtmpp+22,2048);
    transpose4_ooff_simd256(x256+46,xtmpp+23,2048);
    transpose4_ooff_simd256(x256+48,xtmpp+24,2048);
    transpose4_ooff_simd256(x256+50,xtmpp+25,2048);
    transpose4_ooff_simd256(x256+52,xtmpp+26,2048);
    transpose4_ooff_simd256(x256+54,xtmpp+27,2048);
    transpose4_ooff_simd256(x256+56,xtmpp+28,2048);
    transpose4_ooff_simd256(x256+58,xtmpp+29,2048);
    transpose4_ooff_simd256(x256+60,xtmpp+30,2048);
    transpose4_ooff_simd256(x256+62,xtmpp+31,2048);
    x256+=64;
    xtmpp+=32;
  }

  idft16384((int16_t*)(xtmp),(int16_t*)ytmp,1);
  idft16384((int16_t*)(xtmp+2048),(int16_t*)(ytmp+2048),1);


  for (i=0; i<2048; i++) {
    ibfly2_256(ytmpp,ytmpp+2048,
	       y256p,y256p+2048,
	       tw32768_256p);
    tw32768_256p++;
    y256p++;
    ytmpp++;
  }

  if (scale>0) {
    y256p = y256;

    for (i=0; i<256; i++) {
      y256p[0]  = mulhi_int16_simd256(y256p[0],ONE_OVER_SQRT2_Q15_128);
      y256p[1]  = mulhi_int16_simd256(y256p[1],ONE_OVER_SQRT2_Q15_128);
      y256p[2]  = mulhi_int16_simd256(y256p[2],ONE_OVER_SQRT2_Q15_128);
      y256p[3]  = mulhi_int16_simd256(y256p[3],ONE_OVER_SQRT2_Q15_128);
      y256p[4]  = mulhi_int16_simd256(y256p[4],ONE_OVER_SQRT2_Q15_128);
      y256p[5]  = mulhi_int16_simd256(y256p[5],ONE_OVER_SQRT2_Q15_128);
      y256p[6]  = mulhi_int16_simd256(y256p[6],ONE_OVER_SQRT2_Q15_128);
      y256p[7]  = mulhi_int16_simd256(y256p[7],ONE_OVER_SQRT2_Q15_128);
      y256p[8]  = mulhi_int16_simd256(y256p[8],ONE_OVER_SQRT2_Q15_128);
      y256p[9]  = mulhi_int16_simd256(y256p[9],ONE_OVER_SQRT2_Q15_128);
      y256p[10] = mulhi_int16_simd256(y256p[10],ONE_OVER_SQRT2_Q15_128);
      y256p[11] = mulhi_int16_simd256(y256p[11],ONE_OVER_SQRT2_Q15_128);
      y256p[12] = mulhi_int16_simd256(y256p[12],ONE_OVER_SQRT2_Q15_128);
      y256p[13] = mulhi_int16_simd256(y256p[13],ONE_OVER_SQRT2_Q15_128);
      y256p[14] = mulhi_int16_simd256(y256p[14],ONE_OVER_SQRT2_Q15_128);
      y256p[15] = mulhi_int16_simd256(y256p[15],ONE_OVER_SQRT2_Q15_128);
      y256p+=16;
    }
  }

}

int16_t twa18432[12288] __attribute__((aligned(32)));
int16_t twb18432[12288] __attribute__((aligned(32)));
// 6144 x 3
void dft18432(int16_t *input, int16_t *output,unsigned char scale) {

  int i,i2,j;
  uint32_t tmp[3][6144] __attribute__((aligned(32)));
  uint32_t tmpo[3][6144] __attribute__((aligned(32)));
  simd_q15_t *y128p=(simd_q15_t*)output;
  simd_q15_t ONE_OVER_SQRT3_Q15_128 = set1_int16(ONE_OVER_SQRT3_Q15);

  for (i=0,j=0; i<6144; i++) {
    tmp[0][i] = ((uint32_t *)input)[j++];
    tmp[1][i] = ((uint32_t *)input)[j++];
    tmp[2][i] = ((uint32_t *)input)[j++];
  }

  dft6144((int16_t*)(tmp[0]),(int16_t*)(tmpo[0]),scale);
  dft6144((int16_t*)(tmp[1]),(int16_t*)(tmpo[1]),scale);
  dft6144((int16_t*)(tmp[2]),(int16_t*)(tmpo[2]),scale);

  for (i=0,i2=0; i<12288; i+=8,i2+=4)  {
    bfly3((simd_q15_t*)(&tmpo[0][i2]),(simd_q15_t*)(&tmpo[1][i2]),(simd_q15_t*)(&tmpo[2][i2]),
          (simd_q15_t*)(output+i),(simd_q15_t*)(output+12288+i),(simd_q15_t*)(output+24576+i),
          (simd_q15_t*)(twa18432+i),(simd_q15_t*)(twb18432+i));
  }
  if (scale==1) {
    for (i=0; i<288; i++) {
      y128p[0]  = mulhi_int16(y128p[0],ONE_OVER_SQRT3_Q15_128);
      y128p[1]  = mulhi_int16(y128p[1],ONE_OVER_SQRT3_Q15_128);
      y128p[2]  = mulhi_int16(y128p[2],ONE_OVER_SQRT3_Q15_128);
      y128p[3]  = mulhi_int16(y128p[3],ONE_OVER_SQRT3_Q15_128);
      y128p[4]  = mulhi_int16(y128p[4],ONE_OVER_SQRT3_Q15_128);
      y128p[5]  = mulhi_int16(y128p[5],ONE_OVER_SQRT3_Q15_128);
      y128p[6]  = mulhi_int16(y128p[6],ONE_OVER_SQRT3_Q15_128);
      y128p[7]  = mulhi_int16(y128p[7],ONE_OVER_SQRT3_Q15_128);
      y128p[8]  = mulhi_int16(y128p[8],ONE_OVER_SQRT3_Q15_128);
      y128p[9]  = mulhi_int16(y128p[9],ONE_OVER_SQRT3_Q15_128);
      y128p[10] = mulhi_int16(y128p[10],ONE_OVER_SQRT3_Q15_128);
      y128p[11] = mulhi_int16(y128p[11],ONE_OVER_SQRT3_Q15_128);
      y128p[12] = mulhi_int16(y128p[12],ONE_OVER_SQRT3_Q15_128);
      y128p[13] = mulhi_int16(y128p[13],ONE_OVER_SQRT3_Q15_128);
      y128p[14] = mulhi_int16(y128p[14],ONE_OVER_SQRT3_Q15_128);
      y128p[15] = mulhi_int16(y128p[15],ONE_OVER_SQRT3_Q15_128);
      y128p+=16;
    }
  }
}

void idft18432(int16_t *input, int16_t *output,unsigned char scale) {

  int i,i2,j;
  uint32_t tmp[3][6144] __attribute__((aligned(32)));
  uint32_t tmpo[3][6144] __attribute__((aligned(32)));
  simd_q15_t *y128p=(simd_q15_t*)output;
  simd_q15_t ONE_OVER_SQRT3_Q15_128 = set1_int16(ONE_OVER_SQRT3_Q15);

  for (i=0,j=0; i<6144; i++) {
    tmp[0][i] = ((uint32_t *)input)[j++];
    tmp[1][i] = ((uint32_t *)input)[j++];
    tmp[2][i] = ((uint32_t *)input)[j++];
  }

  idft6144((int16_t*)(tmp[0]),(int16_t*)(tmpo[0]),scale);
  idft6144((int16_t*)(tmp[1]),(int16_t*)(tmpo[1]),scale);
  idft6144((int16_t*)(tmp[2]),(int16_t*)(tmpo[2]),scale);

  for (i=0,i2=0; i<12288; i+=8,i2+=4)  {
    ibfly3((simd_q15_t*)(&tmpo[0][i2]),(simd_q15_t*)(&tmpo[1][i2]),(simd_q15_t*)(&tmpo[2][i2]),
	   (simd_q15_t*)(output+i),(simd_q15_t*)(output+12288+i),(simd_q15_t*)(output+24576+i),
	   (simd_q15_t*)(twa18432+i),(simd_q15_t*)(twb18432+i));
  }
  if (scale==1) {
    for (i=0; i<288; i++) {
      y128p[0]  = mulhi_int16(y128p[0],ONE_OVER_SQRT3_Q15_128);
      y128p[1]  = mulhi_int16(y128p[1],ONE_OVER_SQRT3_Q15_128);
      y128p[2]  = mulhi_int16(y128p[2],ONE_OVER_SQRT3_Q15_128);
      y128p[3]  = mulhi_int16(y128p[3],ONE_OVER_SQRT3_Q15_128);
      y128p[4]  = mulhi_int16(y128p[4],ONE_OVER_SQRT3_Q15_128);
      y128p[5]  = mulhi_int16(y128p[5],ONE_OVER_SQRT3_Q15_128);
      y128p[6]  = mulhi_int16(y128p[6],ONE_OVER_SQRT3_Q15_128);
      y128p[7]  = mulhi_int16(y128p[7],ONE_OVER_SQRT3_Q15_128);
      y128p[8]  = mulhi_int16(y128p[8],ONE_OVER_SQRT3_Q15_128);
      y128p[9]  = mulhi_int16(y128p[9],ONE_OVER_SQRT3_Q15_128);
      y128p[10] = mulhi_int16(y128p[10],ONE_OVER_SQRT3_Q15_128);
      y128p[11] = mulhi_int16(y128p[11],ONE_OVER_SQRT3_Q15_128);
      y128p[12] = mulhi_int16(y128p[12],ONE_OVER_SQRT3_Q15_128);
      y128p[13] = mulhi_int16(y128p[13],ONE_OVER_SQRT3_Q15_128);
      y128p[14] = mulhi_int16(y128p[14],ONE_OVER_SQRT3_Q15_128);
      y128p[15] = mulhi_int16(y128p[15],ONE_OVER_SQRT3_Q15_128);
      y128p+=16;
    }
  }
}


int16_t twa24576[16384] __attribute__((aligned(32)));
int16_t twb24576[16384] __attribute__((aligned(32)));
// 8192 x 3
void dft24576(int16_t *input, int16_t *output,unsigned char scale)
{
  int i,i2,j;
  uint32_t tmp[3][8192] __attribute__((aligned(32)));
  uint32_t tmpo[3][8192] __attribute__((aligned(32)));
  simd_q15_t *y128p=(simd_q15_t*)output;
  simd_q15_t ONE_OVER_SQRT3_Q15_128 = set1_int16(ONE_OVER_SQRT3_Q15);

  for (i=0,j=0; i<8192; i++) {
    tmp[0][i] = ((uint32_t *)input)[j++];
    tmp[1][i] = ((uint32_t *)input)[j++];
    tmp[2][i] = ((uint32_t *)input)[j++];
  }

  dft8192((int16_t*)(tmp[0]),(int16_t*)(tmpo[0]),1);
  dft8192((int16_t*)(tmp[1]),(int16_t*)(tmpo[1]),1);
  dft8192((int16_t*)(tmp[2]),(int16_t*)(tmpo[2]),1);
  /*
  for (i=1; i<8192; i++) {
    tmpo[0][i] = tmpo[0][i<<1];
    tmpo[1][i] = tmpo[1][i<<1];
    tmpo[2][i] = tmpo[2][i<<1];
    }*/
#ifndef MR_MAIN
  if (LOG_DUMPFLAG(DEBUG_DFT)) {
    LOG_M("dft24576out0.m","o0",tmpo[0],8192,1,1);
    LOG_M("dft24576out1.m","o1",tmpo[1],8192,1,1);
    LOG_M("dft24576out2.m","o2",tmpo[2],8192,1,1);
  }
#endif
  for (i=0,i2=0; i<16384; i+=8,i2+=4)  {
    bfly3((simd_q15_t*)(&tmpo[0][i2]),(simd_q15_t*)(&tmpo[1][i2]),(simd_q15_t*)(&tmpo[2][i2]),
          (simd_q15_t*)(output+i),(simd_q15_t*)(output+16384+i),(simd_q15_t*)(output+32768+i),
          (simd_q15_t*)(twa24576+i),(simd_q15_t*)(twb24576+i));
  }


  if (scale==1) {
    for (i=0; i<384; i++) {
      y128p[0]  = mulhi_int16(y128p[0],ONE_OVER_SQRT3_Q15_128);
      y128p[1]  = mulhi_int16(y128p[1],ONE_OVER_SQRT3_Q15_128);
      y128p[2]  = mulhi_int16(y128p[2],ONE_OVER_SQRT3_Q15_128);
      y128p[3]  = mulhi_int16(y128p[3],ONE_OVER_SQRT3_Q15_128);
      y128p[4]  = mulhi_int16(y128p[4],ONE_OVER_SQRT3_Q15_128);
      y128p[5]  = mulhi_int16(y128p[5],ONE_OVER_SQRT3_Q15_128);
      y128p[6]  = mulhi_int16(y128p[6],ONE_OVER_SQRT3_Q15_128);
      y128p[7]  = mulhi_int16(y128p[7],ONE_OVER_SQRT3_Q15_128);
      y128p[8]  = mulhi_int16(y128p[8],ONE_OVER_SQRT3_Q15_128);
      y128p[9]  = mulhi_int16(y128p[9],ONE_OVER_SQRT3_Q15_128);
      y128p[10] = mulhi_int16(y128p[10],ONE_OVER_SQRT3_Q15_128);
      y128p[11] = mulhi_int16(y128p[11],ONE_OVER_SQRT3_Q15_128);
      y128p[12] = mulhi_int16(y128p[12],ONE_OVER_SQRT3_Q15_128);
      y128p[13] = mulhi_int16(y128p[13],ONE_OVER_SQRT3_Q15_128);
      y128p[14] = mulhi_int16(y128p[14],ONE_OVER_SQRT3_Q15_128);
      y128p[15] = mulhi_int16(y128p[15],ONE_OVER_SQRT3_Q15_128);
      y128p+=16;
    }
  }
#ifndef MR_MAIN
  if (LOG_DUMPFLAG(DEBUG_DFT)) {
     LOG_M("out.m","out",output,24576,1,1);
  }
#endif
}

void idft24576(int16_t *input, int16_t *output,unsigned char scale)
{
  int i,i2,j;
  uint32_t tmp[3][8192] __attribute__((aligned(32)));
  uint32_t tmpo[3][8192] __attribute__((aligned(32)));
  simd_q15_t *y128p=(simd_q15_t*)output;
  simd_q15_t ONE_OVER_SQRT3_Q15_128 = set1_int16(ONE_OVER_SQRT3_Q15);

  for (i=0,j=0; i<8192; i++) {
    tmp[0][i] = ((uint32_t *)input)[j++];
    tmp[1][i] = ((uint32_t *)input)[j++];
    tmp[2][i] = ((uint32_t *)input)[j++];
  }

  idft8192((int16_t*)(tmp[0]),(int16_t*)(tmpo[0]),1);
  idft8192((int16_t*)(tmp[1]),(int16_t*)(tmpo[1]),1);
  idft8192((int16_t*)(tmp[2]),(int16_t*)(tmpo[2]),1);
 #ifndef MR_MAIN 
  if (LOG_DUMPFLAG(DEBUG_DFT)) {
    LOG_M("idft24576in.m","in",input,24576,1,1);
    LOG_M("idft24576out0.m","o0",tmpo[0],8192,1,1);
    LOG_M("idft24576out1.m","o1",tmpo[1],8192,1,1);
    LOG_M("idft24576out2.m","o2",tmpo[2],8192,1,1);
  }
#endif
  for (i=0,i2=0; i<16384; i+=8,i2+=4)  {
    ibfly3((simd_q15_t*)(&tmpo[0][i2]),(simd_q15_t*)(&tmpo[1][i2]),((simd_q15_t*)&tmpo[2][i2]),
          (simd_q15_t*)(output+i),(simd_q15_t*)(output+16384+i),(simd_q15_t*)(output+32768+i),
          (simd_q15_t*)(twa24576+i),(simd_q15_t*)(twb24576+i));
  }
  if (scale==1) {
    for (i=0; i<384; i++) {
      y128p[0]  = mulhi_int16(y128p[0],ONE_OVER_SQRT3_Q15_128);
      y128p[1]  = mulhi_int16(y128p[1],ONE_OVER_SQRT3_Q15_128);
      y128p[2]  = mulhi_int16(y128p[2],ONE_OVER_SQRT3_Q15_128);
      y128p[3]  = mulhi_int16(y128p[3],ONE_OVER_SQRT3_Q15_128);
      y128p[4]  = mulhi_int16(y128p[4],ONE_OVER_SQRT3_Q15_128);
      y128p[5]  = mulhi_int16(y128p[5],ONE_OVER_SQRT3_Q15_128);
      y128p[6]  = mulhi_int16(y128p[6],ONE_OVER_SQRT3_Q15_128);
      y128p[7]  = mulhi_int16(y128p[7],ONE_OVER_SQRT3_Q15_128);
      y128p[8]  = mulhi_int16(y128p[8],ONE_OVER_SQRT3_Q15_128);
      y128p[9]  = mulhi_int16(y128p[9],ONE_OVER_SQRT3_Q15_128);
      y128p[10] = mulhi_int16(y128p[10],ONE_OVER_SQRT3_Q15_128);
      y128p[11] = mulhi_int16(y128p[11],ONE_OVER_SQRT3_Q15_128);
      y128p[12] = mulhi_int16(y128p[12],ONE_OVER_SQRT3_Q15_128);
      y128p[13] = mulhi_int16(y128p[13],ONE_OVER_SQRT3_Q15_128);
      y128p[14] = mulhi_int16(y128p[14],ONE_OVER_SQRT3_Q15_128);
      y128p[15] = mulhi_int16(y128p[15],ONE_OVER_SQRT3_Q15_128);
      y128p+=16;
    }
  }
#ifndef MR_MAIN
  if (LOG_DUMPFLAG(DEBUG_DFT)) {
    LOG_M("idft24576out.m","out",output,24576,1,1);
  }
#endif
}

int16_t twa36864[24576] __attribute__((aligned(32)));
int16_t twb36864[24576] __attribute__((aligned(32)));

// 12288 x 3
void dft36864(int16_t *input, int16_t *output,uint8_t scale) {

  int i,i2,j;
  uint32_t tmp[3][12288] __attribute__((aligned(32)));
  uint32_t tmpo[3][12288] __attribute__((aligned(32)));
  simd_q15_t *y128p=(simd_q15_t*)output;
  simd_q15_t ONE_OVER_SQRT3_Q15_128 = set1_int16(ONE_OVER_SQRT3_Q15);

  for (i=0,j=0; i<12288; i++) {
    tmp[0][i] = ((uint32_t *)input)[j++];
    tmp[1][i] = ((uint32_t *)input)[j++];
    tmp[2][i] = ((uint32_t *)input)[j++];
  }

  dft12288((int16_t*)(tmp[0]),(int16_t*)(tmpo[0]),1);
  dft12288((int16_t*)(tmp[1]),(int16_t*)(tmpo[1]),1);
  dft12288((int16_t*)(tmp[2]),(int16_t*)(tmpo[2]),1);
#ifndef MR_MAIN
  if (LOG_DUMPFLAG(DEBUG_DFT)) {
    LOG_M("dft36864out0.m","o0",tmpo[0],12288,1,1);
    LOG_M("dft36864out1.m","o1",tmpo[1],12288,1,1);
    LOG_M("dft36864out2.m","o2",tmpo[2],12288,1,1);
  }
#endif
  for (i=0,i2=0; i<24576; i+=8,i2+=4)  {
    bfly3((simd_q15_t*)(&tmpo[0][i2]),(simd_q15_t*)(&tmpo[1][i2]),(simd_q15_t*)(&tmpo[2][i2]),
          (simd_q15_t*)(output+i),(simd_q15_t*)(output+24576+i),(simd_q15_t*)(output+49152+i),
          (simd_q15_t*)(twa36864+i),(simd_q15_t*)(twb36864+i));
  }

  if (scale==1) {
    for (i=0; i<576; i++) {
      y128p[0]  = mulhi_int16(y128p[0],ONE_OVER_SQRT3_Q15_128);
      y128p[1]  = mulhi_int16(y128p[1],ONE_OVER_SQRT3_Q15_128);
      y128p[2]  = mulhi_int16(y128p[2],ONE_OVER_SQRT3_Q15_128);
      y128p[3]  = mulhi_int16(y128p[3],ONE_OVER_SQRT3_Q15_128);
      y128p[4]  = mulhi_int16(y128p[4],ONE_OVER_SQRT3_Q15_128);
      y128p[5]  = mulhi_int16(y128p[5],ONE_OVER_SQRT3_Q15_128);
      y128p[6]  = mulhi_int16(y128p[6],ONE_OVER_SQRT3_Q15_128);
      y128p[7]  = mulhi_int16(y128p[7],ONE_OVER_SQRT3_Q15_128);
      y128p[8]  = mulhi_int16(y128p[8],ONE_OVER_SQRT3_Q15_128);
      y128p[9]  = mulhi_int16(y128p[9],ONE_OVER_SQRT3_Q15_128);
      y128p[10] = mulhi_int16(y128p[10],ONE_OVER_SQRT3_Q15_128);
      y128p[11] = mulhi_int16(y128p[11],ONE_OVER_SQRT3_Q15_128);
      y128p[12] = mulhi_int16(y128p[12],ONE_OVER_SQRT3_Q15_128);
      y128p[13] = mulhi_int16(y128p[13],ONE_OVER_SQRT3_Q15_128);
      y128p[14] = mulhi_int16(y128p[14],ONE_OVER_SQRT3_Q15_128);
      y128p[15] = mulhi_int16(y128p[15],ONE_OVER_SQRT3_Q15_128);
      y128p+=16;
    }
  }
#ifndef MR_MAIN
  if (LOG_DUMPFLAG(DEBUG_DFT)) {
     LOG_M("out.m","out",output,36864,1,1);
  }
#endif
}

void idft36864(int16_t *input, int16_t *output,uint8_t scale) {

  int i,i2,j;
  uint32_t tmp[3][12288] __attribute__((aligned(32)));
  uint32_t tmpo[3][12288] __attribute__((aligned(32)));
  simd_q15_t *y128p=(simd_q15_t*)output;
  simd_q15_t ONE_OVER_SQRT3_Q15_128 = set1_int16(ONE_OVER_SQRT3_Q15);

  for (i=0,j=0; i<12288; i++) {
    tmp[0][i] = ((uint32_t *)input)[j++];
    tmp[1][i] = ((uint32_t *)input)[j++];
    tmp[2][i] = ((uint32_t *)input)[j++];
  }

  idft12288((int16_t*)(tmp[0]),(int16_t*)(tmpo[0]),1);
  idft12288((int16_t*)(tmp[1]),(int16_t*)(tmpo[1]),1);
  idft12288((int16_t*)(tmp[2]),(int16_t*)(tmpo[2]),1);

  for (i=0,i2=0; i<24576; i+=8,i2+=4)  {
    ibfly3((simd_q15_t*)(&tmpo[0][i2]),(simd_q15_t*)(&tmpo[1][i2]),((simd_q15_t*)&tmpo[2][i2]),
          (simd_q15_t*)(output+i),(simd_q15_t*)(output+24576+i),(simd_q15_t*)(output+49152+i),
          (simd_q15_t*)(twa36864+i),(simd_q15_t*)(twb36864+i));
  }
  if (scale==1) {
    for (i=0; i<576; i++) {
      y128p[0]  = mulhi_int16(y128p[0],ONE_OVER_SQRT3_Q15_128);
      y128p[1]  = mulhi_int16(y128p[1],ONE_OVER_SQRT3_Q15_128);
      y128p[2]  = mulhi_int16(y128p[2],ONE_OVER_SQRT3_Q15_128);
      y128p[3]  = mulhi_int16(y128p[3],ONE_OVER_SQRT3_Q15_128);
      y128p[4]  = mulhi_int16(y128p[4],ONE_OVER_SQRT3_Q15_128);
      y128p[5]  = mulhi_int16(y128p[5],ONE_OVER_SQRT3_Q15_128);
      y128p[6]  = mulhi_int16(y128p[6],ONE_OVER_SQRT3_Q15_128);
      y128p[7]  = mulhi_int16(y128p[7],ONE_OVER_SQRT3_Q15_128);
      y128p[8]  = mulhi_int16(y128p[8],ONE_OVER_SQRT3_Q15_128);
      y128p[9]  = mulhi_int16(y128p[9],ONE_OVER_SQRT3_Q15_128);
      y128p[10] = mulhi_int16(y128p[10],ONE_OVER_SQRT3_Q15_128);
      y128p[11] = mulhi_int16(y128p[11],ONE_OVER_SQRT3_Q15_128);
      y128p[12] = mulhi_int16(y128p[12],ONE_OVER_SQRT3_Q15_128);
      y128p[13] = mulhi_int16(y128p[13],ONE_OVER_SQRT3_Q15_128);
      y128p[14] = mulhi_int16(y128p[14],ONE_OVER_SQRT3_Q15_128);
      y128p[15] = mulhi_int16(y128p[15],ONE_OVER_SQRT3_Q15_128);
      y128p+=16;
    }
  }
}

int16_t twa49152[32768] __attribute__((aligned(32)));
int16_t twb49152[32768] __attribute__((aligned(32)));

// 16384 x 3
void dft49152(int16_t *input, int16_t *output,uint8_t scale) {

  int i,i2,j;
  uint32_t tmp[3][16384] __attribute__((aligned(32)));
  uint32_t tmpo[3][16384] __attribute__((aligned(32)));
  simd_q15_t *y128p=(simd_q15_t*)output;
  simd_q15_t ONE_OVER_SQRT3_Q15_128 = set1_int16(ONE_OVER_SQRT3_Q15);

  for (i=0,j=0; i<16384; i++) {
    tmp[0][i] = ((uint32_t *)input)[j++];
    tmp[1][i] = ((uint32_t *)input)[j++];
    tmp[2][i] = ((uint32_t *)input)[j++];
  }

  dft16384((int16_t*)(tmp[0]),(int16_t*)(tmpo[0]),1);
  dft16384((int16_t*)(tmp[1]),(int16_t*)(tmpo[1]),1);
  dft16384((int16_t*)(tmp[2]),(int16_t*)(tmpo[2]),1);

  for (i=0,i2=0; i<32768; i+=8,i2+=4)  {
    bfly3((simd_q15_t*)(&tmpo[0][i2]),(simd_q15_t*)(&tmpo[1][i2]),((simd_q15_t*)&tmpo[2][i2]),
          (simd_q15_t*)(output+i),(simd_q15_t*)(output+32768+i),(simd_q15_t*)(output+65536+i),
          (simd_q15_t*)(twa49152+i),(simd_q15_t*)(twb49152+i));
  }
  if (scale==1) {
    for (i=0; i<768; i++) {
      y128p[0]  = mulhi_int16(y128p[0],ONE_OVER_SQRT3_Q15_128);
      y128p[1]  = mulhi_int16(y128p[1],ONE_OVER_SQRT3_Q15_128);
      y128p[2]  = mulhi_int16(y128p[2],ONE_OVER_SQRT3_Q15_128);
      y128p[3]  = mulhi_int16(y128p[3],ONE_OVER_SQRT3_Q15_128);
      y128p[4]  = mulhi_int16(y128p[4],ONE_OVER_SQRT3_Q15_128);
      y128p[5]  = mulhi_int16(y128p[5],ONE_OVER_SQRT3_Q15_128);
      y128p[6]  = mulhi_int16(y128p[6],ONE_OVER_SQRT3_Q15_128);
      y128p[7]  = mulhi_int16(y128p[7],ONE_OVER_SQRT3_Q15_128);
      y128p[8]  = mulhi_int16(y128p[8],ONE_OVER_SQRT3_Q15_128);
      y128p[9]  = mulhi_int16(y128p[9],ONE_OVER_SQRT3_Q15_128);
      y128p[10] = mulhi_int16(y128p[10],ONE_OVER_SQRT3_Q15_128);
      y128p[11] = mulhi_int16(y128p[11],ONE_OVER_SQRT3_Q15_128);
      y128p[12] = mulhi_int16(y128p[12],ONE_OVER_SQRT3_Q15_128);
      y128p[13] = mulhi_int16(y128p[13],ONE_OVER_SQRT3_Q15_128);
      y128p[14] = mulhi_int16(y128p[14],ONE_OVER_SQRT3_Q15_128);
      y128p[15] = mulhi_int16(y128p[15],ONE_OVER_SQRT3_Q15_128);
      y128p+=16;
    }
  }
}

void idft49152(int16_t *input, int16_t *output,uint8_t scale) {

   int i,i2,j;
  uint32_t tmp[3][16384] __attribute__((aligned(32)));
  uint32_t tmpo[3][16384] __attribute__((aligned(32)));
  simd_q15_t *y128p=(simd_q15_t*)output;
  simd_q15_t ONE_OVER_SQRT3_Q15_128 = set1_int16(ONE_OVER_SQRT3_Q15);

  for (i=0,j=0; i<16384; i++) {
    tmp[0][i] = ((uint32_t *)input)[j++];
    tmp[1][i] = ((uint32_t *)input)[j++];
    tmp[2][i] = ((uint32_t *)input)[j++];
  }

  idft16384((int16_t*)(tmp[0]),(int16_t*)(tmpo[0]),1);
  idft16384((int16_t*)(tmp[1]),(int16_t*)(tmpo[1]),1);
  idft16384((int16_t*)(tmp[2]),(int16_t*)(tmpo[2]),1);

  for (i=0,i2=0; i<32768; i+=8,i2+=4)  {
    ibfly3((simd_q15_t*)(&tmpo[0][i2]),(simd_q15_t*)(&tmpo[1][i2]),((simd_q15_t*)&tmpo[2][i2]),
	   (simd_q15_t*)(output+i),(simd_q15_t*)(output+32768+i),(simd_q15_t*)(output+65536+i),
	   (simd_q15_t*)(twa49152+i),(simd_q15_t*)(twb49152+i));
  }
  if (scale==1) {
    for (i=0; i<768; i++) {
      y128p[0]  = mulhi_int16(y128p[0],ONE_OVER_SQRT3_Q15_128);
      y128p[1]  = mulhi_int16(y128p[1],ONE_OVER_SQRT3_Q15_128);
      y128p[2]  = mulhi_int16(y128p[2],ONE_OVER_SQRT3_Q15_128);
      y128p[3]  = mulhi_int16(y128p[3],ONE_OVER_SQRT3_Q15_128);
      y128p[4]  = mulhi_int16(y128p[4],ONE_OVER_SQRT3_Q15_128);
      y128p[5]  = mulhi_int16(y128p[5],ONE_OVER_SQRT3_Q15_128);
      y128p[6]  = mulhi_int16(y128p[6],ONE_OVER_SQRT3_Q15_128);
      y128p[7]  = mulhi_int16(y128p[7],ONE_OVER_SQRT3_Q15_128);
      y128p[8]  = mulhi_int16(y128p[8],ONE_OVER_SQRT3_Q15_128);
      y128p[9]  = mulhi_int16(y128p[9],ONE_OVER_SQRT3_Q15_128);
      y128p[10] = mulhi_int16(y128p[10],ONE_OVER_SQRT3_Q15_128);
      y128p[11] = mulhi_int16(y128p[11],ONE_OVER_SQRT3_Q15_128);
      y128p[12] = mulhi_int16(y128p[12],ONE_OVER_SQRT3_Q15_128);
      y128p[13] = mulhi_int16(y128p[13],ONE_OVER_SQRT3_Q15_128);
      y128p[14] = mulhi_int16(y128p[14],ONE_OVER_SQRT3_Q15_128);
      y128p[15] = mulhi_int16(y128p[15],ONE_OVER_SQRT3_Q15_128);
      y128p+=16;
    }
  }
}

int16_t tw65536[3*2*16384] __attribute__((aligned(32)));

void idft65536(int16_t *x,int16_t *y,unsigned char scale)
{

  simd256_q15_t xtmp[8192],ytmp[8192],*tw65536_256p=(simd256_q15_t *)tw65536,*x256=(simd256_q15_t *)x,*y256=(simd256_q15_t *)y,*y256p=(simd256_q15_t *)y;
  simd256_q15_t *ytmpp = &ytmp[0];
  int i,j;

  for (i=0,j=0; i<8192; i+=4,j++) {
    transpose16_ooff_simd256(x256+i,xtmp+j,2048);
  }


  idft16384((int16_t*)(xtmp),(int16_t*)(ytmp),1);
  idft16384((int16_t*)(xtmp+2048),(int16_t*)(ytmp+2048),1);
  idft16384((int16_t*)(xtmp+4096),(int16_t*)(ytmp+4096),1);
  idft16384((int16_t*)(xtmp+6144),(int16_t*)(ytmp+6144),1);

  for (i=0; i<2048; i++) {
    ibfly4_256(ytmpp,ytmpp+2048,ytmpp+4096,ytmpp+6144,
           y256p,y256p+2048,y256p+4096,y256p+6144,
           tw65536_256p,tw65536_256p+4096,tw65536_256p+8192);
    tw65536_256p++;
    y256p++;
    ytmpp++;
  }

  if (scale>0) {

    for (i=0; i<512; i++) {
      y256[0]  = shiftright_int16_simd256(y256[0],scale);
      y256[1]  = shiftright_int16_simd256(y256[1],scale);
      y256[2]  = shiftright_int16_simd256(y256[2],scale);
      y256[3]  = shiftright_int16_simd256(y256[3],scale);
      y256[4]  = shiftright_int16_simd256(y256[4],scale);
      y256[5]  = shiftright_int16_simd256(y256[5],scale);
      y256[6]  = shiftright_int16_simd256(y256[6],scale);
      y256[7]  = shiftright_int16_simd256(y256[7],scale);
      y256[8]  = shiftright_int16_simd256(y256[8],scale);
      y256[9]  = shiftright_int16_simd256(y256[9],scale);
      y256[10] = shiftright_int16_simd256(y256[10],scale);
      y256[11] = shiftright_int16_simd256(y256[11],scale);
      y256[12] = shiftright_int16_simd256(y256[12],scale);
      y256[13] = shiftright_int16_simd256(y256[13],scale);
      y256[14] = shiftright_int16_simd256(y256[14],scale);
      y256[15] = shiftright_int16_simd256(y256[15],scale);

      y256+=16;
    }

  }

}

int16_t twa98304[65536] __attribute__((aligned(32)));
int16_t twb98304[65536] __attribute__((aligned(32)));
// 32768 x 3
void dft98304(int16_t *input, int16_t *output,uint8_t scale) {

  int i,i2,j;
  uint32_t tmp[3][32768] __attribute__((aligned(32)));
  uint32_t tmpo[3][32768] __attribute__((aligned(32)));
  simd_q15_t *y128p=(simd_q15_t*)output;
  simd_q15_t ONE_OVER_SQRT3_Q15_128 = set1_int16(ONE_OVER_SQRT3_Q15);

  for (i=0,j=0; i<32768; i++) {
    tmp[0][i] = ((uint32_t *)input)[j++];
    tmp[1][i] = ((uint32_t *)input)[j++];
    tmp[2][i] = ((uint32_t *)input)[j++];
  }

  dft32768((int16_t*)(tmp[0]),(int16_t*)(tmpo[0]),1);
  dft32768((int16_t*)(tmp[1]),(int16_t*)(tmpo[1]),1);
  dft32768((int16_t*)(tmp[2]),(int16_t*)(tmpo[2]),1);

  for (i=0,i2=0; i<65536; i+=8,i2+=4)  {
    bfly3((simd_q15_t*)(&tmpo[0][i2]),(simd_q15_t*)(&tmpo[1][i2]),((simd_q15_t*)&tmpo[2][i2]),
          (simd_q15_t*)(output+i),(simd_q15_t*)(output+65536+i),(simd_q15_t*)(output+131072+i),
          (simd_q15_t*)(twa98304+i),(simd_q15_t*)(twb98304+i));
  }
  if (scale==1) {
    for (i=0; i<1536; i++) {
      y128p[0]  = mulhi_int16(y128p[0],ONE_OVER_SQRT3_Q15_128);
      y128p[1]  = mulhi_int16(y128p[1],ONE_OVER_SQRT3_Q15_128);
      y128p[2]  = mulhi_int16(y128p[2],ONE_OVER_SQRT3_Q15_128);
      y128p[3]  = mulhi_int16(y128p[3],ONE_OVER_SQRT3_Q15_128);
      y128p[4]  = mulhi_int16(y128p[4],ONE_OVER_SQRT3_Q15_128);
      y128p[5]  = mulhi_int16(y128p[5],ONE_OVER_SQRT3_Q15_128);
      y128p[6]  = mulhi_int16(y128p[6],ONE_OVER_SQRT3_Q15_128);
      y128p[7]  = mulhi_int16(y128p[7],ONE_OVER_SQRT3_Q15_128);
      y128p[8]  = mulhi_int16(y128p[8],ONE_OVER_SQRT3_Q15_128);
      y128p[9]  = mulhi_int16(y128p[9],ONE_OVER_SQRT3_Q15_128);
      y128p[10] = mulhi_int16(y128p[10],ONE_OVER_SQRT3_Q15_128);
      y128p[11] = mulhi_int16(y128p[11],ONE_OVER_SQRT3_Q15_128);
      y128p[12] = mulhi_int16(y128p[12],ONE_OVER_SQRT3_Q15_128);
      y128p[13] = mulhi_int16(y128p[13],ONE_OVER_SQRT3_Q15_128);
      y128p[14] = mulhi_int16(y128p[14],ONE_OVER_SQRT3_Q15_128);
      y128p[15] = mulhi_int16(y128p[15],ONE_OVER_SQRT3_Q15_128);
      y128p+=16;
    }
  }
}

void idft98304(int16_t *input, int16_t *output,uint8_t scale) {

  int i,i2,j;
  uint32_t tmp[3][32768] __attribute__((aligned(32)));
  uint32_t tmpo[3][32768] __attribute__((aligned(32)));
  simd_q15_t *y128p=(simd_q15_t*)output;
  simd_q15_t ONE_OVER_SQRT3_Q15_128 = set1_int16(ONE_OVER_SQRT3_Q15);

  for (i=0,j=0; i<32768; i++) {
    tmp[0][i] = ((uint32_t *)input)[j++];
    tmp[1][i] = ((uint32_t *)input)[j++];
    tmp[2][i] = ((uint32_t *)input)[j++];
  }

  idft32768((int16_t*)(tmp[0]),(int16_t*)(tmpo[0]),1);
  idft32768((int16_t*)(tmp[1]),(int16_t*)(tmpo[1]),1);
  idft32768((int16_t*)(tmp[2]),(int16_t*)(tmpo[2]),1);

  for (i=0,i2=0; i<65536; i+=8,i2+=4)  {
    ibfly3((simd_q15_t*)(&tmpo[0][i2]),(simd_q15_t*)(&tmpo[1][i2]),((simd_q15_t*)&tmpo[2][i2]),
	   (simd_q15_t*)(output+i),(simd_q15_t*)(output+65536+i),(simd_q15_t*)(output+131072+i),
	   (simd_q15_t*)(twa98304+i),(simd_q15_t*)(twb98304+i));
  }
  if (scale==1) {
    for (i=0; i<1536; i++) {
      y128p[0]  = mulhi_int16(y128p[0],ONE_OVER_SQRT3_Q15_128);
      y128p[1]  = mulhi_int16(y128p[1],ONE_OVER_SQRT3_Q15_128);
      y128p[2]  = mulhi_int16(y128p[2],ONE_OVER_SQRT3_Q15_128);
      y128p[3]  = mulhi_int16(y128p[3],ONE_OVER_SQRT3_Q15_128);
      y128p[4]  = mulhi_int16(y128p[4],ONE_OVER_SQRT3_Q15_128);
      y128p[5]  = mulhi_int16(y128p[5],ONE_OVER_SQRT3_Q15_128);
      y128p[6]  = mulhi_int16(y128p[6],ONE_OVER_SQRT3_Q15_128);
      y128p[7]  = mulhi_int16(y128p[7],ONE_OVER_SQRT3_Q15_128);
      y128p[8]  = mulhi_int16(y128p[8],ONE_OVER_SQRT3_Q15_128);
      y128p[9]  = mulhi_int16(y128p[9],ONE_OVER_SQRT3_Q15_128);
      y128p[10] = mulhi_int16(y128p[10],ONE_OVER_SQRT3_Q15_128);
      y128p[11] = mulhi_int16(y128p[11],ONE_OVER_SQRT3_Q15_128);
      y128p[12] = mulhi_int16(y128p[12],ONE_OVER_SQRT3_Q15_128);
      y128p[13] = mulhi_int16(y128p[13],ONE_OVER_SQRT3_Q15_128);
      y128p[14] = mulhi_int16(y128p[14],ONE_OVER_SQRT3_Q15_128);
      y128p[15] = mulhi_int16(y128p[15],ONE_OVER_SQRT3_Q15_128);
      y128p+=16;
    }
  }
}

 
///  THIS SECTION IS FOR ALL PUSCH DFTS (i.e. radix 2^a * 3^b * 4^c * 5^d)
///  They use twiddles for 4-way parallel DFTS (i.e. 4 DFTS with interleaved input/output)

static int16_t const W1_12s[8] __attribute__((aligned(32))) = {28377, -16383, 28377, -16383, 28377, -16383, 28377, -16383};
static int16_t const W2_12s[8] __attribute__((aligned(32))) = {16383, -28377, 16383, -28377, 16383, -28377, 16383, -28377};
static int16_t const W3_12s[8] __attribute__((aligned(32))) = {0, -32767, 0, -32767, 0, -32767, 0, -32767};
static int16_t const W4_12s[8] __attribute__((aligned(32))) = {-16383, -28377, -16383, -28377, -16383, -28377, -16383, -28377};
static int16_t const W6_12s[8] __attribute__((aligned(32))) = {-32767, 0, -32767, 0, -32767, 0, -32767, 0};

simd_q15_t *const W1_12=(simd_q15_t *)W1_12s;
simd_q15_t *const W2_12=(simd_q15_t *)W2_12s;
simd_q15_t *const W3_12=(simd_q15_t *)W3_12s;
simd_q15_t *const W4_12=(simd_q15_t *)W4_12s;
simd_q15_t *const W6_12=(simd_q15_t *)W6_12s;

__attribute__((always_inline)) static inline void dft12f(simd_q15_t *x0,
                                                         simd_q15_t *x1,
                                                         simd_q15_t *x2,
                                                         simd_q15_t *x3,
                                                         simd_q15_t *x4,
                                                         simd_q15_t *x5,
                                                         simd_q15_t *x6,
                                                         simd_q15_t *x7,
                                                         simd_q15_t *x8,
                                                         simd_q15_t *x9,
                                                         simd_q15_t *x10,
                                                         simd_q15_t *x11,
                                                         simd_q15_t *y0,
                                                         simd_q15_t *y1,
                                                         simd_q15_t *y2,
                                                         simd_q15_t *y3,
                                                         simd_q15_t *y4,
                                                         simd_q15_t *y5,
                                                         simd_q15_t *y6,
                                                         simd_q15_t *y7,
                                                         simd_q15_t *y8,
                                                         simd_q15_t *y9,
                                                         simd_q15_t *y10,
                                                         simd_q15_t *y11)
{


  simd_q15_t tmp_dft12[12];

  // msg("dft12\n");

  bfly4_tw1(x0,
            x3,
            x6,
            x9,
            tmp_dft12,
            tmp_dft12+3,
            tmp_dft12+6,
            tmp_dft12+9);

  bfly4_tw1(x1,
            x4,
            x7,
            x10,
            tmp_dft12+1,
            tmp_dft12+4,
            tmp_dft12+7,
            tmp_dft12+10);


  bfly4_tw1(x2,
            x5,
            x8,
            x11,
            tmp_dft12+2,
            tmp_dft12+5,
            tmp_dft12+8,
            tmp_dft12+11);

  //  k2=0;
  bfly3_tw1(tmp_dft12,
            tmp_dft12+1,
            tmp_dft12+2,
            y0,
            y4,
            y8);



  //  k2=1;
  bfly3(tmp_dft12+3,
        tmp_dft12+4,
        tmp_dft12+5,
        y1,
        y5,
        y9,
        W1_12,
        W2_12);



  //  k2=2;
  bfly3(tmp_dft12+6,
        tmp_dft12+7,
        tmp_dft12+8,
        y2,
        y6,
        y10,
        W2_12,
        W4_12);

  //  k2=3;
  bfly3(tmp_dft12+9,
        tmp_dft12+10,
        tmp_dft12+11,
        y3,
        y7,
        y11,
        W3_12,
        W6_12);

}




void dft12(int16_t *x,int16_t *y ,unsigned char scale_flag)
{

  simd_q15_t *x128 = (simd_q15_t *)x,*y128 = (simd_q15_t *)y;
  dft12f(&x128[0],
         &x128[1],
         &x128[2],
         &x128[3],
         &x128[4],
         &x128[5],
         &x128[6],
         &x128[7],
         &x128[8],
         &x128[9],
         &x128[10],
         &x128[11],
         &y128[0],
         &y128[1],
         &y128[2],
         &y128[3],
         &y128[4],
         &y128[5],
         &y128[6],
         &y128[7],
         &y128[8],
         &y128[9],
         &y128[10],
         &y128[11]);

}

static const int16_t W1_12s_256[16] __attribute__((aligned(32))) =
    {28377, -16383, 28377, -16383, 28377, -16383, 28377, -16383, 28377, -16383, 28377, -16383, 28377, -16383, 28377, -16383};
static const int16_t W2_12s_256[16] __attribute__((aligned(32))) =
    {16383, -28377, 16383, -28377, 16383, -28377, 16383, -28377, 16383, -28377, 16383, -28377, 16383, -28377, 16383, -28377};
static const int16_t W3_12s_256[16]
    __attribute__((aligned(32))) = {0, -32767, 0, -32767, 0, -32767, 0, -32767, 0, -32767, 0, -32767, 0, -32767, 0, -32767};
static const int16_t W4_12s_256[16] __attribute__((aligned(32))) = {-16383,
                                                                    -28377,
                                                                    -16383,
                                                                    -28377,
                                                                    -16383,
                                                                    -28377,
                                                                    -16383,
                                                                    -28377,
                                                                    -16383,
                                                                    -28377,
                                                                    -16383,
                                                                    -28377,
                                                                    -16383,
                                                                    -28377,
                                                                    -16383,
                                                                    -28377};
static const int16_t W6_12s_256[16]
    __attribute__((aligned(32))) = {-32767, 0, -32767, 0, -32767, 0, -32767, 0, -32767, 0, -32767, 0, -32767, 0, -32767, 0};

simd256_q15_t * const W1_12_256=(simd256_q15_t *)W1_12s_256;
simd256_q15_t * const W2_12_256=(simd256_q15_t *)W2_12s_256;
simd256_q15_t * const W3_12_256=(simd256_q15_t *)W3_12s_256;
simd256_q15_t * const W4_12_256=(simd256_q15_t *)W4_12s_256;
simd256_q15_t * const W6_12_256=(simd256_q15_t *)W6_12s_256;

__attribute__((always_inline)) static inline void dft12f_simd256(simd256_q15_t *x0,
                                                                 simd256_q15_t *x1,
                                                                 simd256_q15_t *x2,
                                                                 simd256_q15_t *x3,
                                                                 simd256_q15_t *x4,
                                                                 simd256_q15_t *x5,
                                                                 simd256_q15_t *x6,
                                                                 simd256_q15_t *x7,
                                                                 simd256_q15_t *x8,
                                                                 simd256_q15_t *x9,
                                                                 simd256_q15_t *x10,
                                                                 simd256_q15_t *x11,
                                                                 simd256_q15_t *y0,
                                                                 simd256_q15_t *y1,
                                                                 simd256_q15_t *y2,
                                                                 simd256_q15_t *y3,
                                                                 simd256_q15_t *y4,
                                                                 simd256_q15_t *y5,
                                                                 simd256_q15_t *y6,
                                                                 simd256_q15_t *y7,
                                                                 simd256_q15_t *y8,
                                                                 simd256_q15_t *y9,
                                                                 simd256_q15_t *y10,
                                                                 simd256_q15_t *y11)
{


  simd256_q15_t tmp_dft12[12];

  // msg("dft12\n");

  bfly4_tw1_256(x0,
		x3,
		x6,
		x9,
		tmp_dft12,
		tmp_dft12+3,
		tmp_dft12+6,
		tmp_dft12+9);


  bfly4_tw1_256(x1,
		x4,
		x7,
		x10,
		tmp_dft12+1,
		tmp_dft12+4,
		tmp_dft12+7,
		tmp_dft12+10);
  

  bfly4_tw1_256(x2,
		x5,
		x8,
		x11,
		tmp_dft12+2,
		tmp_dft12+5,
		tmp_dft12+8,
		tmp_dft12+11);
  
  //  k2=0;
  bfly3_tw1_256(tmp_dft12,
		tmp_dft12+1,
		tmp_dft12+2,
		y0,
		y4,
		y8);
  
  
  
  //  k2=1;
  bfly3_256(tmp_dft12+3,
	    tmp_dft12+4,
	    tmp_dft12+5,
	    y1,
	    y5,
	    y9,
	    W1_12_256,
	    W2_12_256);
  
  
  
  //  k2=2;
  bfly3_256(tmp_dft12+6,
	    tmp_dft12+7,
	    tmp_dft12+8,
	    y2,
	    y6,
	    y10,
	    W2_12_256,
	    W4_12_256);
  
  //  k2=3;
  bfly3_256(tmp_dft12+9,
	    tmp_dft12+10,
	    tmp_dft12+11,
	    y3,
	    y7,
	    y11,
	    W3_12_256,
	    W6_12_256);
  
}




void dft12_simd256(int16_t *x,int16_t *y)
{

  simd256_q15_t *x256 = (simd256_q15_t *)x,*y256 = (simd256_q15_t *)y;
  dft12f_simd256(&x256[0],
                 &x256[1],
                 &x256[2],
                 &x256[3],
                 &x256[4],
                 &x256[5],
                 &x256[6],
                 &x256[7],
                 &x256[8],
                 &x256[9],
                 &x256[10],
                 &x256[11],
                 &y256[0],
                 &y256[1],
                 &y256[2],
                 &y256[3],
                 &y256[4],
                 &y256[5],
                 &y256[6],
                 &y256[7],
                 &y256[8],
                 &y256[9],
                 &y256[10],
                 &y256[11]);

}

static int16_t tw24[88]__attribute__((aligned(32)));

void dft24(int16_t *x,int16_t *y,unsigned char scale_flag)
{

  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *tw128=(simd_q15_t *)&tw24[0];
  simd_q15_t ytmp128[24];//=&ytmp128array[0];
  int i,j,k;

  //  msg("dft24\n");
  dft12f(x128,
         x128+2,
         x128+4,
         x128+6,
         x128+8,
         x128+10,
         x128+12,
         x128+14,
         x128+16,
         x128+18,
         x128+20,
         x128+22,
         ytmp128,
         ytmp128+2,
         ytmp128+4,
         ytmp128+6,
         ytmp128+8,
         ytmp128+10,
         ytmp128+12,
         ytmp128+14,
         ytmp128+16,
         ytmp128+18,
         ytmp128+20,
         ytmp128+22);
  //  msg("dft24b\n");

  dft12f(x128+1,
         x128+3,
         x128+5,
         x128+7,
         x128+9,
         x128+11,
         x128+13,
         x128+15,
         x128+17,
         x128+19,
         x128+21,
         x128+23,
         ytmp128+1,
         ytmp128+3,
         ytmp128+5,
         ytmp128+7,
         ytmp128+9,
         ytmp128+11,
         ytmp128+13,
         ytmp128+15,
         ytmp128+17,
         ytmp128+19,
         ytmp128+21,
         ytmp128+23);

  //  msg("dft24c\n");

  bfly2_tw1(ytmp128,
            ytmp128+1,
            y128,
            y128+12);

  //  msg("dft24d\n");

  for (i=2,j=1,k=0; i<24; i+=2,j++,k++) {

    bfly2(ytmp128+i,
          ytmp128+i+1,
          y128+j,
          y128+j+12,
          tw128+k);
    //    msg("dft24e\n");
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[1]);

    for (i=0; i<24; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

}

static int16_t twa36[88]__attribute__((aligned(32)));
static int16_t twb36[88]__attribute__((aligned(32)));

void dft36(int16_t *x,int16_t *y,unsigned char scale_flag)
{

  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa36[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb36[0];
  simd_q15_t ytmp128[36];//&ytmp128array[0];


  int i,j,k;

  dft12f(x128,
         x128+3,
         x128+6,
         x128+9,
         x128+12,
         x128+15,
         x128+18,
         x128+21,
         x128+24,
         x128+27,
         x128+30,
         x128+33,
         ytmp128,
         ytmp128+3,
         ytmp128+6,
         ytmp128+9,
         ytmp128+12,
         ytmp128+15,
         ytmp128+18,
         ytmp128+21,
         ytmp128+24,
         ytmp128+27,
         ytmp128+30,
         ytmp128+33);

  dft12f(x128+1,
         x128+4,
         x128+7,
         x128+10,
         x128+13,
         x128+16,
         x128+19,
         x128+22,
         x128+25,
         x128+28,
         x128+31,
         x128+34,
         ytmp128+1,
         ytmp128+4,
         ytmp128+7,
         ytmp128+10,
         ytmp128+13,
         ytmp128+16,
         ytmp128+19,
         ytmp128+22,
         ytmp128+25,
         ytmp128+28,
         ytmp128+31,
         ytmp128+34);

  dft12f(x128+2,
         x128+5,
         x128+8,
         x128+11,
         x128+14,
         x128+17,
         x128+20,
         x128+23,
         x128+26,
         x128+29,
         x128+32,
         x128+35,
         ytmp128+2,
         ytmp128+5,
         ytmp128+8,
         ytmp128+11,
         ytmp128+14,
         ytmp128+17,
         ytmp128+20,
         ytmp128+23,
         ytmp128+26,
         ytmp128+29,
         ytmp128+32,
         ytmp128+35);


  bfly3_tw1(ytmp128,
            ytmp128+1,
            ytmp128+2,
            y128,
            y128+12,
            y128+24);

  for (i=3,j=1,k=0; i<36; i+=3,j++,k++) {

    bfly3(ytmp128+i,
          ytmp128+i+1,
          ytmp128+i+2,
          y128+j,
          y128+j+12,
          y128+j+24,
          twa128+k,
          twb128+k);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[2]);

    for (i=0; i<36; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

}

static int16_t twa48[88]__attribute__((aligned(32)));
static int16_t twb48[88]__attribute__((aligned(32)));
static int16_t twc48[88]__attribute__((aligned(32)));

void dft48(int16_t *x, int16_t *y,unsigned char scale_flag)
{

  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa48[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb48[0];
  simd_q15_t *twc128=(simd_q15_t *)&twc48[0];
  simd_q15_t ytmp128[48];//=&ytmp128array[0];
  int i,j,k;


  dft12f(x128,
         x128+4,
         x128+8,
         x128+12,
         x128+16,
         x128+20,
         x128+24,
         x128+28,
         x128+32,
         x128+36,
         x128+40,
         x128+44,
         ytmp128,
         ytmp128+4,
         ytmp128+8,
         ytmp128+12,
         ytmp128+16,
         ytmp128+20,
         ytmp128+24,
         ytmp128+28,
         ytmp128+32,
         ytmp128+36,
         ytmp128+40,
         ytmp128+44);


  dft12f(x128+1,
         x128+5,
         x128+9,
         x128+13,
         x128+17,
         x128+21,
         x128+25,
         x128+29,
         x128+33,
         x128+37,
         x128+41,
         x128+45,
         ytmp128+1,
         ytmp128+5,
         ytmp128+9,
         ytmp128+13,
         ytmp128+17,
         ytmp128+21,
         ytmp128+25,
         ytmp128+29,
         ytmp128+33,
         ytmp128+37,
         ytmp128+41,
         ytmp128+45);


  dft12f(x128+2,
         x128+6,
         x128+10,
         x128+14,
         x128+18,
         x128+22,
         x128+26,
         x128+30,
         x128+34,
         x128+38,
         x128+42,
         x128+46,
         ytmp128+2,
         ytmp128+6,
         ytmp128+10,
         ytmp128+14,
         ytmp128+18,
         ytmp128+22,
         ytmp128+26,
         ytmp128+30,
         ytmp128+34,
         ytmp128+38,
         ytmp128+42,
         ytmp128+46);


  dft12f(x128+3,
         x128+7,
         x128+11,
         x128+15,
         x128+19,
         x128+23,
         x128+27,
         x128+31,
         x128+35,
         x128+39,
         x128+43,
         x128+47,
         ytmp128+3,
         ytmp128+7,
         ytmp128+11,
         ytmp128+15,
         ytmp128+19,
         ytmp128+23,
         ytmp128+27,
         ytmp128+31,
         ytmp128+35,
         ytmp128+39,
         ytmp128+43,
         ytmp128+47);



  bfly4_tw1(ytmp128,
            ytmp128+1,
            ytmp128+2,
            ytmp128+3,
            y128,
            y128+12,
            y128+24,
            y128+36);



  for (i=4,j=1,k=0; i<48; i+=4,j++,k++) {

    bfly4(ytmp128+i,
          ytmp128+i+1,
          ytmp128+i+2,
          ytmp128+i+3,
          y128+j,
          y128+j+12,
          y128+j+24,
          y128+j+36,
          twa128+k,
          twb128+k,
          twc128+k);

  }

  if (scale_flag == 1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[3]);

    for (i=0; i<48; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

}

static int16_t twa60[88]__attribute__((aligned(32)));
static int16_t twb60[88]__attribute__((aligned(32)));
static int16_t twc60[88]__attribute__((aligned(32)));
static int16_t twd60[88]__attribute__((aligned(32)));

void dft60(int16_t *x,int16_t *y,unsigned char scale)
{

  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa60[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb60[0];
  simd_q15_t *twc128=(simd_q15_t *)&twc60[0];
  simd_q15_t *twd128=(simd_q15_t *)&twd60[0];
  simd_q15_t ytmp128[60];//=&ytmp128array[0];
  int i,j,k;

  dft12f(x128,
         x128+5,
         x128+10,
         x128+15,
         x128+20,
         x128+25,
         x128+30,
         x128+35,
         x128+40,
         x128+45,
         x128+50,
         x128+55,
         ytmp128,
         ytmp128+5,
         ytmp128+10,
         ytmp128+15,
         ytmp128+20,
         ytmp128+25,
         ytmp128+30,
         ytmp128+35,
         ytmp128+40,
         ytmp128+45,
         ytmp128+50,
         ytmp128+55);

  dft12f(x128+1,
         x128+6,
         x128+11,
         x128+16,
         x128+21,
         x128+26,
         x128+31,
         x128+36,
         x128+41,
         x128+46,
         x128+51,
         x128+56,
         ytmp128+1,
         ytmp128+6,
         ytmp128+11,
         ytmp128+16,
         ytmp128+21,
         ytmp128+26,
         ytmp128+31,
         ytmp128+36,
         ytmp128+41,
         ytmp128+46,
         ytmp128+51,
         ytmp128+56);

  dft12f(x128+2,
         x128+7,
         x128+12,
         x128+17,
         x128+22,
         x128+27,
         x128+32,
         x128+37,
         x128+42,
         x128+47,
         x128+52,
         x128+57,
         ytmp128+2,
         ytmp128+7,
         ytmp128+12,
         ytmp128+17,
         ytmp128+22,
         ytmp128+27,
         ytmp128+32,
         ytmp128+37,
         ytmp128+42,
         ytmp128+47,
         ytmp128+52,
         ytmp128+57);

  dft12f(x128+3,
         x128+8,
         x128+13,
         x128+18,
         x128+23,
         x128+28,
         x128+33,
         x128+38,
         x128+43,
         x128+48,
         x128+53,
         x128+58,
         ytmp128+3,
         ytmp128+8,
         ytmp128+13,
         ytmp128+18,
         ytmp128+23,
         ytmp128+28,
         ytmp128+33,
         ytmp128+38,
         ytmp128+43,
         ytmp128+48,
         ytmp128+53,
         ytmp128+58);

  dft12f(x128+4,
         x128+9,
         x128+14,
         x128+19,
         x128+24,
         x128+29,
         x128+34,
         x128+39,
         x128+44,
         x128+49,
         x128+54,
         x128+59,
         ytmp128+4,
         ytmp128+9,
         ytmp128+14,
         ytmp128+19,
         ytmp128+24,
         ytmp128+29,
         ytmp128+34,
         ytmp128+39,
         ytmp128+44,
         ytmp128+49,
         ytmp128+54,
         ytmp128+59);

  bfly5_tw1(ytmp128,
            ytmp128+1,
            ytmp128+2,
            ytmp128+3,
            ytmp128+4,
            y128,
            y128+12,
            y128+24,
            y128+36,
            y128+48);

  for (i=5,j=1,k=0; i<60; i+=5,j++,k++) {

    bfly5(ytmp128+i,
          ytmp128+i+1,
          ytmp128+i+2,
          ytmp128+i+3,
          ytmp128+i+4,
          y128+j,
          y128+j+12,
          y128+j+24,
          y128+j+36,
          y128+j+48,
          twa128+k,
          twb128+k,
          twc128+k,
          twd128+k);
  }

  if (scale == 1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[4]);

    for (i=0; i<60; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
//      printf("y[%d] = (%d,%d)\n",i,((int16_t*)&y128[i])[0],((int16_t*)&y128[i])[1]);
    }
  }

}

static int16_t tw72[280]__attribute__((aligned(32)));

void dft72(int16_t *x,int16_t *y,unsigned char scale_flag)
{

  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *tw128=(simd_q15_t *)&tw72[0];
  simd_q15_t x2128[72];// = (simd_q15_t *)&x2128array[0];

  simd_q15_t ytmp128[72];//=&ytmp128array2[0];

  for (i=0,j=0; i<36; i++,j+=2) {
    x2128[i]    = x128[j];    // even inputs
    x2128[i+36] = x128[j+1];  // odd inputs
  }

  dft36((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft36((int16_t *)(x2128+36),(int16_t *)(ytmp128+36),1);

  bfly2_tw1(ytmp128,ytmp128+36,y128,y128+36);

  for (i=1,j=0; i<36; i++,j++) {
    bfly2(ytmp128+i,
          ytmp128+36+i,
          y128+i,
          y128+36+i,
          tw128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[5]);

    for (i=0; i<72; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

}

static int16_t tw96[376]__attribute__((aligned(32)));

void dft96(int16_t *x,int16_t *y,unsigned char scale_flag)
{


  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *tw128=(simd_q15_t *)&tw96[0];
  simd_q15_t x2128[96];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[96];//=&ytmp128array2[0];


  for (i=0,j=0; i<48; i++,j+=2) {
    x2128[i]    = x128[j];
    x2128[i+48] = x128[j+1];
  }

  dft48((int16_t *)x2128,(int16_t *)ytmp128,0);
  dft48((int16_t *)(x2128+48),(int16_t *)(ytmp128+48),0);


  bfly2_tw1(ytmp128,ytmp128+48,y128,y128+48);

  for (i=1,j=0; i<48; i++,j++) {
    bfly2(ytmp128+i,
          ytmp128+48+i,
          y128+i,
          y128+48+i,
          tw128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[6]);

    for (i=0; i<96; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

}

static int16_t twa108[280]__attribute__((aligned(32)));
static int16_t twb108[280]__attribute__((aligned(32)));

void dft108(int16_t *x,int16_t *y,unsigned char scale_flag)
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa108[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb108[0];
  simd_q15_t x2128[108];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[108];//=&ytmp128array2[0];


  for (i=0,j=0; i<36; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+36] = x128[j+1];
    x2128[i+72] = x128[j+2];
  }

  dft36((int16_t *)x2128,(int16_t *)ytmp128,0);
  dft36((int16_t *)(x2128+36),(int16_t *)(ytmp128+36),0);
  dft36((int16_t *)(x2128+72),(int16_t *)(ytmp128+72),0);

  bfly3_tw1(ytmp128,ytmp128+36,ytmp128+72,y128,y128+36,y128+72);

  for (i=1,j=0; i<36; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+36+i,
          ytmp128+72+i,
          y128+i,
          y128+36+i,
          y128+72+i,
          twa128+j,
          twb128+j);

  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[7]);

    for (i=0; i<108; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

}

static int16_t tw120[472]__attribute__((aligned(32)));
void dft120(int16_t *x,int16_t *y, unsigned char scale_flag)
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *tw128=(simd_q15_t *)&tw120[0];
  simd_q15_t x2128[120];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[120];//=&ytmp128array2[0];

  for (i=0,j=0; i<60; i++,j+=2) {
    x2128[i]    = x128[j];
    x2128[i+60] = x128[j+1];
  }

  dft60((int16_t *)x2128,(int16_t *)ytmp128,0);
  dft60((int16_t *)(x2128+60),(int16_t *)(ytmp128+60),0);


  bfly2_tw1(ytmp128,ytmp128+60,y128,y128+60);

  for (i=1,j=0; i<60; i++,j++) {
    bfly2(ytmp128+i,
          ytmp128+60+i,
          y128+i,
          y128+60+i,
          tw128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[8]);

    for (i=0; i<120; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

}

static int16_t twa144[376]__attribute__((aligned(32)));
static int16_t twb144[376]__attribute__((aligned(32)));

void dft144(int16_t *x,int16_t *y,unsigned char scale_flag)
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa144[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb144[0];
  simd_q15_t x2128[144];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[144];//=&ytmp128array2[0];



  for (i=0,j=0; i<48; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+48] = x128[j+1];
    x2128[i+96] = x128[j+2];
  }

  dft48((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft48((int16_t *)(x2128+48),(int16_t *)(ytmp128+48),1);
  dft48((int16_t *)(x2128+96),(int16_t *)(ytmp128+96),1);

  bfly3_tw1(ytmp128,ytmp128+48,ytmp128+96,y128,y128+48,y128+96);

  for (i=1,j=0; i<48; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+48+i,
          ytmp128+96+i,
          y128+i,
          y128+48+i,
          y128+96+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[9]);

    for (i=0; i<144; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

}

static int16_t twa180[472]__attribute__((aligned(32)));
static int16_t twb180[472]__attribute__((aligned(32)));

void dft180(int16_t *x,int16_t *y,unsigned char scale_flag)
{

  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa180[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb180[0];
  simd_q15_t x2128[180];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[180];//=&ytmp128array2[0];



  for (i=0,j=0; i<60; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+60] = x128[j+1];
    x2128[i+120] = x128[j+2];
  }

  dft60((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft60((int16_t *)(x2128+60),(int16_t *)(ytmp128+60),1);
  dft60((int16_t *)(x2128+120),(int16_t *)(ytmp128+120),1);

  bfly3_tw1(ytmp128,ytmp128+60,ytmp128+120,y128,y128+60,y128+120);

  for (i=1,j=0; i<60; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+60+i,
          ytmp128+120+i,
          y128+i,
          y128+60+i,
          y128+120+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[10]);

    for (i=0; i<180; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

}


static int16_t twa216[568]__attribute__((aligned(32)));
static int16_t twb216[568]__attribute__((aligned(32)));

void dft216(int16_t *x,int16_t *y,unsigned char scale_flag)
{

  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa216[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb216[0];
  simd_q15_t x2128[216];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[216];//=&ytmp128array3[0];



  for (i=0,j=0; i<72; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+72] = x128[j+1];
    x2128[i+144] = x128[j+2];
  }

  dft72((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft72((int16_t *)(x2128+72),(int16_t *)(ytmp128+72),1);
  dft72((int16_t *)(x2128+144),(int16_t *)(ytmp128+144),1);

  bfly3_tw1(ytmp128,ytmp128+72,ytmp128+144,y128,y128+72,y128+144);

  for (i=1,j=0; i<72; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+72+i,
          ytmp128+144+i,
          y128+i,
          y128+72+i,
          y128+144+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[12]);

    for (i=0; i<216; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

}

static int16_t twa240[472]__attribute__((aligned(32)));
static int16_t twb240[472]__attribute__((aligned(32)));
static int16_t twc240[472]__attribute__((aligned(32)));

void dft240(int16_t *x,int16_t *y,unsigned char scale_flag)
{

  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa240[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb240[0];
  simd_q15_t *twc128=(simd_q15_t *)&twc240[0];
  simd_q15_t x2128[240];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[240];//=&ytmp128array2[0];



  for (i=0,j=0; i<60; i++,j+=4) {
    x2128[i]    = x128[j];
    x2128[i+60] = x128[j+1];
    x2128[i+120] = x128[j+2];
    x2128[i+180] = x128[j+3];
  }

  dft60((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft60((int16_t *)(x2128+60),(int16_t *)(ytmp128+60),1);
  dft60((int16_t *)(x2128+120),(int16_t *)(ytmp128+120),1);
  dft60((int16_t *)(x2128+180),(int16_t *)(ytmp128+180),1);

  bfly4_tw1(ytmp128,ytmp128+60,ytmp128+120,ytmp128+180,y128,y128+60,y128+120,y128+180);

  for (i=1,j=0; i<60; i++,j++) {
    bfly4(ytmp128+i,
          ytmp128+60+i,
          ytmp128+120+i,
          ytmp128+180+i,
          y128+i,
          y128+60+i,
          y128+120+i,
          y128+180+i,
          twa128+j,
          twb128+j,
          twc128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[13]);

    for (i=0; i<240; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

}

static int16_t twa288[760]__attribute__((aligned(32)));
static int16_t twb288[760]__attribute__((aligned(32)));

void dft288(int16_t *x,int16_t *y,unsigned char scale_flag)
{

  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa288[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb288[0];
  simd_q15_t x2128[288];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[288];//=&ytmp128array3[0];



  for (i=0,j=0; i<96; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+96] = x128[j+1];
    x2128[i+192] = x128[j+2];
  }

  dft96((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft96((int16_t *)(x2128+96),(int16_t *)(ytmp128+96),1);
  dft96((int16_t *)(x2128+192),(int16_t *)(ytmp128+192),1);

  bfly3_tw1(ytmp128,ytmp128+96,ytmp128+192,y128,y128+96,y128+192);

  for (i=1,j=0; i<96; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+96+i,
          ytmp128+192+i,
          y128+i,
          y128+96+i,
          y128+192+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<288; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

}

static int16_t twa300[472]__attribute__((aligned(32)));
static int16_t twb300[472]__attribute__((aligned(32)));
static int16_t twc300[472]__attribute__((aligned(32)));
static int16_t twd300[472]__attribute__((aligned(32)));

void dft300(int16_t *x,int16_t *y,unsigned char scale_flag)
{

  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa300[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb300[0];
  simd_q15_t *twc128=(simd_q15_t *)&twc300[0];
  simd_q15_t *twd128=(simd_q15_t *)&twd300[0];
  simd_q15_t x2128[300];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[300];//=&ytmp128array2[0];



  for (i=0,j=0; i<60; i++,j+=5) {
    x2128[i]    = x128[j];
    x2128[i+60] = x128[j+1];
    x2128[i+120] = x128[j+2];
    x2128[i+180] = x128[j+3];
    x2128[i+240] = x128[j+4];
  }

  dft60((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft60((int16_t *)(x2128+60),(int16_t *)(ytmp128+60),1);
  dft60((int16_t *)(x2128+120),(int16_t *)(ytmp128+120),1);
  dft60((int16_t *)(x2128+180),(int16_t *)(ytmp128+180),1);
  dft60((int16_t *)(x2128+240),(int16_t *)(ytmp128+240),1);

  bfly5_tw1(ytmp128,ytmp128+60,ytmp128+120,ytmp128+180,ytmp128+240,y128,y128+60,y128+120,y128+180,y128+240);

  for (i=1,j=0; i<60; i++,j++) {
    bfly5(ytmp128+i,
          ytmp128+60+i,
          ytmp128+120+i,
          ytmp128+180+i,
          ytmp128+240+i,
          y128+i,
          y128+60+i,
          y128+120+i,
          y128+180+i,
          y128+240+i,
          twa128+j,
          twb128+j,
          twc128+j,
          twd128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[15]);

    for (i=0; i<300; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

}

static int16_t twa324[107*2*4];
static int16_t twb324[107*2*4];

void dft324(int16_t *x,int16_t *y,unsigned char scale_flag)  // 108 x 3
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa324[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb324[0];
  simd_q15_t x2128[324];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[324];//=&ytmp128array3[0];



  for (i=0,j=0; i<108; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+108] = x128[j+1];
    x2128[i+216] = x128[j+2];
  }

  dft108((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft108((int16_t *)(x2128+108),(int16_t *)(ytmp128+108),1);
  dft108((int16_t *)(x2128+216),(int16_t *)(ytmp128+216),1);

  bfly3_tw1(ytmp128,ytmp128+108,ytmp128+216,y128,y128+108,y128+216);

  for (i=1,j=0; i<108; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+108+i,
          ytmp128+216+i,
          y128+i,
          y128+108+i,
          y128+216+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<324; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};

static int16_t twa360[119*2*4];
static int16_t twb360[119*2*4];

void dft360(int16_t *x,int16_t *y,unsigned char scale_flag)  // 120 x 3
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa360[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb360[0];
  simd_q15_t x2128[360];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[360];//=&ytmp128array3[0];



  for (i=0,j=0; i<120; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+120] = x128[j+1];
    x2128[i+240] = x128[j+2];
  }

  dft120((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft120((int16_t *)(x2128+120),(int16_t *)(ytmp128+120),1);
  dft120((int16_t *)(x2128+240),(int16_t *)(ytmp128+240),1);

  bfly3_tw1(ytmp128,ytmp128+120,ytmp128+240,y128,y128+120,y128+240);

  for (i=1,j=0; i<120; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+120+i,
          ytmp128+240+i,
          y128+i,
          y128+120+i,
          y128+240+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<360; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};

static int16_t twa432[107*2*4];
static int16_t twb432[107*2*4];
static int16_t twc432[107*2*4];

void dft432(int16_t *x,int16_t *y,unsigned char scale_flag)  // 108 x 4
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa432[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb432[0];
  simd_q15_t *twc128=(simd_q15_t *)&twc432[0];
  simd_q15_t x2128[432];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[432];//=&ytmp128array2[0];


  for (i=0,j=0; i<108; i++,j+=4) {
    x2128[i]    = x128[j];
    x2128[i+108] = x128[j+1];
    x2128[i+216] = x128[j+2];
    x2128[i+324] = x128[j+3];
  }

  dft108((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft108((int16_t *)(x2128+108),(int16_t *)(ytmp128+108),1);
  dft108((int16_t *)(x2128+216),(int16_t *)(ytmp128+216),1);
  dft108((int16_t *)(x2128+324),(int16_t *)(ytmp128+324),1);

  bfly4_tw1(ytmp128,ytmp128+108,ytmp128+216,ytmp128+324,y128,y128+108,y128+216,y128+324);

  for (i=1,j=0; i<108; i++,j++) {
    bfly4(ytmp128+i,
          ytmp128+108+i,
          ytmp128+216+i,
          ytmp128+324+i,
          y128+i,
          y128+108+i,
          y128+216+i,
          y128+324+i,
          twa128+j,
          twb128+j,
          twc128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(16384); // dft_norm_table[13]);

    for (i=0; i<432; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};
static int16_t twa480[119*2*4];
static int16_t twb480[119*2*4];
static int16_t twc480[119*2*4];

void dft480(int16_t *x,int16_t *y,unsigned char scale_flag)  // 120 x 4
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa480[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb480[0];
  simd_q15_t *twc128=(simd_q15_t *)&twc480[0];
  simd_q15_t x2128[480];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[480];//=&ytmp128array2[0];



  for (i=0,j=0; i<120; i++,j+=4) {
    x2128[i]    = x128[j];
    x2128[i+120] = x128[j+1];
    x2128[i+240] = x128[j+2];
    x2128[i+360] = x128[j+3];
  }

  dft120((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft120((int16_t *)(x2128+120),(int16_t *)(ytmp128+120),1);
  dft120((int16_t *)(x2128+240),(int16_t *)(ytmp128+240),1);
  dft120((int16_t *)(x2128+360),(int16_t *)(ytmp128+360),1);

  bfly4_tw1(ytmp128,ytmp128+120,ytmp128+240,ytmp128+360,y128,y128+120,y128+240,y128+360);

  for (i=1,j=0; i<120; i++,j++) {
    bfly4(ytmp128+i,
          ytmp128+120+i,
          ytmp128+240+i,
          ytmp128+360+i,
          y128+i,
          y128+120+i,
          y128+240+i,
          y128+360+i,
          twa128+j,
          twb128+j,
          twc128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(16384); // dft_norm_table[13]);

    for (i=0; i<480; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};


static int16_t twa540[179*2*4];
static int16_t twb540[179*2*4];

void dft540(int16_t *x,int16_t *y,unsigned char scale_flag)  // 180 x 3
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa540[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb540[0];
  simd_q15_t x2128[540];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[540];//=&ytmp128array3[0];



  for (i=0,j=0; i<180; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+180] = x128[j+1];
    x2128[i+360] = x128[j+2];
  }

  dft180((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft180((int16_t *)(x2128+180),(int16_t *)(ytmp128+180),1);
  dft180((int16_t *)(x2128+360),(int16_t *)(ytmp128+360),1);

  bfly3_tw1(ytmp128,ytmp128+180,ytmp128+360,y128,y128+180,y128+360);

  for (i=1,j=0; i<180; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+180+i,
          ytmp128+360+i,
          y128+i,
          y128+180+i,
          y128+360+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<540; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};

static int16_t twa576[191*2*4];
static int16_t twb576[191*2*4];

void dft576(int16_t *x,int16_t *y,unsigned char scale_flag)  // 192 x 3
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa576[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb576[0];
  simd_q15_t x2128[576];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[576];//=&ytmp128array3[0];



  for (i=0,j=0; i<192; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+192] = x128[j+1];
    x2128[i+384] = x128[j+2];
  }


  dft192((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft192((int16_t *)(x2128+192),(int16_t *)(ytmp128+192),1);
  dft192((int16_t *)(x2128+384),(int16_t *)(ytmp128+384),1);

  bfly3_tw1(ytmp128,ytmp128+192,ytmp128+384,y128,y128+192,y128+384);

  for (i=1,j=0; i<192; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+192+i,
          ytmp128+384+i,
          y128+i,
          y128+192+i,
          y128+384+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<576; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};


static int16_t twa600[299*2*4];

void dft600(int16_t *x,int16_t *y,unsigned char scale_flag)  // 300 x 2
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *tw128=(simd_q15_t *)&twa600[0];
  simd_q15_t x2128[600];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[600];//=&ytmp128array2[0];


  for (i=0,j=0; i<300; i++,j+=2) {
    x2128[i]    = x128[j];
    x2128[i+300] = x128[j+1];
  }

  dft300((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft300((int16_t *)(x2128+300),(int16_t *)(ytmp128+300),1);


  bfly2_tw1(ytmp128,ytmp128+300,y128,y128+300);

  for (i=1,j=0; i<300; i++,j++) {
    bfly2(ytmp128+i,
          ytmp128+300+i,
          y128+i,
          y128+300+i,
          tw128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(ONE_OVER_SQRT2_Q15);

    for (i=0; i<600; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};


static int16_t twa648[215*2*4];
static int16_t twb648[215*2*4];

void dft648(int16_t *x,int16_t *y,unsigned char scale_flag)  // 216 x 3
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa648[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb648[0];
  simd_q15_t x2128[648];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[648];//=&ytmp128array3[0];



  for (i=0,j=0; i<216; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+216] = x128[j+1];
    x2128[i+432] = x128[j+2];
  }

  dft216((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft216((int16_t *)(x2128+216),(int16_t *)(ytmp128+216),1);
  dft216((int16_t *)(x2128+432),(int16_t *)(ytmp128+432),1);

  bfly3_tw1(ytmp128,ytmp128+216,ytmp128+432,y128,y128+216,y128+432);

  for (i=1,j=0; i<216; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+216+i,
          ytmp128+432+i,
          y128+i,
          y128+216+i,
          y128+432+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<648; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};


static int16_t twa720[179*2*4];
static int16_t twb720[179*2*4];
static int16_t twc720[179*2*4];


void dft720(int16_t *x,int16_t *y,unsigned char scale_flag)  // 180 x 4
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa720[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb720[0];
  simd_q15_t *twc128=(simd_q15_t *)&twc720[0];
  simd_q15_t x2128[720];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[720];//=&ytmp128array2[0];



  for (i=0,j=0; i<180; i++,j+=4) {
    x2128[i]    = x128[j];
    x2128[i+180] = x128[j+1];
    x2128[i+360] = x128[j+2];
    x2128[i+540] = x128[j+3];
  }

  dft180((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft180((int16_t *)(x2128+180),(int16_t *)(ytmp128+180),1);
  dft180((int16_t *)(x2128+360),(int16_t *)(ytmp128+360),1);
  dft180((int16_t *)(x2128+540),(int16_t *)(ytmp128+540),1);

  bfly4_tw1(ytmp128,ytmp128+180,ytmp128+360,ytmp128+540,y128,y128+180,y128+360,y128+540);

  for (i=1,j=0; i<180; i++,j++) {
    bfly4(ytmp128+i,
          ytmp128+180+i,
          ytmp128+360+i,
          ytmp128+540+i,
          y128+i,
          y128+180+i,
          y128+360+i,
          y128+540+i,
          twa128+j,
          twb128+j,
          twc128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(16384); // dft_norm_table[13]);

    for (i=0; i<720; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};

static int16_t twa864[287*2*4];
static int16_t twb864[287*2*4];

void dft864(int16_t *x,int16_t *y,unsigned char scale_flag)  // 288 x 3
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa864[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb864[0];
  simd_q15_t x2128[864];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[864];//=&ytmp128array3[0];



  for (i=0,j=0; i<288; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+288] = x128[j+1];
    x2128[i+576] = x128[j+2];
  }

  dft288((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft288((int16_t *)(x2128+288),(int16_t *)(ytmp128+288),1);
  dft288((int16_t *)(x2128+576),(int16_t *)(ytmp128+576),1);

  bfly3_tw1(ytmp128,ytmp128+288,ytmp128+576,y128,y128+288,y128+576);

  for (i=1,j=0; i<288; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+288+i,
          ytmp128+576+i,
          y128+i,
          y128+288+i,
          y128+576+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<864; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};

static int16_t twa900[299*2*4];
static int16_t twb900[299*2*4];

void dft900(int16_t *x,int16_t *y,unsigned char scale_flag)  // 300 x 3
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa900[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb900[0];
  simd_q15_t x2128[900];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[900];//=&ytmp128array3[0];



  for (i=0,j=0; i<300; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+300] = x128[j+1];
    x2128[i+600] = x128[j+2];
  }

  dft300((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft300((int16_t *)(x2128+300),(int16_t *)(ytmp128+300),1);
  dft300((int16_t *)(x2128+600),(int16_t *)(ytmp128+600),1);

  bfly3_tw1(ytmp128,ytmp128+300,ytmp128+600,y128,y128+300,y128+600);

  for (i=1,j=0; i<300; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+300+i,
          ytmp128+600+i,
          y128+i,
          y128+300+i,
          y128+600+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<900; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};


static int16_t twa960[239*2*4];
static int16_t twb960[239*2*4];
static int16_t twc960[239*2*4];


void dft960(int16_t *x,int16_t *y,unsigned char scale_flag)  // 240 x 4
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa960[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb960[0];
  simd_q15_t *twc128=(simd_q15_t *)&twc960[0];
  simd_q15_t x2128[960];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[960];//=&ytmp128array2[0];



  for (i=0,j=0; i<240; i++,j+=4) {
    x2128[i]    = x128[j];
    x2128[i+240] = x128[j+1];
    x2128[i+480] = x128[j+2];
    x2128[i+720] = x128[j+3];
  }

  dft240((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft240((int16_t *)(x2128+240),(int16_t *)(ytmp128+240),1);
  dft240((int16_t *)(x2128+480),(int16_t *)(ytmp128+480),1);
  dft240((int16_t *)(x2128+720),(int16_t *)(ytmp128+720),1);

  bfly4_tw1(ytmp128,ytmp128+240,ytmp128+480,ytmp128+720,y128,y128+240,y128+480,y128+720);

  for (i=1,j=0; i<240; i++,j++) {
    bfly4(ytmp128+i,
          ytmp128+240+i,
          ytmp128+480+i,
          ytmp128+720+i,
          y128+i,
          y128+240+i,
          y128+480+i,
          y128+720+i,
          twa128+j,
          twb128+j,
          twc128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(16384); // dft_norm_table[13]);

    for (i=0; i<960; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};


static int16_t twa972[323*2*4];
static int16_t twb972[323*2*4];

void dft972(int16_t *x,int16_t *y,unsigned char scale_flag)  // 324 x 3
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa972[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb972[0];
  simd_q15_t x2128[972];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[972];//=&ytmp128array3[0];



  for (i=0,j=0; i<324; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+324] = x128[j+1];
    x2128[i+648] = x128[j+2];
  }

  dft324((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft324((int16_t *)(x2128+324),(int16_t *)(ytmp128+324),1);
  dft324((int16_t *)(x2128+648),(int16_t *)(ytmp128+648),1);

  bfly3_tw1(ytmp128,ytmp128+324,ytmp128+648,y128,y128+324,y128+648);

  for (i=1,j=0; i<324; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+324+i,
          ytmp128+648+i,
          y128+i,
          y128+324+i,
          y128+648+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<972; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};

static int16_t twa1080[359*2*4];
static int16_t twb1080[359*2*4];

void dft1080(int16_t *x,int16_t *y,unsigned char scale_flag)  // 360 x 3
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa1080[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb1080[0];
  simd_q15_t x2128[1080];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[1080];//=&ytmp128array3[0];



  for (i=0,j=0; i<360; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+360] = x128[j+1];
    x2128[i+720] = x128[j+2];
  }

  dft360((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft360((int16_t *)(x2128+360),(int16_t *)(ytmp128+360),1);
  dft360((int16_t *)(x2128+720),(int16_t *)(ytmp128+720),1);

  bfly3_tw1(ytmp128,ytmp128+360,ytmp128+720,y128,y128+360,y128+720);

  for (i=1,j=0; i<360; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+360+i,
          ytmp128+720+i,
          y128+i,
          y128+360+i,
          y128+720+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<1080; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};

static int16_t twa1152[287*2*4];
static int16_t twb1152[287*2*4];
static int16_t twc1152[287*2*4];

void dft1152(int16_t *x,int16_t *y,unsigned char scale_flag)  // 288 x 4
{

  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa1152[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb1152[0];
  simd_q15_t *twc128=(simd_q15_t *)&twc1152[0];
  simd_q15_t x2128[1152];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[1152];//=&ytmp128array2[0];



  for (i=0,j=0; i<288; i++,j+=4) {
    x2128[i]    = x128[j];
    x2128[i+288] = x128[j+1];
    x2128[i+576] = x128[j+2];
    x2128[i+864] = x128[j+3];
  }

  dft288((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft288((int16_t *)(x2128+288),(int16_t *)(ytmp128+288),1);
  dft288((int16_t *)(x2128+576),(int16_t *)(ytmp128+576),1);
  dft288((int16_t *)(x2128+864),(int16_t *)(ytmp128+864),1);

  bfly4_tw1(ytmp128,ytmp128+288,ytmp128+576,ytmp128+864,y128,y128+288,y128+576,y128+864);

  for (i=1,j=0; i<288; i++,j++) {
    bfly4(ytmp128+i,
          ytmp128+288+i,
          ytmp128+576+i,
          ytmp128+864+i,
          y128+i,
          y128+288+i,
          y128+576+i,
          y128+864+i,
          twa128+j,
          twb128+j,
          twc128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(16384); // dft_norm_table[13]);

    for (i=0; i<1152; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};

int16_t twa1200[4784];
int16_t twb1200[4784];
int16_t twc1200[4784];

void dft1200(int16_t *x,int16_t *y,unsigned char scale_flag)
{

  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa1200[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb1200[0];
  simd_q15_t *twc128=(simd_q15_t *)&twc1200[0];
  simd_q15_t x2128[1200];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[1200];//=&ytmp128array2[0];



  for (i=0,j=0; i<300; i++,j+=4) {
    x2128[i]    = x128[j];
    x2128[i+300] = x128[j+1];
    x2128[i+600] = x128[j+2];
    x2128[i+900] = x128[j+3];
  }

  dft300((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft300((int16_t *)(x2128+300),(int16_t *)(ytmp128+300),1);
  dft300((int16_t *)(x2128+600),(int16_t *)(ytmp128+600),1);
  dft300((int16_t *)(x2128+900),(int16_t *)(ytmp128+900),1);

  bfly4_tw1(ytmp128,ytmp128+300,ytmp128+600,ytmp128+900,y128,y128+300,y128+600,y128+900);

  for (i=1,j=0; i<300; i++,j++) {
    bfly4(ytmp128+i,
          ytmp128+300+i,
          ytmp128+600+i,
          ytmp128+900+i,
          y128+i,
          y128+300+i,
          y128+600+i,
          y128+900+i,
          twa128+j,
          twb128+j,
          twc128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(16384); // dft_norm_table[13]);
    for (i=0; i<1200; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

}


static int16_t twa1296[431*2*4];
static int16_t twb1296[431*2*4];

void dft1296(int16_t *x,int16_t *y,unsigned char scale_flag) //432 * 3
{

  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa1296[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb1296[0];
  simd_q15_t x2128[1296];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[1296];//=&ytmp128array3[0];



  for (i=0,j=0; i<432; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+432] = x128[j+1];
    x2128[i+864] = x128[j+2];
  }

  dft432((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft432((int16_t *)(x2128+432),(int16_t *)(ytmp128+432),1);
  dft432((int16_t *)(x2128+864),(int16_t *)(ytmp128+864),1);

  bfly3_tw1(ytmp128,ytmp128+432,ytmp128+864,y128,y128+432,y128+864);

  for (i=1,j=0; i<432; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+432+i,
          ytmp128+864+i,
          y128+i,
          y128+432+i,
          y128+864+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<1296; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};


static int16_t twa1440[479*2*4];
static int16_t twb1440[479*2*4];

void dft1440(int16_t *x,int16_t *y,unsigned char scale_flag)  // 480 x 3
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa1440[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb1440[0];
  simd_q15_t x2128[1440];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[1440];//=&ytmp128array3[0];



  for (i=0,j=0; i<480; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+480] = x128[j+1];
    x2128[i+960] = x128[j+2];
  }

  dft480((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft480((int16_t *)(x2128+480),(int16_t *)(ytmp128+480),1);
  dft480((int16_t *)(x2128+960),(int16_t *)(ytmp128+960),1);

  bfly3_tw1(ytmp128,ytmp128+480,ytmp128+960,y128,y128+480,y128+960);

  for (i=1,j=0; i<480; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+480+i,
          ytmp128+960+i,
          y128+i,
          y128+480+i,
          y128+960+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<1440; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};

static int16_t twa1500[2392]__attribute__((aligned(32)));
static int16_t twb1500[2392]__attribute__((aligned(32)));
static int16_t twc1500[2392]__attribute__((aligned(32)));
static int16_t twd1500[2392]__attribute__((aligned(32)));

void dft1500(int16_t *x,int16_t *y,unsigned char scale_flag)
{

  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa1500[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb1500[0];
  simd_q15_t *twc128=(simd_q15_t *)&twc1500[0];
  simd_q15_t *twd128=(simd_q15_t *)&twd1500[0];
  simd_q15_t x2128[1500];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[1500];//=&ytmp128array2[0];



  for (i=0,j=0; i<300; i++,j+=5) {
    x2128[i]    = x128[j];
    x2128[i+300] = x128[j+1];
    x2128[i+600] = x128[j+2];
    x2128[i+900] = x128[j+3];
    x2128[i+1200] = x128[j+4];
  }

  dft300((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft300((int16_t *)(x2128+300),(int16_t *)(ytmp128+300),1);
  dft300((int16_t *)(x2128+600),(int16_t *)(ytmp128+600),1);
  dft300((int16_t *)(x2128+900),(int16_t *)(ytmp128+900),1);
  dft300((int16_t *)(x2128+1200),(int16_t *)(ytmp128+1200),1);

  bfly5_tw1(ytmp128,ytmp128+300,ytmp128+600,ytmp128+900,ytmp128+1200,y128,y128+300,y128+600,y128+900,y128+1200);

  for (i=1,j=0; i<300; i++,j++) {
    bfly5(ytmp128+i,
          ytmp128+300+i,
          ytmp128+600+i,
          ytmp128+900+i,
          ytmp128+1200+i,
          y128+i,
          y128+300+i,
          y128+600+i,
          y128+900+i,
          y128+1200+i,
          twa128+j,
          twb128+j,
          twc128+j,
          twd128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[15]);

    for (i=0; i<1500; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

}

static int16_t twa1620[539*2*4];
static int16_t twb1620[539*2*4];

void dft1620(int16_t *x,int16_t *y,unsigned char scale_flag)  // 540 x 3
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa1620[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb1620[0];
  simd_q15_t x2128[1620];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[1620];//=&ytmp128array3[0];



  for (i=0,j=0; i<540; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+540] = x128[j+1];
    x2128[i+1080] = x128[j+2];
  }

  dft540((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft540((int16_t *)(x2128+540),(int16_t *)(ytmp128+540),1);
  dft540((int16_t *)(x2128+1080),(int16_t *)(ytmp128+1080),1);

  bfly3_tw1(ytmp128,ytmp128+540,ytmp128+1080,y128,y128+540,y128+1080);

  for (i=1,j=0; i<540; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+540+i,
          ytmp128+1080+i,
          y128+i,
          y128+540+i,
          y128+1080+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<1620; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};

static int16_t twa1728[575*2*4];
static int16_t twb1728[575*2*4];

void dft1728(int16_t *x,int16_t *y,unsigned char scale_flag)  // 576 x 3
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa1728[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb1728[0];
  simd_q15_t x2128[1728];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[1728];//=&ytmp128array3[0];



  for (i=0,j=0; i<576; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+576] = x128[j+1];
    x2128[i+1152] = x128[j+2];
  }

  dft576((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft576((int16_t *)(x2128+576),(int16_t *)(ytmp128+576),1);
  dft576((int16_t *)(x2128+1152),(int16_t *)(ytmp128+1152),1);

  bfly3_tw1(ytmp128,ytmp128+576,ytmp128+1152,y128,y128+576,y128+1152);

  for (i=1,j=0; i<576; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+576+i,
          ytmp128+1152+i,
          y128+i,
          y128+576+i,
          y128+1152+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<1728; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};

static int16_t twa1800[599*2*4];
static int16_t twb1800[599*2*4];

void dft1800(int16_t *x,int16_t *y,unsigned char scale_flag)  // 600 x 3
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa1800[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb1800[0];
  simd_q15_t x2128[1800];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[1800];//=&ytmp128array3[0];



  for (i=0,j=0; i<600; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+600] = x128[j+1];
    x2128[i+1200] = x128[j+2];
  }

  dft600((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft600((int16_t *)(x2128+600),(int16_t *)(ytmp128+600),1);
  dft600((int16_t *)(x2128+1200),(int16_t *)(ytmp128+1200),1);

  bfly3_tw1(ytmp128,ytmp128+600,ytmp128+1200,y128,y128+600,y128+1200);

  for (i=1,j=0; i<600; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+600+i,
          ytmp128+1200+i,
          y128+i,
          y128+600+i,
          y128+1200+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<1800; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};

static int16_t twa1920[479*2*4];
static int16_t twb1920[479*2*4];
static int16_t twc1920[479*2*4];

void dft1920(int16_t *x,int16_t *y,unsigned char scale_flag)  // 480 x 4
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa1920[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb1920[0];
  simd_q15_t *twc128=(simd_q15_t *)&twc1920[0];
  simd_q15_t x2128[1920];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[1920];//=&ytmp128array2[0];



  for (i=0,j=0; i<480; i++,j+=4) {
    x2128[i]    = x128[j];
    x2128[i+480] = x128[j+1];
    x2128[i+960] = x128[j+2];
    x2128[i+1440] = x128[j+3];
  }

  dft480((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft480((int16_t *)(x2128+480),(int16_t *)(ytmp128+480),1);
  dft480((int16_t *)(x2128+960),(int16_t *)(ytmp128+960),1);
  dft480((int16_t *)(x2128+1440),(int16_t *)(ytmp128+1440),1);

  bfly4_tw1(ytmp128,ytmp128+480,ytmp128+960,ytmp128+1440,y128,y128+480,y128+960,y128+1440);

  for (i=1,j=0; i<480; i++,j++) {
    bfly4(ytmp128+i,
          ytmp128+480+i,
          ytmp128+960+i,
          ytmp128+1440+i,
          y128+i,
          y128+480+i,
          y128+960+i,
          y128+1440+i,
          twa128+j,
          twb128+j,
          twc128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[13]);
    for (i=0; i<1920; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};

static int16_t twa1944[647*2*4];
static int16_t twb1944[647*2*4];

void dft1944(int16_t *x,int16_t *y,unsigned char scale_flag)  // 648 x 3
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa1944[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb1944[0];
  simd_q15_t x2128[1944];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[1944];//=&ytmp128array3[0];



  for (i=0,j=0; i<648; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+648] = x128[j+1];
    x2128[i+1296] = x128[j+2];
  }

  dft648((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft648((int16_t *)(x2128+648),(int16_t *)(ytmp128+648),1);
  dft648((int16_t *)(x2128+1296),(int16_t *)(ytmp128+1296),1);

  bfly3_tw1(ytmp128,ytmp128+648,ytmp128+1296,y128,y128+648,y128+1296);

  for (i=1,j=0; i<648; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+648+i,
          ytmp128+1296+i,
          y128+i,
          y128+648+i,
          y128+1296+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<1944; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};

static int16_t twa2160[719*2*4];
static int16_t twb2160[719*2*4];

void dft2160(int16_t *x,int16_t *y,unsigned char scale_flag)  // 720 x 3
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa2160[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb2160[0];
  simd_q15_t x2128[2160];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[2160];//=&ytmp128array3[0];



  for (i=0,j=0; i<720; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+720] = x128[j+1];
    x2128[i+1440] = x128[j+2];
  }

  dft720((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft720((int16_t *)(x2128+720),(int16_t *)(ytmp128+720),1);
  dft720((int16_t *)(x2128+1440),(int16_t *)(ytmp128+1440),1);

  bfly3_tw1(ytmp128,ytmp128+720,ytmp128+1440,y128,y128+720,y128+1440);

  for (i=1,j=0; i<720; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+720+i,
          ytmp128+1440+i,
          y128+i,
          y128+720+i,
          y128+1440+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<2160; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};

static int16_t twa2304[767*2*4];
static int16_t twb2304[767*2*4];

void dft2304(int16_t *x,int16_t *y,unsigned char scale_flag)  // 768 x 3
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa2304[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb2304[0];
  simd_q15_t x2128[2304];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[2304];//=&ytmp128array3[0];



  for (i=0,j=0; i<768; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+768] = x128[j+1];
    x2128[i+1536] = x128[j+2];
  }

  dft768((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft768((int16_t *)(x2128+768),(int16_t *)(ytmp128+768),1);
  dft768((int16_t *)(x2128+1536),(int16_t *)(ytmp128+1536),1);

  bfly3_tw1(ytmp128,ytmp128+768,ytmp128+1536,y128,y128+768,y128+1536);

  for (i=1,j=0; i<768; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+768+i,
          ytmp128+1536+i,
          y128+i,
          y128+768+i,
          y128+1536+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<2304; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};

static int16_t twa2400[599*2*4];
static int16_t twb2400[599*2*4];
static int16_t twc2400[599*2*4];

void dft2400(int16_t *x,int16_t *y,unsigned char scale_flag)  // 600 x 4
{

  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa2400[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb2400[0];
  simd_q15_t *twc128=(simd_q15_t *)&twc2400[0];
  simd_q15_t x2128[2400];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[2400];//=&ytmp128array2[0];



  for (i=0,j=0; i<600; i++,j+=4) {
    x2128[i]    = x128[j];
    x2128[i+600] = x128[j+1];
    x2128[i+1200] = x128[j+2];
    x2128[i+1800] = x128[j+3];
  }

  dft600((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft600((int16_t *)(x2128+600),(int16_t *)(ytmp128+600),1);
  dft600((int16_t *)(x2128+1200),(int16_t *)(ytmp128+1200),1);
  dft600((int16_t *)(x2128+1800),(int16_t *)(ytmp128+1800),1);

  bfly4_tw1(ytmp128,ytmp128+600,ytmp128+1200,ytmp128+1800,y128,y128+600,y128+1200,y128+1800);

  for (i=1,j=0; i<600; i++,j++) {
    bfly4(ytmp128+i,
          ytmp128+600+i,
          ytmp128+1200+i,
          ytmp128+1800+i,
          y128+i,
          y128+600+i,
          y128+1200+i,
          y128+1800+i,
          twa128+j,
          twb128+j,
          twc128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[13]);
    for (i=0; i<2400; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};

static int16_t twa2592[863*2*4];
static int16_t twb2592[863*2*4];

void dft2592(int16_t *x,int16_t *y,unsigned char scale_flag)  // 864 x 3
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa2592[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb2592[0];
  simd_q15_t x2128[2592];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[2592];//=&ytmp128array3[0];



  for (i=0,j=0; i<864; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+864] = x128[j+1];
    x2128[i+1728] = x128[j+2];
  }

  dft864((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft864((int16_t *)(x2128+864),(int16_t *)(ytmp128+864),1);
  dft864((int16_t *)(x2128+1728),(int16_t *)(ytmp128+1728),1);

  bfly3_tw1(ytmp128,ytmp128+864,ytmp128+1728,y128,y128+864,y128+1728);

  for (i=1,j=0; i<864; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+864+i,
          ytmp128+1728+i,
          y128+i,
          y128+864+i,
          y128+1728+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<2592; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};

static int16_t twa2700[899*2*4];
static int16_t twb2700[899*2*4];

void dft2700(int16_t *x,int16_t *y,unsigned char scale_flag)  // 900 x 3
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa2700[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb2700[0];
  simd_q15_t x2128[2700];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[2700];//=&ytmp128array3[0];



  for (i=0,j=0; i<900; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+900] = x128[j+1];
    x2128[i+1800] = x128[j+2];
  }

  dft900((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft900((int16_t *)(x2128+900),(int16_t *)(ytmp128+900),1);
  dft900((int16_t *)(x2128+1800),(int16_t *)(ytmp128+1800),1);

  bfly3_tw1(ytmp128,ytmp128+900,ytmp128+1800,y128,y128+900,y128+1800);

  for (i=1,j=0; i<900; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+900+i,
          ytmp128+1800+i,
          y128+i,
          y128+900+i,
          y128+1800+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<2700; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};

static int16_t twa2880[959*2*4];
static int16_t twb2880[959*2*4];

void dft2880(int16_t *x,int16_t *y,unsigned char scale_flag)  // 960 x 3
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa2880[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb2880[0];
  simd_q15_t x2128[2880];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[2880];//=&ytmp128array3[0];



  for (i=0,j=0; i<960; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+960] = x128[j+1];
    x2128[i+1920] = x128[j+2];
  }

  dft960((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft960((int16_t *)(x2128+960),(int16_t *)(ytmp128+960),1);
  dft960((int16_t *)(x2128+1920),(int16_t *)(ytmp128+1920),1);

  bfly3_tw1(ytmp128,ytmp128+960,ytmp128+1920,y128,y128+960,y128+1920);

  for (i=1,j=0; i<960; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+960+i,
          ytmp128+1920+i,
          y128+i,
          y128+960+i,
          y128+1920+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<2880; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};

static int16_t twa2916[971*2*4];
static int16_t twb2916[971*2*4];

void dft2916(int16_t *x,int16_t *y,unsigned char scale_flag)  // 972 x 3
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa2916[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb2916[0];
  simd_q15_t x2128[2916];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[2916];//=&ytmp128array3[0];



  for (i=0,j=0; i<972; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+972] = x128[j+1];
    x2128[i+1944] = x128[j+2];
  }

  dft972((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft972((int16_t *)(x2128+972),(int16_t *)(ytmp128+972),1);
  dft972((int16_t *)(x2128+1944),(int16_t *)(ytmp128+1944),1);

  bfly3_tw1(ytmp128,ytmp128+972,ytmp128+1944,y128,y128+972,y128+1944);

  for (i=1,j=0; i<972; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+972+i,
          ytmp128+1944+i,
          y128+i,
          y128+972+i,
          y128+1944+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<2916; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};

static int16_t twa3000[599*8]__attribute__((aligned(32)));
static int16_t twb3000[599*8]__attribute__((aligned(32)));
static int16_t twc3000[599*8]__attribute__((aligned(32)));
static int16_t twd3000[599*8]__attribute__((aligned(32)));

void dft3000(int16_t *x,int16_t *y,unsigned char scale_flag) // 600 * 5
{

  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa3000[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb3000[0];
  simd_q15_t *twc128=(simd_q15_t *)&twc3000[0];
  simd_q15_t *twd128=(simd_q15_t *)&twd3000[0];
  simd_q15_t x2128[3000];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[3000];//=&ytmp128array2[0];



  for (i=0,j=0; i<600; i++,j+=5) {
    x2128[i]    = x128[j];
    x2128[i+600] = x128[j+1];
    x2128[i+1200] = x128[j+2];
    x2128[i+1800] = x128[j+3];
    x2128[i+2400] = x128[j+4];
  }

  dft600((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft600((int16_t *)(x2128+600),(int16_t *)(ytmp128+600),1);
  dft600((int16_t *)(x2128+1200),(int16_t *)(ytmp128+1200),1);
  dft600((int16_t *)(x2128+1800),(int16_t *)(ytmp128+1800),1);
  dft600((int16_t *)(x2128+2400),(int16_t *)(ytmp128+2400),1);

  bfly5_tw1(ytmp128,ytmp128+600,ytmp128+1200,ytmp128+1800,ytmp128+2400,y128,y128+600,y128+1200,y128+1800,y128+2400);

  for (i=1,j=0; i<600; i++,j++) {
    bfly5(ytmp128+i,
          ytmp128+600+i,
          ytmp128+1200+i,
          ytmp128+1800+i,
          ytmp128+2400+i,
          y128+i,
          y128+600+i,
          y128+1200+i,
          y128+1800+i,
          y128+2400+i,
          twa128+j,
          twb128+j,
          twc128+j,
          twd128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[15]);

    for (i=0; i<3000; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

}

static int16_t twa3240[1079*2*4];
static int16_t twb3240[1079*2*4];

void dft3240(int16_t *x,int16_t *y,unsigned char scale_flag)  // 1080 x 3
{
  int i,j;
  simd_q15_t *x128=(simd_q15_t *)x;
  simd_q15_t *y128=(simd_q15_t *)y;
  simd_q15_t *twa128=(simd_q15_t *)&twa3240[0];
  simd_q15_t *twb128=(simd_q15_t *)&twb3240[0];
  simd_q15_t x2128[3240];// = (simd_q15_t *)&x2128array[0];
  simd_q15_t ytmp128[3240];//=&ytmp128array3[0];



  for (i=0,j=0; i<1080; i++,j+=3) {
    x2128[i]    = x128[j];
    x2128[i+1080] = x128[j+1];
    x2128[i+2160] = x128[j+2];
  }

  dft1080((int16_t *)x2128,(int16_t *)ytmp128,1);
  dft1080((int16_t *)(x2128+1080),(int16_t *)(ytmp128+1080),1);
  dft1080((int16_t *)(x2128+2160),(int16_t *)(ytmp128+2160),1);

  bfly3_tw1(ytmp128,ytmp128+1080,ytmp128+2160,y128,y128+1080,y128+2160);

  for (i=1,j=0; i<1080; i++,j++) {
    bfly3(ytmp128+i,
          ytmp128+1080+i,
          ytmp128+2160+i,
          y128+i,
          y128+1080+i,
          y128+2160+i,
          twa128+j,
          twb128+j);
  }

  if (scale_flag==1) {
    const simd_q15_t norm128 = set1_int16(dft_norm_table[14]);

    for (i=0; i<3240; i++) {
      y128[i] = mulhi_int16(y128[i],norm128);
    }
  }

};

void init_rad4(int N,int16_t *tw) {

  int16_t *twa = tw;
  int16_t *twb = twa+(N/2);
  int16_t *twc = twb+(N/2);
  int i;

  for (i=0;i<(N/4);i++) {
    *twa = (int16_t)round(32767.0*cos(2*M_PI*i/N)); twa++;
    *twa = -(int16_t)round(32767.0*sin(2*M_PI*i/N)); twa++;
    *twb = (int16_t)round(32767.0*cos(2*M_PI*2*i/N)); twb++;
    *twb = -(int16_t)round(32767.0*sin(2*M_PI*2*i/N)); twb++;
    *twc = (int16_t)round(32767.0*cos(2*M_PI*3*i/N)); twc++;
    *twc = -(int16_t)round(32767.0*sin(2*M_PI*3*i/N)); twc++;
  }
}
void init_rad4_rep(int N,int16_t *twa,int16_t *twb,int16_t *twc) {

  int i,j;

  for (i=1;i<(N/4);i++) {
    twa[0] = (int16_t)round(32767.0*cos(2*M_PI*i/N));
    twa[1] = -(int16_t)round(32767.0*sin(2*M_PI*i/N));
    twb[0] = (int16_t)round(32767.0*cos(2*M_PI*2*i/N));
    twb[1] = -(int16_t)round(32767.0*sin(2*M_PI*2*i/N));
    twc[0] = (int16_t)round(32767.0*cos(2*M_PI*3*i/N));
    twc[1] = -(int16_t)round(32767.0*sin(2*M_PI*3*i/N));
    for (j=1;j<4;j++) {
      ((int32_t*)twa)[j]=((int32_t*)twa)[0];
      ((int32_t*)twb)[j]=((int32_t*)twb)[0];
      ((int32_t*)twc)[j]=((int32_t*)twc)[0];
    }
    twa+=8;
    twb+=8;
    twc+=8;
  }
}

void init_rad2(int N,int16_t *tw) {

  int16_t *twa = tw;
  int i;

  for (i=0;i<(N>>1);i++) {
    *twa = (int16_t)round(32767.0*cos(2*M_PI*i/N)); twa++;
    *twa = -(int16_t)round(32767.0*sin(2*M_PI*i/N)); twa++;
  }
}

void init_rad2_rep(int N,int16_t *twa) {

  int i,j;

  for (i=1;i<(N/2);i++) {
    twa[0] = (int16_t)round(32767.0*cos(2*M_PI*i/N));
    twa[1] = -(int16_t)round(32767.0*sin(2*M_PI*i/N));
    for (j=1;j<4;j++) {
      ((int32_t*)twa)[j]=((int32_t*)twa)[0];
    }
    twa+=8;
  }
}

void init_rad3(int N,int16_t *twa,int16_t *twb) {

  int i;

  for (i=0;i<(N/3);i++) {
    *twa = (int16_t)round(32767.0*cos(2*M_PI*i/N)); twa++;
    *twa = -(int16_t)round(32767.0*sin(2*M_PI*i/N)); twa++;
    *twb = (int16_t)round(32767.0*cos(2*M_PI*2*i/N)); twb++;
    *twb = -(int16_t)round(32767.0*sin(2*M_PI*2*i/N)); twb++;
  }
}

void init_rad3_rep(int N,int16_t *twa,int16_t *twb) {

  int i,j;

  for (i=1;i<(N/3);i++) {
    twa[0] = (int16_t)round(32767.0*cos(2*M_PI*i/N));
    twa[1] = -(int16_t)round(32767.0*sin(2*M_PI*i/N));
    twb[0] = (int16_t)round(32767.0*cos(2*M_PI*2*i/N));
    twb[1] = -(int16_t)round(32767.0*sin(2*M_PI*2*i/N));
    for (j=1;j<4;j++) {
      ((int32_t*)twa)[j]=((int32_t*)twa)[0];
      ((int32_t*)twb)[j]=((int32_t*)twb)[0];
    }
    twa+=8;
    twb+=8;
  }
}

void init_rad5_rep(int N,int16_t *twa,int16_t *twb,int16_t *twc,int16_t *twd) {

  int i,j;

  for (i=1;i<(N/5);i++) {
    twa[0] = (int16_t)round(32767.0*cos(2*M_PI*i/N));
    twa[1] = -(int16_t)round(32767.0*sin(2*M_PI*i/N));
    twb[0] = (int16_t)round(32767.0*cos(2*M_PI*2*i/N));
    twb[1] = -(int16_t)round(32767.0*sin(2*M_PI*2*i/N));
    twc[0] = (int16_t)round(32767.0*cos(2*M_PI*3*i/N));
    twc[1] = -(int16_t)round(32767.0*sin(2*M_PI*3*i/N));
    twd[0] = (int16_t)round(32767.0*cos(2*M_PI*4*i/N));
    twd[1] = -(int16_t)round(32767.0*sin(2*M_PI*4*i/N));
    for (j=1;j<4;j++) {
      ((int32_t*)twa)[j]=((int32_t*)twa)[0];
      ((int32_t*)twb)[j]=((int32_t*)twb)[0];
      ((int32_t*)twc)[j]=((int32_t*)twc)[0];
      ((int32_t*)twd)[j]=((int32_t*)twd)[0];
    }
    twa+=8;
    twb+=8;
    twc+=8;
    twd+=8;
  }
}
/*----------------------------------------------------------------*/
/* dft library entry points:                                      */

int dfts_autoinit(void)
{

  init_rad2(32768,tw32768);
  init_rad4(65536,tw65536);


  init_rad3(18432,twa18432,twb18432);
  init_rad3(24576,twa24576,twb24576);
  init_rad3(36864,twa36864,twb36864);
  init_rad3(49152,twa49152,twb49152);
  init_rad3(98304,twa98304,twb98304);


  init_rad2_rep(24,tw24);
  init_rad3_rep(36,twa36,twb36);
  init_rad4_rep(48,twa48,twb48,twc48);
  init_rad5_rep(60,twa60,twb60,twc60,twd60);
  init_rad2_rep(72,tw72);
  init_rad2_rep(96,tw96);
  init_rad3_rep(108,twa108,twb108);
  init_rad2_rep(120,tw120);
  init_rad3_rep(144,twa144,twb144);
  init_rad3_rep(180,twa180,twb180);
  init_rad3_rep(216,twa216,twb216);
  init_rad4_rep(240,twa240,twb240,twc240);
  init_rad3_rep(288,twa288,twb288);
  init_rad5_rep(300,twa300,twb300,twc300,twd300);
  init_rad3_rep(324,twa324,twb324);
  init_rad3_rep(360,twa360,twb360);
  init_rad4_rep(432,twa432,twb432,twc432);
  init_rad4_rep(480,twa480,twb480,twc480);
  init_rad3_rep(540,twa540,twb540);
  init_rad3_rep(576,twa576,twb576);
  init_rad2_rep(600,twa600);
  init_rad3_rep(648,twa648,twb648);
  init_rad4_rep(720,twa720,twb720,twc720);
  init_rad3_rep(864,twa864,twb864);
  init_rad3_rep(900,twa900,twb900);
  init_rad4_rep(960,twa960,twb960,twc960);
  init_rad3_rep(972,twa972,twb972);
  init_rad3_rep(1080,twa1080,twb1080);
  init_rad4_rep(1152,twa1152,twb1152,twc1152);
  init_rad4_rep(1200,twa1200,twb1200,twc1200);
  init_rad3_rep(1296,twa1296,twb1296);
  init_rad3_rep(1440,twa1440,twb1440);
  init_rad5_rep(1500,twa1500,twb1500,twc1500,twd1500);
  init_rad3_rep(1620,twa1620,twb1620);
  init_rad3_rep(1728,twa1728,twb1728);
  init_rad3_rep(1800,twa1800,twb1800);
  init_rad4_rep(1920,twa1920,twb1920, twc1920);
  init_rad3_rep(1944,twa1944,twb1944);
  init_rad3_rep(2160,twa2160,twb2160);
  init_rad3_rep(2304,twa2304,twb2304);
  init_rad4_rep(2400,twa2400,twb2400,twc2400);
  init_rad3_rep(2592,twa2592,twb2592);
  init_rad3_rep(2700,twa2700,twb2700);
  init_rad3_rep(2880,twa2880,twb2880);
  init_rad3_rep(2916,twa2916,twb2916);
  init_rad5_rep(3000,twa3000,twb3000,twc3000,twd3000);
  init_rad3_rep(3240,twa3240,twb3240);

  return 0;
}



#ifndef MR_MAIN

void dft_implementation(uint8_t sizeidx, int16_t *input, int16_t *output, unsigned char scale_flag)
{
  AssertFatal((sizeidx >= 0 && sizeidx<DFT_SIZE_IDXTABLESIZE),"Invalid dft size index %i\n",sizeidx);
        int algn=0xF;
        if ( (dft_ftab[sizeidx].size%3) != 0 ) // there is no AVX2 implementation for multiples of 3 DFTs
          algn=0x1F;
        AssertFatal(((intptr_t)output&algn)==0,"Buffers should be aligned %p",output);
        if (((intptr_t)input)&algn) {
          LOG_D(PHY, "DFT called with input not aligned, add a memcpy, size %d\n", sizeidx);
          int sz=dft_ftab[sizeidx].size;
          if (sizeidx==DFT_12) // This case does 8 DFTs in //
            sz*=8;
          int16_t tmp[sz*2] __attribute__ ((aligned(32))); // input and output are not in right type (int16_t instead of c16_t)
          memcpy(tmp, input, sizeof tmp);
          dft_ftab[sizeidx].func(tmp,output,scale_flag);
        } else
          dft_ftab[sizeidx].func(input,output,scale_flag);
};

void idft_implementation(uint8_t sizeidx, int16_t *input, int16_t *output, unsigned char scale_flag)
{
  AssertFatal((sizeidx>=0 && sizeidx<DFT_SIZE_IDXTABLESIZE),"Invalid idft size index %i\n",sizeidx);
        int algn=0xF;
	algn=0x1F;
        AssertFatal( ((intptr_t)output&algn)==0,"Buffers should be 16 bytes aligned %p",output);
        if (((intptr_t)input)&algn ) {  
          LOG_D(PHY, "DFT called with input not aligned, add a memcpy\n");
          int sz=idft_ftab[sizeidx].size;
          int16_t tmp[sz*2] __attribute__ ((aligned(32))); // input and output are not in right type (int16_t instead of c16_t)
          memcpy(tmp, input, sizeof tmp);
          idft_ftab[sizeidx].func(tmp,output,scale_flag);
        } else
          idft_ftab[sizeidx].func(input,output,scale_flag);
};

#endif

/*---------------------------------------------------------------------------------------*/

#ifdef MR_MAIN
#include <string.h>
#include <stdio.h>

#define LOG_M write_output
int write_output(const char *fname,const char *vname,void *data,int length,int dec,char format)
{

  FILE *fp=NULL;
  int i;


  printf("Writing %d elements of type %d to %s\n",length,format,fname);


  if (format == 10 || format ==11 || format == 12 || format == 13 || format == 14) {
    fp = fopen(fname,"a+");
  } else if (format != 10 && format !=11  && format != 12 && format != 13 && format != 14) {
    fp = fopen(fname,"w+");
  }



  if (fp== NULL) {
    printf("[OPENAIR][FILE OUTPUT] Cannot open file %s\n",fname);
    return(-1);
  }

  if (format != 10 && format !=11  && format != 12 && format != 13 && format != 14)
    fprintf(fp,"%s = [",vname);


  switch (format) {
  case 0:   // real 16-bit

    for (i=0; i<length; i+=dec) {
      fprintf(fp,"%d\n",((short *)data)[i]);
    }

    break;

  case 1:  // complex 16-bit
  case 13:
  case 14:
  case 15:

    for (i=0; i<length<<1; i+=(2*dec)) {
      fprintf(fp,"%d + j*(%d)\n",((short *)data)[i],((short *)data)[i+1]);

    }


    break;

  case 2:  // real 32-bit
    for (i=0; i<length; i+=dec) {
      fprintf(fp,"%d\n",((int *)data)[i]);
    }

    break;

  case 3: // complex 32-bit
    for (i=0; i<length<<1; i+=(2*dec)) {
      fprintf(fp,"%d + j*(%d)\n",((int *)data)[i],((int *)data)[i+1]);
    }

    break;

  case 4: // real 8-bit
    for (i=0; i<length; i+=dec) {
      fprintf(fp,"%d\n",((char *)data)[i]);
    }

    break;

  case 5: // complex 8-bit
    for (i=0; i<length<<1; i+=(2*dec)) {
      fprintf(fp,"%d + j*(%d)\n",((char *)data)[i],((char *)data)[i+1]);
    }

    break;

  case 6:  // real 64-bit
    for (i=0; i<length; i+=dec) {
      fprintf(fp,"%lld\n",((long long*)data)[i]);
    }

    break;

  case 7: // real double
    for (i=0; i<length; i+=dec) {
      fprintf(fp,"%g\n",((double *)data)[i]);
    }

    break;

  case 8: // complex double
    for (i=0; i<length<<1; i+=2*dec) {
      fprintf(fp,"%g + j*(%g)\n",((double *)data)[i], ((double *)data)[i+1]);
    }

    break;

  case 9: // real unsigned 8-bit
    for (i=0; i<length; i+=dec) {
      fprintf(fp,"%d\n",((unsigned char *)data)[i]);
    }

    break;


  case 10 : // case eren 16 bit complex :

    for (i=0; i<length<<1; i+=(2*dec)) {

      if((i < 2*(length-1)) && (i > 0))
        fprintf(fp,"%d + j*(%d),",((short *)data)[i],((short *)data)[i+1]);
      else if (i == 2*(length-1))
        fprintf(fp,"%d + j*(%d);",((short *)data)[i],((short *)data)[i+1]);
      else if (i == 0)
        fprintf(fp,"\n%d + j*(%d),",((short *)data)[i],((short *)data)[i+1]);



    }

    break;

  case 11 : //case eren 16 bit real for channel magnitudes:
    for (i=0; i<length; i+=dec) {

      if((i <(length-1))&& (i > 0))
        fprintf(fp,"%d,",((short *)data)[i]);
      else if (i == (length-1))
        fprintf(fp,"%d;",((short *)data)[i]);
      else if (i == 0)
        fprintf(fp,"\n%d,",((short *)data)[i]);
    }

    printf("\n erennnnnnnnnnnnnnn: length :%d",length);
    break;

  case 12 : // case eren for log2_maxh real unsigned 8 bit
    fprintf(fp,"%d \n",((unsigned char *)&data)[0]);
    break;

  }

  if (format != 10 && format !=11 && format !=12 && format != 13 && format != 15) {
    fprintf(fp,"];\n");
    fclose(fp);
    return(0);
  } else if (format == 10 || format ==11 || format == 12 || format == 13 || format == 15) {
    fclose(fp);
    return(0);
  }

  return 0;
}


int main(int argc, char**argv)
{


  time_stats_t ts;
  simd256_q15_t x[16384],x2[16384],y[16384],tw0,tw1,tw2,tw3;
  int i;
  simd_q15_t *x128=(simd_q15_t*)x,*y128=(simd_q15_t*)y;

  dfts_autoinit();

  set_taus_seed(0);
  cpu_meas_enabled = 1;
  /*
     ((int16_t *)&tw0)[0] = 32767;
     ((int16_t *)&tw0)[1] = 0;
     ((int16_t *)&tw0)[2] = 32767;
     ((int16_t *)&tw0)[3] = 0;
     ((int16_t *)&tw0)[4] = 32767;
     ((int16_t *)&tw0)[5] = 0;
     ((int16_t *)&tw0)[6] = 32767;
     ((int16_t *)&tw0)[7] = 0;

     ((int16_t *)&tw1)[0] = 32767;
     ((int16_t *)&tw1)[1] = 0;
     ((int16_t *)&tw1)[2] = 32767;
     ((int16_t *)&tw1)[3] = 0;
     ((int16_t *)&tw1)[4] = 32767;
     ((int16_t *)&tw1)[5] = 0;
     ((int16_t *)&tw1)[6] = 32767;
     ((int16_t *)&tw1)[7] = 0;

     ((int16_t *)&tw2)[0] = 32767;
     ((int16_t *)&tw2)[1] = 0;
     ((int16_t *)&tw2)[2] = 32767;
     ((int16_t *)&tw2)[3] = 0;
     ((int16_t *)&tw2)[4] = 32767;
     ((int16_t *)&tw2)[5] = 0;
     ((int16_t *)&tw2)[6] = 32767;
     ((int16_t *)&tw2)[7] = 0;

     ((int16_t *)&tw3)[0] = 32767;
     ((int16_t *)&tw3)[1] = 0;
     ((int16_t *)&tw3)[2] = 32767;
     ((int16_t *)&tw3)[3] = 0;
     ((int16_t *)&tw3)[4] = 32767;
     ((int16_t *)&tw3)[5] = 0;
     ((int16_t *)&tw3)[6] = 32767;
     ((int16_t *)&tw3)[7] = 0;
  */
  for (i = 0; i < 300; i++) {
    x[i] = simde_mm256_set1_epi32(taus());
    x[i] = simde_mm256_srai_epi16(x[i], 4);
    }
      /*
    bfly2_tw1(x,x+1,y,y+1);
    printf("(%d,%d) (%d,%d) => (%d,%d) (%d,%d)\n",((int16_t*)&x[0])[0],((int16_t*)&x[0])[1],((int16_t*)&x[1])[0],((int16_t*)&x[1])[1],((int16_t*)&y[0])[0],((int16_t*)&y[0])[1],((int16_t*)&y[1])[0],((int16_t*)&y[1])[1]);
    printf("(%d,%d) (%d,%d) => (%d,%d) (%d,%d)\n",((int16_t*)&x[0])[0],((int16_t*)&x[0])[1],((int16_t*)&x[1])[0],((int16_t*)&x[1])[1],((int16_t*)&y[0])[2],((int16_t*)&y[0])[3],((int16_t*)&y[1])[2],((int16_t*)&y[1])[3]);
    printf("(%d,%d) (%d,%d) => (%d,%d) (%d,%d)\n",((int16_t*)&x[0])[0],((int16_t*)&x[0])[1],((int16_t*)&x[1])[0],((int16_t*)&x[1])[1],((int16_t*)&y[0])[4],((int16_t*)&y[0])[5],((int16_t*)&y[1])[4],((int16_t*)&y[1])[5]);
    printf("(%d,%d) (%d,%d) => (%d,%d) (%d,%d)\n",((int16_t*)&x[0])[0],((int16_t*)&x[0])[1],((int16_t*)&x[1])[0],((int16_t*)&x[1])[1],((int16_t*)&y[0])[6],((int16_t*)&y[0])[7],((int16_t*)&y[1])[6],((int16_t*)&y[1])[7]);
    bfly2(x,x+1,y,y+1, &tw0);
    printf("0(%d,%d) (%d,%d) => (%d,%d) (%d,%d)\n",((int16_t*)&x[0])[0],((int16_t*)&x[0])[1],((int16_t*)&x[1])[0],((int16_t*)&x[1])[1],((int16_t*)&y[0])[0],((int16_t*)&y[0])[1],((int16_t*)&y[1])[0],((int16_t*)&y[1])[1]);
    printf("1(%d,%d) (%d,%d) => (%d,%d) (%d,%d)\n",((int16_t*)&x[0])[0],((int16_t*)&x[0])[1],((int16_t*)&x[1])[0],((int16_t*)&x[1])[1],((int16_t*)&y[0])[2],((int16_t*)&y[0])[3],((int16_t*)&y[1])[2],((int16_t*)&y[1])[3]);
    printf("2(%d,%d) (%d,%d) => (%d,%d) (%d,%d)\n",((int16_t*)&x[0])[0],((int16_t*)&x[0])[1],((int16_t*)&x[1])[0],((int16_t*)&x[1])[1],((int16_t*)&y[0])[4],((int16_t*)&y[0])[5],((int16_t*)&y[1])[4],((int16_t*)&y[1])[5]);
    printf("3(%d,%d) (%d,%d) => (%d,%d) (%d,%d)\n",((int16_t*)&x[0])[0],((int16_t*)&x[0])[1],((int16_t*)&x[1])[0],((int16_t*)&x[1])[1],((int16_t*)&y[0])[6],((int16_t*)&y[0])[7],((int16_t*)&y[1])[6],((int16_t*)&y[1])[7]);
    bfly2(x,x+1,y,y+1, &tw0);

    bfly3_tw1(x,x+1,x+2,y, y+1,y+2);
    printf("0(%d,%d) (%d,%d) (%d %d) => (%d,%d) (%d,%d) (%d %d)\n",((int16_t*)&x[0])[0],((int16_t*)&x[0])[1],((int16_t*)&x[1])[0],((int16_t*)&x[1])[1],((int16_t*)&x[2])[0],((int16_t*)&x[2])[1],((int16_t*)&y[0])[0],((int16_t*)&y[0])[1],((int16_t*)&y[1])[0],((int16_t*)&y[1])[1],((int16_t*)&y[2])[0],((int16_t*)&y[2])[1]);
    printf("1(%d,%d) (%d,%d) (%d %d) => (%d,%d) (%d,%d) (%d %d)\n",((int16_t*)&x[0])[0],((int16_t*)&x[0])[1],((int16_t*)&x[1])[0],((int16_t*)&x[1])[1],((int16_t*)&x[2])[0],((int16_t*)&x[2])[1],((int16_t*)&y[0])[2],((int16_t*)&y[0])[3],((int16_t*)&y[1])[2],((int16_t*)&y[1])[3],((int16_t*)&y[2])[2],((int16_t*)&y[2])[3]);
    printf("2(%d,%d) (%d,%d) (%d %d) => (%d,%d) (%d,%d) (%d %d)\n",((int16_t*)&x[0])[0],((int16_t*)&x[0])[1],((int16_t*)&x[1])[0],((int16_t*)&x[1])[1],((int16_t*)&x[2])[0],((int16_t*)&x[2])[1],((int16_t*)&y[0])[4],((int16_t*)&y[0])[5],((int16_t*)&y[1])[4],((int16_t*)&y[1])[5],((int16_t*)&y[2])[4],((int16_t*)&y[2])[5]);
    printf("3(%d,%d) (%d,%d) (%d %d) => (%d,%d) (%d,%d) (%d %d)\n",((int16_t*)&x[0])[0],((int16_t*)&x[0])[1],((int16_t*)&x[1])[0],((int16_t*)&x[1])[1],((int16_t*)&x[2])[0],((int16_t*)&x[2])[1],((int16_t*)&y[0])[6],((int16_t*)&y[0])[7],((int16_t*)&y[1])[6],((int16_t*)&y[1])[7],((int16_t*)&y[2])[6],((int16_t*)&y[2])[7]);
    bfly3(x,x+1,x+2,y, y+1,y+2,&tw0,&tw1);

    printf("0(%d,%d) (%d,%d) (%d %d) => (%d,%d) (%d,%d) (%d %d)\n",((int16_t*)&x[0])[0],((int16_t*)&x[0])[1],((int16_t*)&x[1])[0],((int16_t*)&x[1])[1],((int16_t*)&x[2])[0],((int16_t*)&x[2])[1],((int16_t*)&y[0])[0],((int16_t*)&y[0])[1],((int16_t*)&y[1])[0],((int16_t*)&y[1])[1],((int16_t*)&y[2])[0],((int16_t*)&y[2])[1]);
    printf("1(%d,%d) (%d,%d) (%d %d) => (%d,%d) (%d,%d) (%d %d)\n",((int16_t*)&x[0])[0],((int16_t*)&x[0])[1],((int16_t*)&x[1])[0],((int16_t*)&x[1])[1],((int16_t*)&x[2])[0],((int16_t*)&x[2])[1],((int16_t*)&y[0])[2],((int16_t*)&y[0])[3],((int16_t*)&y[1])[2],((int16_t*)&y[1])[3],((int16_t*)&y[2])[2],((int16_t*)&y[2])[3]);
    printf("2(%d,%d) (%d,%d) (%d %d) => (%d,%d) (%d,%d) (%d %d)\n",((int16_t*)&x[0])[0],((int16_t*)&x[0])[1],((int16_t*)&x[1])[0],((int16_t*)&x[1])[1],((int16_t*)&x[2])[0],((int16_t*)&x[2])[1],((int16_t*)&y[0])[4],((int16_t*)&y[0])[5],((int16_t*)&y[1])[4],((int16_t*)&y[1])[5],((int16_t*)&y[2])[4],((int16_t*)&y[2])[5]);
    printf("3(%d,%d) (%d,%d) (%d %d) => (%d,%d) (%d,%d) (%d %d)\n",((int16_t*)&x[0])[0],((int16_t*)&x[0])[1],((int16_t*)&x[1])[0],((int16_t*)&x[1])[1],((int16_t*)&x[2])[0],((int16_t*)&x[2])[1],((int16_t*)&y[0])[6],((int16_t*)&y[0])[7],((int16_t*)&y[1])[6],((int16_t*)&y[1])[7],((int16_t*)&y[2])[6],((int16_t*)&y[2])[7]);


    bfly4_tw1(x,x+1,x+2,x+3,y, y+1,y+2,y+3);
    printf("(%d,%d) (%d,%d) (%d %d) (%d,%d) => (%d,%d) (%d,%d) (%d %d) (%d,%d)\n",
     ((int16_t*)&x[0])[0],((int16_t*)&x[0])[1],((int16_t*)&x[1])[0],((int16_t*)&x[1])[1],
     ((int16_t*)&x[2])[0],((int16_t*)&x[2])[1],((int16_t*)&x[3])[0],((int16_t*)&x[3])[1],
     ((int16_t*)&y[0])[0],((int16_t*)&y[0])[1],((int16_t*)&y[1])[0],((int16_t*)&y[1])[1],
     ((int16_t*)&y[2])[0],((int16_t*)&y[2])[1],((int16_t*)&y[3])[0],((int16_t*)&y[3])[1]);

    bfly4(x,x+1,x+2,x+3,y, y+1,y+2,y+3,&tw0,&tw1,&tw2);
    printf("0(%d,%d) (%d,%d) (%d %d) (%d,%d) => (%d,%d) (%d,%d) (%d %d) (%d,%d)\n",
     ((int16_t*)&x[0])[0],((int16_t*)&x[0])[1],((int16_t*)&x[1])[0],((int16_t*)&x[1])[1],
     ((int16_t*)&x[2])[0],((int16_t*)&x[2])[1],((int16_t*)&x[3])[0],((int16_t*)&x[3])[1],
     ((int16_t*)&y[0])[0],((int16_t*)&y[0])[1],((int16_t*)&y[1])[0],((int16_t*)&y[1])[1],
     ((int16_t*)&y[2])[0],((int16_t*)&y[2])[1],((int16_t*)&y[3])[0],((int16_t*)&y[3])[1]);
    printf("1(%d,%d) (%d,%d) (%d %d) (%d,%d) => (%d,%d) (%d,%d) (%d %d) (%d,%d)\n",
     ((int16_t*)&x[0])[0],((int16_t*)&x[0])[1],((int16_t*)&x[1])[0],((int16_t*)&x[1])[1],
     ((int16_t*)&x[2])[0],((int16_t*)&x[2])[1],((int16_t*)&x[3])[0],((int16_t*)&x[3])[1],
     ((int16_t*)&y[0])[2],((int16_t*)&y[0])[3],((int16_t*)&y[1])[2],((int16_t*)&y[1])[3],
     ((int16_t*)&y[2])[2],((int16_t*)&y[2])[3],((int16_t*)&y[3])[2],((int16_t*)&y[3])[3]);
    printf("2(%d,%d) (%d,%d) (%d %d) (%d,%d) => (%d,%d) (%d,%d) (%d %d) (%d,%d)\n",
     ((int16_t*)&x[0])[0],((int16_t*)&x[0])[1],((int16_t*)&x[1])[0],((int16_t*)&x[1])[1],
     ((int16_t*)&x[2])[0],((int16_t*)&x[2])[1],((int16_t*)&x[3])[0],((int16_t*)&x[3])[1],
     ((int16_t*)&y[0])[4],((int16_t*)&y[0])[5],((int16_t*)&y[1])[4],((int16_t*)&y[1])[5],
     ((int16_t*)&y[2])[4],((int16_t*)&y[2])[5],((int16_t*)&y[3])[4],((int16_t*)&y[3])[5]);
    printf("3(%d,%d) (%d,%d) (%d %d) (%d,%d) => (%d,%d) (%d,%d) (%d %d) (%d,%d)\n",
     ((int16_t*)&x[0])[0],((int16_t*)&x[0])[1],((int16_t*)&x[1])[0],((int16_t*)&x[1])[1],
     ((int16_t*)&x[2])[6],((int16_t*)&x[2])[7],((int16_t*)&x[3])[6],((int16_t*)&x[3])[7],
     ((int16_t*)&y[0])[6],((int16_t*)&y[0])[7],((int16_t*)&y[1])[6],((int16_t*)&y[1])[7],
     ((int16_t*)&y[2])[0],((int16_t*)&y[2])[1],((int16_t*)&y[3])[0],((int16_t*)&y[3])[1]);

    bfly5_tw1(x,x+1,x+2,x+3,x+4,y,y+1,y+2,y+3,y+4);

    for (i=0;i<5;i++)
      printf("%d,%d,",
       ((int16_t*)&x[i])[0],((int16_t*)&x[i])[1]);
    printf("\n");
    for (i=0;i<5;i++)
      printf("%d,%d,",
       ((int16_t*)&y[i])[0],((int16_t*)&y[i])[1]);
    printf("\n");

    bfly5(x,x+1,x+2,x+3,x+4,y, y+1,y+2,y+3,y+4,&tw0,&tw1,&tw2,&tw3);
    for (i=0;i<5;i++)
      printf("%d,%d,",
       ((int16_t*)&x[i])[0],((int16_t*)&x[i])[1]);
    printf("\n");
    for (i=0;i<5;i++)
      printf("%d,%d,",
       ((int16_t*)&y[i])[0],((int16_t*)&y[i])[1]);
    printf("\n");


    printf("\n\n12-point\n");
    dft12f(x,
     x+1,
     x+2,
     x+3,
     x+4,
     x+5,
     x+6,
     x+7,
     x+8,
     x+9,
     x+10,
     x+11,
     y,
     y+1,
     y+2,
     y+3,
     y+4,
     y+5,
     y+6,
     y+7,
     y+8,
     y+9,
     y+10,
     y+11);


    printf("X: ");
    for (i=0;i<12;i++)
      printf("%d,%d,",((int16_t*)(&x[i]))[0],((int16_t *)(&x[i]))[1]);
    printf("\nY:");
    for (i=0;i<12;i++)
      printf("%d,%d,",((int16_t*)(&y[i]))[0],((int16_t *)(&y[i]))[1]);
    printf("\n");

 */

    for (i=0;i<32;i++) {
      ((int16_t*)x)[i] = (int16_t)((taus()&0xffff))>>5;
    }
    memset((void*)&y[0],0,16*4);
    idft16((int16_t *)x,(int16_t *)y);
    printf("\n\n16-point\n");
    printf("X: ");
    for (i=0;i<4;i++)
      printf("%d,%d,%d,%d,%d,%d,%d,%d,",((int16_t*)&x[i])[0],((int16_t *)&x[i])[1],((int16_t*)&x[i])[2],((int16_t *)&x[i])[3],((int16_t*)&x[i])[4],((int16_t*)&x[i])[5],((int16_t*)&x[i])[6],((int16_t*)&x[i])[7]);
    printf("\nY:");

    for (i=0;i<4;i++)
      printf("%d,%d,%d,%d,%d,%d,%d,%d,",((int16_t*)&y[i])[0],((int16_t *)&y[i])[1],((int16_t*)&y[i])[2],((int16_t *)&y[i])[3],((int16_t*)&y[i])[4],((int16_t *)&y[i])[5],((int16_t*)&y[i])[6],((int16_t *)&y[i])[7]);
    printf("\n");
 
  memset((void*)&x[0],0,2048*4);
      
  for (i=0; i<2048; i+=4) {
     ((int16_t*)x)[i<<1] = 1024;
     ((int16_t*)x)[1+(i<<1)] = 0;
     ((int16_t*)x)[2+(i<<1)] = 0;
     ((int16_t*)x)[3+(i<<1)] = 1024;
     ((int16_t*)x)[4+(i<<1)] = -1024;
     ((int16_t*)x)[5+(i<<1)] = 0;
     ((int16_t*)x)[6+(i<<1)] = 0;
     ((int16_t*)x)[7+(i<<1)] = -1024;
     }
  /*
  for (i=0; i<2048; i+=2) {
     ((int16_t*)x)[i<<1] = 1024;
     ((int16_t*)x)[1+(i<<1)] = 0;
     ((int16_t*)x)[2+(i<<1)] = -1024;
     ((int16_t*)x)[3+(i<<1)] = 0;
     }
       
  for (i=0;i<2048*2;i++) {
    ((int16_t*)x)[i] = i/2;//(int16_t)((taus()&0xffff))>>5;
  }
     */
  memset((void*)&x[0],0,64*sizeof(int32_t));
  for (i=2;i<36;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  for (i=(128-36);i<128;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  idft64((int16_t *)x,(int16_t *)y,1);
  

  printf("64-point\n");
  printf("X: ");
  for (i=0;i<8;i++)
    print_shorts256("",((int16_t *)x)+(i*16));

  printf("\nY:");

  for (i=0;i<8;i++)
    print_shorts256("",((int16_t *)y)+(i*16));
  printf("\n");

  


  idft64((int16_t *)x,(int16_t *)y,1);
  idft64((int16_t *)x,(int16_t *)y,1);
  idft64((int16_t *)x,(int16_t *)y,1);
  reset_meas(&ts);

  for (i=0; i<10000000; i++) {
    start_meas(&ts);
    idft64((int16_t *)x,(int16_t *)y,1);
    stop_meas(&ts);

  }
  /*
  printf("\n\n64-point (%f cycles, #trials %d)\n",(double)ts.diff/(double)ts.trials,ts.trials);
  //  LOG_M("x64.m","x64",x,64,1,1);
  LOG_M("y64.m","y64",y,64,1,1);
  LOG_M("x64.m","x64",x,64,1,1);
  */
/*
  printf("X: ");
  for (i=0;i<16;i++)
    printf("%d,%d,%d,%d,%d,%d,%d,%d,",((int16_t*)&x[i])[0],((int16_t *)&x[i])[1],((int16_t*)&x[i])[2],((int16_t *)&x[i])[3],((int16_t*)&x[i])[4],((int16_t*)&x[i])[5],((int16_t*)&x[i])[6],((int16_t*)&x[i])[7]);
  printf("\nY:");

  for (i=0;i<16;i++)
    printf("%d,%d,%d,%d,%d,%d,%d,%d,",((int16_t*)&y[i])[0],((int16_t *)&y[i])[1],((int16_t*)&y[i])[2],((int16_t *)&y[i])[3],((int16_t*)&y[i])[4],((int16_t *)&y[i])[5],((int16_t*)&y[i])[6],((int16_t *)&y[i])[7]);
  printf("\n");

  idft64((int16_t*)y,(int16_t*)x,1);
  printf("X: ");
  for (i=0;i<16;i++)
    printf("%d,%d,%d,%d,%d,%d,%d,%d,",((int16_t*)&x[i])[0],((int16_t *)&x[i])[1],((int16_t*)&x[i])[2],((int16_t *)&x[i])[3],((int16_t*)&x[i])[4],((int16_t*)&x[i])[5],((int16_t*)&x[i])[6],((int16_t*)&x[i])[7]);
 
  for (i=0; i<256; i++) {
    ((int16_t*)x)[i] = (int16_t)((taus()&0xffff))>>5;
  }
*/
  
  memset((void*)&x[0],0,128*4);
  for (i=2;i<72;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  for (i=(256-72);i<256;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  reset_meas(&ts);

  for (i=0; i<10000; i++) {
    start_meas(&ts);
    idft128((int16_t *)x,(int16_t *)y,1);
    stop_meas(&ts);
  }

  printf("\n\n128-point(%f cycles)\n",(double)ts.diff/(double)ts.trials);
  LOG_M("y128.m","y128",y,128,1,1);
  LOG_M("x128.m","x128",x,128,1,1);
/*
  printf("X: ");
   for (i=0;i<32;i++)
     printf("%d,%d,%d,%d,%d,%d,%d,%d,",((int16_t*)&x[i])[0],((int16_t *)&x[i])[1],((int16_t*)&x[i])[2],((int16_t *)&x[i])[3],((int16_t*)&x[i])[4],((int16_t*)&x[i])[5],((int16_t*)&x[i])[6],((int16_t*)&x[i])[7]);
   printf("\nY:");

   for (i=0;i<32;i++)
     printf("%d,%d,%d,%d,%d,%d,%d,%d,",((int16_t*)&y[i])[0],((int16_t *)&y[i])[1],((int16_t*)&y[i])[2],((int16_t *)&y[i])[3],((int16_t*)&y[i])[4],((int16_t *)&y[i])[5],((int16_t*)&y[i])[6],((int16_t *)&y[i])[7]);
   printf("\n");
*/

  /*
  for (i=0; i<512; i++) {
    ((int16_t*)x)[i] = (int16_t)((taus()&0xffff))>>5;
  }
  
  memset((void*)&y[0],0,256*4);
  */
  memset((void*)&x[0],0,256*sizeof(int32_t));
  for (i=2;i<144;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  for (i=(512-144);i<512;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  reset_meas(&ts);

  for (i=0; i<10000; i++) {
    start_meas(&ts);
    idft256((int16_t *)x,(int16_t *)y,1);
    stop_meas(&ts);
  }

  printf("\n\n256-point(%f cycles)\n",(double)ts.diff/(double)ts.trials);
  LOG_M("y256.m","y256",y,256,1,1);
  LOG_M("x256.m","x256",x,256,1,1);

  memset((void*)&x[0],0,512*sizeof(int32_t));
  for (i=2;i<302;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  for (i=(1024-300);i<1024;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }

  reset_meas(&ts);
  for (i=0; i<10000; i++) {
    start_meas(&ts);
    idft512((int16_t *)x,(int16_t *)y,1);
    stop_meas(&ts);
  }

  printf("\n\n512-point(%f cycles)\n",(double)ts.diff/(double)ts.trials);
  LOG_M("y512.m","y512",y,512,1,1);
  LOG_M("x512.m","x512",x,512,1,1);
  /*
  printf("X: ");
  for (i=0;i<64;i++)
    printf("%d,%d,%d,%d,%d,%d,%d,%d,",((int16_t*)&x[i])[0],((int16_t *)&x[i])[1],((int16_t*)&x[i])[2],((int16_t *)&x[i])[3],((int16_t*)&x[i])[4],((int16_t*)&x[i])[5],((int16_t*)&x[i])[6],((int16_t*)&x[i])[7]);
  printf("\nY:");

  for (i=0;i<64;i++)
    printf("%d,%d,%d,%d,%d,%d,%d,%d,",((int16_t*)&y[i])[0],((int16_t *)&y[i])[1],((int16_t*)&y[i])[2],((int16_t *)&y[i])[3],((int16_t*)&y[i])[4],((int16_t *)&y[i])[5],((int16_t*)&y[i])[6],((int16_t *)&y[i])[7]);
  printf("\n");
  */

  memset((void*)x,0,1024*sizeof(int32_t));
  for (i=2;i<602;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  for (i=2*724;i<2048;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  reset_meas(&ts);

  for (i=0; i<10000; i++) {
    start_meas(&ts);
    idft1024((int16_t *)x,(int16_t *)y,1);
    stop_meas(&ts);
  }

  printf("\n\n1024-point(%f cycles)\n",(double)ts.diff/(double)ts.trials);
  LOG_M("y1024.m","y1024",y,1024,1,1);
  LOG_M("x1024.m","x1024",x,1024,1,1);


  memset((void*)x,0,1536*sizeof(int32_t));
  for (i=2;i<1202;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  for (i=2*(1536-600);i<3072;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  reset_meas(&ts);

  for (i=0; i<10000; i++) {
    start_meas(&ts);
    idft1536((int16_t *)x,(int16_t *)y,1);
    stop_meas(&ts);
  }

  printf("\n\n1536-point(%f cycles)\n",(double)ts.diff/(double)ts.trials);
  write_output("y1536.m","y1536",y,1536,1,1);
  write_output("x1536.m","x1536",x,1536,1,1);


  memset((void*)x,0,2048*sizeof(int32_t));
  for (i=2;i<1202;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  for (i=2*(2048-600);i<4096;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  reset_meas(&ts);

  for (i=0; i<10000; i++) {
    start_meas(&ts);
    dft2048((int16_t *)x,(int16_t *)y,1);
    stop_meas(&ts);
  }

  printf("\n\n2048-point(%f cycles)\n",(double)ts.diff/(double)ts.trials);
  LOG_M("y2048.m","y2048",y,2048,1,1);
  LOG_M("x2048.m","x2048",x,2048,1,1);

// NR 80Mhz, 217 PRB, 3/4 sampling
  memset((void*)x, 0, 3072*sizeof(int32_t));
  for (i=2;i<2506;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  for (i=2*(3072-1252);i<6144;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }

  reset_meas(&ts);

  for (i=0; i<10000; i++) {
    start_meas(&ts);
    idft3072((int16_t *)x,(int16_t *)y,1);
    stop_meas(&ts);
  }

  printf("\n\n3072-point(%f cycles)\n",(double)ts.diff/(double)ts.trials);
  write_output("y3072.m","y3072",y,3072,1,1);
  write_output("x3072.m","x3072",x,3072,1,1);


  memset((void*)x,0,4096*sizeof(int32_t));
  for (i=0;i<2400;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  for (i=2*(4096-1200);i<8192;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  reset_meas(&ts);

  for (i=0; i<10000; i++) {
    start_meas(&ts);
    idft4096((int16_t *)x,(int16_t *)y,1);
    stop_meas(&ts);
  }

  printf("\n\n4096-point(%f cycles)\n",(double)ts.diff/(double)ts.trials);
  LOG_M("y4096.m","y4096",y,4096,1,1);
  LOG_M("x4096.m","x4096",x,4096,1,1);

  dft4096((int16_t *)y,(int16_t *)x2,1);
  LOG_M("x4096_2.m","x4096_2",x2,4096,1,1);

// NR 160Mhz, 434 PRB, 3/4 sampling
  memset((void*)x, 0, 6144*sizeof(int32_t));
  for (i=2;i<5010;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  for (i=2*(6144-2504);i<12288;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }

  reset_meas(&ts);

  for (i=0; i<10000; i++) {
    start_meas(&ts);
    idft6144((int16_t *)x,(int16_t *)y,1);
    stop_meas(&ts);
  }

  printf("\n\n6144-point(%f cycles)\n",(double)ts.diff/(double)ts.trials);
  write_output("y6144.m","y6144",y,6144,1,1);
  write_output("x6144.m","x6144",x,6144,1,1);

  memset((void*)x,0,8192*sizeof(int32_t));
  for (i=2;i<4802;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  for (i=2*(8192-2400);i<16384;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  reset_meas(&ts);
  for (i=0; i<10000; i++) {
    start_meas(&ts);
    idft8192((int16_t *)x,(int16_t *)y,1);
    stop_meas(&ts);
  }

  printf("\n\n8192-point(%f cycles)\n",(double)ts.diff/(double)ts.trials);
  LOG_M("y8192.m","y8192",y,8192,1,1);
  LOG_M("x8192.m","x8192",x,8192,1,1);

  memset((void*)x,0,16384*sizeof(int32_t));
  for (i=2;i<9602;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  for (i=2*(16384-4800);i<32768;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  reset_meas(&ts);
  for (i=0; i<10000; i++) {
    start_meas(&ts);
    dft16384((int16_t *)x,(int16_t *)y,1);
    stop_meas(&ts);
  }

  printf("\n\n16384-point(%f cycles)\n",(double)ts.diff/(double)ts.trials);
  LOG_M("y16384.m","y16384",y,16384,1,1);
  LOG_M("x16384.m","x16384",x,16384,1,1);

  memset((void*)x,0,1536*sizeof(int32_t));
  for (i=2;i<1202;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  for (i=2*(1536-600);i<3072;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  reset_meas(&ts);
  for (i=0; i<10000; i++) {
    start_meas(&ts);
    idft1536((int16_t *)x,(int16_t *)y,1);
    stop_meas(&ts);
  }

  printf("\n\n1536-point(%f cycles)\n",(double)ts.diff/(double)ts.trials);
  LOG_M("y1536.m","y1536",y,1536,1,1);
  LOG_M("x1536.m","x1536",x,1536,1,1);

  printf("\n\n1536-point(%f cycles)\n",(double)ts.diff/(double)ts.trials);
  LOG_M("y8192.m","y8192",y,8192,1,1);
  LOG_M("x8192.m","x8192",x,8192,1,1);

  memset((void*)x,0,3072*sizeof(int32_t));
  for (i=2;i<1202;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  for (i=2*(3072-600);i<3072;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  reset_meas(&ts);
  for (i=0; i<10000; i++) {
    start_meas(&ts);
    idft3072((int16_t *)x,(int16_t *)y,1);
    stop_meas(&ts);
  }

  printf("\n\n3072-point(%f cycles)\n",(double)ts.diff/(double)ts.trials);
  LOG_M("y3072.m","y3072",y,3072,1,1);
  LOG_M("x3072.m","x3072",x,3072,1,1);

  memset((void*)x,0,6144*sizeof(int32_t));
  for (i=2;i<4802;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  for (i=2*(6144-2400);i<12288;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  reset_meas(&ts);
  for (i=0; i<10000; i++) {
    start_meas(&ts);
    idft6144((int16_t *)x,(int16_t *)y,1);
    stop_meas(&ts);
  }

  printf("\n\n6144-point(%f cycles)\n",(double)ts.diff/(double)ts.trials);
  LOG_M("y6144.m","y6144",y,6144,1,1);
  LOG_M("x6144.m","x6144",x,6144,1,1);

  memset((void*)x,0,12288*sizeof(int32_t));
  for (i=2;i<9602;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  for (i=2*(12288-4800);i<24576;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  reset_meas(&ts);
  for (i=0; i<10000; i++) {
    start_meas(&ts);
    idft12288((int16_t *)x,(int16_t *)y,1);
    stop_meas(&ts);
  }

  printf("\n\n12288-point(%f cycles)\n",(double)ts.diff/(double)ts.trials);
  LOG_M("y12288.m","y12288",y,12288,1,1);
  LOG_M("x12288.m","x12288",x,12288,1,1);

  memset((void*)x,0,18432*sizeof(int32_t));
  for (i=2;i<14402;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  for (i=2*(18432-7200);i<36864;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  reset_meas(&ts);
  for (i=0; i<10000; i++) {
    start_meas(&ts);
    idft18432((int16_t *)x,(int16_t *)y,1);
    stop_meas(&ts);
  }

  printf("\n\n18432-point(%f cycles)\n",(double)ts.diff/(double)ts.trials);
  LOG_M("y18432.m","y18432",y,18432,1,1);
  LOG_M("x18432.m","x18432",x,18432,1,1);

  memset((void*)x,0,24576*sizeof(int32_t));
  for (i=2;i<19202;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  for (i=2*(24576-19200);i<49152;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  reset_meas(&ts);
  for (i=0; i<10000; i++) {
    start_meas(&ts);
    idft24576((int16_t *)x,(int16_t *)y,1);
    stop_meas(&ts);
  }

  printf("\n\n24576-point(%f cycles)\n",(double)ts.diff/(double)ts.trials);
  LOG_M("y24576.m","y24576",y,24576,1,1);
  LOG_M("x24576.m","x24576",x,24576,1,1);


  memset((void*)x,0,2*18432*sizeof(int32_t));
  for (i=2;i<(2*14402);i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  for (i=2*(36864-14400);i<(36864*2);i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  reset_meas(&ts);
  for (i=0; i<10000; i++) {
    start_meas(&ts);
    dft36864((int16_t *)x,(int16_t *)y,1);
    stop_meas(&ts);
  }

  printf("\n\n36864-point(%f cycles)\n",(double)ts.diff/(double)ts.trials);
  LOG_M("y36864.m","y36864",y,36864,1,1);
  LOG_M("x36864.m","x36864",x,36864,1,1);


  memset((void*)x,0,49152*sizeof(int32_t));
  for (i=2;i<28402;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  } 
  for (i=2*(49152-14400);i<98304;i++) {
    if ((taus() & 1)==0)
      ((int16_t*)x)[i] = 364;
    else
      ((int16_t*)x)[i] = -364;
  }
  reset_meas(&ts);
  for (i=0; i<10000; i++) {
    start_meas(&ts);
    idft49152((int16_t *)x,(int16_t *)y,1);
    stop_meas(&ts);
  }

  printf("\n\n49152-point(%f cycles)\n",(double)ts.diff/(double)ts.trials);
  LOG_M("y49152.m","y49152",y,49152,1,1);
  LOG_M("x49152.m","x49152",x,49152,1,1);

  return(0);
}


#endif
#endif
