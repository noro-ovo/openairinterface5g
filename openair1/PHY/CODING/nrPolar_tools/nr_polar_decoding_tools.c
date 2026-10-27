/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */
#include "PHY/CODING/nrPolar_tools/nr_polar_defs.h"

static inline void updateBit(uint8_t listSize,
                             uint16_t row,
                             uint16_t col,
                             uint16_t xlen,
                             uint8_t ylen,
                             int zlen,
                             uint8_t bit[xlen][ylen][zlen],
                             uint8_t bitU[xlen][ylen])
{
  const uint offset = (xlen / (pow(2, (ylen - col))));

  for (uint i = 0; i < listSize; i++) {
    if (((row) % (2 * offset)) >= offset) {
      if (bitU[row][col - 1] == 0)
        updateBit(listSize, row, (col - 1), xlen, ylen, zlen, bit, bitU);
      bit[row][col][i] = bit[row][col - 1][i];
    } else {
      if (bitU[row][col - 1] == 0)
        updateBit(listSize, row, (col - 1), xlen, ylen, zlen, bit, bitU);
      if (bitU[row + offset][col - 1] == 0)
        updateBit(listSize, (row + offset), (col - 1), xlen, ylen, zlen, bit, bitU);
      bit[row][col][i] = ((bit[row][col - 1][i] + bit[row + offset][col - 1][i]) % 2);
    }
  }

  bitU[row][col] = 1;
}

static inline void
computeLLR(uint16_t row, uint16_t col, uint8_t i, uint16_t offset, int xlen, int ylen, int zlen, double llr[xlen][ylen][zlen])
{
  double a = llr[row][col + 1][i];
  double b = llr[row + offset][col + 1][i];
  llr[row][col][i] = log((exp(a + b) + 1) / (exp(a) + exp(b))); // eq. (8a)
}

void updateLLR(uint8_t listSize,
               uint16_t row,
               uint16_t col,
               uint16_t xlen,
               uint8_t ylen,
               int zlen,
               double llr[xlen][ylen][zlen],
               uint8_t llrU[xlen][ylen],
               uint8_t bit[xlen][ylen][zlen],
               uint8_t bitU[xlen][ylen])
{
  const uint offset = (xlen / (pow(2, (ylen - col - 1))));
  for (uint i = 0; i < listSize; i++) {
    if ((row % (2 * offset)) >= offset) {
      if (bitU[row - offset][col] == 0)
        updateBit(listSize, (row - offset), col, xlen, ylen, zlen, bit, bitU);
      if (llrU[row - offset][col + 1] == 0)
        updateLLR(listSize, (row - offset), (col + 1), xlen, ylen, zlen, llr, llrU, bit, bitU);
      if (llrU[row][col + 1] == 0)
        updateLLR(listSize, row, (col + 1), xlen, ylen, zlen, llr, llrU, bit, bitU);
      llr[row][col][i] = (pow((-1), bit[row - offset][col][i]) * llr[row - offset][col + 1][i]) + llr[row][col + 1][i];
    } else {
      if (llrU[row][col + 1] == 0)
        updateLLR(listSize, row, (col + 1), xlen, ylen, zlen, llr, llrU, bit, bitU);
      if (llrU[row + offset][col + 1] == 0)
        updateLLR(listSize, (row + offset), (col + 1), xlen, ylen, zlen, llr, llrU, bit, bitU);
      computeLLR(row, col, i, offset, xlen, ylen, zlen, llr);
    }
  }
  llrU[row][col] = 1;

  //	printf("LLR (a %f, b %f): llr[%d][%d] %f\n",32*a,32*b,col,row,32*llr[col][row]);
}

void updatePathMetric(double *pathMetric,
                      uint8_t listSize,
                      uint8_t bitValue,
                      uint16_t row,
                      int xlen,
                      int ylen,
                      int zlen,
                      double llr[xlen][ylen][zlen])
{
  const int multiplier = (2 * bitValue) - 1;
  for (uint i = 0; i < listSize; i++)
    pathMetric[i] += log(1 + exp(multiplier * llr[row][0][i])); // eq. (11b)
}

void updatePathMetric2(double *pathMetric,
                       uint8_t listSize,
                       uint16_t row,
                       int xlen,
                       int ylen,
                       int zlen,
                       double llr[xlen][ylen][zlen])
{
  double tempPM[listSize];
  memcpy(tempPM, pathMetric, (sizeof(double) * listSize));

  uint bitValue = 0;
  int multiplier = (2 * bitValue) - 1;
  for (uint i = 0; i < listSize; i++)
    pathMetric[i] += log(1 + exp(multiplier * llr[row][0][i])); // eq. (11b)

  bitValue = 1;
  multiplier = (2 * bitValue) - 1;
  for (uint i = listSize; i < 2 * listSize; i++)
    pathMetric[i] = tempPM[(i - listSize)] + log(1 + exp(multiplier * llr[row][0][(i - listSize)])); // eq. (11b)
}
