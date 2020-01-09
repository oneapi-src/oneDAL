/* file: kernel_blas.cl */
/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef __BLAS_KERNELS_CL__
#define __BLAS_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE(name, src) static const char* name = #src;

DECLARE_SOURCE(clKernelGemm,

__kernel void blas_sgemm_small(const uint k,
                               const algorithmFPType alpha,
                               const __global algorithmFPType *a, const uint lda_col, const uint lda_row, const uint offsetA,
                               const __global algorithmFPType *b, const uint ldb_col, const uint ldb_row, const uint offsetB,
                               const algorithmFPType beta,
                               __global algorithmFPType *c, const uint ldc_col, const ldc_row, const uint offsetC)
{

    const uint rows = get_global_id(0);
    const uint cols = get_global_id(1);

    algorithmFPType sum = (algorithmFPType)0;
    for (uint i = 0; i < k; i++)
    {
        sum += a[i*lda_row + rows*lda_col + offsetA]*b[cols*ldb_row + i*ldb_col + offsetB];
    }

    c[rows*ldc_col + cols*ldc_row + offsetC] = alpha * sum + beta * c[rows*ldc_col + cols*ldc_row + offsetC];
}

__kernel void blas_sgemm_without_sum(const uint k,
                               const algorithmFPType alpha,
                               const __global algorithmFPType *a, const uint lda_col, const uint lda_row, const uint offsetA,
                               const __global algorithmFPType *b, const uint ldb_col, const uint ldb_row, const uint offsetB,
                               const algorithmFPType beta,
                               __global algorithmFPType *c, const uint ldc_col, const uint ldc_row, const uint offsetC)
{

    const uint rows = get_global_id(0);
    const uint cols = get_global_id(1);

    algorithmFPType sum = (algorithmFPType)0;
    for (uint i = 0; i < k; i++)
    {
        sum += a[i*lda_row + rows*lda_col + offsetA]*b[cols*ldb_row + i*ldb_col + offsetB];
    }

    c[rows*ldc_col + cols*ldc_row + offsetC] = alpha * sum;
}

__kernel void blas_sgemv_small(const uint k,
                               const algorithmFPType alpha,
                               const __global algorithmFPType *a, const uint lda_col, const uint lda_row,
                               const __global algorithmFPType *x,
                               const algorithmFPType beta,
                               __global algorithmFPType *y) {

    const uint row = get_global_id(0);

    algorithmFPType sum = (algorithmFPType)0;
    for (uint i = 0; i < k; i++)
    {
        sum += a[i*lda_row + row*lda_col] * x[i];
    }

    y[row] = alpha * sum + beta * y[row];
}

__kernel void blas_sgemm(const uint k,
                         const algorithmFPType alpha,
                         const __global algorithmFPType *a, const uint lda_col, const uint lda_row,
                         const __global algorithmFPType *b, const uint ldb_col, const uint ldb_row,
                         const algorithmFPType beta,
                         __global algorithmFPType *c, const uint ldc_col, const uint ldc_row) {

    const size_t BLOCK_SIZE = 4;

    const int row = get_local_id(0);
    const int col = get_local_id(1);

    const int globalRow = BLOCK_SIZE*get_group_id(0) + row;
    const int globalCol = BLOCK_SIZE*get_group_id(1) + col;

    __local algorithmFPType Asub[BLOCK_SIZE][BLOCK_SIZE];
    __local algorithmFPType Bsub[BLOCK_SIZE][BLOCK_SIZE];

    algorithmFPType sum = (algorithmFPType)0;

    const int numTiles = k/BLOCK_SIZE;
    for (int t = 0; t < numTiles; t++) {

        const int tiledRow = BLOCK_SIZE*t + row;
        const int tiledCol = BLOCK_SIZE*t + col;

        Asub[col][row] = a[globalRow*lda_col + tiledCol*lda_row];
        Bsub[col][row] = b[tiledRow*ldb_col + globalCol*ldb_row];

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum += Asub[i][row] * Bsub[col][i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[globalRow*ldc_col + globalCol*ldc_row] = alpha * sum + beta * c[globalRow*ldc_col + globalCol*ldc_row];
}

);

DECLARE_SOURCE(clKernelAxpy,

__kernel void blas_axpy(const algorithmFPType a,
                        const __global algorithmFPType *x,
                        const int incx,
                        __global algorithmFPType *y,
                        const int incy) {
    const int i_x = (get_local_id(0)) * incx;
    const int i_y = (get_local_id(0)) * incy;
    y[i_y] += x[i_x] * a;
}

);

#endif // __BLAS_KERNELS_CL__
