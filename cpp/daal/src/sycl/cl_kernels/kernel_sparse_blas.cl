/* file: kernel_sparse_blas.cl */
/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef __SPARSE_BLAS_KERNELS_CL__
#define __SPARSE_BLAS_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE(name, src) static const char * name = #src;

DECLARE_SOURCE(
    clKernelSpGemm,

    algorithmFPType dot_product(__global const algorithmFPType * const aValues, __global const algorithmFPType * const bValues,
                                __global const ulong * const aCols, __global const ulong * const bCols, ulong aRowCur, ulong aRowEnd, ulong bRowCur,
                                ulong bRowEnd) {
        algorithmFPType localSum = (algorithmFPType)0;

        while ((aRowCur < aRowEnd) && (bRowCur < bRowEnd))
        {
            const ulong aCurIdx = aCols[aRowCur];
            const ulong bCurIdx = bCols[bRowCur];

            if (aCurIdx == bCurIdx)
            {
                localSum += aValues[aRowCur] * bValues[bRowCur];
                aRowCur++;
                bRowCur++;
            }
            else if (aCurIdx < bCurIdx)
            {
                aRowCur++;
            }
            else
            {
                bRowCur++;
            }
        }
        return localSum;
    }

    __kernel void spmm_kernel_without_sum(const algorithmFPType alpha, __global const algorithmFPType * const aValues,
                                          __global const ulong * const aCols, __global const ulong * const aRowInd,
                                          __global const algorithmFPType * const bValues, __global const ulong * const bCols,
                                          __global const ulong * const bRowInd, __global algorithmFPType * c, const ulong ldC, const ulong offsetC,
                                          const algorithmFPType beta) {
        const ulong i = get_global_id(0);
        const ulong j = get_global_id(1);

        const ulong aRowCur = aRowInd[i] - 1;
        const ulong aRowEnd = aRowInd[i + 1] - 1;

        const ulong bRowCur = bRowInd[j] - 1;
        const ulong bRowEnd = bRowInd[j + 1] - 1;

        const algorithmFPType dotProduct = dot_product(aValues, bValues, aCols, bCols, aRowCur, aRowEnd, bRowCur, bRowEnd);
        c[i * ldC + j + offsetC]         = alpha * dotProduct;
    }

    __kernel void spmm_kernel(const algorithmFPType alpha, __global const algorithmFPType * const aValues, __global const ulong * const aCols,
                              __global const ulong * const aRowInd, __global const algorithmFPType * const bValues,
                              __global const ulong * const bCols, __global const ulong * const bRowInd, __global algorithmFPType * c, const ulong ldC,
                              const ulong offsetC, const algorithmFPType beta) {
        const ulong i = get_global_id(0);
        const ulong j = get_global_id(1);

        const ulong aRowCur = aRowInd[i] - 1;
        const ulong aRowEnd = aRowInd[i + 1] - 1;

        const ulong bRowCur = bRowInd[j] - 1;
        const ulong bRowEnd = bRowInd[j + 1] - 1;

        const algorithmFPType dotProduct = dot_product(aValues, bValues, aCols, bCols, aRowCur, aRowEnd, bRowCur, bRowEnd);
        c[i * ldC + j + offsetC]         = alpha * dotProduct + beta * c[i * ldC + j + offsetC];
    }

);

#endif // __SPARSE_BLAS_KERNELS_CL__
