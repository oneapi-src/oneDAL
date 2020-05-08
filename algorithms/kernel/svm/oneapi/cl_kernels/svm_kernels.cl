/* file: svm_kernels.cl */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

/*
//++
//  Implementation of SVM kernels.
//--
*/

#ifndef __SVM_KERNELS_CL__
#define __SVM_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE_DAAL(name, src) static const char *(name) = #src;

DECLARE_SOURCE_DAAL(
    clKernelSVM,

    __kernel void makeInversion(const __global algorithmFPType * const x, __global algorithmFPType * res) {
        const uint i = get_global_id(0);
        res[i]       = -x[i];
    }

    __kernel void makeRange(__global uint * x) {
        const uint i = get_global_id(0);
        x[i]         = i;
    }

    __kernel void checkUpper(const __global algorithmFPType * const y, const __global algorithmFPType * const alpha, const algorithmFPType C,
                             __global uint * indicator) {
        const uint i = get_global_id(0);
        indicator[i] = (y[i] > 0 && alpha[i] < C) || (y[i] < 0 && alpha[i] > 0);
    }

    __kernel void checkLower(const __global algorithmFPType * const y, const __global algorithmFPType * const alpha, const algorithmFPType C,
                             __global uint * indicator) {
        const uint i = get_global_id(0);
        indicator[i] = (y[i] > 0 && alpha[i] > 0) || (y[i] < 0 && alpha[i] < C);
    }

    __kernel void checkBorder(const __global algorithmFPType * const alpha, const algorithmFPType C, __global uint * indicator) {
        const uint i                 = get_global_id(0);
        const algorithmFPType alphai = alpha[i];
        indicator[i]                 = 0 < alphai && alphai < C;
    }

    __kernel void checkNonZeroBinary(const __global algorithmFPType * const alpha, __global uint * indicator) {
        const uint i = get_global_id(0);
        indicator[i] = alpha[i] != (algorithmFPType)0;
    }

    __kernel void resetIndicatorWithZeros(const __global uint * const ind, __global uint * indicator) {
        const uint i      = get_global_id(0);
        indicator[ind[i]] = 0;
    }

    __kernel void copyDataByIndices(const __global algorithmFPType * const x, const __global uint * const xInd, const uint ldx,
                                    __global algorithmFPType * newX) {
        const uint index = get_global_id(1);
        const uint jCol  = get_global_id(0);

        const uint iRow = xInd[index];

        const __global algorithmFPType * const xi = &x[iRow * ldx];
        __global algorithmFPType * newXi          = &newX[index * ldx];

        newXi[jCol] = xi[jCol];
    }

    __kernel void copyDataByIndicesInt(const __global algorithmFPType * const x, const __global int * const xInd, const uint ldx,
                                       __global algorithmFPType * newX) {
        const int index = get_global_id(1);
        const int jCol  = get_global_id(0);

        const int iRow = xInd[index];

        const __global algorithmFPType * const xi = &x[iRow * ldx];
        __global algorithmFPType * newXi          = &newX[index * ldx];

        newXi[jCol] = xi[jCol];
    }

    __kernel void computeDualCoeffs(const __global algorithmFPType * const y, __global algorithmFPType * a) {
        const uint i = get_global_id(0);
        a[i]         = a[i] * y[i];
    }

);

#undef DECLARE_SOURCE_DAAL

#endif
