/* file: kernel_function.cl */
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
//  Implementation of Kernel Function OpenCL kernels.
//--
*/

#ifndef __KERNEL_FUNCTION_KERNELS_CL__
#define __KERNEL_FUNCTION_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE_DAAL(name, src) static const char *(name) = #src;

DECLARE_SOURCE_DAAL(
    clKernelKF,

    __kernel void computeRBF(const __global algorithmFPType * const sqrA1, const __global algorithmFPType * const sqrA2, const uint ld,
                             const algorithmFPType expThreshold, const algorithmFPType coeff, __global algorithmFPType * rbf) {
        const uint i = get_global_id(0);
        const uint j = get_global_id(1);

        const algorithmFPType sqrA1i = sqrA1[i];
        const algorithmFPType sqrA2j = sqrA2[j];
        const algorithmFPType rbfij  = rbf[i * ld + j];
        const algorithmFPType arg    = fmax((rbfij + sqrA1i + sqrA2j) * coeff, expThreshold);

        rbf[i * ld + j] = exp(arg);
    }

);

#undef DECLARE_SOURCE_DAAL

#endif
