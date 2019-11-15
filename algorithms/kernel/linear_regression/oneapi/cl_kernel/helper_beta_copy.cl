/* file: helper_beta_copy.cl */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of Linead Regression OpenCL kernels.
//--
*/

#ifndef __HELPER_BETA_COPY_CL__
#define __HELPER_BETA_COPY_CL__

#include <string.h>

#define DECLARE_SOURCE(name, src) static const char* name = #src;

DECLARE_SOURCE(clKernelHelperBetaCopy,

__kernel void copyBeta(const __global algorithmFPType *src, uint nCols, uint nColsSrc, __global algorithmFPType *dst, uint intercept)
{
    uint idxY = get_global_id(0);
    uint idxX = get_global_id(1);

    if(idxX == 0)
    {
        if(intercept == 1)
        {
            dst[idxY*nCols] = src[idxY*nColsSrc + nColsSrc- 1];
        }
    }
    else
    {
        dst[idxY*nCols + idxX] = src[idxY*nColsSrc + idxX - 1];
    }
}

);

#endif // __HELPER_BETA_COPY_CL__
