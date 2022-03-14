/* file: reduce_results.cl */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of copy kernels.
//--
*/

#ifndef __REDUCE_RESULTS_CL__
#define __REDUCE_RESULTS_CL__

#include <string.h>

#define DECLARE_SOURCE(name, src) static const char * name = #src;

DECLARE_SOURCE(
    clKernelCopy,

    __kernel void reduceResults(__global algorithmFPType * dst, uint dstOffset, uint dstStride, const __global algorithmFPType * src, uint srcOffset,
                                uint srcStride) {
        const uint valIdx = get_global_id(0);

        dst[dstStride * valIdx + dstOffset] += src[srcStride * valIdx + srcOffset];
    }

);

#endif // __REDUCE_RESULTS_CL__
