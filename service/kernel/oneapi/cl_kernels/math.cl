/* file: math.cl */
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
//  Implementation of math kernels.
//--
*/

#ifndef __MATH_CL__
#define __MATH_CL__

#define DECLARE_SOURCE(name, src) static const char* name = #src;

DECLARE_SOURCE(clKernelMath,

__kernel void vLog(
    const __global algorithmFPType* const x,
    __global algorithmFPType* result)
{
    const uint i = get_global_id(0);
    result[i] = log(x[i]);
}

);

#endif
