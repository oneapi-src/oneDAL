/* file: logistic_regression_dense_default.cl */
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
//  Implementation of Logistic Regression OpenCL kernels.
//--
*/

#ifndef __LOGISTIC_REGRESSION_KERNELS_CL__
#define __LOGISTIC_REGRESSION_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE_DAAL(name, src) static const char* (name) = #src;

DECLARE_SOURCE_DAAL(clKernelLogisticResgression,

__kernel void heaviside(
  const __global algorithmFPType* const x,
  __global algorithmFPType* result)
{
  const uint i = get_global_id(0);

  algorithmFPType zero = (algorithmFPType)0;
  algorithmFPType one = (algorithmFPType)1;

  result[i] = x[i] >= zero ? one : zero;
}

__kernel void argMax(
  const __global algorithmFPType* const x,
  __global algorithmFPType* result,
  const uint p)
{
  const uint i = get_global_id(0);

  algorithmFPType maxVal = x[i*p + 0];
  uint maxIdx = 0;

  for (uint j = 1; j < p; j++)
  {
      if (maxVal < x[i*p + j])
      {
          maxVal = x[i*p + j];
          maxIdx = j;
      }
  }

  result[i] = (algorithmFPType)maxIdx;
}

);

#undef DECLARE_SOURCE_DAAL

#endif
