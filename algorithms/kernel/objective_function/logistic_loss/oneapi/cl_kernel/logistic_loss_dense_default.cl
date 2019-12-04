/* file: logistic_loss_dense_default.cl */
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
//  Implementation of Log Loss OpenCL kernels.
//--
*/

#ifndef __OBJECTIVE_LOGISTIC_LOSS_KERNELS_CL__
#define __OBJECTIVE_LOGISTIC_LOSS_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE_DAAL(name, src) static const char* (name) = #src;

DECLARE_SOURCE_DAAL(clKernelLogLoss,

inline void __sum(__global algorithmFPType* partialSums, __local algorithmFPType *localSum)
{
  const uint global_group_id = get_group_id(0);
  const uint group_size = get_local_size(0);
  const uint local_id = get_local_id(0);

  for (uint stride = group_size / 2; stride > 0; stride /=2)
  {
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < stride)
    {
      localSum[local_id] += localSum[local_id + stride];
    }
  }

  if (local_id == 0)
  {
    partialSums[global_group_id] = localSum[0];
  }
}

__kernel void logLoss(const __global algorithmFPType* const y,
  const __global algorithmFPType* const sigma, __global algorithmFPType* result)
{
    const uint i = get_global_id(0);
    const algorithmFPType one = (algorithmFPType)1.0;

    result[i] = y[i]*log(sigma[i]) + (one - y[i])*log(one - sigma[i]);
}

__kernel void sigmoid(const __global algorithmFPType* const xb,
  const algorithmFPType expThreshold,
  const uint calculateInverse,
  __global algorithmFPType* result)
{
    const uint i = get_global_id(0);
    const algorithmFPType one = (algorithmFPType)1.0;

    const algorithmFPType f = fmax(-xb[i], expThreshold);
    const algorithmFPType p = one / (one + exp(f));

    if (calculateInverse != 0)
    {
        const uint firstColIdx = 2*i;

        result[firstColIdx] = one - p;
        result[firstColIdx+1] = p;
    }
    else
    {
        result[i] = p;
    }
}

__kernel void hessian(const __global algorithmFPType* const x, const uint ldx,
  const __global algorithmFPType* const sigma, const uint n, __global algorithmFPType* h,
  const uint ldh, const uint offset, const algorithmFPType alpha)
{
    const uint row = get_global_id(0);
    const uint col = get_global_id(1);
    const algorithmFPType one = (algorithmFPType)1.0;

    if (col < row) return;

    algorithmFPType sum = (algorithmFPType)0.0;

    for (uint i = 0; i < n; i++)
    {
        sum += x[i * ldx + row] * x[i * ldx + col] * sigma[i] * (one - sigma[i]);
    }

    h[(row + offset)*ldh + (col + offset)] = sum*alpha;
    h[(col + offset)*ldh + (row + offset)] = sum*alpha;
}

__kernel void hessianIntercept(const __global algorithmFPType* const x, const uint ldx,
  const __global algorithmFPType* const sigma, const uint n, __global algorithmFPType* h,
  const uint ldh, const algorithmFPType alpha)
{
    const uint row = get_global_id(0);
    const algorithmFPType one = (algorithmFPType)1.0;

    algorithmFPType sum = (algorithmFPType)0.0;
    for (uint i = 0; i < n; i++)
    {
        sum += x[i * ldx + row] * sigma[i] * (one - sigma[i]);
    }

    h[(row + 1)*ldh] = sum*alpha;
    h[(row + 1)] = sum*alpha;
}

__kernel void hessianInterceptH0(const __global algorithmFPType* const sigma, const uint n,
    __global algorithmFPType* partialSums)
{
  __local algorithmFPType localSum[LOCAL_SUM_SIZE];

  uint local_id = get_local_id(0);
  uint global_id = get_global_id(0);
  const algorithmFPType one = (algorithmFPType)1.0;

  if (global_id >= n)
  {
    localSum[local_id] = (algorithmFPType)0;
  }
  else
  {
    localSum[local_id] = sigma[global_id] * (one - sigma[global_id]);
  }

  __sum(partialSums, localSum);
}

__kernel void hessianRegulization(__global algorithmFPType* h, const uint ldh,
  const algorithmFPType beta)
{
   // not regulated b0
    const uint row = get_global_id(0) + 1;
    h[row*ldh + row] += beta;
}

);

#undef DECLARE_SOURCE_DAAL

#endif
