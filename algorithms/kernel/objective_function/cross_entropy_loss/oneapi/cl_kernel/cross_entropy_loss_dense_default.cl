/* file: cross_entropy_loss_dense_default.cl */
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
//  Implementation of Cross-Entropy Loss OpenCL kernels.
//--
*/

#ifndef __CROSS_ENTROPY_LOSS_KERNELS_CL__
#define __CROSS_ENTROPY_LOSS_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE_DAAL(name, src) static const char* (name) = #src;

DECLARE_SOURCE_DAAL(clKernelCrossEntropyLoss,

void __softmax(
    const __global algorithmFPType* const xi,
    __global algorithmFPType* resi,
    const uint nClasses,
    const algorithmFPType expThreshold)
{

    algorithmFPType maxxi = xi[0];
    for (uint j = 1; j < nClasses; j++)
    {
        maxxi = fmax(maxxi, xi[j]);
    }

    algorithmFPType sum = (algorithmFPType)0;
    for (uint j = 0; j < nClasses; j++)
    {
        const algorithmFPType arg = fmax(xi[j] - maxxi, expThreshold);
        const algorithmFPType value = exp(arg);
        sum += value;
        resi[j] = value;
    }

    for (uint j = 0; j < nClasses; j++)
    {
        resi[j] = resi[j] / sum;
    }
}


__kernel void softmax(
    const __global algorithmFPType* const x,
    __global algorithmFPType* result,
    const uint nClasses,
    const algorithmFPType expThreshold)
{
    const uint i = get_global_id(0);
    const __global algorithmFPType* const xi = &x[i*nClasses];
    __global algorithmFPType* resi = &result[i*nClasses];
    __softmax(xi, resi, nClasses, expThreshold);
}

__kernel void softmaxAndUpdateProba(
    const __global algorithmFPType* const x,
    const __global algorithmFPType* const y,
    __global algorithmFPType* result,
    const uint nClasses,
    const algorithmFPType expThreshold)
{
    const uint i = get_global_id(0);
    const __global algorithmFPType* const xi = &x[i*nClasses];
    __global algorithmFPType* resi = &result[i*nClasses];

    __softmax(xi, resi, nClasses, expThreshold);

    resi[(uint)y[i]] -= 1;
}

__kernel void crossEntropy(
const __global algorithmFPType* const y,
const __global algorithmFPType* const s,
__global algorithmFPType* result,
const uint nClasses)
{
    const uint i = get_global_id(0);

    result[i] = log(s[i*nClasses  + (uint)y[i]]);
}

__kernel void updateProba(
const __global algorithmFPType* const y,
__global algorithmFPType* sigma,
const uint nClasses, const algorithmFPType value)
{
    const uint i = get_global_id(0);

    sigma[i*nClasses + (uint)y[i]] += value;
}

__kernel void hessian(const __global algorithmFPType* const x, const uint ldx,
const __global algorithmFPType* const prob, const uint n, __global algorithmFPType* h,
const uint nBeta, const uint nClasses, const uint offset, const algorithmFPType alpha)
{
    const uint row = get_global_id(0);
    const uint col = get_global_id(1);

    if (col < row) return;

    const algorithmFPType one = (algorithmFPType)1.0;

    const uint t = row / nBeta;
    const uint p = col / nBeta;
    const uint j = row % nBeta;
    const uint q = col % nBeta;

    if (j == 0 || q == 0)
    {
        h[row*nBeta + col] = (algorithmFPType)0;
        h[col*nBeta + row] = (algorithmFPType)0;
    return;
    }

    algorithmFPType sum = (algorithmFPType)0.0;
    for (uint i = 0; i < n; i++)
    {

        const algorithmFPType pt = p == t ? one - prob[i*nClasses + t] : -prob[i*nClasses + t];
        sum += x[i*ldx + j]*x[i*ldx + q]*prob[i*nClasses + p]*pt;
    }

    h[row*nBeta + col] = sum*alpha;
    h[col*nBeta + row] = sum*alpha;

}

);

#undef DECLARE_SOURCE_DAAL

#endif
