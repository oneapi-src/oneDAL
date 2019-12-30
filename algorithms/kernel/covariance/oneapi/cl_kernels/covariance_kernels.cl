/* file: covariance_kernels.cl */
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

/*
//++
//  Implementation of Covariance OpenCL kernels.
//--
*/

#ifndef __COVARIANCE_KERNELS_CL__
#define __COVARIANCE_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE(name, src) static const char* name = #src;

DECLARE_SOURCE(covariance_kernels,

bool isFirstDataBlock(const algorithmFPType nObservations)
{
    return (nObservations < (algorithmFPType)(0.5));
}

__kernel void mergeCrossProduct(uint nFeatures,
                               __global const algorithmFPType *partialCrossProduct,
                               __global const algorithmFPType *partialSums,
                               algorithmFPType partialNObservations,
                               __global algorithmFPType *crossProduct,
                               __global const algorithmFPType *sums,
                               algorithmFPType nObservations)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    if ((i < nFeatures) && (j < nFeatures))
    {
        algorithmFPType invPartialNObs = (algorithmFPType)(1.0) / partialNObservations;
        algorithmFPType invNObs = (algorithmFPType)(1.0) / nObservations;
        algorithmFPType invNewNObs = (algorithmFPType)(1.0) / (nObservations + partialNObservations);

        crossProduct[i * nFeatures + j] += partialCrossProduct[i * nFeatures + j];
        crossProduct[i * nFeatures + j] += partialSums[i] * partialSums[j] * invPartialNObs;
        crossProduct[i * nFeatures + j] += sums[i] * sums[j] * invNObs;
        crossProduct[i * nFeatures + j] -= (partialSums[i] + sums[i]) * (partialSums[j] + sums[j]) * invNewNObs;
    }
}

__kernel void prepareMeansAndCrossProductDiag(unsigned int nFeatures,
                                              algorithmFPType nObservations,
                                              __global algorithmFPType* crossProduct,
                                              __global algorithmFPType* diagCrossProduct,
                                              __global algorithmFPType* sums,
                                              __global algorithmFPType* mean)
{
    const int tid = get_global_id(0);
    const algorithmFPType invNObservations = (algorithmFPType)(1.0) / nObservations;

    diagCrossProduct[tid] = crossProduct[tid*nFeatures+tid];
    mean[tid] = sums[tid] * invNObservations;
}

__kernel void finalize(unsigned int nFeatures,
                       algorithmFPType nObservations,
                       __global algorithmFPType* crossProduct,
                       __global algorithmFPType* cov,
                       __global algorithmFPType* diagCrossProduct,
                       unsigned int isOutputCorrelationMatrix)
{
    algorithmFPType invNObservationsM1 = (algorithmFPType)(1.0);

    if (nObservations > (algorithmFPType)(1.0))
    {
        invNObservationsM1 = (algorithmFPType)(1.0) / (nObservations - (algorithmFPType)(1.0));
    }

    const int global_row_id = get_global_id(0);
    const int global_col_id = get_global_id(1);

    if ((global_row_id < nFeatures) && (global_col_id < nFeatures))
    {
        algorithmFPType covElement = (algorithmFPType)(1.0);

        algorithmFPType crossProductRowElement = diagCrossProduct[global_row_id];
        algorithmFPType crossProductColElement = diagCrossProduct[global_col_id];

        algorithmFPType crossProductElement = crossProduct[global_row_id*nFeatures + global_col_id];

        if (!isOutputCorrelationMatrix)
        {
            covElement = crossProductElement * invNObservationsM1;
        }
        else if (global_row_id != global_col_id)
        {
            algorithmFPType sqrtRowElement = (algorithmFPType)(1.0) / sqrt(crossProductRowElement);
            algorithmFPType sqrtColElement = (algorithmFPType)(1.0) / sqrt(crossProductColElement);

            covElement = crossProductElement * sqrtRowElement * sqrtColElement;
        }

        cov[global_row_id*nFeatures + global_col_id] = covElement;
    }
}

);

#endif
