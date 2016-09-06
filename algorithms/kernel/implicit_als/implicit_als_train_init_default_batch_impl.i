/* file: implicit_als_train_init_default_batch_impl.i */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of defaultDense method for impicit ALS initialization
//--
*/

#ifndef __IMPLICIT_ALS_TRAIN_INIT_DEFAULT_BATCH_IMPL_I__
#define __IMPLICIT_ALS_TRAIN_INIT_DEFAULT_BATCH_IMPL_I__

#include "service_rng.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace init
{
namespace internal
{

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSInitKernelBase<algorithmFPType, cpu>::randFactors(
            size_t nItems, size_t nFactors, algorithmFPType *itemsFactors, int *buffer,
            daal::internal::IntRng<int, cpu> &rng)
{
    int randMax = 1000;
    algorithmFPType invRandMax = 1.0 / (algorithmFPType)randMax;

    for (size_t i = 0; i < nItems; i++)
    {
        rng.uniform(nFactors - 1, 0, randMax, buffer);
        for (size_t j = 1; j < nFactors; j++)
        {
            itemsFactors[i*nFactors + j] = invRandMax * (algorithmFPType)buffer[j-1];
        }
    }
}

}
}
}
}
}
}

#endif
