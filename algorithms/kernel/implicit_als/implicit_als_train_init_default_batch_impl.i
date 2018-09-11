/* file: implicit_als_train_init_default_batch_impl.i */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of defaultDense method for impicit ALS initialization
//--
*/

#ifndef __IMPLICIT_ALS_TRAIN_INIT_DEFAULT_BATCH_IMPL_I__
#define __IMPLICIT_ALS_TRAIN_INIT_DEFAULT_BATCH_IMPL_I__

#include "uniform_kernel.h"
#include "uniform_impl.i"

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
services::Status ImplicitALSInitKernelBase<algorithmFPType, cpu>::randFactors(
            size_t nItems, size_t nFactors, algorithmFPType *itemsFactors, int *buffer, engines::BatchBase &engine)
{
    int randMax = 1000;
    const algorithmFPType invRandMax = 1.0 / algorithmFPType(randMax);

    Status s;

    for (size_t i = 0; i < nItems; i++)
    {
        DAAL_CHECK_STATUS(s, (distributions::uniform::internal::UniformKernelDefault<int, cpu>::compute(0, randMax, engine, nFactors - 1, buffer)));
        for (size_t j = 1; j < nFactors; j++)
            itemsFactors[i*nFactors + j] = invRandMax * algorithmFPType(buffer[j-1]);
        }
    return s;
}

}
}
}
}
}
}

#endif
