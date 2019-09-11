/* file: implicit_als_train_init_default_batch_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
#include "service_error_handling.h"

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
services::Status ImplicitALSInitKernelBase<algorithmFPType, cpu>::randFactors(size_t nItems, size_t nFactors,
        algorithmFPType *itemsFactors, engines::BatchBase &engine)
{
    const size_t nTheads = threader_get_threads_number();
    const size_t nBlocks = nTheads;
    const size_t blockSize = (nItems*nFactors) / nBlocks;
    const size_t lastBlockSize = (nItems*nFactors) - (nBlocks-1)*blockSize;

    TArray<services::SharedPtr<engines::BatchBase>, cpu> engines(nBlocks-1);
    for(size_t i = 0; i < nBlocks-1; i++)
    {
        engines[i] = engine.clone();
    }

    daal::SafeStatus safeStatus;
    daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock)
    {
        const size_t start  = blockSize * iBlock;
        const size_t nElems = (iBlock != nBlocks-1) ? blockSize : lastBlockSize;
        algorithmFPType* const arr = itemsFactors + start;

        if (iBlock != 0)
        {
            engines[iBlock-1]->skipAhead(start);
            safeStatus |= distributions::uniform::internal::UniformKernelDefault<algorithmFPType, cpu>::compute(0.0f, 1.0f, *engines[iBlock-1], nElems, arr );
        }
        else
        {
            safeStatus |= distributions::uniform::internal::UniformKernelDefault<algorithmFPType, cpu>::compute(0.0f, 1.0f, engine, nElems, arr );
        }
    });

    return safeStatus.detach();
}

}
}
}
}
}
}

#endif
