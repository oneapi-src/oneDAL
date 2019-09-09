/* file: layers_threading.h */
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

#ifndef __LAYERS_THREADING_H__
#define __LAYERS_THREADING_H__

#include "tensor.h"
#include "threading.h"
#include "service_tensor.h"
#include "service_numeric_table.h"
#include "service_error_handling.h"
#include "service_mkl_tensor.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace internal
{

inline void getNumberOfFixedDims(const TensorOffsetLayout &inputLayout, const Collection<size_t> &dims, size_t &fDimN, const size_t minElementsNumInBlock)
{
    const Collection<size_t> &inputOffsets = inputLayout.getOffsets();
    size_t nDims = dims.size();

    for(int idx = nDims - 1; idx >= 0; idx--)
    {
        if (inputOffsets[idx] > minElementsNumInBlock)
        {
            fDimN = idx + 1;
            break;
        }
    }
}

inline void getFixedDimsIndexes(const size_t fDimN, size_t *fDims, const Collection<size_t> &dims, const size_t i)
{
    size_t offsetAfter = dims[fDimN - 1];

    /* Get last fixed dim index as the remainder of the division */
    fDims[fDimN - 1] = i % dims[fDimN - 1];

    /* Count indexes starting from the penultimate element of the fDims[] array*/
    for(size_t j = fDimN - 1; j > 0; j--)
    {
        size_t totalOffset = offsetAfter * dims[j - 1];
        size_t nTimes = i / totalOffset;

        fDims[j - 1] = (i - totalOffset * nTimes) / offsetAfter;

        offsetAfter *= dims[j - 1];
    }
}

#undef __DAAL_MAKE_TENSOR_THREADSAFE
#define __DAAL_MAKE_TENSOR_THREADSAFE(TensorPtr)                            \
    {                                                                       \
        if (dynamic_cast<MklTensor<float>*>(TensorPtr))                     \
        {                                                                   \
            dynamic_cast<MklTensor<float>*>(TensorPtr)->syncDnnToPlain();   \
        }                                                                   \
        if (dynamic_cast<MklTensor<double>*>(TensorPtr))                    \
        {                                                                   \
            dynamic_cast<MklTensor<double>*>(TensorPtr)->syncDnnToPlain();  \
        }                                                                   \
    }

template<CpuType cpu, typename F>
Status computeImpl(const Tensor &inputTensor, const F &processBlock, const size_t minElementsNumInBlock = 997)
{
    __DAAL_MAKE_TENSOR_THREADSAFE(const_cast<Tensor*>(&inputTensor));

    const Collection<size_t> &dims = inputTensor.getDimensions();
    TensorOffsetLayout inputLayout = inputTensor.createRawSubtensorLayout();

    size_t fDimN = 0;
    getNumberOfFixedDims(inputLayout, dims, fDimN, minElementsNumInBlock);

    if(fDimN == 0)
    {
        return processBlock(fDimN, 0, dims[fDimN], inputLayout);
    }
    else
    {
        size_t nBlocks = inputTensor.getSize(0, fDimN);

        SafeStatus safeStat;
        daal::threader_for(nBlocks, nBlocks, [ =, &safeStat, &dims ](size_t i)
        {
            TArray<size_t, cpu> fdimsBlock(fDimN);
            size_t *fDims = fdimsBlock.get();
            DAAL_CHECK_THR(fDims, ErrorMemoryAllocationFailed);

            getFixedDimsIndexes(fDimN, fDims, dims, i);
            Status localStatus = processBlock(fDimN, fDims, dims[fDimN], inputLayout);
            DAAL_CHECK_STATUS_THR(localStatus);
        } );
        DAAL_CHECK_SAFE_STATUS();
    }
    return Status();
}

} // internal
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
