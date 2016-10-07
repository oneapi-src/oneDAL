/* file: layers_threading.h */
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

#ifndef __LAYERS_THREADING_H__
#define __LAYERS_THREADING_H__

#include "tensor.h"
#include "threading.h"
#include "service_tensor.h"
#include "service_numeric_table.h"

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

inline void getNumberOfFixedDims(TensorOffsetLayout &inputLayout, const Collection<size_t> &dims, size_t &fDimN, size_t minElementsNumInBlock)
{
    const Collection<size_t> &inputOffsets = inputLayout.getOffsets();
    size_t nDims = dims.size();

    for(int idx = nDims - 1; idx >= 0; idx--)
    {
        if (inputOffsets[idx] > minElementsNumInBlock)
        {
            if (idx == 0)
            {
                fDimN = 1;
            }
            else
            {
                fDimN = idx + 1;
            }
            break;
        }
    }
}

inline void getFixedDimsIndexes(size_t fDimN, size_t *fDims, const Collection<size_t> &dims, size_t i)
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

template<CpuType cpu, typename F>
void computeImpl(Tensor *inputTensor, services::KernelErrorCollection *errors, const F &processBlock, size_t minElementsNumInBlock = 997)
{
    const Collection<size_t> &dims = inputTensor->getDimensions();
    TensorOffsetLayout inputLayout = inputTensor->createRawSubtensorLayout();

    size_t fDimN = 0;
    getNumberOfFixedDims(inputLayout, dims, fDimN, minElementsNumInBlock);

    if(fDimN == 0)
    {
        processBlock(fDimN, 0, dims[fDimN], inputLayout);
    }
    else
    {
        size_t nBlocks = inputTensor->getSize(0, fDimN);

        daal::threader_for(nBlocks, nBlocks, [ = ](size_t i)
        {
            SmartPtr<cpu> fDimsPtr(fDimN * sizeof(size_t));
            size_t *fDims = (size_t *)fDimsPtr.get();
            if (!fDims) { errors->add(services::ErrorMemoryAllocationFailed); return; }

            getFixedDimsIndexes(fDimN, fDims, dims, i);
            processBlock(fDimN, fDims, dims[fDimN], inputLayout);
        } );
    }
}

} // internal
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
