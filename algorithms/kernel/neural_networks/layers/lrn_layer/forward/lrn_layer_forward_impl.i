/* file: lrn_layer_forward_impl.i */
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
//  Implementation of the forward local response normalization layer
//--
*/

#ifndef __LRN_LAYER_FORWARD_IMPL_I__
#define __LRN_LAYER_FORWARD_IMPL_I__

#include "service_numeric_table.h"
#include "layers_threading.h"

using namespace daal::internal;
using namespace daal::services;
using namespace daal::algorithms::neural_networks::layers::internal;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace lrn
{
namespace forward
{
namespace internal
{
template<CpuType cpu, typename algorithmFPType>
int shiftFixedDims(size_t *fixedDims, int shiftValue, size_t pos, size_t dimSize, size_t *fixedDimsShifted, size_t nDims)
{
    int innerDimPosition = fixedDims[pos];
    if(innerDimPosition + shiftValue < 0) { return 1; }
    if(innerDimPosition + shiftValue >= dimSize) { return 2; }

    fixedDimsShifted[pos] = fixedDims[pos] + shiftValue;
    return 0;
}

template<typename algorithmFPType, Method method, CpuType cpu>
void LRNKernel<algorithmFPType, method, cpu>::compute(Tensor *inputTensor, const lrn::Parameter *parameter, Tensor *sMinusBetaTensor,
        Tensor *resultTensor)
{
    algorithmFPType kappa = parameter->kappa;
    algorithmFPType alpha = parameter->alpha;
    algorithmFPType beta = parameter->beta;
    const size_t minElementsNumInBlock = 3000;
    const size_t nAdjust = parameter->nAdjust;
    const size_t halfAdjust = nAdjust / 2;
    const size_t leftAdjust = nAdjust - halfAdjust - 1;
    const size_t rightAdjust = halfAdjust + 1;

    if(0.0 == beta)
    {
        computeImpl<cpu>(inputTensor, this->_errors.get(), [ = ](size_t nFixedDims, size_t *fixedDims, size_t nRowsToProcess,
                         const TensorOffsetLayout & layout)
        {
            ReadSubtensor<algorithmFPType, cpu, Tensor> inputBlock(*inputTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
            const algorithmFPType *inputArray = inputBlock.get();

            WriteOnlySubtensor<algorithmFPType, cpu, Tensor> resultBlock(*resultTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
            algorithmFPType *resultArray = resultBlock.get();

            WriteOnlySubtensor<algorithmFPType, cpu, Tensor> sMinusBetaBlock(*sMinusBetaTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
            algorithmFPType *sMinusBetaArray = sMinusBetaBlock.get();

            size_t nElements = inputBlock.getSize();

            daal::services::daal_memcpy_s(resultArray, nElements * sizeof(algorithmFPType), inputArray, nElements * sizeof(algorithmFPType));
            for(size_t j = 0; j < nElements; j++)
            {
                sMinusBetaArray[j] = 1;
            }
        });
        return;
    }

    size_t targetInd, targetDim;
    {
        TensorOffsetLayout defaultInputLayout = inputTensor->createDefaultSubtensorLayout();
        const Collection<size_t> &defaultIndices = defaultInputLayout.getIndices();
        ReadRows<algorithmFPType, cpu, NumericTable> targetDimBD(parameter->dimension.get(), 0, 1);
        targetDim = targetDimBD.get()[0];
        targetInd = defaultIndices[targetDim];
    }

    const Collection<size_t> &dims = inputTensor->getDimensions();
    size_t targetDimSize = dims[targetDim];

    TensorOffsetLayout rawInputLayout = inputTensor->createRawSubtensorLayout();
    const Collection<size_t> &rawIndices = rawInputLayout.getIndices();
    size_t nDims = dims.size();

    size_t firstFreeDim = 0;
    getNumberOfFixedDims(rawInputLayout, dims, firstFreeDim, minElementsNumInBlock);
    size_t nFixedDims = firstFreeDim;

    size_t targetDimPosition = nDims;
    for(size_t i = 0; i < nDims; i++)
    {
        if(rawIndices[i] == targetInd)
        {
            targetDimPosition = i;
        }
    }
    DAAL_CHECK(targetDimPosition != nDims, ErrorIncorrectParameter);


    if(targetDimPosition >= firstFreeDim)
    {
        computeImpl<cpu>(inputTensor, this->_errors.get(), [ = ](size_t nFixedDims, size_t *fixedDims, size_t nRowsToProcess,
                         const TensorOffsetLayout & layout)
        {
            ReadSubtensor<algorithmFPType, cpu, Tensor> inputBlock(*inputTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
            const algorithmFPType *inputArray = inputBlock.get();

            WriteOnlySubtensor<algorithmFPType, cpu, Tensor> resultBlock(*resultTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
            algorithmFPType *resultArray = resultBlock.get();

            WriteOnlySubtensor<algorithmFPType, cpu, Tensor> sMinusBetaBlock(*sMinusBetaTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
            algorithmFPType *sMinusBetaArray = sMinusBetaBlock.get();

            size_t nElements = inputBlock.getSize();

            size_t offsetBefore = 1;
            size_t offsetAfter = 1;
            for (size_t i = firstFreeDim; i < targetDimPosition; i++)
            {
                offsetBefore *= dims[i];
            }
            for (size_t i = targetDimPosition + 1; i < nDims; i++)
            {
                offsetAfter *= dims[i];
            }

            for(size_t j = 0; j < nElements; j++)
            {
                resultArray[j] = inputArray[j] * inputArray[j];
                sMinusBetaArray[j] = 0;
            }

            size_t leftAdjustLocal = leftAdjust;
            size_t leftAdjustMax = (leftAdjustLocal > targetDimSize) ? targetDimSize : leftAdjustLocal;
            for(size_t inner = 1; inner <= leftAdjustMax; inner++)
            {
                for (size_t i = 0; i < offsetBefore; i++)
                {
                    for (size_t k = 0; k < targetDimSize - inner; k++)
                    {
                        PRAGMA_SIMD_ASSERT
                        for (size_t j = 0; j < offsetAfter; j++)
                        {
                            size_t indexK = (i * targetDimSize + k) * offsetAfter + j;
                            size_t index = indexK + inner * offsetAfter;
                            sMinusBetaArray[index] += resultArray[indexK];
                        }
                    }
                }
            }

            for(size_t inner = 0; inner < rightAdjust; inner++)
            {
                for (size_t i = 0; i < offsetBefore; i++)
                {
                    for (size_t k = inner; k < targetDimSize; k++)
                    {
                        PRAGMA_SIMD_ASSERT
                        for (size_t j = 0; j < offsetAfter; j++)
                        {
                            size_t indexK = (i * targetDimSize + k) * offsetAfter + j;
                            size_t index = indexK - inner * offsetAfter;
                            sMinusBetaArray[index] += resultArray[indexK];
                        }
                    }
                }
            }

            PRAGMA_SIMD_ASSERT
            for(size_t j = 0; j < nElements; j++)
            {
                sMinusBetaArray[j] = kappa + alpha * sMinusBetaArray[j];
            }

            daal::internal::Math<algorithmFPType, cpu>::vPowxAsLnExp(nElements, sMinusBetaArray, -beta, sMinusBetaArray);

            PRAGMA_SIMD_ASSERT
            for(size_t i = 0; i < nElements; i++)
            {
                resultArray[i] = sMinusBetaArray[i] * inputArray[i];
            }
        }, minElementsNumInBlock);
    }
    else
    {
        computeImpl<cpu>(inputTensor, this->_errors.get(), [ = ]( size_t nFixedDims, size_t *fixedDims, size_t nRowsToProcess,
                         const TensorOffsetLayout & layout)
        {
            SmartPtr<cpu> shiftFixedDimsPtr(nFixedDims * sizeof(size_t));
            size_t *fixedDimsShifted = (size_t *)shiftFixedDimsPtr.get();
            WriteOnlySubtensor<algorithmFPType, cpu, Tensor> resultBlock(*resultTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
            algorithmFPType *resultArray = resultBlock.get();
            PRAGMA_SIMD_ASSERT
            for(size_t i = 0; i < nFixedDims; i++)
            {
                fixedDimsShifted[i] = fixedDims[i];
            }

            WriteOnlySubtensor<algorithmFPType, cpu, Tensor> sMinusBetaBlock(*sMinusBetaTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
            algorithmFPType *sMinusBetaArray = sMinusBetaBlock.get();

            ReadSubtensor<algorithmFPType, cpu, Tensor> inputBlock(*inputTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
            const algorithmFPType *inputArrayAligned = inputBlock.get();

            size_t nElements = sMinusBetaBlock.getSize();
            PRAGMA_SIMD_ASSERT
            for(size_t j = 0; j < nElements; j++)
            {
                sMinusBetaArray[j] = 0;
            }

            int l = -((int)leftAdjust);
            int r = rightAdjust;
            for(int shiftValue = l; shiftValue < r; shiftValue++)
            {
                const algorithmFPType *inputArray = inputArrayAligned;
                ReadSubtensor<algorithmFPType, cpu, Tensor> inputBlockLocal;
                if(shiftValue != 0)
                {
                    int retCode = shiftFixedDims<cpu, algorithmFPType>(fixedDims, shiftValue, targetDimPosition, targetDimSize, fixedDimsShifted, nFixedDims);
                    if(retCode != 0) { continue; }
                    inputBlockLocal.set(*inputTensor, nFixedDims, fixedDimsShifted, 0, nRowsToProcess, layout);
                    inputArray = inputBlockLocal.get();
                }
                PRAGMA_SIMD_ASSERT
                for (size_t i = 0; i < nElements; i++)
                {
                    sMinusBetaArray[i] += inputArray[i] * inputArray[i];
                }
            }
            PRAGMA_SIMD_ASSERT
            for(size_t j = 0; j < nElements; j++)
            {
                sMinusBetaArray[j] = kappa + alpha * sMinusBetaArray[j];
            }

            daal::internal::Math<algorithmFPType, cpu>::vPowxAsLnExp(nElements, sMinusBetaArray, -beta, sMinusBetaArray);

            PRAGMA_SIMD_ASSERT
            for(size_t i = 0; i < nElements; i++)
            {
                resultArray[i] = sMinusBetaArray[i] * inputArrayAligned[i];
            }
        }, minElementsNumInBlock);
    }
}

} // internal
} // forward
} // namespace lrn
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
