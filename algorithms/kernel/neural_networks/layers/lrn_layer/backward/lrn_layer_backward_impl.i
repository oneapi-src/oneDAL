/* file: lrn_layer_backward_impl.i */
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
//  Implementation of the backward local response normalization layer
//--
*/
/*
    Let x is p-dimesinal tensor with size n1 x n2 x ... x ... x np stored in one memory array
    So x(i1, i2, ... , np) = x[ i1 * (n2 * n3 * ... * np) + i2 * (n3 * ... * np) + ... + i(p-1) * np + ip]
    Let ind(x(i1, i2, ... , np)) = i1 * (n2 * n3 * ... * np) + i2 * (n3 * ... * np) + ... + i(p-1) * np + ip
    We choose k - target dimension, then
    ind(x(i1, i2, ... ik, ... , np)) =
                               = i1 * (n2 * n3 * ... * np) + i2 * (n3 * ... * np) + ... + ik * (n(k+1) * ... * np) + ... + i(p-1) * np + ip
    and
    ind(x(i1, i2, ... ik + k', ... , np)) =
                               = i1 * (n2 * n3 * ... * np) + i2 * (n3 * ... * np) + ... + (ik + k') * (n(k+1) * ... * np) + ... + i(p-1) * np + ip
                               = ind(x(i1, i2, ... ik, ... , np)) + k' * (n(k+1) * ... * np)
    dimOffset(k) = (n(k+1) * ... * np)
    curOffsetTargetZero = ind(x(i1, i2, ... , i(k-1), 0, i(k+1), ... , np))
*/

#ifndef __LRN_LAYER_BACKWARD_IMPL_I__
#define __LRN_LAYER_BACKWARD_IMPL_I__

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
namespace backward
{
namespace internal
{

template<CpuType cpu, typename algorithmFPType>
int shiftFixedDims(size_t *fixedDims, int shiftValue, size_t pos, size_t dimSize, size_t *dimsShifted, size_t nDims)
{
    int innerDimPosition = fixedDims[pos];
    if(innerDimPosition + shiftValue < 0) { return 1; }
    if(innerDimPosition + shiftValue >= dimSize) { return 2; }

    dimsShifted[pos] = fixedDims[pos] + shiftValue;
    return 0;
}

template<typename algorithmFPType, Method method, CpuType cpu>
void LRNKernel<algorithmFPType, method, cpu>::compute(Tensor *auxDataTensor, Tensor *sMinusBetaTensor, Tensor *inputGradientTensor, Tensor *gradientTensor,
        const lrn::Parameter *parameter)
{
    const size_t minElementsNumInBlock = 3000;
    const algorithmFPType kappa = parameter->kappa;
    const algorithmFPType alpha = parameter->alpha;
    const algorithmFPType beta = parameter->beta;
    const algorithmFPType ab2 = 2.0 * alpha * beta;
    const algorithmFPType toMinusBetaPower = (-beta - 1) / -beta;
    const size_t nAdjust = parameter->nAdjust;
    const size_t halfAdjust = nAdjust / 2;
    const size_t rightAdjust = nAdjust - halfAdjust;
    const size_t leftAdjust = halfAdjust;

    if(0.0 == beta)
    {
        computeImpl<cpu>(inputGradientTensor, this->_errors.get(), [ = ](size_t nFixedDims, size_t *fixedDims, size_t nRowsToProcess,
                         const TensorOffsetLayout & layout)
        {
            ReadSubtensor<algorithmFPType, cpu, Tensor> inputGradientBlock(*inputGradientTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
            const algorithmFPType *inputGradientArray = inputGradientBlock.get();

            WriteOnlySubtensor<algorithmFPType, cpu, Tensor> gradientBlock(*gradientTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
            algorithmFPType *gradientArray = gradientBlock.get();

            size_t nElements = inputGradientBlock.getSize();

            daal::services::daal_memcpy_s(gradientArray, nElements * sizeof(algorithmFPType), inputGradientArray, nElements * sizeof(algorithmFPType));
        });
        return;
    }

    size_t targetInd, targetDim;
    {
        TensorOffsetLayout defaultinputGradientLayout = inputGradientTensor->createDefaultSubtensorLayout();
        const Collection<size_t> &defaultIndices = defaultinputGradientLayout.getIndices();
        ReadRows<algorithmFPType, cpu, NumericTable> targetDimBD(parameter->dimension.get(), 0, 1);
        targetDim = targetDimBD.get()[0];
        targetInd = defaultIndices[targetDim];
    }

    const Collection<size_t> &dims = inputGradientTensor->getDimensions();
    size_t targetDimSize = dims[targetDim];

    TensorOffsetLayout rawInputGradientLayout = inputGradientTensor->createRawSubtensorLayout();
    const Collection<size_t> &rawIndices = rawInputGradientLayout.getIndices();
    size_t nDims = dims.size();

    size_t firstFreeDim = 0;
    getNumberOfFixedDims(rawInputGradientLayout, dims, firstFreeDim, minElementsNumInBlock);
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
        computeImpl<cpu>(inputGradientTensor, this->_errors.get(), [ = ](size_t nFixedDims, size_t *fixedDims, size_t nRowsToProcess,
                         const TensorOffsetLayout & layout)
        {
            ReadSubtensor<algorithmFPType, cpu, Tensor> inputGradientBlock(*inputGradientTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
            const algorithmFPType *inputGradientArray = inputGradientBlock.get();

            ReadSubtensor<algorithmFPType, cpu, Tensor> auxDataBlock(*auxDataTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
            const algorithmFPType *auxDataArray = auxDataBlock.get();

            ReadSubtensor<algorithmFPType, cpu, Tensor> sMinusBetaBlock(*sMinusBetaTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
            const algorithmFPType *sMinusBetaArray = sMinusBetaBlock.get();

            WriteOnlySubtensor<algorithmFPType, cpu, Tensor> gradientBlock(*gradientTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
            algorithmFPType *gradientArray = gradientBlock.get();

            size_t nElements = inputGradientBlock.getSize();

            TSmartPtr<algorithmFPType, cpu> sMinusBetaMinusOnePtr(nElements);
            algorithmFPType *sMinusBetaMinusOneArray = sMinusBetaMinusOnePtr.get();

            Math<algorithmFPType, cpu>::vPowxAsLnExp(nElements, const_cast<algorithmFPType *>(sMinusBetaArray), toMinusBetaPower, sMinusBetaMinusOneArray);

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
                gradientArray[j] = 0;
            }

            size_t leftAdjustLocal = leftAdjust;
            size_t leftAdjustMax = (leftAdjustLocal > targetDimSize) ? targetDimSize : leftAdjustLocal;
            for(int inner = 1; inner <= leftAdjustMax; inner++)
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
                            gradientArray[index] += inputGradientArray[indexK] * auxDataArray[indexK] * sMinusBetaMinusOneArray[indexK];
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
                            gradientArray[index] += inputGradientArray[indexK] * auxDataArray[indexK] * sMinusBetaMinusOneArray[indexK];
                        }
                    }
                }
            }

            PRAGMA_SIMD_ASSERT
            for(size_t j = 0; j < nElements; j++)
            {
                gradientArray[j] = inputGradientArray[j] * sMinusBetaArray[j] - ab2 * auxDataArray[j] * gradientArray[j];
            }
        }, minElementsNumInBlock);
    }
    else
    {
        computeImpl<cpu>(inputGradientTensor, this->_errors.get(), [ = ]( size_t nFixedDims, size_t *fixedDims, size_t nRowsToProcess,
                         const TensorOffsetLayout & layout)
        {
            SmartPtr<cpu> shiftFixedDimsPtr(nFixedDims * sizeof(size_t));
            size_t *fixedDimsShifted = (size_t *)shiftFixedDimsPtr.get();
            WriteOnlySubtensor<algorithmFPType, cpu, Tensor> gradientBlock(*gradientTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
            algorithmFPType *gradientArray = gradientBlock.get();
            size_t nElements = gradientBlock.getSize();
            PRAGMA_SIMD_ASSERT
            for(size_t j = 0; j < nElements; j++)
            {
                gradientArray[j] = 0;
            }
            PRAGMA_SIMD_ASSERT
            for(size_t i = 0; i < nFixedDims; i++)
            {
                fixedDimsShifted[i] = fixedDims[i];
            }

            ReadSubtensor<algorithmFPType, cpu, Tensor> inputGradientBlock(*inputGradientTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
            const algorithmFPType *inputGradientArray = inputGradientBlock.get();

            ReadSubtensor<algorithmFPType, cpu, Tensor> auxDataBlock(*auxDataTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
            const algorithmFPType *auxDataArray = auxDataBlock.get();

            ReadSubtensor<algorithmFPType, cpu, Tensor> sMinusBetaBlock(*sMinusBetaTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
            const algorithmFPType *sMinusBetaArray = sMinusBetaBlock.get();

            TSmartPtr<algorithmFPType, cpu> sMinusBetaMinusOnePtr(nElements);
            algorithmFPType *sMinusBetaMinusOneArray = sMinusBetaMinusOnePtr.get();

            int l = -((int)leftAdjust);
            int r = rightAdjust;
            for(int shiftValue = l; shiftValue < r; shiftValue++)
            {
                const algorithmFPType *inputGradientArrayShifted = inputGradientArray;
                const algorithmFPType *auxDataArrayShifted = auxDataArray;
                const algorithmFPType *sMinusBetaArrayShifted = sMinusBetaArray;
                ReadSubtensor<algorithmFPType, cpu, Tensor> inputGradientBlockLocal;
                ReadSubtensor<algorithmFPType, cpu, Tensor> auxDataBlockLocal;
                ReadSubtensor<algorithmFPType, cpu, Tensor> sMinusBetaBlockLocal;

                if(shiftValue != 0)
                {
                    int retCode = shiftFixedDims<cpu, algorithmFPType>(fixedDims, shiftValue, targetDimPosition, targetDimSize, fixedDimsShifted, nFixedDims);
                    if(retCode != 0) { continue; }
                    inputGradientBlockLocal.set(*inputGradientTensor, nFixedDims, fixedDimsShifted, 0, nRowsToProcess, layout);
                    inputGradientArrayShifted = inputGradientBlockLocal.get();

                    auxDataBlockLocal.set(*auxDataTensor, nFixedDims, fixedDimsShifted, 0, nRowsToProcess, layout);
                    auxDataArrayShifted = auxDataBlockLocal.get();

                    sMinusBetaBlockLocal.set(*sMinusBetaTensor, nFixedDims, fixedDimsShifted, 0, nRowsToProcess, layout);
                    sMinusBetaArrayShifted = sMinusBetaBlockLocal.get();
                }

                Math<algorithmFPType, cpu>::vPowxAsLnExp(nElements, const_cast<algorithmFPType *>(sMinusBetaArrayShifted), toMinusBetaPower, sMinusBetaMinusOneArray);

                PRAGMA_SIMD_ASSERT
                for (size_t i = 0; i < nElements; i++)
                {
                    gradientArray[i] += inputGradientArrayShifted[i] * auxDataArrayShifted[i] * sMinusBetaMinusOneArray[i];
                }
            }
            PRAGMA_SIMD_ASSERT
            for(size_t j = 0; j < nElements; j++)
            {
                gradientArray[j] = inputGradientArray[j] * sMinusBetaArray[j] - ab2 * auxDataArray[j] * gradientArray[j];
            }
        }, minElementsNumInBlock);
    }
}

} // internal
} // backward
} // namespace lrn
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
