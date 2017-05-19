/* file: lrn_layer_backward_impl.i */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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

#include "service_dnn.h"
#include "service_mkl_tensor.h"

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
services::Status LRNKernel<algorithmFPType, method, cpu>::compute(Tensor *auxDataTensor, Tensor *sMinusBetaTensor, Tensor *inputGradientTensor, Tensor *gradientTensor,
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

    MklTensor<algorithmFPType> *auxDataMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(auxDataTensor);
    MklTensor<algorithmFPType> *inputGradientMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(inputGradientTensor);
    MklTensor<algorithmFPType> *sMinusBetaMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(sMinusBetaTensor);
    MklTensor<algorithmFPType> *gradientMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(gradientTensor);

    if (sMinusBetaMklTensor != 0 && !sMinusBetaMklTensor->isPlainLayout())
    {
        algorithmFPType* lrnRes[dnnResourceNumber] = {0};
        dnnError_t err;

        dnnLayout_t inputLayout;
        ReadSubtensor<algorithmFPType, cpu> inputBlock;

        if (auxDataMklTensor != 0)
        {
            inputLayout = (dnnLayout_t)auxDataMklTensor->getDnnLayout();
            lrnRes[dnnResourceSrc] = auxDataMklTensor->getDnnArray();
        }
        else
        {
            const services::Collection<size_t>& inDims = auxDataTensor->getDimensions();
            size_t nDims = inDims.size();
            size_t *inputSize = new size_t[nDims];
            size_t *inputStrides = new size_t[nDims];
            inputSize    [0] = inDims [nDims - 1];
            inputStrides [0] = 1;
            for(size_t i = 1; i < nDims; i++)
            {
                inputSize    [i] = inDims[nDims - 1 - i];
                inputStrides [i] = inputStrides[i - 1] * inputSize[i - 1];
            }
            err = dnn::xLayoutCreate(&inputLayout, nDims, inputSize, inputStrides); ON_ERR(err);

            inputBlock.set(auxDataTensor, 0, 0, 0, inDims[0]);
            lrnRes[dnnResourceSrc] = const_cast<algorithmFPType*>(inputBlock.get());

            delete [] inputSize;
            delete [] inputStrides;
        }

        dnnLayout_t inputGradLayout;
        ReadSubtensor<algorithmFPType, cpu> inputGradBlock;

        if (inputGradientMklTensor != 0)
        {
            inputGradLayout = (dnnLayout_t)inputGradientMklTensor->getDnnLayout();
            lrnRes[dnnResourceDiffDst] = inputGradientMklTensor->getDnnArray();
        }
        else
        {
            const services::Collection<size_t>& inGradDims = inputGradientTensor->getDimensions();
            size_t nDims = inGradDims.size();
            size_t *inputGradSize = new size_t[nDims];
            size_t *inputGradStrides = new size_t[nDims];
            inputGradSize    [0] = inGradDims[nDims - 1];
            inputGradStrides [0] = 1;
            for(size_t i = 1; i < nDims; i++)
            {
                inputGradSize    [i] = inGradDims[nDims - 1 - i];
                inputGradStrides [i] = inputGradStrides[i - 1] * inputGradSize[i - 1];
            }
            err = dnn::xLayoutCreate(&inputGradLayout, nDims, inputGradSize, inputGradStrides); ON_ERR(err);

            inputGradBlock.set(inputGradientTensor, 0, 0, 0, inGradDims[0]);
            lrnRes[dnnResourceDiffDst] = const_cast<algorithmFPType*>(inputGradBlock.get());

            delete [] inputGradSize;
            delete [] inputGradStrides;
        }

        if (lrnPrim == NULL)
        {
            err = dnn::xLRNCreateBackward( &lrnPrim, inputGradLayout, inputLayout, nAdjust, alpha * nAdjust, beta, kappa); ON_ERR(err);
        }

        dnnLayout_t workspaceLayout;
        err = dnn::xLayoutCreateFromPrimitive(&workspaceLayout, lrnPrim, dnnResourceWorkspace); ON_ERR(err);
        sMinusBetaMklTensor->setDnnLayout(workspaceLayout);
        lrnRes[dnnResourceWorkspace] = sMinusBetaMklTensor->getDnnArray();

        dnnLayout_t gradLayout;
        err = dnn::xLayoutCreateFromPrimitive(&gradLayout, lrnPrim, dnnResourceDiffSrc); ON_ERR(err);
        WriteOnlySubtensor<algorithmFPType, cpu> gradBlock;

        if (gradientMklTensor != 0)
        {
            gradientMklTensor->setDnnLayout(gradLayout);
            lrnRes[dnnResourceDiffSrc] = gradientMklTensor->getDnnArray();

            err = dnn::xExecute(lrnPrim, (void**)lrnRes); ON_ERR(err);
        }
        else
        {
            const services::Collection<size_t>& outDims = gradientTensor->getDimensions();
            size_t nDims = outDims.size();
            size_t *gradSize = new size_t[nDims];
            size_t *gradStrides = new size_t[nDims];
            gradSize   [0] = outDims[nDims - 1];
            gradStrides[0] = 1;
            for(size_t i = 1; i < nDims; i++)
            {
                gradSize    [i] = outDims[nDims - 1 - i];
                gradStrides [i] = gradStrides[i - 1] * gradSize[i - 1];
            }
            err = dnn::xLayoutCreate(&gradLayout, nDims, gradSize, gradStrides); ON_ERR(err);

            gradBlock.set(gradientTensor, 0, 0, 0, outDims[0]);
            algorithmFPType *gradArray = gradBlock.get();

            LayoutConvertor<algorithmFPType, cpu> cvFromInnerOutput(&lrnRes[dnnResourceDiffSrc], gradLayout, false, &gradArray, gradLayout, true); ON_ERR(cvFromInnerOutput.err);

            err = dnn::xExecute(lrnPrim, (void**)lrnRes);

            cvFromInnerOutput.convert(); ON_ERR(cvFromInnerOutput.err);
            delete [] gradSize;
            delete [] gradStrides;
        }
    }
    else
    {
        if(0.0 == beta)
        {
            __DAAL_MAKE_TENSOR_THREADSAFE(gradientTensor)

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
            DAAL_RETURN_STATUS()
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
            __DAAL_MAKE_TENSOR_THREADSAFE(auxDataTensor)
            __DAAL_MAKE_TENSOR_THREADSAFE(sMinusBetaTensor)
            __DAAL_MAKE_TENSOR_THREADSAFE(gradientTensor)

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

                TArray<algorithmFPType, cpu> sMinusBetaMinusOnePtr(nElements);
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
            __DAAL_MAKE_TENSOR_THREADSAFE(auxDataTensor)
            __DAAL_MAKE_TENSOR_THREADSAFE(sMinusBetaTensor)
            __DAAL_MAKE_TENSOR_THREADSAFE(gradientTensor)

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

                TArray<algorithmFPType, cpu> sMinusBetaMinusOnePtr(nElements);
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
    DAAL_RETURN_STATUS()
}

} // internal
} // backward
} // namespace lrn
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
