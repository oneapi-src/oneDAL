/* file: lrn_layer_forward_impl.i */
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
//  Implementation of the forward local response normalization layer
//--
*/

#ifndef __LRN_LAYER_FORWARD_IMPL_I__
#define __LRN_LAYER_FORWARD_IMPL_I__

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
services::Status LRNKernel<algorithmFPType, method, cpu>::compute(const Tensor &inputTensor, const lrn::Parameter &parameter, Tensor &sMinusBetaTensor,
        Tensor &resultTensor)
{
    const algorithmFPType kappa = parameter.kappa;
    const algorithmFPType alpha = parameter.alpha;
    const algorithmFPType beta = parameter.beta;
    const size_t minElementsNumInBlock = 3000;
    const size_t nAdjust = parameter.nAdjust;
    const size_t halfAdjust = nAdjust / 2;
    const size_t leftAdjust = nAdjust - halfAdjust - 1;
    const size_t rightAdjust = halfAdjust + 1;

    MklTensor<algorithmFPType> *inputMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(const_cast<Tensor *>(&inputTensor));
    MklTensor<algorithmFPType> *sMinusBetaMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(&sMinusBetaTensor);
    MklTensor<algorithmFPType> *resultMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(&resultTensor);

    if (inputMklTensor != 0 && sMinusBetaMklTensor != 0 && resultMklTensor != 0)
    {
        dnnLayout_t inputLayout = (dnnLayout_t)inputMklTensor->getDnnLayout();
        dnnLayout_t workspaceLayout;
        dnnLayout_t resultLayout;

        dnnError_t err;

        if (lrnPrim == NULL)
        {
            err = dnn::xLRNCreateForward( &lrnPrim, inputLayout, nAdjust, alpha * nAdjust, beta, kappa); ON_ERR(err);
        }

        err = dnn::xLayoutCreateFromPrimitive(&workspaceLayout, lrnPrim, dnnResourceWorkspace); ON_ERR(err);
        if (sMinusBetaMklTensor->getDataMemoryStatus() != TensorIface::notAllocated)
        {
            sMinusBetaMklTensor->freeDataMemory();
        }
        sMinusBetaMklTensor->setDnnLayout(workspaceLayout);
        if (sMinusBetaMklTensor->getDataMemoryStatus() == TensorIface::notAllocated)
        {
            sMinusBetaMklTensor->allocateDataMemory();
        }

        err = dnn::xLayoutCreateFromPrimitive(&resultLayout, lrnPrim, dnnResourceDst); ON_ERR(err);
        resultMklTensor->setDnnLayout(resultLayout);

        algorithmFPType *lrnRes[dnnResourceNumber] = {0};

        lrnRes[dnnResourceSrc] = inputMklTensor->getDnnArray();
        lrnRes[dnnResourceWorkspace] = sMinusBetaMklTensor->getDnnArray();
        lrnRes[dnnResourceDst] = resultMklTensor->getDnnArray();

        err = dnn::xExecute(lrnPrim, (void **)lrnRes); ON_ERR(err);
    }
    else
    {
        if(0.0 == beta)
        {
            __DAAL_MAKE_TENSOR_THREADSAFE(&resultTensor)
            __DAAL_MAKE_TENSOR_THREADSAFE(&sMinusBetaTensor)

            return computeImpl<cpu>(inputTensor, [ =, &inputTensor, &sMinusBetaTensor, &resultTensor ](size_t nFixedDims, size_t *fixedDims, size_t nRowsToProcess,
                                          const TensorOffsetLayout & layout) -> Status
            {
                ReadSubtensor<algorithmFPType, cpu, Tensor> inputBlock(const_cast<Tensor &>(inputTensor), nFixedDims, fixedDims, 0, nRowsToProcess, layout);
                DAAL_CHECK_BLOCK_STATUS(inputBlock);
                const algorithmFPType *inputArray = inputBlock.get();

                WriteOnlySubtensor<algorithmFPType, cpu, Tensor> resultBlock(resultTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
                DAAL_CHECK_BLOCK_STATUS(resultBlock);
                algorithmFPType *resultArray = resultBlock.get();

                WriteOnlySubtensor<algorithmFPType, cpu, Tensor> sMinusBetaBlock(sMinusBetaTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
                DAAL_CHECK_BLOCK_STATUS(sMinusBetaBlock);
                algorithmFPType *sMinusBetaArray = sMinusBetaBlock.get();

                const size_t nElements = inputBlock.getSize();

                daal::services::internal::daal_memcpy_s(resultArray, nElements * sizeof(algorithmFPType),
                                                        inputArray, nElements * sizeof(algorithmFPType));
                for(size_t j = 0; j < nElements; j++)
                {
                    sMinusBetaArray[j] = 1;
                }
                return Status();
            });
        }

        size_t targetInd, targetDim;
        {
            TensorOffsetLayout defaultInputLayout = inputTensor.createDefaultSubtensorLayout();
            const Collection<size_t> &defaultIndices = defaultInputLayout.getIndices();
            ReadRows<algorithmFPType, cpu, NumericTable> targetDimBD(parameter.dimension.get(), 0, 1);
            DAAL_CHECK_BLOCK_STATUS(targetDimBD);
            targetDim = targetDimBD.get()[0];
            targetInd = defaultIndices[targetDim];
        }

        const Collection<size_t> &dims = inputTensor.getDimensions();
        size_t targetDimSize = dims[targetDim];

        TensorOffsetLayout rawInputLayout = inputTensor.createRawSubtensorLayout();
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
            __DAAL_MAKE_TENSOR_THREADSAFE(&resultTensor)
            __DAAL_MAKE_TENSOR_THREADSAFE(&sMinusBetaTensor)

            return computeImpl<cpu>(inputTensor, [ =, &inputTensor, &sMinusBetaTensor, &resultTensor ](size_t nFixedDims, size_t *fixedDims, size_t nRowsToProcess,
                                          const TensorOffsetLayout & layout) -> Status
            {
                ReadSubtensor<algorithmFPType, cpu, Tensor> inputBlock(const_cast<Tensor &>(inputTensor), nFixedDims, fixedDims, 0, nRowsToProcess, layout);
                DAAL_CHECK_BLOCK_STATUS(inputBlock);
                const algorithmFPType *inputArray = inputBlock.get();

                WriteOnlySubtensor<algorithmFPType, cpu, Tensor> resultBlock(resultTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
                DAAL_CHECK_BLOCK_STATUS(resultBlock);
                algorithmFPType *resultArray = resultBlock.get();

                WriteOnlySubtensor<algorithmFPType, cpu, Tensor> sMinusBetaBlock(sMinusBetaTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
                DAAL_CHECK_BLOCK_STATUS(sMinusBetaBlock);
                algorithmFPType *sMinusBetaArray = sMinusBetaBlock.get();

                const size_t nElements = inputBlock.getSize();

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

                const size_t leftAdjustLocal = leftAdjust;
                const size_t leftAdjustMax = (leftAdjustLocal > targetDimSize) ? targetDimSize : leftAdjustLocal;
                for(size_t inner = 1; inner <= leftAdjustMax; inner++)
                {
                    for (size_t i = 0; i < offsetBefore; i++)
                    {
                        for (size_t k = 0; k < targetDimSize - inner; k++)
                        {
                            PRAGMA_VECTOR_ALWAYS
                            for (size_t j = 0; j < offsetAfter; j++)
                            {
                                const size_t indexK = (i * targetDimSize + k) * offsetAfter + j;
                                const size_t index = indexK + inner * offsetAfter;
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
                            PRAGMA_VECTOR_ALWAYS
                            for (size_t j = 0; j < offsetAfter; j++)
                            {
                                const size_t indexK = (i * targetDimSize + k) * offsetAfter + j;
                                const size_t index = indexK - inner * offsetAfter;
                                sMinusBetaArray[index] += resultArray[indexK];
                            }
                        }
                    }
                }

                PRAGMA_VECTOR_ALWAYS
                for(size_t j = 0; j < nElements; j++)
                {
                    sMinusBetaArray[j] = kappa + alpha * sMinusBetaArray[j];
                }

                Math<algorithmFPType, cpu>::vPowxAsLnExp(nElements, sMinusBetaArray, -beta, sMinusBetaArray);

                PRAGMA_VECTOR_ALWAYS
                for(size_t i = 0; i < nElements; i++)
                {
                    resultArray[i] = sMinusBetaArray[i] * inputArray[i];
                }
                return Status();
            }, minElementsNumInBlock);
        }
        else
        {
            __DAAL_MAKE_TENSOR_THREADSAFE(&resultTensor)
            __DAAL_MAKE_TENSOR_THREADSAFE(&sMinusBetaTensor)

            return computeImpl<cpu>(inputTensor, [ =, &inputTensor, &sMinusBetaTensor, &resultTensor ]( size_t nFixedDims, size_t *fixedDims, size_t nRowsToProcess,
                                          const TensorOffsetLayout & layout) -> Status
            {
                TArray<size_t, cpu> shiftFixedDimsPtr(nFixedDims);
                size_t *fixedDimsShifted = (size_t *)shiftFixedDimsPtr.get();
                DAAL_CHECK_MALLOC(fixedDimsShifted);

                WriteOnlySubtensor<algorithmFPType, cpu, Tensor> resultBlock(resultTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
                DAAL_CHECK_BLOCK_STATUS(resultBlock)
                algorithmFPType *resultArray = resultBlock.get();

                PRAGMA_VECTOR_ALWAYS
                for(size_t i = 0; i < nFixedDims; i++)
                {
                    fixedDimsShifted[i] = fixedDims[i];
                }

                WriteOnlySubtensor<algorithmFPType, cpu, Tensor> sMinusBetaBlock(sMinusBetaTensor, nFixedDims, fixedDims, 0, nRowsToProcess, layout);
                DAAL_CHECK_BLOCK_STATUS(sMinusBetaBlock)
                algorithmFPType *sMinusBetaArray = sMinusBetaBlock.get();

                ReadSubtensor<algorithmFPType, cpu, Tensor> inputBlock(const_cast<Tensor &>(inputTensor), nFixedDims, fixedDims, 0, nRowsToProcess, layout);
                DAAL_CHECK_BLOCK_STATUS(inputBlock)
                const algorithmFPType *inputArrayAligned = inputBlock.get();

                size_t nElements = sMinusBetaBlock.getSize();
                PRAGMA_VECTOR_ALWAYS
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

                        inputBlockLocal.set(const_cast<Tensor&>(inputTensor), nFixedDims, fixedDimsShifted, 0, nRowsToProcess, layout);
                        DAAL_CHECK_BLOCK_STATUS(inputBlockLocal)
                        inputArray = inputBlockLocal.get();
                    }
                    PRAGMA_VECTOR_ALWAYS
                    for (size_t i = 0; i < nElements; i++)
                    {
                        sMinusBetaArray[i] += inputArray[i] * inputArray[i];
                    }
                }
                PRAGMA_VECTOR_ALWAYS
                for(size_t j = 0; j < nElements; j++)
                {
                    sMinusBetaArray[j] = kappa + alpha * sMinusBetaArray[j];
                }

                Math<algorithmFPType, cpu>::vPowxAsLnExp(nElements, sMinusBetaArray, -beta, sMinusBetaArray);

                PRAGMA_VECTOR_ALWAYS
                for(size_t i = 0; i < nElements; i++)
                {
                    resultArray[i] = sMinusBetaArray[i] * inputArrayAligned[i];
                }
                return Status();
            }, minElementsNumInBlock);
        }
    }
    return Status();
}

} // internal
} // forward
} // namespace lrn
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
