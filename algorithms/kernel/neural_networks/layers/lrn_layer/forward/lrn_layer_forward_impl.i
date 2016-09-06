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

using namespace daal::internal;
using namespace daal::services;

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

template<typename algorithmFPType, Method method, CpuType cpu>
void LRNKernel<algorithmFPType, method, cpu>::compute(Tensor *inputTensor, const lrn::Parameter *parameter, Tensor *sMinusBetaTensor, Tensor *resultTensor)
{
    BlockDescriptor<int> dimBlock;
    parameter->dimension->getBlockOfRows(0, 1, readOnly, dimBlock);
    int *dimArray = dimBlock.getBlockPtr();
    size_t targetDim = dimArray[0];
    parameter->dimension->releaseBlockOfRows(dimBlock);

    const services::Collection<size_t> &dims = inputTensor->getDimensions();
    size_t nInputRows = dims[0];
    size_t nElements = inputTensor->getSize();

    algorithmFPType kappa = parameter->kappa;
    algorithmFPType alpha = parameter->alpha;
    algorithmFPType beta = parameter->beta;
    const algorithmFPType zero = 0.0;
    const algorithmFPType one = 1.0;

    SubtensorDescriptor<algorithmFPType> inputBlock;
    inputTensor->getSubtensor(0, 0, 0, nInputRows, readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTensor->getSubtensor(0, 0, 0, nInputRows, writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> sMinusBetaBlock;
    sMinusBetaTensor->getSubtensor(0, 0, 0, nInputRows, writeOnly, sMinusBetaBlock);
    algorithmFPType *sMinusBetaArray = sMinusBetaBlock.getPtr();

    if(zero == beta)
    {
        daal::services::daal_memcpy_s(resultArray, nElements * sizeof(algorithmFPType), inputArray, nElements * sizeof(algorithmFPType));
        for(int j = 0; j < nElements; j++)
        {
            sMinusBetaArray[j] = one;
        }
        inputTensor->releaseSubtensor(inputBlock);
        resultTensor->releaseSubtensor(resultBlock);
        sMinusBetaTensor->releaseSubtensor(sMinusBetaBlock);
        return;
    }

    size_t nDims = dims.size();

    const size_t nAdjust = parameter->nAdjust;
    const size_t halfAdjust = nAdjust / 2;
    const size_t leftAdjust = nAdjust - halfAdjust - 1;
    const size_t rightAdjust = halfAdjust + 1;

    size_t dimensionSize = dims[targetDim];

    size_t offsetBefore = 1;
    size_t offsetAfter = 1;
    for (size_t i = 0; i < targetDim; i++)
    {
        offsetBefore *= dims[i];
    }
    for (size_t i = targetDim + 1; i < nDims; i++)
    {
        offsetAfter *= dims[i];
    }

    for(size_t j = 0; j < nElements; j++)
    {
        resultArray[j] = inputArray[j] * inputArray[j];
        sMinusBetaArray[j] = 0;
    }

    size_t leftAdjustMax = (leftAdjust > dimensionSize) ? dimensionSize : leftAdjust;
    for(size_t inner = 1; inner <= leftAdjustMax; inner++)
    {
        for (size_t i = 0; i < offsetBefore; i++)
        {
            for (size_t k = 0; k < dimensionSize - inner; k++)
            {
                for (size_t j = 0; j < offsetAfter; j++)
                {
                    size_t indexK = (i * dimensionSize + k) * offsetAfter + j;
                    size_t index = (i * dimensionSize + (k + inner)) * offsetAfter + j;
                    sMinusBetaArray[index] += resultArray[indexK];
                }
            }
        }
    }

    for(size_t inner = 0; inner < rightAdjust; inner++)
    {
        for (size_t i = 0; i < offsetBefore; i++)
        {
            for (size_t k = inner; k < dimensionSize; k++)
            {
                for (size_t j = 0; j < offsetAfter; j++)
                {
                    size_t indexK = (i * dimensionSize + k) * offsetAfter + j;
                    size_t index = (i * dimensionSize + (k - inner)) * offsetAfter + j;
                    sMinusBetaArray[index] += resultArray[indexK];
                }
            }
        }
    }

    for(size_t j = 0; j < nElements; j++)
    {
        sMinusBetaArray[j] = kappa + alpha * sMinusBetaArray[j];
    }

    daal::internal::Math<algorithmFPType,cpu>::vPowx(nElements, sMinusBetaArray, -beta, sMinusBetaArray);

    for(size_t i = 0; i < nElements; i++)
    {
        resultArray[i] = sMinusBetaArray[i] * inputArray[i];
    }

    inputTensor->releaseSubtensor(inputBlock);
    resultTensor->releaseSubtensor(resultBlock);
    sMinusBetaTensor->releaseSubtensor(sMinusBetaBlock);
}

} // internal
} // forward
} // namespace lrn
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
