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
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void LRNKernel<algorithmFPType, method, cpu>::compute(Tensor *dataTensor, Tensor *sMinusBetaTensor, Tensor *inputGradientTensor, Tensor *gradientTensor,
                                                      const lrn::Parameter *parameter)
{
    BlockDescriptor<int> dimBlock;
    parameter->dimension->getBlockOfRows(0, 1, readOnly, dimBlock);
    int *dimArray = dimBlock.getBlockPtr();
    size_t targetDim = dimArray[0];
    parameter->dimension->releaseBlockOfRows(dimBlock);

    const services::Collection<size_t> &dims = dataTensor->getDimensions();
    size_t nInputRows = dims[0];
    size_t nElements = dataTensor->getSize();

    algorithmFPType kappa = parameter->kappa;
    algorithmFPType alpha = parameter->alpha;
    algorithmFPType beta = parameter->beta;

    SubtensorDescriptor<algorithmFPType> inputGradientBlock;
    inputGradientTensor->getSubtensor(0, 0, 0, nInputRows, readOnly, inputGradientBlock);
    algorithmFPType *inputGradientArray = inputGradientBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> gradientBlock;
    gradientTensor->getSubtensor(0, 0, 0, nInputRows, writeOnly, gradientBlock);
    algorithmFPType *gradientArray = gradientBlock.getPtr();

    if(0.0 == beta)
    {
        daal::services::daal_memcpy_s(gradientArray, nElements * sizeof(algorithmFPType), inputGradientArray, nElements * sizeof(algorithmFPType));
        inputGradientTensor->releaseSubtensor(inputGradientBlock);
        gradientTensor->releaseSubtensor(gradientBlock);
        return;
    }

    algorithmFPType *sMBetaM1Array = (algorithmFPType *)daal::services::daal_malloc(nElements * sizeof(algorithmFPType));
    if(!sMBetaM1Array)
    {
        inputGradientTensor->releaseSubtensor(inputGradientBlock);
        gradientTensor->releaseSubtensor(gradientBlock);
        this->_errors->add(ErrorMemoryAllocationFailed);
        return;
    }

    SubtensorDescriptor<algorithmFPType> dataBlock;
    dataTensor->getSubtensor(0, 0, 0, nInputRows, readOnly, dataBlock);
    algorithmFPType *dataArray = dataBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> sMinusBetaBlock;
    sMinusBetaTensor->getSubtensor(0, 0, 0, nInputRows, readOnly, sMinusBetaBlock);
    algorithmFPType *sMinusBetaArray = sMinusBetaBlock.getPtr();

    size_t nDims = dims.size();
    Collection<size_t> curDims(nDims);
    for(int i = 0; i < nDims; i++)
    {
        curDims[i] = 0;
    }

    const size_t nAdjust = parameter->nAdjust;
    size_t leftAdjust = (nAdjust / 2);
    size_t rightAdjust = nAdjust - leftAdjust;

    size_t dimOffset = getDimOffset(targetDim, dims);
    size_t idx, curOffset = 0, curOffsetLarger = 0, innerIdx;
    size_t targetDimSize = dims[targetDim];
    size_t targetDimIdx = 0;

    daal::services::daal_memcpy_s(gradientArray, nElements * sizeof(algorithmFPType), sMinusBetaArray, nElements * sizeof(algorithmFPType));
    daal::internal::Math<algorithmFPType,cpu>::vPowx(nElements, sMinusBetaArray, (-beta - 1) / -beta, sMBetaM1Array);

    for(int i = 0; i < nDims; i++)
    {
        curDims[i] = 0;
    }

    algorithmFPType sum;
    algorithmFPType ab2 = 2.0 * alpha * beta;
    for(size_t j = 0; j < nElements; j++)
    {
        targetDimIdx = curDims[targetDim];
        curOffsetLarger = j - targetDimIdx * dimOffset;

        size_t startIdx = (targetDimIdx > leftAdjust) ? targetDimIdx - leftAdjust : 0;
        size_t endIdx = (targetDimIdx + rightAdjust < targetDimSize) ? targetDimIdx + rightAdjust : targetDimSize;
        sum = 0;
        for(size_t k = startIdx; k < endIdx; k++)
        {
            idx = curOffsetLarger + k * dimOffset;
            sum += inputGradientArray[idx] * sMBetaM1Array[idx] * dataArray[idx];
        }

        gradientArray[j] = inputGradientArray[j] * gradientArray[j] - ab2 * dataArray[j] * sum;

        for(size_t d = 1; d < nDims + 1; d++)
        {
            curDims[nDims - d]++;
            if(curDims[nDims - d] < dims[nDims - d]) { break; }
            curDims[nDims - d] = 0;
        }
    }

    daal::services::daal_free(sMBetaM1Array);
    dataTensor->releaseSubtensor(dataBlock);
    inputGradientTensor->releaseSubtensor(inputGradientBlock);
    gradientTensor->releaseSubtensor(gradientBlock);
    sMinusBetaTensor->releaseSubtensor(sMinusBetaBlock);
}

template<typename algorithmFPType, Method method, CpuType cpu>
inline size_t LRNKernel<algorithmFPType, method, cpu>::getDimOffset(size_t k, const Collection<size_t> &full)
{
    size_t n = full.size();
    size_t offset = 1;
    for(size_t d = k + 1; d < n; d++)
    {
        offset *= full[d];
    }
    return offset;
}


} // internal
} // backward
} // namespace lrn
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
