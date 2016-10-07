/* file: dropout_layer_forward_impl.i */
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
//  Implementation of the forward dropout layer
//--
*/

#ifndef __DROPOUT_LAYER_FORWARD_IMPL_I__
#define __DROPOUT_LAYER_FORWARD_IMPL_I__

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
namespace dropout
{
namespace forward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void DropoutKernel<algorithmFPType, method, cpu>::compute(Tensor *inputTensor, Tensor *resultTensor, Tensor *maskTensor, const dropout::Parameter *parameter)
{
    daal::internal::BaseRNGs<cpu> baseRng(parameter->seed);
    daal::internal::RNGs<int, cpu> rng;

    retainRatio = parameter->retainRatio;
    inverseRetainRatio = (algorithmFPType) 1.0 / retainRatio;

    const services::Collection<size_t> &dims = inputTensor->getDimensions();
    size_t nInputRows = dims[0];

    size_t nBlocks = nInputRows / _nRowsInBlock;
    size_t nRowsInLastBlock = nInputRows - nBlocks * _nRowsInBlock;

    size_t nElementsInRow = inputTensor->getSize() / nInputRows;
    size_t nRowsInBuffer = (nBlocks > 0) ? _nRowsInBlock : nRowsInLastBlock;
    rngBuffer = (int *)daal_malloc(nElementsInRow * nRowsInBuffer * sizeof(int));
    if(!rngBuffer) {this->_errors->add(ErrorMemoryAllocationFailed); return;}

    if(parameter->predictionStage == false)
    {
        for(size_t block = 0; block < nBlocks; block++)
        {
            processBlock(inputTensor, block * _nRowsInBlock, _nRowsInBlock, resultTensor, maskTensor, rng, baseRng);
        }
        if(nRowsInLastBlock > 0)
        {
            processBlock(inputTensor, nBlocks * _nRowsInBlock, nRowsInLastBlock, resultTensor, maskTensor, rng, baseRng);
        }
    }
    else
    {
        for(size_t block = 0; block < nBlocks; block++)
        {
            processBlockPrediction(inputTensor, block * _nRowsInBlock, _nRowsInBlock, resultTensor, rng, baseRng);
        }
        if(nRowsInLastBlock > 0)
        {
            processBlockPrediction(inputTensor, nBlocks * _nRowsInBlock, nRowsInLastBlock, resultTensor, rng, baseRng);
        }

    }

    daal_free(rngBuffer);
}

template<typename algorithmFPType, Method method, CpuType cpu>
inline void DropoutKernel<algorithmFPType, method, cpu>::processBlock(
    Tensor *inputTensor,
    size_t nProcessedRows,
    size_t nRowsInCurrentBlock,
    Tensor *resultTensor,
    Tensor *maskTensor,
    daal::internal::RNGs<int, cpu> &rng,
    daal::internal::BaseRNGs<cpu> &baseRng)
{
    SubtensorDescriptor<algorithmFPType> inputBlock;
    inputTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> maskBlock;
    maskTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, writeOnly, maskBlock);
    algorithmFPType *maskArray = maskBlock.getPtr();

   size_t nDataElements = inputBlock.getSize();

    int errCode = rng.bernoulli(nDataElements, rngBuffer, baseRng, retainRatio);
    if(errCode) { this->_errors->add(ErrorIncorrectErrorcodeFromGenerator); }

    for(size_t i = 0; i < nDataElements; i++)
    {
        maskArray[i] = rngBuffer[i] * inverseRetainRatio;
        resultArray[i] = inputArray[i] * maskArray[i];
    }

    maskTensor->releaseSubtensor(maskBlock);
    inputTensor->releaseSubtensor(inputBlock);
    resultTensor->releaseSubtensor(resultBlock);
}

template<typename algorithmFPType, Method method, CpuType cpu>
inline void DropoutKernel<algorithmFPType, method, cpu>::processBlockPrediction(
    Tensor *inputTensor,
    size_t nProcessedRows,
    size_t nRowsInCurrentBlock,
    Tensor *resultTensor,
    daal::internal::RNGs<int, cpu> &rng, daal::internal::BaseRNGs<cpu> &baseRng)
{
    SubtensorDescriptor<algorithmFPType> inputBlock;
    inputTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    const int nDataElements = inputBlock.getSize();

    int errCode = rng.bernoulli(nDataElements, rngBuffer, baseRng, retainRatio);
    if(errCode) { this->_errors->add(ErrorIncorrectErrorcodeFromGenerator); }

    for(size_t i = 0; i < nDataElements; i++)
    {
        resultArray[i] = inputArray[i] * rngBuffer[i] * inverseRetainRatio;
    }

    inputTensor->releaseSubtensor(inputBlock);
    resultTensor->releaseSubtensor(resultBlock);
}

} // internal
} // forward
} // namespace dropout
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
