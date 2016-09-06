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
void DropoutKernel<algorithmFPType, method, cpu>::compute(const dropout::forward::Input *input, const dropout::Parameter *parameter,
        dropout::forward::Result *result)
{
    SharedPtr<Tensor> inputTable = input->get(layers::forward::data);
    SharedPtr<Tensor> resultTable = result->get(layers::forward::value);

    daal::internal::IntRng<int, cpu> rng(parameter->seed);
    retainRatio = parameter->retainRatio;
    inverseRetainRatio = (algorithmFPType) 1.0 / retainRatio;

    const services::Collection<size_t> &dims = inputTable->getDimensions();
    size_t nInputRows = dims[0];

    size_t nBlocks = nInputRows / _nRowsInBlock;
    size_t nRowsInLastBlock = nInputRows - nBlocks * _nRowsInBlock;

    size_t nElementsInRow = inputTable->getSize() / nInputRows;
    size_t nRowsInBuffer = (nBlocks > 0) ? _nRowsInBlock : nRowsInLastBlock;
    rngBuffer = (int *)daal_malloc(nElementsInRow * nRowsInBuffer * sizeof(int));
    if(!rngBuffer) {this->_errors->add(ErrorMemoryAllocationFailed); return;}

    if(parameter->predictionStage == false)
    {
        SharedPtr<Tensor> maskTable = result->get(auxRetainMask);
        for(size_t block = 0; block < nBlocks; block++)
        {
            processBlock(inputTable, block * _nRowsInBlock, _nRowsInBlock, resultTable, maskTable, rng);
        }
        if(nRowsInLastBlock > 0)
        {
            processBlock(inputTable, nBlocks * _nRowsInBlock, nRowsInLastBlock, resultTable, maskTable, rng);
        }
    }
    else
    {
        for(size_t block = 0; block < nBlocks; block++)
        {
            processBlockPrediction(inputTable, block * _nRowsInBlock, _nRowsInBlock, resultTable, rng);
        }
        if(nRowsInLastBlock > 0)
        {
            processBlockPrediction(inputTable, nBlocks * _nRowsInBlock, nRowsInLastBlock, resultTable, rng);
        }

    }

    daal_free(rngBuffer);
}

template<typename algorithmFPType, Method method, CpuType cpu>
inline void DropoutKernel<algorithmFPType, method, cpu>::processBlock(
    SharedPtr<Tensor> inputTable,
    size_t nProcessedRows,
    size_t nRowsInCurrentBlock,
    SharedPtr<Tensor> resultTable,
    SharedPtr<Tensor> maskTable,
    daal::internal::IntRng<int, cpu> &rng)
{
    SubtensorDescriptor<algorithmFPType> inputBlock;
    inputTable->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTable->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> maskBlock;
    maskTable->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, writeOnly, maskBlock);
    algorithmFPType *maskArray = maskBlock.getPtr();

    size_t nDataElements = inputBlock.getSize();

    rng.bernoulli(nDataElements, rngBuffer, retainRatio);

    for(size_t i = 0; i < nDataElements; i++)
    {
        maskArray[i] = rngBuffer[i] * inverseRetainRatio;
        resultArray[i] = inputArray[i] * maskArray[i];
    }

    maskTable->releaseSubtensor(maskBlock);
    inputTable->releaseSubtensor(inputBlock);
    resultTable->releaseSubtensor(resultBlock);
}

template<typename algorithmFPType, Method method, CpuType cpu>
inline void DropoutKernel<algorithmFPType, method, cpu>::processBlockPrediction(
    SharedPtr<Tensor> inputTable,
    size_t nProcessedRows,
    size_t nRowsInCurrentBlock,
    SharedPtr<Tensor> resultTable,
    daal::internal::IntRng<int, cpu> &rng)
{
    SubtensorDescriptor<algorithmFPType> inputBlock;
    inputTable->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTable->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    size_t nDataElements = inputBlock.getSize();

    rng.bernoulli(nDataElements, rngBuffer, retainRatio);

    for(size_t i = 0; i < nDataElements; i++)
    {
        resultArray[i] = inputArray[i] * rngBuffer[i] * inverseRetainRatio;
    }

    inputTable->releaseSubtensor(inputBlock);
    resultTable->releaseSubtensor(resultBlock);
}

} // internal
} // forward
} // namespace dropout
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
