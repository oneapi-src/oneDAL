/* file: dropout_layer_forward_impl.i */
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
//  Implementation of the forward dropout layer
//--
*/

#ifndef __DROPOUT_LAYER_FORWARD_IMPL_I__
#define __DROPOUT_LAYER_FORWARD_IMPL_I__

#include "service_tensor.h"

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
services::Status DropoutKernel<algorithmFPType, method, cpu>::initialize(const dropout::Parameter *parameter)
{
    baseRng.reset(new daal::internal::BaseRNGs<cpu>(parameter->seed));
    rngs.reset(new daal::internal::RNGs<int, cpu>());
    return services::Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DropoutKernel<algorithmFPType, method, cpu>::reset()
{
    baseRng.reset();
    rngs.reset();
    return services::Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DropoutKernel<algorithmFPType, method, cpu>::compute(
    Tensor *inputTensor,
    Tensor *resultTensor,
    Tensor *maskTensor,
    const dropout::Parameter *parameter)
{
    if (parameter->predictionStage && inputTensor == resultTensor) { return services::Status(); }

    retainRatio = parameter->retainRatio;
    inverseRetainRatio = (algorithmFPType) 1.0 / retainRatio;

    const services::Collection<size_t> &dims = inputTensor->getDimensions();
    size_t nInputRows = dims[0];

    size_t nBlocks = nInputRows / _nRowsInBlock;
    size_t nRowsInLastBlock = nInputRows - nBlocks * _nRowsInBlock;

    size_t nElementsInRow = inputTensor->getSize() / nInputRows;
    size_t nRowsInBuffer = (nBlocks > 0) ? _nRowsInBlock : nRowsInLastBlock;

    daal::internal::TArray<int, cpu> rngBuffer(nElementsInRow * nRowsInBuffer);
    if (!rngBuffer.get()) return services::Status(ErrorMemoryAllocationFailed);

    if (parameter->predictionStage == false)
    {
        for (size_t block = 0; block < nBlocks; block++)
        {
            processBlock(inputTensor, block * _nRowsInBlock, _nRowsInBlock,
                resultTensor, maskTensor, rngBuffer.get());
        }
        if (nRowsInLastBlock > 0)
        {
            processBlock(inputTensor, nBlocks * _nRowsInBlock, nRowsInLastBlock,
                resultTensor, maskTensor, rngBuffer.get());
        }
    }
    else
    {
        for (size_t block = 0; block < nBlocks; block++)
        {
            processBlockPrediction(inputTensor, block * _nRowsInBlock, _nRowsInBlock, resultTensor);
        }
        if (nRowsInLastBlock > 0)
        {
            processBlockPrediction(inputTensor, nBlocks * _nRowsInBlock, nRowsInLastBlock, resultTensor);
        }
    }
    DAAL_RETURN_STATUS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
inline void DropoutKernel<algorithmFPType, method, cpu>::processBlock(
    Tensor *inputTensor,
    size_t nProcessedRows,
    size_t nRowsInCurrentBlock,
    Tensor *resultTensor,
    Tensor *maskTensor,
    int *rngBuffer)
{
    daal::internal::ReadSubtensor<algorithmFPType, cpu> inputSubtensor(
        inputTensor, 0, 0, nProcessedRows, nRowsInCurrentBlock);
    daal::internal::WriteOnlySubtensor<algorithmFPType, cpu> resultSubtensor(
        resultTensor, 0, 0, nProcessedRows, nRowsInCurrentBlock);
    daal::internal::WriteOnlySubtensor<algorithmFPType, cpu> maskSubtensor(
        maskTensor, 0, 0, nProcessedRows, nRowsInCurrentBlock);

    const algorithmFPType *inputArray = inputSubtensor.get();
    algorithmFPType *resultArray      = resultSubtensor.get();
    algorithmFPType *maskArray        = maskSubtensor.get();

    size_t nDataElements = inputSubtensor.getSize();

    int errCode = rngs->bernoulli(nDataElements, rngBuffer, *baseRng, retainRatio);
    if (errCode) { this->_errors->add(ErrorIncorrectErrorcodeFromGenerator); return; }

    for (size_t i = 0; i < nDataElements; i++)
    {
        maskArray[i] = rngBuffer[i] * inverseRetainRatio;
        resultArray[i] = inputArray[i] * maskArray[i];
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
inline void DropoutKernel<algorithmFPType, method, cpu>::processBlockPrediction(
    Tensor *inputTensor,
    size_t nProcessedRows,
    size_t nRowsInCurrentBlock,
    Tensor *resultTensor)
{
    daal::internal::ReadSubtensor<algorithmFPType, cpu> inputSubtensor(
        inputTensor, 0, 0, nProcessedRows, nRowsInCurrentBlock);
    daal::internal::WriteOnlySubtensor<algorithmFPType, cpu> resultSubtensor(
        resultTensor, 0, 0, nProcessedRows, nRowsInCurrentBlock);

    const algorithmFPType *inputArray = inputSubtensor.get();
    algorithmFPType *resultArray      = resultSubtensor.get();

    const int nDataElements = inputSubtensor.getSize();
    for (size_t i = 0; i < nDataElements; i++)
    {
        resultArray[i] = inputArray[i];
    }
}

} // internal
} // forward
} // namespace dropout
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
