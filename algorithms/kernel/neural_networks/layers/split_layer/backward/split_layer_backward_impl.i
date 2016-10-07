/* file: split_layer_backward_impl.i */
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
//  Implementation of split algorithm
//--
*/

#ifndef __SPLIT_LAYER_BACKWARD_IMPL_I__
#define __SPLIT_LAYER_BACKWARD_IMPL_I__

#include "threading.h"

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace split
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void SplitKernel<algorithmFPType, method, cpu>::compute(Tensor *inputTensors[], Tensor *resultTensor, size_t nInputs)
{
    const services::Collection<size_t> &dims = inputTensors[0]->getDimensions();
    size_t nInputRows = dims[0];

    size_t nBlocks = nInputRows / _nRowsInBlock;
    size_t nRowsInLastBlock = nInputRows - nBlocks * _nRowsInBlock;

    for(size_t block = 0; block < nBlocks; block++)
    {
        processBlockInit(inputTensors[0], block * _nRowsInBlock, _nRowsInBlock, resultTensor);
    }
    if(nRowsInLastBlock > 0)
    {
        processBlockInit(inputTensors[0], nBlocks * _nRowsInBlock, nRowsInLastBlock, resultTensor);
    }

    for(int i = 1; i < nInputs; i++)
    {
        Tensor *inputTensor = inputTensors[i];

        for(size_t block = 0; block < nBlocks; block++)
        {
            processBlock(inputTensor, block * _nRowsInBlock, _nRowsInBlock, resultTensor);
        }
        if(nRowsInLastBlock > 0)
        {
            processBlock(inputTensor, nBlocks * _nRowsInBlock, nRowsInLastBlock, resultTensor);
        }
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
inline void SplitKernel<algorithmFPType, method, cpu>::processBlock(Tensor *inputTensor,
                                                                    size_t nProcessedRows,
                                                                    size_t nRowsInCurrentBlock,
                                                                    Tensor *resultTensor)
{
    SubtensorDescriptor<algorithmFPType> inputBlock;
    inputTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, readWrite, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    size_t nDataElements = inputBlock.getSize();
    for(size_t i = 0; i < nDataElements; i++)
    {
        resultArray[i] += inputArray[i];
    }

    inputTensor->releaseSubtensor(inputBlock);
    resultTensor->releaseSubtensor(resultBlock);
}

template<typename algorithmFPType, Method method, CpuType cpu>
inline void SplitKernel<algorithmFPType, method, cpu>::processBlockInit(Tensor *inputTensor,
                                                                        size_t nProcessedRows,
                                                                        size_t nRowsInCurrentBlock,
                                                                        Tensor *resultTensor)
{
    SubtensorDescriptor<algorithmFPType> inputBlock;
    inputTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    size_t nDataElements = inputBlock.getSize();
    for(size_t i = 0; i < nDataElements; i++)
    {
        resultArray[i] = inputArray[i];
    }

    inputTensor->releaseSubtensor(inputBlock);
    resultTensor->releaseSubtensor(resultBlock);
}

} // namespace internal
} // namespace backward
} // namespace split
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
