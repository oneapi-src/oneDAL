/* file: split_layer_forward_impl.i */
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

#ifndef __SPLIT_LAYER_FORWARD_IMPL_I__
#define __SPLIT_LAYER_FORWARD_IMPL_I__

#include "service_blas.h"
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
namespace forward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void SplitKernel<algorithmFPType, method, cpu>::compute(Tensor *inputTensor, Tensor *resultTensors[], size_t nOutputs)
{
    const services::Collection<size_t> &dims = inputTensor->getDimensions();
    size_t nInputRows = dims[0];

    size_t nBlocks = nInputRows / _nRowsInBlock;
    nBlocks += (nBlocks * _nRowsInBlock != nInputRows);

    for(int i = 0; i < nOutputs; i++)
    {
        Tensor *resultTensor = resultTensors[i];

        daal::threader_for(nBlocks, nBlocks, [ = ](int block)
        {
            size_t nRowsToProcess = _nRowsInBlock;
            if( block == nBlocks - 1 )
            {
                nRowsToProcess = nInputRows - block * _nRowsInBlock;
            }

            processBlock(inputTensor, block * _nRowsInBlock, nRowsToProcess, resultTensor);
        } );
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
inline void SplitKernel<algorithmFPType, method, cpu>::processBlock(Tensor *inputTensor,
                                                                    size_t nProcessedRows, size_t nRowsInCurrentBlock,
                                                                    Tensor *resultTensor)
{
    SubtensorDescriptor<algorithmFPType> inputBlock;
    inputTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    algorithmFPType zero = (algorithmFPType)0;
    size_t nDataElements = inputBlock.getSize();
    for(size_t i = 0; i < nDataElements; i++)
    {
        resultArray[i] = inputArray[i];
    }

    inputTensor->releaseSubtensor(inputBlock);
    resultTensor->releaseSubtensor(resultBlock);
}

} // namespace internal
} // namespace forward
} // namespace split
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
