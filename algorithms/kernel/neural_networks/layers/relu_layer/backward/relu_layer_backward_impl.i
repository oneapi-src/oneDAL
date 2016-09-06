/* file: relu_layer_backward_impl.i */
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
//  Implementation of relu algorithm
//--
*/

#ifndef __RELU_LAYER_BACKWARD_IMPL_I__
#define __RELU_LAYER_BACKWARD_IMPL_I__

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
namespace relu
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void ReLUKernel<algorithmFPType, method, cpu>::compute(const relu::backward::Input *input, relu::backward::Result *result)
{
    SharedPtr<Tensor> inputGradientTensor = input->get(layers::backward::inputGradient);
    SharedPtr<Tensor> forwardDataTensor = input->get(relu::auxData);

    SharedPtr<Tensor> resultTensor = result->get(layers::backward::gradient);

    const services::Collection<size_t> &dims = inputGradientTensor->getDimensions();
    size_t nInputRows = dims[0];

    size_t nBlocks = nInputRows / _nRowsInBlock;
    nBlocks += (nBlocks * _nRowsInBlock != nInputRows);

    daal::threader_for(nBlocks, nBlocks, [ = ](int block)
    {
        size_t nRowsToProcess = _nRowsInBlock;
        if( block == nBlocks - 1 )
        {
            nRowsToProcess = nInputRows - block * _nRowsInBlock;
        }

        processBlock(inputGradientTensor, forwardDataTensor, block * _nRowsInBlock, nRowsToProcess, resultTensor);
    } );
}

template<typename algorithmFPType, Method method, CpuType cpu>
inline void ReLUKernel<algorithmFPType, method, cpu>::processBlock(SharedPtr<Tensor> inputGradientTensor,
                                                                   SharedPtr<Tensor> forwardDataTensor,
                                                                   size_t nProcessedRows,
                                                                   size_t nRowsInCurrentBlock,
                                                                   SharedPtr<Tensor> resultTensor)
{
    SubtensorDescriptor<algorithmFPType> inputGradientBlock;
    inputGradientTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, readOnly, inputGradientBlock);
    algorithmFPType *inputGradientArray = inputGradientBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> forwardBlock;
    forwardDataTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, readOnly, forwardBlock);
    algorithmFPType *forwardDataArray = forwardBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    algorithmFPType zero = (algorithmFPType)0;
    size_t nDataElements = inputGradientBlock.getSize();
    for(size_t i = 0; i < nDataElements; i++)
    {
        if(forwardDataArray[i] > zero)
        {
            resultArray[i] = inputGradientArray[i];
        }
        else
        {
            resultArray[i] = zero;
        }
    }

    inputGradientTensor->releaseSubtensor(inputGradientBlock);
    forwardDataTensor->releaseSubtensor(forwardBlock);
    resultTensor->releaseSubtensor(resultBlock);
}

} // namespace internal
} // namespace backward
} // namespace relu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
