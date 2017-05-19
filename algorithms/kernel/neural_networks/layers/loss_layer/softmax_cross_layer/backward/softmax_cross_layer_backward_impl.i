/* file: softmax_cross_layer_backward_impl.i */
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
//  Implementation of the backward softmax cross layer
//--
*/

#ifndef __SOFTMAX_CROSS_LAYER_BACKWARD_IMPL_I__
#define __SOFTMAX_CROSS_LAYER_BACKWARD_IMPL_I__

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
namespace loss
{
namespace softmax_cross
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status SoftmaxCrossKernel<algorithmFPType, method, cpu>::compute(Tensor *probTensor, Tensor *groundTruthTensor,
                                                               const softmax_cross::Parameter *parameter, Tensor *resultTensor)
{
    size_t nRows = groundTruthTensor->getDimensionSize(0);
    _dim = parameter->dimension;

    size_t nBlocks = nRows / _nRowsInBlock;
    nBlocks += (nBlocks * _nRowsInBlock != nRows);

    daal::tls<Error *> threadLocalError( [ = ]()-> Error* { return new Error(); } );

    __DAAL_MAKE_TENSOR_THREADSAFE(probTensor)
    __DAAL_MAKE_TENSOR_THREADSAFE(groundTruthTensor)

    daal::threader_for(nBlocks, nBlocks, [ =, &threadLocalError ](int block)
    {
        size_t nRowsToProcess = _nRowsInBlock;
        if( block == nBlocks - 1 )
        {
            nRowsToProcess = nRows - block * _nRowsInBlock;
        }

        Error *localError = threadLocalError.local();
        processBlock(probTensor, groundTruthTensor, block * _nRowsInBlock, nRowsToProcess, resultTensor, localError);
        if(localError->id() != NoErrorMessageFound) {return;}
    }
                      );

    threadLocalError.reduce( [ = ](Error * e)-> void
    {
        if(e->id() != NoErrorMessageFound)
        {
            SharedPtr<Error> eCopy = SharedPtr<Error>(new Error(*e));
            this->_errors->add(eCopy);
        }
        delete e;
    });
    DAAL_RETURN_STATUS()
}

template<typename algorithmFPType, Method method, CpuType cpu>
inline void SoftmaxCrossKernel<algorithmFPType, method, cpu>::processBlock(Tensor *probTensor,
                                                                           Tensor *groundTruthTensor,
                                                                           size_t nProcessedRows, size_t nRowsInCurrentBlock,
                                                                           Tensor *gradientTensor,
                                                                           Error *localError)
{
    SubtensorDescriptor<int> groundTruthBlock;
    groundTruthTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, readOnly, groundTruthBlock);
    int *groundTruthArray = groundTruthBlock.getPtr();
    if(!groundTruthArray)
    {
        localError->setId(ErrorMemoryAllocationFailed);
        return;
    }

    SubtensorDescriptor<algorithmFPType> probBlock;
    probTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, readOnly, probBlock);
    algorithmFPType *probArray = probBlock.getPtr();
    if(!probArray)
    {
        groundTruthTensor->releaseSubtensor(groundTruthBlock);
        localError->setId(ErrorMemoryAllocationFailed);
        return;
    }

    SubtensorDescriptor<algorithmFPType> gradientBlock;
    gradientTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, writeOnly, gradientBlock);
    algorithmFPType *gradientArray = gradientBlock.getPtr();
    if(!gradientArray)
    {
        groundTruthTensor->releaseSubtensor(groundTruthBlock);
        probTensor->releaseSubtensor(probBlock);
        localError->setId(ErrorMemoryAllocationFailed);
        return;
    }

    size_t nDataElements = probBlock.getSize();
    for(size_t i = 0; i < nDataElements; i++)
    {
        gradientArray[i] = probArray[i];
    }

    Collection<size_t> softmaxDim = probTensor->getDimensions();
    size_t nFeatures = softmaxDim[_dim];
    algorithmFPType one = 1.0;

    size_t offsetBefore = (_dim == 0 ? 1 : probTensor->getSize(0, _dim));
    size_t nDims = softmaxDim.size() - _dim - 1;
    size_t offsetAfter = (_dim == softmaxDim.size() - 1 ? 1 : probTensor->getSize(_dim + 1, nDims));

    size_t jSample = offsetBefore / softmaxDim[0];

    for(size_t j = 0; j < nRowsInCurrentBlock * jSample; j++)
    {
        for(size_t k = 0; k < offsetAfter; k++)
        {
            gradientArray[(j * nFeatures + groundTruthArray[j * offsetAfter + k])*offsetAfter + k] -= one;
        }
    }

    groundTruthTensor->releaseSubtensor(groundTruthBlock);
    probTensor->releaseSubtensor(probBlock);
    gradientTensor->releaseSubtensor(gradientBlock);
}

} // internal
} // backward
} // namespace softmax_cross
} // namespace loss
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
