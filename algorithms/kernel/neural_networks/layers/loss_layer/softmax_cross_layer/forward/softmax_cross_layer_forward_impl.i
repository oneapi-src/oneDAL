/* file: softmax_cross_layer_forward_impl.i */
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
//  Implementation of the forward softmax cross layer
//--
*/

#ifndef __SOFTMAX_CROSS_LAYER_FORWARD_IMPL_I__
#define __SOFTMAX_CROSS_LAYER_FORWARD_IMPL_I__

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
namespace forward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status SoftmaxCrossKernel<algorithmFPType, method, cpu>::compute(Tensor *inputTensor, Tensor *groundTruthTensor, const softmax_cross::Parameter *parameter,
                                                               Tensor *probabilitiesTensor, Tensor *resultTensor)
{
    _eps = parameter->accuracyThreshold;
    _dim = parameter->dimension;

    size_t nInputRows = inputTensor->getDimensionSize(0);

    size_t nBlocks = nInputRows / _nRowsInBlock;
    nBlocks += (nBlocks * _nRowsInBlock != nInputRows);

    daal::tls<algorithmFPType *> blockLoss( [ = ]()-> algorithmFPType*
    {
        algorithmFPType *lossValue = new algorithmFPType;
        *lossValue = 0;
        return lossValue;
    } );

    daal::tls<Error *> threadLocalError( [ = ]()-> Error* { return new Error(); } );

    __DAAL_MAKE_TENSOR_THREADSAFE(inputTensor)
    __DAAL_MAKE_TENSOR_THREADSAFE(groundTruthTensor)

    daal::threader_for(nBlocks, nBlocks, [ =, &blockLoss, &threadLocalError ](int block)
    {
        size_t nRowsToProcess = _nRowsInBlock;
        if( block == nBlocks - 1 )
        {
            nRowsToProcess = nInputRows - block * _nRowsInBlock;
        }

        algorithmFPType *loss = blockLoss.local();
        Error *localError = threadLocalError.local();
        *loss += processBlock(inputTensor, groundTruthTensor, block * _nRowsInBlock, nRowsToProcess, probabilitiesTensor, localError);
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
    if(!this->_errors->isEmpty()) DAAL_RETURN_STATUS()

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTensor->getSubtensor(0, 0, 0, 1, writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();
    resultArray[0] = (algorithmFPType) 0;

    blockLoss.reduce( [ =, &resultArray ](algorithmFPType * partialLoss)-> void
    {
        resultArray[0] += (*partialLoss);
        delete partialLoss;
    }
                    );

    size_t dimsSize = inputTensor->getSize() / inputTensor->getDimensionSize(_dim);

    resultArray[0] = -1.0 * resultArray[0] / dimsSize;

    resultTensor->releaseSubtensor(resultBlock);
    DAAL_RETURN_STATUS()
}

template<typename algorithmFPType, Method method, CpuType cpu>
inline algorithmFPType SoftmaxCrossKernel<algorithmFPType, method, cpu>::processBlock(Tensor *inputTensor,
                                                                                      Tensor *groundTruthTensor,
                                                                                      size_t nProcessedRows, size_t nRowsInCurrentBlock,
                                                                                      Tensor *probabilitiesTensor,
                                                                                      Error *localError)
{
    SubtensorDescriptor<algorithmFPType> inputBlock;
    inputTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getPtr();
    if(!inputArray)
    {
        localError->setId(ErrorMemoryAllocationFailed);
        return 0;
    }

    SubtensorDescriptor<algorithmFPType> probBlock;
    probabilitiesTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, writeOnly, probBlock);
    algorithmFPType *probArray = probBlock.getPtr();
    if(!probArray)
    {
        inputTensor->releaseSubtensor(inputBlock);
        localError->setId(ErrorMemoryAllocationFailed);
        return 0;
    }

    Collection<size_t> softmaxDim = inputTensor->getDimensions();
    softmaxDim[0] = nRowsInCurrentBlock;

    SharedPtr<HomogenTensor<algorithmFPType> > softmaxInput(new HomogenTensor<algorithmFPType>(softmaxDim, inputArray));
    SharedPtr<HomogenTensor<algorithmFPType> > softmaxProb(new HomogenTensor<algorithmFPType>(softmaxDim, probArray));

    softmax::forward::internal::SoftmaxKernel<algorithmFPType, softmax::defaultDense, cpu> softmaxKernel;
    softmax::Parameter softmaxKernelParameter;
    softmaxKernelParameter.dimension = _dim;
    softmaxKernelParameter.predictionStage = true;

    softmaxKernel.compute(softmaxInput.get(), &softmaxKernelParameter, softmaxProb.get());

    inputTensor->releaseSubtensor(inputBlock);

    SubtensorDescriptor<int> groundTruthBlock;
    groundTruthTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, readOnly, groundTruthBlock);
    int *groundTruthArray = groundTruthBlock.getPtr();
    if(!groundTruthArray)
    {
        probabilitiesTensor->releaseSubtensor(probBlock);
        localError->setId(ErrorMemoryAllocationFailed);
        return 0;
    }

    algorithmFPType partialLoss = 0;
    size_t nFeatures = softmaxDim[_dim];

    size_t offsetBefore = (_dim == 0 ? 1 : inputTensor->getSize(0, _dim));
    size_t nDims = softmaxDim.size() - _dim - 1;
    size_t offsetAfter = (_dim == softmaxDim.size() - 1 ? 1 : inputTensor->getSize(_dim + 1, nDims));

    softmaxDim[0] = inputTensor->getDimensionSize(0);

    size_t jSample = offsetBefore / softmaxDim[0];

    for(size_t j = 0; j < nRowsInCurrentBlock * jSample; j++)
    {
        for(size_t k = 0; k < offsetAfter; k++)
        {
            partialLoss += Math<algorithmFPType, cpu>::sLog(Math<algorithmFPType, cpu>::sMax(probArray[(j * nFeatures + groundTruthArray[j * offsetAfter + k]) * offsetAfter + k], _eps));
        }
    }

    groundTruthTensor->releaseSubtensor(groundTruthBlock);
    probabilitiesTensor->releaseSubtensor(probBlock);

    return partialLoss;
}

} // namespace internal
} // namespace forward
} // namespace softmax_cross
} // namespace loss
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
