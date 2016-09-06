/* file: softmax_cross_layer_forward_impl.i */
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
void SoftmaxCrossKernel<algorithmFPType, method, cpu>::compute(const softmax_cross::forward::Input *input, const softmax_cross::Parameter *parameter,
        softmax_cross::forward::Result *result)
{
    SharedPtr<Tensor> inputTensor = input->get(layers::forward::data);
    SharedPtr<Tensor> groundTruthTensor = input->get(loss::forward::groundTruth);
    SharedPtr<Tensor> probabilitiesTensor = result->get(auxProbabilities);
    SharedPtr<Tensor> resultTensor = result->get(layers::forward::value);
    _eps = parameter->accuracyThreshold;

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
    if(!this->_errors->isEmpty()) {return;}

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

    resultArray[0] = -1.0 * resultArray[0] / nInputRows;

    resultTensor->releaseSubtensor(resultBlock);
}

template<typename algorithmFPType, Method method, CpuType cpu>
inline algorithmFPType SoftmaxCrossKernel<algorithmFPType, method, cpu>::processBlock(SharedPtr<Tensor> inputTensor,
        SharedPtr<Tensor> groundTruthTensor,
        size_t nProcessedRows, size_t nRowsInCurrentBlock,
        SharedPtr<Tensor> probabilitiesTensor,
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

    Collection<size_t> softmaxDim(2);
    softmaxDim[0] = nRowsInCurrentBlock;
    softmaxDim[1] = inputTensor->getSize(1, inputTensor->getNumberOfDimensions() - 1);

    SharedPtr<HomogenTensor<algorithmFPType> > softmaxInput(new HomogenTensor<algorithmFPType>(softmaxDim, inputArray));
    SharedPtr<HomogenTensor<algorithmFPType> > softmaxProb(new HomogenTensor<algorithmFPType>(softmaxDim, probArray));

    softmax::forward::internal::SoftmaxKernel<algorithmFPType, softmax::defaultDense, cpu> softmaxKernel;
    softmax::forward::Input softmaxKernelInput;
    softmax::forward::Result softmaxKernelResult;
    softmax::Parameter softmaxKernelParameter;

    softmaxKernelInput.set(layers::forward::data, softmaxInput);
    softmaxKernelResult.set(layers::forward::value, softmaxProb);
    softmaxKernelResult.set(layers::forward::resultForBackward, services::SharedPtr<LayerData>(new LayerData()));
    softmaxKernelResult.set(softmax::auxValue, softmaxProb);
    softmaxKernelParameter.dimension = 1;

    softmaxKernel.compute(&softmaxKernelInput, &softmaxKernelParameter, &softmaxKernelResult);

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
    size_t nFeatures = softmaxDim[1];
    for(size_t i = 0; i < nRowsInCurrentBlock; i++)
    {
        partialLoss += Math<algorithmFPType, cpu>::sLog(Math<algorithmFPType, cpu>::sMax(probArray[i * nFeatures + groundTruthArray[i]], _eps));
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
