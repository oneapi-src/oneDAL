/* file: softmax_cross_layer_backward_impl.i */
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
void SoftmaxCrossKernel<algorithmFPType, method, cpu>::compute(const softmax_cross::backward::Input *input,
        const softmax_cross::Parameter *parameter,
        softmax_cross::backward::Result *result)
{
    SharedPtr<Tensor> probTensor = input->get(softmax_cross::auxProbabilities);
    SharedPtr<Tensor> groundTruthTensor = input->get(softmax_cross::auxGroundTruth);
    SharedPtr<Tensor> resultTensor = result->get(layers::backward::gradient);

    size_t nRows = groundTruthTensor->getDimensionSize(0);

    size_t nBlocks = nRows / _nRowsInBlock;
    nBlocks += (nBlocks * _nRowsInBlock != nRows);

    daal::tls<Error *> threadLocalError( [ = ]()-> Error* { return new Error(); } );

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
    if(!this->_errors->isEmpty()) {return;}

}

template<typename algorithmFPType, Method method, CpuType cpu>
inline void SoftmaxCrossKernel<algorithmFPType, method, cpu>::processBlock(SharedPtr<Tensor> probTensor,
        SharedPtr<Tensor> groundTruthTensor,
        size_t nProcessedRows, size_t nRowsInCurrentBlock,
        SharedPtr<Tensor> gradientTensor,
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

    size_t nFeatures = probTensor->getSize(1, probTensor->getNumberOfDimensions() - 1);
    algorithmFPType one = 1.0;
    for(size_t i = 0; i < nRowsInCurrentBlock; i++)
    {
        gradientArray[i * nFeatures + groundTruthArray[i]] -= one;
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
