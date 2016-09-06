/* file: neural_networks_training_service.h */
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

//++
//  Declaration of template function that calculate neural networks.
//--


#ifndef __NEURAL_NETWORKS_TRAINING_SERVICE_H__
#define __NEURAL_NETWORKS_TRAINING_SERVICE_H__

#include "tensor.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace training
{
namespace internal
{
/*

Collection<size_t> dataMeanDims = data->getDimensions();
size_t targetDim = 1;
dataMeanDims[targetDim] = 1;
SharedPtr<HomogenTensor<algorithmFPType> > dataMean(new HomogenTensor<algorithmFPType>(dataMeanDims, Tensor::doAllocate));
int errorCode = mergeTensor<algorithmFPType, cpu>(data, dataMean, targetDim);
if(errorCode)
{
    printf("errorCode = %d\n", errorCode);
}
else
{
    printTensor(dataMean, "dataMean", 10);
}

errorCode = spreadTensor<algorithmFPType, cpu>(dataMean, data, targetDim);
if(errorCode)
{
    printf("errorCode = %d\n", errorCode);
}
else
{
    printTensor(data, "data spread", 10);
}

*/

template<typename algorithmFPType, CpuType cpu>
int mergeTensor(SharedPtr<Tensor> inputTable, SharedPtr<Tensor> resultTable, size_t targetDim = 0)
{
    const Collection<size_t> &inDims  = inputTable->getDimensions();
    const Collection<size_t> &outDims = resultTable->getDimensions();

    if(outDims[targetDim] != 1) { return -1; }
    if(outDims.size() != inDims.size()) { return -2; }
    if(!inputTable) { return -3; }
    if(!resultTable) { return -4; }
    for(size_t i = 0; i < inDims.size(); i++)
    {
        if(i != targetDim)
        {
            if(inDims[i] != outDims[i])
            {
                return -5;
            }
        }
    }

    SubtensorDescriptor<algorithmFPType> inputBlock;
    inputTable->getSubtensor(0, 0, 0, inDims[0], readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTable->getSubtensor(0, 0, 0, outDims[0], writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    size_t offsetBefore = 1;
    size_t offsetAfter = 1;
    size_t dimensionSize = inDims[targetDim];
    for (size_t i = 0; i < targetDim; i++)
    {
        offsetBefore *= inDims[i];
    }
    for (size_t i = targetDim + 1; i < inDims.size(); i++)
    {
        offsetAfter *= inDims[i];
    }

    for (size_t i = 0; i < resultTable->getSize(); i++)
    {
        resultArray[i] = 0;
    }

    for (size_t i = 0; i < offsetBefore; i++)
    {
        for (size_t k = 0; k < dimensionSize; k++)
        {
            for (size_t j = 0; j < offsetAfter; j++)
            {
                size_t indexIn = (i * dimensionSize + k) * offsetAfter + j;
                size_t indexVal = i * offsetAfter + j;
                resultArray[indexVal] += inputArray[indexIn];
            }
        }
    }

    algorithmFPType invM = 1.0 / dimensionSize;
    for (size_t i = 0; i < resultTable->getSize(); i++)
    {
        resultArray[i] *= invM;
    }

    inputTable->releaseSubtensor(inputBlock);
    resultTable->releaseSubtensor(resultBlock);
    return 0;
}

template<typename algorithmFPType, CpuType cpu>
int spreadTensor(SharedPtr<Tensor> inputTable, SharedPtr<Tensor> resultTable, size_t targetDim = 0)
{
    const Collection<size_t> &inDims  = inputTable->getDimensions();
    const Collection<size_t> &outDims = resultTable->getDimensions();

    if(inDims[targetDim] != 1) { return -1; }
    if(outDims.size() != inDims.size()) { return -2; }
    if(!inputTable) { return -3; }
    if(!resultTable) { return -4; }
    for(size_t i = 0; i < inDims.size(); i++)
    {
        if(i != targetDim)
        {
            if(inDims[i] != outDims[i])
            {
                return -5;
            }
        }
    }

    SubtensorDescriptor<algorithmFPType> inputBlock;
    inputTable->getSubtensor(0, 0, 0, inDims[0], readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTable->getSubtensor(0, 0, 0, outDims[0], writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    size_t offsetBefore = 1;
    size_t offsetAfter = 1;
    size_t dimensionSize = outDims[targetDim];
    for (size_t i = 0; i < targetDim; i++)
    {
        offsetBefore *= inDims[i];
    }
    for (size_t i = targetDim + 1; i < inDims.size(); i++)
    {
        offsetAfter *= inDims[i];
    }

    for (size_t i = 0; i < offsetBefore; i++)
    {
        for (size_t k = 0; k < dimensionSize; k++)
        {
            for (size_t j = 0; j < offsetAfter; j++)
            {
                size_t indexVal = (i * dimensionSize + k) * offsetAfter + j;
                size_t indexIn = i * offsetAfter + j;
                resultArray[indexVal] = inputArray[indexIn];
            }
        }
    }

    inputTable->releaseSubtensor(inputBlock);
    resultTable->releaseSubtensor(resultBlock);
    return 0;
}

} // namespace daal::internal
} // namespace training
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
