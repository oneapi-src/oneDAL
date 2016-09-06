/* file: concat_layer_backward_impl.i */
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
//  Implementation of concat algorithm
//--
*/

#ifndef __CONCAT_LAYER_BACKWARD_IMPL_I__
#define __CONCAT_LAYER_BACKWARD_IMPL_I__

#include "service_micro_table.h"

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
namespace concat
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void ConcatKernel<algorithmFPType, method, cpu>::compute(Tensor *inputTensor, NumericTable *forwardOutputTable, const concat::Parameter *parameter,
                                                         Tensor *resultTensors[])
{
    size_t concatDimension = parameter->concatDimension;

    const services::Collection<size_t> &dims = inputTensor->getDimensions();
    size_t nInputRows = dims[0];

    SubtensorDescriptor<algorithmFPType> inputBlock;
    inputTensor->getSubtensor(0, 0, 0, nInputRows, readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getPtr();

    size_t nOutputs = forwardOutputTable->getNumberOfColumns();

    int *auxDims;
    BlockMicroTable<int, readOnly, cpu> inputDims(forwardOutputTable);
    inputDims.getBlockOfRows(0, 1, &auxDims);

    size_t dimsSum = 0;
    for(size_t j = 0; j < nOutputs; j++)
    {
        dimsSum += auxDims[j];
    }

    size_t offsetBefore = 1;
    for(size_t j = 0; j < concatDimension; j++)
    {
        offsetBefore *= dims[j];
    }

    size_t offsetAfter = 1;
    for(size_t j = concatDimension + 1; j < dims.size(); j++)
    {
        offsetAfter *= dims[j];
    }

    size_t sum = 0;
    for(int l = 0; l < nOutputs; l++)
    {
        Tensor *resultTensor = resultTensors[l];

        const services::Collection<size_t> &resDims = resultTensor->getDimensions();

        size_t nResultRows = resDims[0];

        SubtensorDescriptor<algorithmFPType> resultBlock;
        resultTensor->getSubtensor(0, 0, 0, nResultRows, writeOnly, resultBlock);
        algorithmFPType *resultArray = resultBlock.getPtr();

        for(size_t i = 0; i < offsetBefore; i++)
        {
            for(size_t k = 0; k < auxDims[l]; k++)
            {
                for(size_t j = 0; j < offsetAfter; j++)
                {
                    size_t outputIndex = (i * auxDims[l] + k) * offsetAfter + j;
                    size_t inputIndex = (i * dimsSum + k + sum) * offsetAfter + j;
                    resultArray[outputIndex] = inputArray[inputIndex];
                }
            }
        }
        sum += auxDims[l];
        resultTensor->releaseSubtensor(resultBlock);
    }
    inputTensor->releaseSubtensor(inputBlock);
    inputDims.release();
}

} // namespace internal
} // namespace backward
} // namespace concat
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
