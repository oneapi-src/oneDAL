/* file: softmax_layer_forward_impl.i */
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
//  Implementation of softmax layer.
//--
*/

#ifndef __SOFTMAX_LAYER_FORWARD_IMPL_I__
#define __SOFTMAX_LAYER_FORWARD_IMPL_I__

#include "service_data_utils.h"
#include "service_math.h"

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
namespace softmax
{
namespace forward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void SoftmaxKernel<algorithmFPType, method, cpu>::compute(const softmax::forward::Input *input, const softmax::Parameter *parameter,
                                                      softmax::forward::Result *result)
{
    SharedPtr<Tensor> inputTensor = input->get(layers::forward::data);
    SharedPtr<Tensor> resultTensor = result->get(layers::forward::value);

    algorithmFPType minValue = -data_feature_utils::internal::MaxVal<algorithmFPType, cpu>::get();

    size_t dimension = parameter->dimension;

    const services::Collection<size_t>& dims = inputTensor->getDimensions();

    size_t dimensionSize = dims[dimension];

    size_t offsetBefore = 1;
    size_t offsetAfter = 1;

    for(size_t i = 0; i < dimension; i++)
    {
        offsetBefore *= dims[i];
    }
    for(size_t i = dimension + 1; i < dims.size(); i++)
    {
        offsetAfter *= dims[i];
    }

    algorithmFPType * expArray = (algorithmFPType *)services::daal_malloc(dimensionSize * offsetAfter * sizeof(algorithmFPType));
    if(!expArray) {this->_errors->add(services::ErrorMemoryAllocationFailed); return;}

    algorithmFPType * maxArray = (algorithmFPType *)services::daal_malloc(offsetAfter * sizeof(algorithmFPType));
    if(!maxArray) {services::daal_free(expArray); this->_errors->add(services::ErrorMemoryAllocationFailed); return;}

    SubtensorDescriptor<algorithmFPType> inputBlock;
    inputTensor->getSubtensor(0, 0, 0, dims[0], readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTensor->getSubtensor(0, 0, 0, dims[0], writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    for(size_t i = 0; i < offsetBefore; i++)
    {
        //max computation
        for(size_t j = 0; j < offsetAfter; j++)
        {
            maxArray[j] = minValue;
            for(size_t k = 0; k < dimensionSize; k++)
            {
                size_t inputIndex = (i * dimensionSize + k) * offsetAfter + j;
                if(inputArray[inputIndex] > maxArray[j])
                    maxArray[j] = inputArray[inputIndex];
            }
        }

        for(size_t k = 0; k < dimensionSize; k++)
        {
            for(size_t j = 0; j < offsetAfter; j++)
            {
                size_t inputIndex = (i * dimensionSize + k) * offsetAfter + j;
                size_t expIndex = k * offsetAfter + j;
                expArray[expIndex] = inputArray[inputIndex] - maxArray[j];
            }
        }

        daal::internal::Math<algorithmFPType,cpu>::vExp(dimensionSize * offsetAfter, expArray, expArray);

        //sum computation
        for(size_t j = 0; j < offsetAfter; j++)
        {
            maxArray[j] = (algorithmFPType)0;
            for(size_t k = 0; k < dimensionSize; k++)
            {
                size_t expIndex = k * offsetAfter + j;
                maxArray[j] += expArray[expIndex];
            }
        }

        //can be splitted to division and load to result
        for(size_t k = 0; k < dimensionSize; k++)
        {
            for(size_t j = 0; j < offsetAfter; j++)
            {
                size_t resultIndex = (i * dimensionSize + k) * offsetAfter + j;
                size_t expIndex = k * offsetAfter + j;
                resultArray[resultIndex] = expArray[expIndex] / maxArray[j];
            }
        }
    }

    inputTensor->releaseSubtensor(inputBlock);
    resultTensor->releaseSubtensor(resultBlock);

    services::daal_free(expArray);
    services::daal_free(maxArray);
}


//!!!!!!!!!!!!!!!!!!!!!!!!!!!
/*
    algorithmFPType minValue = -data_feature_utils::internal::MaxVal<algorithmFPType, cpu>::get();
    algorithmFPType max;
    algorithmFPType sum = (algorithmFPType)0;

    for(size_t i = 0; i < nRowsInCurrentBlock; i++)
    {
        max = minValue;

        for(size_t j = 0; j < nInputColumns; j++)
        {
            if(max < inputArray[i * nInputColumns + j])
            {
                max = inputArray[i * nInputColumns + j];
            }
            resultArray[i * nInputColumns + j] = inputArray[i * nInputColumns + j];
        }

        for(size_t j = 0; j < nInputColumns; j++)
        {
            resultArray[i * nInputColumns + j] -= max;
        }

        vExp<cpu>(nInputColumns, resultArray + i * nInputColumns, resultArray + i * nInputColumns);

        for(size_t j = 0; j < nInputColumns; j++)
        {
            sum += resultArray[i * nInputColumns + j];
        }

        for(size_t j = 0; j < nInputColumns; j++)
        {
            resultArray[i * nInputColumns + j] /= sum;
        }
        sum = (algorithmFPType)0;
    }
*/

} // internal
} // forward
} // namespace softmax
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
