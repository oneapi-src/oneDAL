/* file: softmax_layer_backward_impl.i */
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

#ifndef __SOFTMAX_LAYER_BACKWARD_IMPL_I__
#define __SOFTMAX_LAYER_BACKWARD_IMPL_I__

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
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void SoftmaxKernel<algorithmFPType, method, cpu>::compute(const softmax::backward::Input *input, const softmax::Parameter *parameter,
                                                      softmax::backward::Result *result)
{
    SharedPtr<Tensor> inputTensor = input->get(layers::backward::inputGradient);
    SharedPtr<Tensor> valueTensor = input->get(softmax::auxValue);
    SharedPtr<Tensor> resultTensor = result->get(layers::backward::gradient);

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

    algorithmFPType * sumArray = (algorithmFPType *)services::daal_malloc(offsetAfter * sizeof(algorithmFPType));
    if(!sumArray) {this->_errors->add(services::ErrorMemoryAllocationFailed); return;}

    SubtensorDescriptor<algorithmFPType> inputBlock;
    inputTensor->getSubtensor(0, 0, 0, dims[0], readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> valueBlock;
    valueTensor->getSubtensor(0, 0, 0, dims[0], readOnly, valueBlock);
    algorithmFPType *valueArray = valueBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTensor->getSubtensor(0, 0, 0, dims[0], writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    for(size_t i = 0; i < offsetBefore; i++)
    {
        for(size_t j = 0; j < offsetAfter; j++)
        {
            sumArray[j] = (algorithmFPType)0;
            for(size_t k = 0; k < dimensionSize; k++)
            {
                size_t index = (i * dimensionSize + k) * offsetAfter + j;
                sumArray[j] += inputArray[index] * valueArray[index];
            }
        }
        for(size_t k = 0; k < dimensionSize; k++)
        {
            for(size_t j = 0; j < offsetAfter; j++)
            {
                size_t index = (i * dimensionSize + k) * offsetAfter + j;
                resultArray[index] = inputArray[index] - sumArray[j];
                resultArray[index] = resultArray[index] * valueArray[index];
            }
        }
    }

    inputTensor->releaseSubtensor(inputBlock);
    valueTensor->releaseSubtensor(valueBlock);
    resultTensor->releaseSubtensor(resultBlock);

    services::daal_free(sumArray);
}

} // internal
} // backward
} // namespace softmax
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
