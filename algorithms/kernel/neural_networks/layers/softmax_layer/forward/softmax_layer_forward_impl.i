/* file: softmax_layer_forward_impl.i */
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
//  Implementation of softmax layer.
//--
*/

#ifndef __SOFTMAX_LAYER_FORWARD_IMPL_I__
#define __SOFTMAX_LAYER_FORWARD_IMPL_I__

#include "service_data_utils.h"
#include "service_math.h"
#include "service_memory.h"
#include "threading.h"

using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

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

/* Threshold for vector exp negative args domain  */
template<typename algorithmFPType> inline algorithmFPType exp_threshold        (void){ return algorithmFPType(0.0); }
template<>                         inline float           exp_threshold<float> (void){ return float (-87.0);  }
template<>                         inline double          exp_threshold<double>(void){ return double(-708.0); }

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status SoftmaxKernel<algorithmFPType, method, cpu>::compute(Tensor *inputTensor, const softmax::Parameter *parameter,
                                                      Tensor *resultTensor)
{
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

    SubtensorDescriptor<algorithmFPType> inputBlock;
    inputTensor->getSubtensor(0, 0, 0, dims[0], readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTensor->getSubtensor(0, 0, 0, dims[0], writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    threader_for(offsetBefore, offsetBefore, [&](size_t i)
    {

        algorithmFPType * expArray = service_scalable_malloc<algorithmFPType,cpu>(dimensionSize * offsetAfter );
        if(!expArray) {  this->_errors->add(services::ErrorMemoryAllocationFailed); return;}

        algorithmFPType * maxArray = service_scalable_malloc<algorithmFPType,cpu>(offsetAfter);
        if(!maxArray) { this->_errors->add(services::ErrorMemoryAllocationFailed); return;}

        for(size_t j = 0; j < offsetAfter; j++)
        {
            maxArray[j] = minValue;
        }

        for(size_t k = 0; k < dimensionSize; k++)
        {
            for(size_t j = 0; j < offsetAfter; j++)
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

                // make all values less than threshold as threshold value
                // to fix slow work on vExp on large negative inputs
              #if (__CPUID__(DAAL_CPU) != __avx512_mic__)
                if( expArray[expIndex] < exp_threshold<algorithmFPType>() )
                {
                    expArray[expIndex] = exp_threshold<algorithmFPType>();
                }
              #endif
            }
        }

        daal::internal::Math<algorithmFPType,cpu>::vExp(dimensionSize * offsetAfter, expArray, expArray);

        for(size_t j = 0; j < offsetAfter; j++)
        {
            maxArray[j] = (algorithmFPType)0;
        }

        for(size_t k = 0; k < dimensionSize; k++)
        {
            for(size_t j = 0; j < offsetAfter; j++)
            {
                size_t expIndex = k * offsetAfter + j;
                maxArray[j] += expArray[expIndex];
            }
        }

        for(size_t j = 0; j < offsetAfter; j++)
        {
            maxArray[j] = ((algorithmFPType)1.0) / maxArray[j];
        }


        for(size_t k = 0; k < dimensionSize; k++)
        {
            for(size_t j = 0; j < offsetAfter; j++)
            {
                size_t resultIndex = (i * dimensionSize + k) * offsetAfter + j;
                size_t expIndex = k * offsetAfter + j;
                resultArray[resultIndex] = expArray[expIndex] * maxArray[j];
            }
        }

        service_scalable_free<algorithmFPType,cpu>(expArray);
        service_scalable_free<algorithmFPType,cpu>(maxArray);
    });

    inputTensor->releaseSubtensor(inputBlock);
    resultTensor->releaseSubtensor(resultBlock);
    DAAL_RETURN_STATUS()
}



} // internal
} // forward
} // namespace softmax
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
