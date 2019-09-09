/* file: softmax_layer_backward_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
#include "service_numeric_table.h"
#include "service_tensor.h"
#include "service_error_handling.h"
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
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status SoftmaxKernel<algorithmFPType, method, cpu>::compute(
    const Tensor &inputTensor,
    const Tensor &valueTensor,
    const softmax::Parameter &parameter,
    Tensor &resultTensor)
{
    const algorithmFPType minValue = -services::internal::MaxVal<algorithmFPType>::get();

    const size_t dimension = parameter.dimension;
    const size_t dimensionSize = inputTensor.getDimensionSize(dimension);
    const size_t offsetInclude = inputTensor.getSize(dimension, inputTensor.getNumberOfDimensions() - dimension);
    const size_t offsetBefore = inputTensor.getSize() / offsetInclude;
    const size_t offsetAfter = offsetInclude / dimensionSize;
    const size_t nBatches = inputTensor.getDimensionSize(0);

    ReadSubtensor<algorithmFPType, cpu> inputBlock(const_cast<Tensor&>(inputTensor), 0, 0, 0, nBatches);
    DAAL_CHECK_BLOCK_STATUS(inputBlock);
    const algorithmFPType *inputArray = inputBlock.get();

    ReadSubtensor<algorithmFPType, cpu> valueBlock(const_cast<Tensor&>(valueTensor), 0, 0, 0, nBatches);
    DAAL_CHECK_BLOCK_STATUS(valueBlock);
    const algorithmFPType *valueArray = valueBlock.get();

    WriteOnlySubtensor<algorithmFPType, cpu> resultBlock(resultTensor, 0, 0, 0, nBatches);
    DAAL_CHECK_BLOCK_STATUS(resultBlock);
    algorithmFPType *resultArray = resultBlock.get();

    SafeStatus safeStat;
    threader_for(offsetBefore, offsetBefore, [&](size_t i)
    {
        TArrayScalableCalloc<algorithmFPType, cpu> sumArrayPtr(dimensionSize * offsetAfter);
        algorithmFPType *sumArray = sumArrayPtr.get();
        DAAL_CHECK_THR(sumArray, ErrorMemoryAllocationFailed);

        for(size_t k = 0; k < dimensionSize; k++)
        {
            for(size_t j = 0; j < offsetAfter; j++)
            {
                const size_t index = (i * dimensionSize + k) * offsetAfter + j;
                sumArray[j] += inputArray[index] * valueArray[index];
            }
        }
        for(size_t k = 0; k < dimensionSize; k++)
        {
            for(size_t j = 0; j < offsetAfter; j++)
            {
                const size_t index = (i * dimensionSize + k) * offsetAfter + j;
                resultArray[index] = inputArray[index] - sumArray[j];
                resultArray[index] = resultArray[index] * valueArray[index];
            }
        }
    });

    return Status();
}

} // internal
} // backward
} // namespace softmax
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
