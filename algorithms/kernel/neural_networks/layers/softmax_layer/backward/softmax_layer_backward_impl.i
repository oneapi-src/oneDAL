/* file: softmax_layer_backward_impl.i */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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
