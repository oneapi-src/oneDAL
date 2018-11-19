/* file: softmax_layer_forward_impl.i */
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

#ifndef __SOFTMAX_LAYER_FORWARD_IMPL_I__
#define __SOFTMAX_LAYER_FORWARD_IMPL_I__

#include "service_data_utils.h"
#include "service_math.h"
#include "service_tensor.h"
#include "service_numeric_table.h"
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
namespace forward
{
namespace internal
{

/* Threshold for vector exp negative args domain  */
template<typename algorithmFPType> inline algorithmFPType exp_threshold        (void) { return algorithmFPType(0.0); }
template<>                         inline float           exp_threshold<float> (void) { return float (-87.0);  }
template<>                         inline double          exp_threshold<double>(void) { return double(-708.0); }

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status SoftmaxKernel<algorithmFPType, method, cpu>::compute(
    const Tensor &inputTensor,
    const softmax::Parameter &parameter,
    Tensor &resultTensor)
{
    const algorithmFPType minValue = -services::internal::MaxVal<algorithmFPType>::get();

    const size_t dimension = parameter.dimension;
    const size_t dimensionSize = inputTensor.getDimensionSize(dimension);
    const size_t offsetInclude = inputTensor.getSize(dimension, inputTensor.getNumberOfDimensions() - dimension);
    const size_t offsetBefore = inputTensor.getSize() / offsetInclude;
    const size_t offsetAfter = offsetInclude / dimensionSize;

    ReadSubtensor<algorithmFPType, cpu> inputBlock(const_cast<Tensor &>(inputTensor), 0, 0, 0, inputTensor.getDimensionSize(0));
    DAAL_CHECK_BLOCK_STATUS(inputBlock);
    const algorithmFPType *inputArray = inputBlock.get();

    WriteOnlySubtensor<algorithmFPType, cpu> resultBlock(resultTensor, 0, 0, 0, inputTensor.getDimensionSize(0));
    DAAL_CHECK_BLOCK_STATUS(resultBlock);
    algorithmFPType *resultArray = resultBlock.get();

    SafeStatus safeStat;
    threader_for(offsetBefore, offsetBefore, [&](size_t i)
    {
        TArrayScalable<algorithmFPType, cpu> expArrayPtr(dimensionSize * offsetAfter);
        algorithmFPType *expArray = expArrayPtr.get();
        DAAL_CHECK_THR(expArray, ErrorMemoryAllocationFailed);

        TArrayScalable<algorithmFPType, cpu> maxArrayPtr(offsetAfter);
        algorithmFPType *maxArray = maxArrayPtr.get();
        DAAL_CHECK_THR(maxArray, ErrorMemoryAllocationFailed);

        for(size_t j = 0; j < offsetAfter; j++)
        {
            maxArray[j] = minValue;
        }

        for(size_t k = 0; k < dimensionSize; k++)
        {
            for(size_t j = 0; j < offsetAfter; j++)
            {
                const size_t inputIndex = (i * dimensionSize + k) * offsetAfter + j;
                if(inputArray[inputIndex] > maxArray[j])
                {
                    maxArray[j] = inputArray[inputIndex];
                }
            }
        }

        for(size_t k = 0; k < dimensionSize; k++)
        {
            for(size_t j = 0; j < offsetAfter; j++)
            {
                const size_t inputIndex = (i * dimensionSize + k) * offsetAfter + j;
                const size_t expIndex = k * offsetAfter + j;
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

        Math<algorithmFPType, cpu>::vExp(dimensionSize * offsetAfter, expArray, expArray);

        for(size_t j = 0; j < offsetAfter; j++)
        {
            maxArray[j] = (algorithmFPType)0;
        }

        for(size_t k = 0; k < dimensionSize; k++)
        {
            for(size_t j = 0; j < offsetAfter; j++)
            {
                const size_t expIndex = k * offsetAfter + j;
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
                const size_t resultIndex = (i * dimensionSize + k) * offsetAfter + j;
                const size_t expIndex = k * offsetAfter + j;
                resultArray[resultIndex] = expArray[expIndex] * maxArray[j];
            }
        }
    });
    return Status();
}



} // internal
} // forward
} // namespace softmax
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
