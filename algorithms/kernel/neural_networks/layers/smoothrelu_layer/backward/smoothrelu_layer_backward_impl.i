/* file: smoothrelu_layer_backward_impl.i */
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
// Implementation of the backward smooth rectifier linear unit (smooth relu) layer
//--
*/

#ifndef __SMOOTHRELU_LAYER_BACKWARD_IMPL_I__
#define __SMOOTHRELU_LAYER_BACKWARD_IMPL_I__

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
namespace smoothrelu
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status SmoothReLUKernel<algorithmFPType, method, cpu>::compute(const Tensor &inputGradientTensor, const Tensor &forwardDataTensor, Tensor &resultTensor)
{
    __DAAL_MAKE_TENSOR_THREADSAFE(const_cast<Tensor *>(&forwardDataTensor))
    __DAAL_MAKE_TENSOR_THREADSAFE(&resultTensor)

    Status s = computeImpl<cpu>(inputGradientTensor, [=, &inputGradientTensor, &forwardDataTensor, &resultTensor](size_t fDimN, size_t *fDims, size_t nRowsToProcess, const TensorOffsetLayout &layout) -> Status
    {
        ReadSubtensor<algorithmFPType, cpu, Tensor> inputBlock(const_cast<Tensor &>(inputGradientTensor), fDimN, fDims, 0, nRowsToProcess, layout);
        DAAL_CHECK_BLOCK_STATUS(inputBlock);
        const algorithmFPType *inputArray = inputBlock.get();

        ReadSubtensor<algorithmFPType, cpu, Tensor> forwardValueBlock(const_cast<Tensor &>(forwardDataTensor), fDimN, fDims, 0, nRowsToProcess, layout);
        DAAL_CHECK_BLOCK_STATUS(forwardValueBlock);
        const algorithmFPType *forwardValueArray = forwardValueBlock.get();

        WriteSubtensor<algorithmFPType, cpu, Tensor> resultBlock(resultTensor, fDimN, fDims, 0, nRowsToProcess, layout);
        DAAL_CHECK_BLOCK_STATUS(resultBlock);
        algorithmFPType *resultArray = resultBlock.get();

        algorithmFPType one = (algorithmFPType)1.0;
        size_t nDataElements = inputBlock.getSize();

        //res = in * 1/(exp(-fO)+1)
       PRAGMA_IVDEP
       PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < nDataElements; i++)
        {
            resultArray[i] = -forwardValueArray[i];

            /* Arguments filtering before vector exponential function call */
            /* There is a known issue that vExp works slowly on large negative arguments (where results are zero or denormals) */
          #if (__CPUID__(DAAL_CPU) != __avx512_mic__)
            if( resultArray[i] < daal::internal::Math<algorithmFPType,cpu>::vExpThreshold() )
            {
                resultArray[i] = daal::internal::Math<algorithmFPType,cpu>::vExpThreshold();
            }
          #endif
        }

        daal::internal::Math<algorithmFPType,cpu>::vExp(nDataElements, resultArray, resultArray);

       PRAGMA_IVDEP
       PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < nDataElements; i++)
        {
            resultArray[i] = one / (resultArray[i] + one);
            resultArray[i] = inputArray[i] * resultArray[i];
        }
        return Status();
    });
    return s;
}

} // namespace internal
} // backward
} // namespace smoothrelu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
