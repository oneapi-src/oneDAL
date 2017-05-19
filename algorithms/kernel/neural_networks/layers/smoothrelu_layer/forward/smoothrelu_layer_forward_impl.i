/* file: smoothrelu_layer_forward_impl.i */
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
// Implementation of the forward smooth rectifier linear unit (smooth relu) layer
//--
*/

#ifndef __SMOOTHRELU_LAYER_FORWARD_IMPL_I__
#define __SMOOTHRELU_LAYER_FORWARD_IMPL_I__

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
namespace forward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status SmoothReLUKernel<algorithmFPType, method, cpu>::compute(Tensor *inputTensor, Tensor *resultTensor)
{
    __DAAL_MAKE_TENSOR_THREADSAFE(resultTensor)

    computeImpl<cpu>(inputTensor, this->_errors.get(), [=](size_t fDimN, size_t *fDims, size_t nRowsToProcess, const TensorOffsetLayout &layout)
    {
        ReadSubtensor<algorithmFPType, cpu, Tensor> inputBlock(*inputTensor, fDimN, fDims, 0, nRowsToProcess, layout);
        const algorithmFPType *inputArray = inputBlock.get();

        WriteSubtensor<algorithmFPType, cpu, Tensor> resultBlock(*resultTensor, fDimN, fDims, 0, nRowsToProcess, layout);
        algorithmFPType *resultArray = resultBlock.get();

        algorithmFPType one = (algorithmFPType)1.0;
        size_t nDataElements = inputBlock.getSize();

        //res = log(1+exp(in))
        daal::internal::Math<algorithmFPType,cpu>::vExp(nDataElements, const_cast<algorithmFPType *>(inputArray), resultArray);
        daal::internal::Math<algorithmFPType,cpu>::vLog1p(nDataElements, resultArray, resultArray);
    });
    DAAL_RETURN_STATUS();
}

} // internal
} // forward
} // namespace smoothrelu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
