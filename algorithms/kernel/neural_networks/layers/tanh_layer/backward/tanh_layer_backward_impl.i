/* file: tanh_layer_backward_impl.i */
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
//  Implementation of hyperbolic tangent algorithm
//--
*/

#ifndef __TANH_LAYER_BACKWARD_IMPL_I__
#define __TANH_LAYER_BACKWARD_IMPL_I__

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace tanh
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status TanhKernel<algorithmFPType, method, cpu>::compute(Tensor *inputTensor, Tensor *forwardOutputTensor, Tensor *resultTensor)
{
    __DAAL_MAKE_TENSOR_THREADSAFE(forwardOutputTensor)
    __DAAL_MAKE_TENSOR_THREADSAFE(resultTensor)

    computeImpl<cpu>(inputTensor, this->_errors.get(), [=](size_t fDimN, size_t *fDims, size_t nRowsToProcess, const TensorOffsetLayout &layout)
    {
        ReadSubtensor<algorithmFPType, cpu, Tensor> inputBlock(*inputTensor, fDimN, fDims, 0, nRowsToProcess, layout);
        const algorithmFPType *inputArray = inputBlock.get();

        ReadSubtensor<algorithmFPType, cpu, Tensor> forwardBlock(*forwardOutputTensor, fDimN, fDims, 0, nRowsToProcess, layout);
        const algorithmFPType *forwardOutputArray = forwardBlock.get();

        WriteSubtensor<algorithmFPType, cpu, Tensor> resultBlock(*resultTensor, fDimN, fDims, 0, nRowsToProcess, layout);
        algorithmFPType *resultArray = resultBlock.get();

        size_t nDataElements = inputBlock.getSize();

       PRAGMA_IVDEP
       PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < nDataElements; i++)
        {
            resultArray[i] = inputArray[i] * ( (algorithmFPType)1 - forwardOutputArray[i] * forwardOutputArray[i] );
        }
    });
    DAAL_RETURN_STATUS();
}

} // namespace internal
} // namespace backward
} // namespace tanh
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
