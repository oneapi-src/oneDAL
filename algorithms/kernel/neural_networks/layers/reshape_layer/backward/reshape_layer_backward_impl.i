/* file: reshape_layer_backward_impl.i */
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
//  Implementation of reshape layer.
//--
*/

#ifndef __RESHAPE_LAYER_BACKWARD_IMPL_I__
#define __RESHAPE_LAYER_BACKWARD_IMPL_I__

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace reshape
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status ReshapeKernel<algorithmFPType, method, cpu>::compute(Tensor *inputTensor, Tensor *resultTensor)
{
    const services::Collection<size_t>& iDims = inputTensor->getDimensions();
    const services::Collection<size_t>& oDims = resultTensor->getDimensions();

    ReadSubtensor<algorithmFPType, cpu> iBlock(inputTensor, 0, 0, 0, iDims[0]);
    const algorithmFPType *iArray = iBlock.get();

    WriteSubtensor<algorithmFPType, cpu, Tensor> rBlock(*resultTensor, 0, 0, 0, oDims[0]);
    algorithmFPType *rArray = rBlock.get();

    size_t nDataElements = iBlock.getSize();
    for(size_t i = 0; i < nDataElements; i++)
    {
        rArray[i] = iArray[i];
    }
    DAAL_RETURN_STATUS()
}

} // internal
} // backward
} // namespace reshape
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
