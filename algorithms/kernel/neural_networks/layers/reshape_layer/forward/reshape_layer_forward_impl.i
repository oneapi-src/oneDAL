/* file: reshape_layer_forward_impl.i */
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
//  Implementation of reshape layer.
//--
*/

#ifndef __RESHAPE_LAYER_FORWARD_IMPL_I__
#define __RESHAPE_LAYER_FORWARD_IMPL_I__

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
namespace forward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status ReshapeKernel<algorithmFPType, method, cpu>::compute(const Tensor &inputTensor, Tensor &resultTensor)
{
    ReadSubtensor<algorithmFPType, cpu> iBlock(const_cast<Tensor&>(inputTensor), 0, 0, 0, inputTensor.getDimensionSize(0));
    DAAL_CHECK_BLOCK_STATUS(iBlock);
    const algorithmFPType *iArray = iBlock.get();

    WriteOnlySubtensor<algorithmFPType, cpu, Tensor> rBlock(resultTensor, 0, 0, 0, resultTensor.getDimensionSize(0));
    DAAL_CHECK_BLOCK_STATUS(rBlock);
    algorithmFPType *rArray = rBlock.get();

    const size_t nDataElements = iBlock.getSize();
    for(size_t i = 0; i < nDataElements; i++)
    {
        rArray[i] = iArray[i];
    }
    return Status();
}

} // internal
} // forward
} // namespace reshape
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
