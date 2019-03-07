/* file: reshape_layer_backward_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
} // backward
} // namespace reshape
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
