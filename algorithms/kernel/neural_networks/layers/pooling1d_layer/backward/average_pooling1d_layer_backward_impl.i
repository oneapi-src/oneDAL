/* file: average_pooling1d_layer_backward_impl.i */
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
//  Implementation of backward pooling layer
//--
*/

#ifndef __AVERAGE_POOLING1D_LAYER_BACKWARD_IMPL_I__
#define __AVERAGE_POOLING1D_LAYER_BACKWARD_IMPL_I__

#include "service_memory.h"
#include "service_blas.h"
#include "service_tensor.h"

#include "pooling1d_layer_impl.i"

using namespace daal::services;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace average_pooling1d
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status PoolingKernel<algorithmFPType, method, cpu>::compute(const Tensor &inputTensor, const average_pooling1d::Parameter &parameter,
                                                          Tensor &gradTensor)
{
    const Collection<size_t> &inputDims = inputTensor.getDimensions();
    const Collection<size_t> &gradDims = gradTensor.getDimensions();

    ReadSubtensor<algorithmFPType, cpu> inputBlock(const_cast<Tensor&>(inputTensor), 0, 0, 0, inputDims[0]);
    DAAL_CHECK_BLOCK_STATUS(inputBlock);
    const algorithmFPType *inputGrad = inputBlock.get();

    WriteOnlySubtensor<algorithmFPType, cpu> gradBlock(gradTensor, 0, 0, 0, gradDims[0]);
    DAAL_CHECK_BLOCK_STATUS(gradBlock);
    algorithmFPType *grad = gradBlock.get();

    const algorithmFPType zero = 0.0;
    daal::services::internal::service_memset<algorithmFPType, cpu>(grad, zero, gradBlock.getSize());

    pooling1d::internal::Parameter par(parameter.index .size[0], parameter.padding   .size[0],
                                       parameter.stride.size[0], parameter.kernelSize.size[0],
                                       gradTensor, gradDims, inputDims);

    const algorithmFPType one = 1.0;
    const algorithmFPType gradMultiplier = one / (algorithmFPType)(par.kernelSize);
    for (DAAL_INT i = 0; i < par.offsetBefore; i++)
    {
        /*
         * Loop by the first kernel dimension
         * f - index of the left upper corner of the kernel
         * fo - index of the output value
         */
        for (DAAL_INT f = -par.padding, fo = 0; fo < par.firstOutSize; f += par.stride, fo++)
        {
            for (DAAL_INT j = 0; j < par.offsetAfter; j++)
            {
                /*
                 * Input value index
                 */
                const DAAL_INT inputIndex = j + par.offsetAfter * (fo + par.firstOutSize * i);
                const algorithmFPType inputValue = gradMultiplier * inputGrad[inputIndex];

                /*
                 * Loops over the kernel
                 */
                for (DAAL_INT fi = f; fi < f + par.kernelSize; fi++)
                {
                    const DAAL_INT gradIndex = j + par.offsetAfter * (fi + par.firstSize * i);
                    const bool paddingFlag = (fi < 0) || (fi >= par.firstSize);

                    if (!paddingFlag)
                    {
                        grad[gradIndex] += inputValue;
                    }
                }
            }
        }
    }
    return Status();
}
} // namespace internal
} // namespace backward
} // namespace average_pooling1d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
