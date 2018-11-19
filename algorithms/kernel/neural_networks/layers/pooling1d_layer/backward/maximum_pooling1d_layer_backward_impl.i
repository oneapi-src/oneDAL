/* file: maximum_pooling1d_layer_backward_impl.i */
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
//  Implementation of backward pooling layer
//--
*/

#ifndef __MAXIMUM_POOLING1D_LAYER_BACKWARD_IMPL_I__
#define __MAXIMUM_POOLING1D_LAYER_BACKWARD_IMPL_I__

#include "service_memory.h"
#include "service_blas.h"
#include "service_tensor.h"
#include "service_numeric_table.h"

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
namespace maximum_pooling1d
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status PoolingKernel<algorithmFPType, method, cpu>::compute(const Tensor &inputGradTensor,
                const Tensor &selectedPosTensor, Tensor &gradTensor,
                const maximum_pooling1d::Parameter &parameter)
{
    const algorithmFPType zero = 0.0;

    const Collection<size_t> &inputDims = inputGradTensor.getDimensions();
    const Collection<size_t> &gradDims = gradTensor.getDimensions();

    ReadSubtensor<algorithmFPType, cpu> inputGradSubtensor(const_cast<Tensor&>(inputGradTensor), 0, 0, 0, inputDims[0]);
    DAAL_CHECK_BLOCK_STATUS(inputGradSubtensor);
    const algorithmFPType *inputGrad = inputGradSubtensor.get();

    ReadSubtensor<int, cpu> selectedPosSubtensor(const_cast<Tensor&>(selectedPosTensor), 0, 0, 0, inputDims[0]);
    DAAL_CHECK_BLOCK_STATUS(selectedPosSubtensor);
    const int *selectedPos = selectedPosSubtensor.get();

    WriteOnlySubtensor<algorithmFPType, cpu> gradSubtensor(gradTensor, 0, 0, 0, gradDims[0]);
    DAAL_CHECK_BLOCK_STATUS(gradSubtensor);
    algorithmFPType *grad = gradSubtensor.get();

    const size_t gradSize = gradTensor.getSize();
    daal::services::internal::service_memset<algorithmFPType, cpu>(grad, zero, gradSize);

    pooling1d::internal::Parameter par(parameter.index .size[0], parameter.padding   .size[0],
                                       parameter.stride.size[0], parameter.kernelSize.size[0],
                                       gradTensor, gradDims, inputDims);

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
                const DAAL_INT fi = f + selectedPos[inputIndex];
                const bool paddingFlag = (fi < 0) || (fi >= par.firstSize);
                if (!paddingFlag && selectedPos[inputIndex] >= 0)
                {
                    DAAL_INT gradIndex = j + par.offsetAfter * (fi + par.firstSize * i);
                    grad[gradIndex] += inputGrad[inputIndex];
                }
            }
        }
    }
    return Status();
}

} // namespace internal
} // namespace backward
} // namespace maximum_pooling1d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
