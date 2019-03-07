/* file: average_pooling1d_layer_forward_impl.i */
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
//  Implementation of forward average pooling layer
//--
*/

#ifndef __AVERAGE_POOLING1D_LAYER_FORWARD_IMPL_I__
#define __AVERAGE_POOLING1D_LAYER_FORWARD_IMPL_I__

#include "service_memory.h"
#include "service_data_utils.h"
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
namespace forward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status PoolingKernel<algorithmFPType, method, cpu>::compute(const Tensor &dataTensor, const average_pooling1d::Parameter &parameter, Tensor &valueTensor)
{
    const algorithmFPType zero = 0.0;
    const algorithmFPType one = 1.0;

    const Collection<size_t> &dims = dataTensor.getDimensions();
    const Collection<size_t> &valueDims = valueTensor.getDimensions();

    ReadSubtensor<algorithmFPType, cpu> dataBlock(const_cast<Tensor&>(dataTensor), 0, 0, 0, dims[0]);
    DAAL_CHECK_BLOCK_STATUS(dataBlock);
    const algorithmFPType *data = dataBlock.get();

    WriteOnlySubtensor<algorithmFPType, cpu> valueBlock(valueTensor, 0, 0, 0, valueDims[0]);
    DAAL_CHECK_BLOCK_STATUS(valueBlock);
    algorithmFPType *value = valueBlock.get();

    pooling1d::internal::Parameter par(parameter.index .size[0], parameter.padding   .size[0],
                                       parameter.stride.size[0], parameter.kernelSize.size[0],
                                       dataTensor, dims, valueDims);

    const algorithmFPType divisor = 1.0 / (algorithmFPType)(par.kernelSize);
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
                 * Resulting value index
                 */
                const DAAL_INT valueIndex = j + par.offsetAfter * (fo + par.firstOutSize * i);

                algorithmFPType average = zero;
                /*
                 * Loop over the kernel
                 */
                for (DAAL_INT fi = f; fi < f + par.kernelSize; fi++)
                {
                    DAAL_INT dataIndex = j + par.offsetAfter * (fi + par.firstSize * i);
                    bool paddingFlag = (fi < 0) || (fi >= par.firstSize);
                    algorithmFPType dataValue = (paddingFlag ? zero : data[dataIndex]);

                    average += dataValue;
                }
                value[valueIndex] = average * divisor;
            }
        }
    }
    return Status();
}

} // namespace internal
} // namespace forward
} // namespace average_pooling1d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
