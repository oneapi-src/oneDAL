/* file: average_pooling1d_layer_forward_impl.i */
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
//  Implementation of forward average pooling layer
//--
*/

#ifndef __AVERAGE_POOLING1D_LAYER_FORWARD_IMPL_I__
#define __AVERAGE_POOLING1D_LAYER_FORWARD_IMPL_I__

#include "service_memory.h"
#include "service_data_utils.h"
#include "service_blas.h"

#include "pooling1d_layer_impl.i"

using namespace daal::services;

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
services::Status PoolingKernel<algorithmFPType, method, cpu>::compute(Tensor *dataTensor, const average_pooling1d::Parameter *parameter, Tensor *valueTensor)
{
    const algorithmFPType zero = 0.0;
    const algorithmFPType one = 1.0;

    const Collection<size_t> &dims = dataTensor->getDimensions();
    const Collection<size_t> &valueDims = valueTensor->getDimensions();
    SubtensorDescriptor<algorithmFPType> dataBlock, valueBlock;
    dataTensor->getSubtensor(0, 0, 0, dims[0], readOnly, dataBlock);
    valueTensor->getSubtensor(0, 0, 0, valueDims[0], writeOnly, valueBlock);

    algorithmFPType *data = dataBlock.getPtr();
    algorithmFPType *value = valueBlock.getPtr();

    pooling1d::internal::Parameter par(parameter->index .size[0], parameter->padding   .size[0],
                                       parameter->stride.size[0], parameter->kernelSize.size[0],
                                       dataTensor, dims, valueDims);

    algorithmFPType divisor = 1.0 / (algorithmFPType)(par.kernelSize);
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
                DAAL_INT valueIndex = j + par.offsetAfter * (fo + par.firstOutSize * i);

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
    dataTensor->releaseSubtensor(dataBlock);
    valueTensor->releaseSubtensor(valueBlock);
    DAAL_RETURN_STATUS()
}

} // namespace internal
} // namespace forward
} // namespace average_pooling1d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
