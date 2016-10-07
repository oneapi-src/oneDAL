/* file: average_pooling2d_layer_forward_impl.i */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

#ifndef __AVERAGE_POOLING2D_LAYER_FORWARD_IMPL_I__
#define __AVERAGE_POOLING2D_LAYER_FORWARD_IMPL_I__

#include "service_memory.h"
#include "service_data_utils.h"
#include "service_blas.h"

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace average_pooling2d
{
namespace forward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::compute(Tensor *dataTensor, const average_pooling2d::Parameter *parameter, Tensor *valueTensor)
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

    pooling2d::internal::Parameter par(parameter->indices.size, parameter->paddings.size,
                                       parameter->strides.size, parameter->kernelSizes.size,
                                       dataTensor, dims, valueDims);

    defaultCompute(par, data, value);

    dataTensor->releaseSubtensor(dataBlock);
    valueTensor->releaseSubtensor(valueBlock);
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::defaultInnerLoop(pooling2d::internal::Parameter &par,
                DAAL_INT i, DAAL_INT f, DAAL_INT k, DAAL_INT s, DAAL_INT j,
                const algorithmFPType *data, algorithmFPType *valuePtr)
{
    const algorithmFPType zero = 0.0;
    algorithmFPType average = zero;

    /*
     * Loops over the kernel
     */
    for (DAAL_INT fi = f; fi < f + par.firstKernelSize; fi++)
    {
        for (DAAL_INT si = s; si < s + par.secondKernelSize; si++)
        {
            DAAL_INT dataIndex = j + par.offsetAfter * (si + par.secondSize * (k + par.offsetBetween * (fi + par.firstSize * i)));
            bool paddingFlag = ((fi < 0) || (fi >= par.firstSize) || (si < 0) || (si >= par.secondSize));
            algorithmFPType dataValue = (paddingFlag ? zero : data[dataIndex]);

            average += dataValue;
        }
    }
    valuePtr[j] = average / (algorithmFPType)(par.firstKernelSize * par.secondKernelSize);
}

} // namespace internal
} // namespace forward
} // namespace average_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
