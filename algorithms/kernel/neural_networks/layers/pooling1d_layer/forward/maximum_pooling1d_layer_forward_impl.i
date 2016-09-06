/* file: maximum_pooling1d_layer_forward_impl.i */
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
//  Implementation of forward pooling layer
//--
*/

#ifndef __MAXIMUM_POOLING1D_LAYER_FORWARD_KERNEL_IMPL_I__
#define __MAXIMUM_POOLING1D_LAYER_FORWARD_KERNEL_IMPL_I__

#include "service_memory.h"
#include "service_data_utils.h"
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
namespace forward
{
namespace internal
{
template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::compute(Tensor *dataTensor, Tensor *valueTensor,
        Tensor *selectedPosTensor, const maximum_pooling1d::Parameter *parameter)
{
    const algorithmFPType zero = 0.0;

    const Collection<size_t> &dims = dataTensor->getDimensions();
    const Collection<size_t> &valueDims = valueTensor->getDimensions();

    ReadSubtensor<algorithmFPType, cpu, Tensor> dataSubtensor(dataTensor, 0, 0, 0, dims[0]);
    WriteOnlySubtensor<algorithmFPType, cpu, Tensor> valueSubtensor(valueTensor, 0, 0, 0, valueDims[0]);

    const algorithmFPType *data = dataSubtensor.get();
    algorithmFPType *value = valueSubtensor.get();

    int *selectedPos = nullptr;
    WriteOnlySubtensor<int, cpu, Tensor> selectedPosSubtensor;
    if(parameter->predictionStage == false)
    {
        selectedPosSubtensor.set(*selectedPosTensor, 0, 0, 0, valueDims[0]);
        selectedPos = selectedPosSubtensor.get();
        size_t selectedPosSize = selectedPosTensor->getSize();
        daal::services::internal::service_memset<int, cpu>(selectedPos, 0, selectedPosSize);
    }


    pooling1d::internal::Parameter par(parameter->index .size[0], parameter->padding   .size[0],
                                       parameter->stride.size[0], parameter->kernelSize.size[0],
                                       dataTensor, dims, valueDims);

    for (MKL_INT i = 0; i < par.offsetBefore; i++)
    {
        /*
         * Loop by the first kernel dimension
         * f - index of the left upper corner of the kernel
         * fo - index of the output value
         */
        for (MKL_INT f = -par.padding, fo = 0; fo < par.firstOutSize; f += par.stride, fo++)
        {
            for (MKL_INT j = 0; j < par.offsetAfter; j++)
            {
                /*
                 * Resulting value index
                 */
                MKL_INT valueIndex = j + par.offsetAfter * (fo + par.firstOutSize * i);

                algorithmFPType max = -(data_feature_utils::internal::MaxVal<algorithmFPType, cpu>::get());
                MKL_INT maxIdx = -1;

                /*
                 * Loop over the kernel
                 */
                for (MKL_INT fi = f; fi < f + par.kernelSize; fi++)
                {
                    MKL_INT dataIndex = j + par.offsetAfter * (fi + par.firstSize * i);
                    bool paddingFlag = (fi < 0) || (fi >= par.firstSize);
                    algorithmFPType dataValue = (paddingFlag ? zero : data[dataIndex]);

                    if (dataValue > max)
                    {
                        max = dataValue;
                        maxIdx = fi - f;
                    }
                }
                value[valueIndex] = max;
                if(selectedPos)
                {
                    selectedPos[valueIndex] = maxIdx;
                }
            }
        }
    }
}

} // namespace internal
} // namespace forward
} // namespace maximum_pooling1d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
