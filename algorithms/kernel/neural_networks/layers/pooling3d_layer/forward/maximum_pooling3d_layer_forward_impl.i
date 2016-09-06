/* file: maximum_pooling3d_layer_forward_impl.i */
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

#ifndef __MAXIMUM_POOLING3D_LAYER_FORWARD_IMPL_I__
#define __MAXIMUM_POOLING3D_LAYER_FORWARD_IMPL_I__

#include "service_memory.h"
#include "service_data_utils.h"
#include "service_blas.h"
#include "service_tensor.h"
#include "service_numeric_table.h"

#include "pooling3d_layer_impl.i"

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
namespace maximum_pooling3d
{
namespace forward
{
namespace internal
{
template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::compute(Tensor *dataTensor, Tensor *valueTensor,
        Tensor *selectedPosTensor, const maximum_pooling3d::Parameter *parameter)
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

    pooling3d::internal::Parameter<cpu> par(parameter->indices.size, parameter->paddings   .size,
                                            parameter->strides.size, parameter->kernelSizes.size,
                                            dataTensor, dims, valueDims);

    MKL_INT ii[nKernelDims + 1];    // index of the input data
    MKL_INT ik[nKernelDims];        // index of the kernel
    MKL_INT iv[nKernelDims];        // index of the value
    MKL_INT valueOffset[nKernelDims + 1];
    MKL_INT dataOffset[nKernelDims + 1];

    for (ii[0] = 0; ii[0] < par.offset[0]; ii[0]++)
    {
        valueOffset[0] = ii[0];
        dataOffset[0]  = ii[0];
        recurrentCompute(0, ii, ik, iv, par.padding, par.stride, par.kernelSize, par.dataSize, par.valueSize,
                         par.offset, dataOffset, valueOffset, data, value, selectedPos);
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::recurrentCompute(size_t d,
        MKL_INT *ii, MKL_INT *ik, MKL_INT *iv, const MKL_INT *padding, const MKL_INT *stride, const MKL_INT *kernelSize,
        const MKL_INT *dataSize, const MKL_INT *valueSize, const MKL_INT *offset, MKL_INT *dataOffset, MKL_INT *valueOffset,
        const algorithmFPType *data, algorithmFPType *value, int *selectedPos)
{
    const algorithmFPType zero = 0.0;
    const algorithmFPType one = 1.0;
    if (d < nKernelDims)
    {
        /*
         * Loops over the d-th kernel dimension
         */
        for (ik[d] = -padding[d], iv[d] = 0; iv[d] < valueSize[d]; ik[d] += stride[d], iv[d]++)
        {
            valueOffset[d + 1] = offset[d + 1] * (iv[d] + valueSize[d] * valueOffset[d]);
            dataOffset[d + 1] = offset[d + 1] * (ik[d] + dataSize[d] * dataOffset[d]);
            for (ii[d + 1] = 0; ii[d + 1] < offset[d + 1]; ii[d + 1]++)
            {
                recurrentCompute(d + 1, ii, ik, iv, padding, stride, kernelSize, dataSize, valueSize,
                                 offset, dataOffset, valueOffset, data, value, selectedPos);
            }
        }
    }
    else
    {
        /*
         * Resulting value index
         */
        MKL_INT valueIndex = ii[nKernelDims] + valueOffset[nKernelDims];

        algorithmFPType max = -(data_feature_utils::internal::MaxVal<algorithmFPType, cpu>::get());
        MKL_INT maxIdx = -1;

        MKL_INT iwk[nKernelDims];              // index of the value within kernel
        MKL_INT iwkShifted[nKernelDims];
        MKL_INT dataKernelOffset[nKernelDims];
        bool paddingFlags[nKernelDims];
        /*
         * Loops over the kernel
         */
        for (iwk[0] = 0, iwkShifted[0] = ik[0]; iwk[0] < kernelSize[0]; iwk[0]++, iwkShifted[0]++)
        {
            paddingFlags[0] = (iwkShifted[0] < 0) || (iwkShifted[0] >= (MKL_INT)dataSize[0]);
            dataKernelOffset[0] = offset[1] * iwk[0];
            for (iwk[1] = 0, iwkShifted[1] = ik[1]; iwk[1] < kernelSize[1]; iwk[1]++, iwkShifted[1]++)
            {
                paddingFlags[1] = (iwkShifted[1] < 0) || (iwkShifted[1] >= (MKL_INT)dataSize[1]);
                dataKernelOffset[1] = offset[2] * (iwk[1] + dataSize[1] * dataKernelOffset[0]);
                for (iwk[2] = 0, iwkShifted[2] = ik[2]; iwk[2] < kernelSize[2]; iwk[2]++, iwkShifted[2]++)
                {
                    paddingFlags[2] = (iwkShifted[2] < 0) || (iwkShifted[2] >= (MKL_INT)dataSize[2]);
                    dataKernelOffset[2] = offset[3] * (iwk[2] + dataSize[2] * dataKernelOffset[1]);
                    MKL_INT dataIndex = ii[3] + dataOffset[3] + dataKernelOffset[2];

                    bool paddingFlag = paddingFlags[0] || paddingFlags[1] || paddingFlags[2];
                    algorithmFPType dataValue = (paddingFlag ? zero : data[dataIndex]);

                    if (dataValue > max)
                    {
                        max = dataValue;
                        maxIdx = (iwk[0] * kernelSize[1] + iwk[1]) * kernelSize[2] + iwk[2];
                    }
                }
            }
        }
        value[valueIndex] = max;
        if(selectedPos)
        {
            selectedPos[valueIndex] = maxIdx;
        }
    }
}

} // namespace internal
} // namespace forward
} // namespace maximum_pooling3d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
