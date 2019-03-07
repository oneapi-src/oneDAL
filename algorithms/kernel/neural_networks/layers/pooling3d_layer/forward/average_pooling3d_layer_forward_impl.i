/* file: average_pooling3d_layer_forward_impl.i */
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

#ifndef __AVERAGE_POOLING3D_LAYER_FORWARD_IMPL_I__
#define __AVERAGE_POOLING3D_LAYER_FORWARD_IMPL_I__

#include "service_sort.h"
#include "service_memory.h"
#include "service_data_utils.h"
#include "service_blas.h"
#include "service_tensor.h"

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
namespace average_pooling3d
{
namespace forward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status PoolingKernel<algorithmFPType, method, cpu>::compute(const Tensor &dataTensor, const average_pooling3d::Parameter &parameter,Tensor &valueTensor)
{
    const Collection<size_t> &dims = dataTensor.getDimensions();
    const Collection<size_t> &valueDims = valueTensor.getDimensions();

    ReadSubtensor<algorithmFPType, cpu> dataBlock(const_cast<Tensor&>(dataTensor), 0, 0, 0, dims[0]);
    DAAL_CHECK_BLOCK_STATUS(dataBlock);
    const algorithmFPType *data = dataBlock.get();

    WriteOnlySubtensor<algorithmFPType, cpu> valueBlock(valueTensor, 0, 0, 0, valueDims[0]);
    DAAL_CHECK_BLOCK_STATUS(valueBlock);
    algorithmFPType *value = valueBlock.get();

    pooling3d::internal::Parameter<cpu> par(parameter.indices.size, parameter.paddings   .size,
                                            parameter.strides.size, parameter.kernelSizes.size,
                                            dataTensor, dims, valueDims);

    algorithmFPType divisor = 1.0;
    for (size_t d = 0; d < nKernelDims; d++)
    {
        divisor *= (algorithmFPType)par.kernelSize[d];
    }
    divisor = 1.0 / divisor;

    DAAL_INT ii[nKernelDims + 1];    // index of the input data
    DAAL_INT ik[nKernelDims];        // index of the kernel
    DAAL_INT iv[nKernelDims];        // index of the value
    DAAL_INT valueOffset[nKernelDims + 1];
    DAAL_INT dataOffset[nKernelDims + 1];

    for (ii[0] = 0; ii[0] < par.offset[0]; ii[0]++)
    {
        valueOffset[0] = 0;
        dataOffset[0]  = 0;
        /*
         * Process the dimensions of input tensor recursively
         */
        recurrentCompute(0, ii, ik, iv, par.padding, par.stride, par.kernelSize, par.dataSize, par.valueSize,
            par.offset, dataOffset, valueOffset, data, value, divisor);
    }
    return Status();
}


template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::recurrentCompute(size_t d,
    DAAL_INT *ii, DAAL_INT *ik, DAAL_INT *iv, const DAAL_INT *padding, const DAAL_INT *stride, const DAAL_INT *kernelSize,
    const DAAL_INT* dataSize, const DAAL_INT* valueSize, const DAAL_INT* offset, DAAL_INT* dataOffset, DAAL_INT* valueOffset,
    const algorithmFPType *data, algorithmFPType *value, algorithmFPType divisor)
{
    const algorithmFPType zero = 0.0;

    if (d < nKernelDims)
    {
        /*
         * Loops over the d-th kernel dimension
         */
        for (ik[d] = -padding[d], iv[d] = 0; iv[d] < valueSize[d]; ik[d] += stride[d], iv[d]++)
        {
            valueOffset[d+1] = offset[d+1] * (iv[d] + valueSize[d] * (ii[d] + valueOffset[d]));
            dataOffset[d+1]  = offset[d+1] * (ik[d] + dataSize[d]  * (ii[d] + dataOffset[d]));

            for (ii[d+1] = 0; ii[d+1] < offset[d+1]; ii[d+1]++)
            {
                recurrentCompute(d + 1, ii, ik, iv, padding, stride, kernelSize, dataSize, valueSize,
                    offset, dataOffset, valueOffset, data, value, divisor);
            }
        }
    }
    else
    {
        /*
         * Resulting value index
         */
        DAAL_INT valueIndex = ii[3] + valueOffset[3];

        algorithmFPType average = zero;

        DAAL_INT iwk[nKernelDims];              // index of the value within kernel
        DAAL_INT iwkShifted[nKernelDims];
        DAAL_INT dataKernelOffset[nKernelDims];
        bool paddingFlags[nKernelDims];
        /*
         * Loops over the kernel
         */
        for (iwk[0] = 0, iwkShifted[0] = ik[0]; iwk[0] < kernelSize[0]; iwk[0]++, iwkShifted[0]++)
        {
            paddingFlags[0] = (iwkShifted[0] < 0) || (iwkShifted[0] >= (DAAL_INT)dataSize[0]);
            dataKernelOffset[0] = offset[1] * iwk[0];
            for (iwk[1] = 0, iwkShifted[1] = ik[1]; iwk[1] < kernelSize[1]; iwk[1]++, iwkShifted[1]++)
            {
                paddingFlags[1] = (iwkShifted[1] < 0) || (iwkShifted[1] >= (DAAL_INT)dataSize[1]);
                dataKernelOffset[1] = offset[2] * (iwk[1] + dataSize[1] * dataKernelOffset[0]);
                for (iwk[2] = 0, iwkShifted[2] = ik[2]; iwk[2] < kernelSize[2]; iwk[2]++, iwkShifted[2]++)
                {
                    paddingFlags[2] = (iwkShifted[2] < 0) || (iwkShifted[2] >= (DAAL_INT)dataSize[2]);
                    dataKernelOffset[2] = offset[3] * (iwk[2] + dataSize[2] * dataKernelOffset[1]);
                    DAAL_INT dataIndex = ii[3] + dataOffset[3] + dataKernelOffset[2];

                    bool paddingFlag = paddingFlags[0] || paddingFlags[1] || paddingFlags[2];
                    algorithmFPType dataValue = (paddingFlag ? zero : data[dataIndex]);

                    average += dataValue;
                }
            }
        }
        value[valueIndex] = average * divisor;
    }
}

} // namespace internal
} // namespace forward
} // namespace average_pooling3d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
