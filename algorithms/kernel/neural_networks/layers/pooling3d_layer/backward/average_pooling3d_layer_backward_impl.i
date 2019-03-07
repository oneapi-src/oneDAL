/* file: average_pooling3d_layer_backward_impl.i */
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
//  Implementation of backward average pooling layer
//--
*/

#ifndef __AVERAGE_POOLING3D_LAYER_BACKWARD_IMPL_I__
#define __AVERAGE_POOLING3D_LAYER_BACKWARD_IMPL_I__

#include "service_sort.h"
#include "service_memory.h"
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
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status PoolingKernel<algorithmFPType, method, cpu>::compute(const Tensor &inputTensor, const average_pooling3d::Parameter &parameter, Tensor &gradTensor)
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

    pooling3d::internal::Parameter<cpu> par(parameter.indices.size, parameter.paddings   .size,
                                            parameter.strides.size, parameter.kernelSizes.size,
                                            gradTensor, gradDims, inputDims);

    const algorithmFPType one = 1.0;
    algorithmFPType gradMultiplier = 1.0;
    for (size_t d = 0; d < nKernelDims; d++)
    {
        gradMultiplier *= (algorithmFPType)par.kernelSize[d];
    }
    gradMultiplier = 1.0 / gradMultiplier;

    DAAL_INT ii[nKernelDims + 1];    // index of the input data
    DAAL_INT ik[nKernelDims];        // index of the kernel
    DAAL_INT iv[nKernelDims];        // index of the value
    DAAL_INT inputOffset[nKernelDims + 1];
    DAAL_INT gradOffset[nKernelDims + 1];

    for (ii[0] = 0; ii[0] < par.offset[0]; ii[0]++)
    {
        inputOffset[0] = 0;
        gradOffset[0]  = 0;

        recurrentCompute(0, ii, ik, iv, par.padding, par.stride, par.kernelSize, par.dataSize, par.valueSize,
            par.offset, gradOffset, inputOffset, inputGrad, grad, gradMultiplier);
    }
    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::recurrentCompute(size_t d,
    DAAL_INT *ii, DAAL_INT *ik, DAAL_INT *iv, const DAAL_INT *padding, const DAAL_INT *stride, const DAAL_INT *kernelSize,
    const DAAL_INT* gradSize, const DAAL_INT* inputSize, const DAAL_INT* offset, DAAL_INT* gradOffset, DAAL_INT* inputOffset,
    const algorithmFPType *inputGrad, algorithmFPType *grad, algorithmFPType gradMultiplier)
{
    const algorithmFPType zero = 0.0;
    if (d < nKernelDims)
    {
        /*
         * Loops over the d-th kernel dimension
         */
        for (ik[d] = -padding[d], iv[d] = 0; iv[d] < inputSize[d]; ik[d] += stride[d], iv[d]++)
        {
            inputOffset[d+1] = offset[d+1] * (iv[d] + inputSize[d] * (ii[d] + inputOffset[d]));
            gradOffset[d+1]  = offset[d+1] * (ik[d] + gradSize[d]  * (ii[d] + gradOffset[d]));

            for (ii[d+1] = 0; ii[d+1] < offset[d+1]; ii[d+1]++)
            {
                recurrentCompute(d + 1, ii, ik, iv, padding, stride, kernelSize, gradSize, inputSize,
                    offset, gradOffset, inputOffset, inputGrad, grad, gradMultiplier);
            }
        }
    }
    else
    {
        /*
         * Input gradient index
         */
        DAAL_INT inputIndex = ii[3] + inputOffset[3];
        algorithmFPType inputValue = gradMultiplier * inputGrad[inputIndex];

        DAAL_INT iwk[nKernelDims];              // index of the gradient within kernel
        DAAL_INT iwkShifted[nKernelDims];
        DAAL_INT gradKernelOffset[nKernelDims];
        bool paddingFlags[nKernelDims];
        /*
         * Loops over the kernel
         */
        for (iwk[0] = 0, iwkShifted[0] = ik[0]; iwk[0] < kernelSize[0]; iwk[0]++, iwkShifted[0]++)
        {
            paddingFlags[0] = (iwkShifted[0] < 0) || (iwkShifted[0] >= gradSize[0]);
            gradKernelOffset[0] = offset[1] * iwk[0];
            for (iwk[1] = 0, iwkShifted[1] = ik[1]; iwk[1] < kernelSize[1]; iwk[1]++, iwkShifted[1]++)
            {
                paddingFlags[1] = (iwkShifted[1] < 0) || (iwkShifted[1] >= gradSize[1]);
                gradKernelOffset[1] = offset[2] * (iwk[1] + gradSize[1] * gradKernelOffset[0]);
                for (iwk[2] = 0, iwkShifted[2] = ik[2]; iwk[2] < kernelSize[2]; iwk[2]++, iwkShifted[2]++)
                {
                    paddingFlags[2] = (iwkShifted[2] < 0) || (iwkShifted[2] >= gradSize[2]);
                    gradKernelOffset[2] = offset[3] * (iwk[2] + gradSize[2] * gradKernelOffset[1]);
                    DAAL_INT gradIndex = ii[3] + gradOffset[3] + gradKernelOffset[2];
                    bool paddingFlag = paddingFlags[0] || paddingFlags[1] || paddingFlags[2];

                    if (!paddingFlag)
                    {
                        grad[gradIndex] += inputValue;
                    }
                }
            }
        }
    }
}

} // namespace internal
} // namespace backward
} // namespace average_pooling3d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
