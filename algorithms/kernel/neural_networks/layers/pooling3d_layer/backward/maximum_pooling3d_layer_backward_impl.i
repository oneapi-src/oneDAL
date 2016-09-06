/* file: maximum_pooling3d_layer_backward_impl.i */
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
//  Implementation of backward pooling layer
//--
*/

#ifndef __MAXIMUM_POOLING3D_LAYER_BACKWARD_IMPL_I__
#define __MAXIMUM_POOLING3D_LAYER_BACKWARD_IMPL_I__

#include "service_sort.h"
#include "service_memory.h"
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
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::compute(Tensor *inputGradTensor,
                Tensor *selectedPosTensor, Tensor *gradTensor,
                const maximum_pooling3d::Parameter *parameter)
{
    const algorithmFPType zero = 0.0;

    const Collection<size_t> &inputDims = inputGradTensor->getDimensions();
    const Collection<size_t> &gradDims = gradTensor->getDimensions();

    ReadSubtensor<algorithmFPType, cpu, Tensor> inputGradSubtensor(inputGradTensor, 0, 0, 0, inputDims[0]);
    ReadSubtensor<int, cpu, Tensor> selectedPosSubtensor(selectedPosTensor, 0, 0, 0, inputDims[0]);
    WriteOnlySubtensor<algorithmFPType, cpu, Tensor> gradSubtensor(gradTensor, 0, 0, 0, gradDims[0]);

    const algorithmFPType *inputGrad = inputGradSubtensor.get();
    const int *selectedPos = selectedPosSubtensor.get();
    algorithmFPType *grad = gradSubtensor.get();

    size_t gradientSize = gradTensor->getSize();
    daal::services::internal::service_memset<algorithmFPType, cpu>(grad, zero, gradientSize);

    pooling3d::internal::Parameter<cpu> par(parameter->indices.size, parameter->paddings   .size,
                                            parameter->strides.size, parameter->kernelSizes.size,
                                            gradTensor, gradDims, inputDims);

    MKL_INT ii[nKernelDims + 1];    // index of the input data
    MKL_INT ik[nKernelDims];        // index of the kernel
    MKL_INT iv[nKernelDims];        // index of the value
    MKL_INT inputOffset[nKernelDims + 1];
    MKL_INT gradOffset[nKernelDims + 1];

    for (ii[0] = 0; ii[0] < par.offset[0]; ii[0]++)
    {
        inputOffset[0] = ii[0];
        gradOffset[0]  = ii[0];
        recurrentCompute(0, ii, ik, iv, par.padding, par.stride, par.kernelSize, par.dataSize, par.valueSize,
            par.offset, gradOffset, inputOffset, inputGrad, grad, selectedPos);
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::recurrentCompute(size_t d,
    MKL_INT *ii, MKL_INT *ik, MKL_INT *iv, const MKL_INT *padding, const MKL_INT *stride, const MKL_INT *kernelSize,
    const MKL_INT* gradSize, const MKL_INT* inputSize, const MKL_INT* offset, MKL_INT* gradOffset, MKL_INT* inputOffset,
    const algorithmFPType *inputGrad, algorithmFPType *grad, const int *selectedPos)
{
    const algorithmFPType zero = 0.0;
    if (d < nKernelDims)
    {
        /*
         * Loops over the d-th kernel dimension
         */
        for (ik[d] = -padding[d], iv[d] = 0; iv[d] < inputSize[d]; ik[d] += stride[d], iv[d]++)
        {
            inputOffset[d+1] = offset[d+1] * (iv[d] + inputSize[d] * inputOffset[d]);
            gradOffset[d+1] = offset[d+1] * (ik[d] + gradSize[d] * gradOffset[d]);
            for (ii[d+1] = 0; ii[d+1] < offset[d+1]; ii[d+1]++)
            {
                recurrentCompute(d + 1, ii, ik, iv, padding, stride, kernelSize, gradSize, inputSize,
                    offset, gradOffset, inputOffset, inputGrad, grad, selectedPos);
            }
        }
    }
    else
    {
        /*
         * Input gradient index
         */
        MKL_INT inputIndex = ii[nKernelDims] + inputOffset[nKernelDims];

        MKL_INT iwk[nKernelDims];              // index of the gradient within kernel
        MKL_INT gradKernelOffset[nKernelDims];

        if (selectedPos[inputIndex] >= 0)
        {
            MKL_INT kernelSize01 = kernelSize[1] * kernelSize[2];
            iwk[0] = selectedPos[inputIndex] / kernelSize01;
            MKL_INT residual = selectedPos[inputIndex] - iwk[0] * kernelSize01;
            iwk[1] = residual / kernelSize[2];
            iwk[2] = residual - iwk[1] * kernelSize[2];
            bool paddingFlag = false;
            for (size_t i = 0; i < nKernelDims; i++)
            {
                MKL_INT iwkShifted = ik[i] + iwk[i];
                paddingFlag = (paddingFlag || (iwkShifted < 0) || (iwkShifted >= gradSize[i]));
            }
            if (!paddingFlag)
            {
                gradKernelOffset[0] = offset[1] * iwk[0];
                gradKernelOffset[1] = offset[2] * (iwk[1] + gradSize[1] * gradKernelOffset[0]);
                gradKernelOffset[2] = offset[3] * (iwk[2] + gradSize[2] * gradKernelOffset[1]);
                MKL_INT gradIndex = ii[nKernelDims] + gradOffset[nKernelDims] + gradKernelOffset[nKernelDims - 1];
                grad[gradIndex] += inputGrad[inputIndex];
            }
        }
    }
}

} // namespace internal
} // namespace backward
} // namespace maximum_pooling3d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
