/* file: maximum_pooling2d_layer_backward_impl.i */
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

#ifndef __MAXIMUM_POOLING2D_LAYER_BACKWARD_IMPL_I__
#define __MAXIMUM_POOLING2D_LAYER_BACKWARD_IMPL_I__

#include "service_memory.h"
#include "service_blas.h"
#include "service_tensor.h"
#include "service_numeric_table.h"
#include "threading.h"

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
namespace maximum_pooling2d
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::compute(Tensor *inputGradTensor,
                Tensor *selectedPosTensor, Tensor *gradTensor,
                const pooling2d::Parameter *parameter)
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

    size_t gradSize = gradTensor->getSize();
    daal::services::internal::service_memset<algorithmFPType, cpu>(grad, zero, gradSize);

    pooling2d::internal::Parameter par(parameter->indices.size, parameter->paddings   .size,
                                            parameter->strides.size, parameter->kernelSizes.size,
                                            gradTensor, gradDims, inputDims);

    size_t nDim = inputDims.size();
    if (par.firstIndex == nDim - 2 && par.secondIndex == nDim - 1 && par.firstPadding == 0 && par.secondPadding == 0)
    {
        indicesLastZeroPaddingsCompute(par, inputGrad, selectedPos, grad);
    }
    else if (par.firstIndex == 0 && par.secondIndex == 1 && par.firstPadding == 0 && par.secondPadding == 0)
    {
        indicesFirstZeroPaddingsCompute(par, inputGrad, selectedPos, grad);
    }
    else
    {
        defaultCompute(par, inputGrad, selectedPos, grad);
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::defaultInnerLoop(
                pooling2d::internal::Parameter &par,
                DAAL_INT i, DAAL_INT f, DAAL_INT k, DAAL_INT s,
                const algorithmFPType *inputGradPtr, const int *selectedPosPtr,
                algorithmFPType *grad)
{
    for (DAAL_INT j = 0; j < par.offsetAfter; j++)
    {
        if (selectedPosPtr[j] >= 0)
        {
            DAAL_INT fOffset = selectedPosPtr[j] / par.secondKernelSize;
            DAAL_INT sOffset = selectedPosPtr[j] - fOffset * par.secondKernelSize;
            DAAL_INT fi = f + fOffset;
            DAAL_INT si = s + sOffset;
            bool paddingFlag = ((fi < 0) || (fi >= par.firstSize) || (si < 0) || (si >= par.secondSize));
            if (!paddingFlag)
            {
                DAAL_INT gradIndex = j + par.offsetAfter * (si + par.secondSize * (k + par.offsetBetween * (fi + par.firstSize * i)));
                grad[gradIndex] += inputGradPtr[j];
            }
        }
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::indicesLastZeroPaddingsCompute(
                pooling2d::internal::Parameter &par,
                const algorithmFPType *inputGrad, const int *selectedPos,
                algorithmFPType *grad)
{
    threader_for(par.offsetBefore, par.offsetBefore, [&](size_t i)
    {
        /*
         * Loop by the first kernel dimension
         * f - index of the left upper corner of the kernel
         * fo - index of the output value
         */
        for (DAAL_INT f = 0, fo = 0; fo < par.firstOutSize; f += par.firstStride, fo++)
        {
            /*
             * Loop by the second kernel dimension
             * s - index of the left upper corner of the kernel
             */
            DAAL_INT inputIndex = par.secondOutSize * (fo + par.firstOutSize * i);
            const algorithmFPType *inputGradPtr = inputGrad + inputIndex;
            const int *selectedPosPtr = selectedPos + inputIndex;
            for (DAAL_INT s = 0, so = 0; so < par.secondOutSize; s += par.secondStride, so++)
            {
                if (selectedPosPtr[so] >= 0)
                {
                    DAAL_INT fOffset = selectedPosPtr[so] / par.secondKernelSize;
                    DAAL_INT sOffset = selectedPosPtr[so] - fOffset * par.secondKernelSize;
                    DAAL_INT fi = f + fOffset;
                    DAAL_INT si = s + sOffset;
                    DAAL_INT gradIndex = si + par.secondSize * (fi + par.firstSize * i);
                    grad[gradIndex] += inputGradPtr[so];
                }
            }
        }
    } );
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::indicesFirstZeroPaddingsCompute(
                pooling2d::internal::Parameter &par,
                const algorithmFPType *inputGrad, const int *selectedPos,
                algorithmFPType *grad)
{
    /*
     * Loop by the first kernel dimension
     * f - index of the left upper corner of the kernel
     * fo - index of the output value
     */
    threader_for(par.firstOutSize, par.firstOutSize, [&](size_t fo)
    {
        DAAL_INT f = fo * par.firstStride;
        /*
         * Loop by the second kernel dimension
         * s - index of the left upper corner of the kernel
         * so - index of the output value
         */
        for (DAAL_INT s = 0, so = 0; so < par.secondOutSize; s += par.secondStride, so++)
        {
            DAAL_INT inputIndex = par.offsetAfter * (so + par.secondOutSize * fo);
            const algorithmFPType *inputGradPtr = inputGrad + inputIndex;
            const int *selectedPosPtr = selectedPos + inputIndex;
            for (DAAL_INT j = 0; j < par.offsetAfter; j++)
            {
                if (selectedPos[inputIndex] >= 0)
                {
                    DAAL_INT fOffset = selectedPosPtr[j] / par.secondKernelSize;
                    DAAL_INT sOffset = selectedPosPtr[j] - fOffset * par.secondKernelSize;
                    DAAL_INT fi = f + fOffset;
                    DAAL_INT si = s + sOffset;
                    DAAL_INT gradIndex = j + par.offsetAfter * (si + par.secondSize * fi);
                    grad[gradIndex] += inputGradPtr[j];
                }
            }
        }
    } );
}

} // namespace internal
} // namespace backward
} // namespace maximum_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
