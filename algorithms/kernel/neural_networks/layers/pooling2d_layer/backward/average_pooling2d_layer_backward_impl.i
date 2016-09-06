/* file: average_pooling2d_layer_backward_impl.i */
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
//  Implementation of backward average pooling layer
//--
*/

#ifndef __AVERAGE_POOLING2D_LAYER_BACKWARD_IMPL_I__
#define __AVERAGE_POOLING2D_LAYER_BACKWARD_IMPL_I__

#include "service_memory.h"
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
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::compute(
    const average_pooling2d::backward::Input *input, const average_pooling2d::Parameter *parameter,
    average_pooling2d::backward::Result *result)
{
    SharedPtr<Tensor> inputTensor = input->get(layers::backward::inputGradient);
    SharedPtr<Tensor> gradTensor = result->get(layers::backward::gradient);

    const Collection<size_t> &inputDims = inputTensor->getDimensions();
    const Collection<size_t> &gradDims = gradTensor->getDimensions();
    SubtensorDescriptor<algorithmFPType> inputBlock, maskBlock, gradBlock;

    inputTensor->getSubtensor(0, 0, 0, inputDims[0], readOnly, inputBlock);
    gradTensor->getSubtensor(0, 0, 0, gradDims[0], writeOnly, gradBlock);

    algorithmFPType *inputGrad = inputBlock.getPtr();
    algorithmFPType *grad = gradBlock.getPtr();

    algorithmFPType zero = 0.0;
    daal::services::internal::service_memset<algorithmFPType, cpu>(grad, zero, gradBlock.getSize());

    pooling2d::internal::Parameter par(parameter->indices.size, parameter->paddings   .size,
                                            parameter->strides.size, parameter->kernelSizes.size,
                                            gradTensor.get(), gradDims, inputDims);

    const algorithmFPType one = 1.0;
    algorithmFPType gradMultiplier = one / (algorithmFPType)(par.firstKernelSize * par.secondKernelSize);
    for (MKL_INT i = 0; i < par.offsetBefore; i++)
    {
        /*
         * Loop by the first kernel dimension
         * f - index of the left upper corner of the kernel
         * fo - index of the output value
         */
        for (MKL_INT f = -par.firstPadding, fo = 0; fo < par.firstOutSize; f += par.firstStride, fo++)
        {
            for (MKL_INT k = 0; k < par.offsetBetween; k++)
            {
                /*
                 * Loop by the second kernel dimension
                 * s - index of the left upper corner of the kernel
                 * so - index of the output value
                 */
                for (MKL_INT s = -par.secondPadding, so = 0; so < par.secondOutSize; s += par.secondStride, so++)
                {
                    for (MKL_INT j = 0; j < par.offsetAfter; j++)
                    {
                        /*
                         * Input value index
                         */
                        MKL_INT inputIndex = j + par.offsetAfter * (so + par.secondOutSize * (k + par.offsetBetween * (fo + par.firstOutSize * i)));
                        algorithmFPType inputValue = gradMultiplier * inputGrad[inputIndex];

                        /*
                         * Loops over the kernel
                         */
                        for (MKL_INT fi = f; fi < f + par.firstKernelSize; fi++)
                        {
                            for (MKL_INT si = s; si < s + par.secondKernelSize; si++)
                            {
                                MKL_INT gradIndex = j + par.offsetAfter * (si + par.secondSize * (k + par.offsetBetween * (fi + par.firstSize * i)));
                                bool paddingFlag = ((fi < 0) || (fi >= par.firstSize) || (si < 0) || (si >= par.secondSize));

                                if (!paddingFlag)
                                {
                                    grad[gradIndex] += inputValue;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    inputTensor->releaseSubtensor(inputBlock);
    gradTensor->releaseSubtensor(gradBlock);
}
} // namespace internal
} // namespace backward
} // namespace average_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
