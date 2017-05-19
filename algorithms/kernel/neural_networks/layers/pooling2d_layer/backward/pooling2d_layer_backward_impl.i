/* file: pooling2d_layer_backward_impl.i */
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
//  Common methods for implementation of backward 2D pooling layers
//--
*/

#ifndef __POOLING2D_LAYER_BACKWARD_IMPL_I__
#define __POOLING2D_LAYER_BACKWARD_IMPL_I__

#include "tensor.h"
#include "kernel.h"
#include "threading.h"
#include "pooling2d_layer_internal_parameter.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace pooling2d
{
namespace backward
{
namespace internal
{
/**
 *  \brief Kernel for backward 2D pooling layer results computation
 */
template<typename algorithmFPType, CpuType cpu>
class PoolingKernel : public Kernel
{
protected:
    void defaultCompute(pooling2d::internal::Parameter &parameter,
                const algorithmFPType *inputGrad, const int *selectedPos,
                algorithmFPType *grad);

    virtual void defaultInnerLoop(pooling2d::internal::Parameter &par,
                DAAL_INT i, DAAL_INT f, DAAL_INT k, DAAL_INT s,
                const algorithmFPType *inputGradPtr, const int *selectedPosPtr,
                algorithmFPType *grad) = 0;
};

template<typename algorithmFPType, CpuType cpu>
void PoolingKernel<algorithmFPType, cpu>::defaultCompute(
                pooling2d::internal::Parameter &par,
                const algorithmFPType *inputGrad, const int *selectedPos,
                algorithmFPType *grad)
{
    threader_for(par.offsetBefore, par.offsetBefore, [&](DAAL_INT i)
    {
        /*
         * Loop by the first kernel dimension
         * f - index of the left upper corner of the kernel
         * fo - index of the output value
         */
        for (DAAL_INT f = -par.firstPadding, fo = 0; fo < par.firstOutSize; f += par.firstStride, fo++)
        {
            for (DAAL_INT k = 0; k < par.offsetBetween; k++)
            {
                /*
                 * Loop by the second kernel dimension
                 * s - index of the left upper corner of the kernel
                 * so - index of the output value
                 */
                for (DAAL_INT s = -par.secondPadding, so = 0; so < par.secondOutSize; s += par.secondStride, so++)
                {
                    DAAL_INT inputIndex = par.offsetAfter * (so + par.secondOutSize * (k + par.offsetBetween * (fo + par.firstOutSize * i)));
                    const algorithmFPType *inputGradPtr = inputGrad + inputIndex;
                    const int *selectedPosPtr = selectedPos + inputIndex;
                    defaultInnerLoop(par, i, f, k, s, inputGradPtr, selectedPosPtr, grad);
                }
            }
        }
    } );
}

} // namespace internal
} // namespace backward
} // namespace pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
