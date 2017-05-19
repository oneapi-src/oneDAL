/* file: pooling2d_layer_forward_impl.i */
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
//  Common methods for implementation of forward 2D pooling layers
//--
*/

#ifndef __POOLING2D_LAYER_FORWARD_IMPL_I__
#define __POOLING2D_LAYER_FORWARD_IMPL_I__

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
namespace forward
{
namespace internal
{
/**
 *  \brief Kernel for forward 2D pooling layer results computation
 */
template<typename algorithmFPType, CpuType cpu>
class PoolingKernel : public Kernel
{
protected:
    void defaultCompute(pooling2d::internal::Parameter &parameter,
                const algorithmFPType *data, algorithmFPType *value, int *selectedPos);

    virtual void defaultInnerLoop(pooling2d::internal::Parameter &par,
                DAAL_INT i, DAAL_INT f, DAAL_INT k, DAAL_INT s, DAAL_INT j,
                const algorithmFPType *data, algorithmFPType *valuePtr, int *selectedPosPtr) = 0;

    void defaultCompute(pooling2d::internal::Parameter &parameter,
                const algorithmFPType *data, algorithmFPType *value);

    virtual void defaultInnerLoop(pooling2d::internal::Parameter &par,
                DAAL_INT i, DAAL_INT f, DAAL_INT k, DAAL_INT s, DAAL_INT j,
                const algorithmFPType *data, algorithmFPType *valuePtr) = 0;
};

template<typename algorithmFPType, CpuType cpu>
void PoolingKernel<algorithmFPType, cpu>::defaultCompute(
            pooling2d::internal::Parameter &par,
            const algorithmFPType *data, algorithmFPType *value, int *selectedPos)
{
    const algorithmFPType zero = 0.0;
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
                    DAAL_INT valueIndex = par.offsetAfter * (so + par.secondOutSize * (k + par.offsetBetween * (fo + par.firstOutSize * i)));
                    algorithmFPType *valuePtr = value + valueIndex;
                    int *selectedPosPtr = selectedPos + valueIndex;
                    for (DAAL_INT j = 0; j < par.offsetAfter; j++)
                    {
                        defaultInnerLoop(par, i, f, k, s, j, data, valuePtr, selectedPosPtr);
                    }
                }
            }
        }
    } );
}

template<typename algorithmFPType, CpuType cpu>
void PoolingKernel<algorithmFPType, cpu>::defaultCompute(
            pooling2d::internal::Parameter &par,
            const algorithmFPType *data, algorithmFPType *value)
{
    const algorithmFPType zero = 0.0;
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
                    DAAL_INT valueIndex = par.offsetAfter * (so + par.secondOutSize * (k + par.offsetBetween * (fo + par.firstOutSize * i)));
                    algorithmFPType *valuePtr = value + valueIndex;
                    for (DAAL_INT j = 0; j < par.offsetAfter; j++)
                    {
                        defaultInnerLoop(par, i, f, k, s, j, data, valuePtr);
                    }
                }
            }
        }
    } );
}

} // namespace internal
} // namespace forward
} // namespace pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
