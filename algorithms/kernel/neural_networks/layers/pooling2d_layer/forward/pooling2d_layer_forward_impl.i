/* file: pooling2d_layer_forward_impl.i */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
    void defaultCompute(const pooling2d::internal::Parameter &parameter,
                const algorithmFPType *data, algorithmFPType *value, int *selectedPos);

    virtual void defaultInnerLoop(const pooling2d::internal::Parameter &par,
                DAAL_INT i, DAAL_INT f, DAAL_INT k, DAAL_INT s, DAAL_INT j,
                const algorithmFPType *data, algorithmFPType *valuePtr, int *selectedPosPtr) = 0;

    void defaultCompute(const pooling2d::internal::Parameter &parameter,
                const algorithmFPType *data, algorithmFPType *value);

    virtual void defaultInnerLoop(const pooling2d::internal::Parameter &par,
                DAAL_INT i, DAAL_INT f, DAAL_INT k, DAAL_INT s, DAAL_INT j,
                const algorithmFPType *data, algorithmFPType *valuePtr) = 0;
};

template<typename algorithmFPType, CpuType cpu>
void PoolingKernel<algorithmFPType, cpu>::defaultCompute(
            const pooling2d::internal::Parameter &par,
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
            const pooling2d::internal::Parameter &par,
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
