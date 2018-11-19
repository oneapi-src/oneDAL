/* file: average_pooling2d_layer_forward_kernel.h */
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

//++
//  Declaration of template function that calculate forward pooling layer results.
//--

#ifndef __AVERAGE_POOLING2D_LAYER_FORWARD_KERNEL_H__
#define __AVERAGE_POOLING2D_LAYER_FORWARD_KERNEL_H__

#include "neural_networks/layers/pooling2d/average_pooling2d_layer_forward.h"
#include "neural_networks/layers/pooling2d/average_pooling2d_layer_forward_types.h"
#include "pooling2d_layer_internal_parameter.h"
#include "tensor.h"
#include "pooling2d_layer_forward_impl.i"
#include "service_dnn.h"
#include "service_dnn_internal.h"
#include "layers_threading.h"

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
namespace average_pooling2d
{
namespace forward
{
namespace internal
{

/**
 *  \brief Kernel for forward pooling layer results computation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class PoolingKernel : public pooling2d::forward::internal::PoolingKernel<algorithmFPType, cpu>
{
public:
    /* Computes the results of forward pooling layer */
    services::Status compute(const Tensor &dataTensor, const average_pooling2d::Parameter &parameter, Tensor &valueTensor);

    services::Status initialize(const services::Collection<size_t> &inDimsFull,
                                const services::Collection<size_t> &outDimsFull);

    ~PoolingKernel()
    {
        if (avePoolPrim)
        {
            dnn::xDelete(avePoolPrim);
        }
    }
protected:
    typedef daal::internal::Dnn<algorithmFPType, cpu> dnn;
    typedef daal::internal::DnnLayout<algorithmFPType, cpu> xDnnLayout;

    using pooling2d::forward::internal::PoolingKernel<algorithmFPType, cpu>::defaultCompute;

    virtual void defaultInnerLoop(const pooling2d::internal::Parameter &par,
                                  DAAL_INT i, DAAL_INT f, DAAL_INT k, DAAL_INT s, DAAL_INT j,
                                  const algorithmFPType *data, algorithmFPType *valuePtr);

    virtual void defaultInnerLoop(const pooling2d::internal::Parameter &par,
                                  DAAL_INT i, DAAL_INT f, DAAL_INT k, DAAL_INT s, DAAL_INT j,
                                  const algorithmFPType *data, algorithmFPType *valuePtr, int *selectedPos)
    {
        defaultInnerLoop(par, i, f, k, s, j, data, valuePtr);
    }

    dnnPrimitive_t avePoolPrim = NULL;

    size_t *outputSize = NULL;
    TArray<size_t, cpu> outputSizePtr;

    size_t *outputStrides = NULL;
    TArray<size_t, cpu> outputStridesPtr;

    xDnnLayout ltUserOutput;
};

} // internal
} // forward
} // average_pooling2d
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
