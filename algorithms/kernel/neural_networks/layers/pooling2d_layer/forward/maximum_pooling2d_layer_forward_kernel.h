/* file: maximum_pooling2d_layer_forward_kernel.h */
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

#ifndef __MAXIMUM_POOLING2D_LAYER_FORWARD_KERNEL_H__
#define __MAXIMUM_POOLING2D_LAYER_FORWARD_KERNEL_H__

#include "neural_networks/layers/pooling2d/maximum_pooling2d_layer_forward.h"
#include "neural_networks/layers/pooling2d/maximum_pooling2d_layer_forward_types.h"
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
namespace maximum_pooling2d
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
    /* Computes the results of forward batch normalization layer */
    services::Status compute(const Tensor &dataTensor, Tensor &valueTensor,
                 Tensor *selectedPosTensor, const maximum_pooling2d::Parameter &parameter);

    services::Status initialize(const services::Collection<size_t>& inDimsFull,
                    const services::Collection<size_t>& outDimsFull);

    ~PoolingKernel()
    {
        if (maxPoolPrim)
        {
            dnn::xDelete(maxPoolPrim);
        }
        if (outputSize)
        {
            delete [] outputSize;
        }
        if (outputStrides)
        {
            delete [] outputStrides;
        }
    }
protected:
    /* Training */
    using pooling2d::forward::internal::PoolingKernel<algorithmFPType, cpu>::defaultCompute;

    virtual void defaultInnerLoop(const pooling2d::internal::Parameter &par,
                DAAL_INT i, DAAL_INT f, DAAL_INT k, DAAL_INT s, DAAL_INT j,
                const algorithmFPType *data, algorithmFPType *valuePtr, int *selectedPosPtr);

    void indicesLastZeroPaddingsCompute(const pooling2d::internal::Parameter &parameter,
                const algorithmFPType *data, algorithmFPType *value,
                int *selectedPos);

    void indicesFirstZeroPaddingsCompute(const pooling2d::internal::Parameter &parameter,
                const algorithmFPType *data, algorithmFPType *value,
                int *selectedPos);

    /* Prediction */
    virtual void defaultInnerLoop(const pooling2d::internal::Parameter &par,
                DAAL_INT i, DAAL_INT f, DAAL_INT k, DAAL_INT s, DAAL_INT j,
                const algorithmFPType *data, algorithmFPType *valuePtr);

    void indicesLastZeroPaddingsCompute(const pooling2d::internal::Parameter &parameter,
                const algorithmFPType *data, algorithmFPType *value);

    void indicesFirstZeroPaddingsCompute(const pooling2d::internal::Parameter &parameter,
                const algorithmFPType *data, algorithmFPType *value);

private:
    typedef daal::internal::Dnn<algorithmFPType, cpu> dnn;
    typedef daal::internal::DnnLayout<algorithmFPType, cpu> xDnnLayout;

    dnnPrimitive_t maxPoolPrim = NULL;

    size_t *outputSize = NULL;
    size_t *outputStrides = NULL;

    xDnnLayout ltUserOutput;
};

} // internal
} // forward
} // maximum_pooling2d
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
