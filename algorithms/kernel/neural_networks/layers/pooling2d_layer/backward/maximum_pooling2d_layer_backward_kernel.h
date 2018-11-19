/* file: maximum_pooling2d_layer_backward_kernel.h */
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
//  Declaration of template function that calculate backward pooling layer relults.
//--


#ifndef __MAXIMUM_POOLING2D_LAYER_BACKWARD_KERNEL_H__
#define __MAXIMUM_POOLING2D_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/pooling2d/maximum_pooling2d_layer_backward.h"
#include "neural_networks/layers/pooling2d/maximum_pooling2d_layer_backward_types.h"
#include "pooling2d_layer_internal_parameter.h"
#include "tensor.h"
#include "pooling2d_layer_backward_impl.i"
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
namespace backward
{
namespace internal
{

/**
 *  \brief Kernel for backward pooling layer results computation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class PoolingKernel : public pooling2d::backward::internal::PoolingKernel<algorithmFPType, cpu>
{
public:
    services::Status compute(const Tensor &inputGradTensor,
                             const Tensor &selectedPosTensor, Tensor &gradTensor, const Tensor *dataTensor,
                             const pooling2d::Parameter &parameter);

    services::Status initialize(const services::Collection<size_t> &inDimsFull,
                                const services::Collection<size_t> &outDimsFull);

    ~PoolingKernel()
    {
        if (maxPoolPrim)
        {
            dnn::xDelete(maxPoolPrim);
        }
    }
protected:
    using pooling2d::backward::internal::PoolingKernel<algorithmFPType, cpu>::defaultCompute;

    virtual void defaultInnerLoop(const pooling2d::internal::Parameter &par,
                                  DAAL_INT i, DAAL_INT f, DAAL_INT k, DAAL_INT s,
                                  const algorithmFPType *inputGradPtr, const int *selectedPosPtr,
                                  algorithmFPType *grad);

    void indicesLastZeroPaddingsCompute(const pooling2d::internal::Parameter &parameter,
                                        const algorithmFPType *inputGrad, const int *selectedPos,
                                        algorithmFPType *grad);

    void indicesFirstZeroPaddingsCompute(const pooling2d::internal::Parameter &parameter,
                                         const algorithmFPType *inputGrad, const int *selectedPos,
                                         algorithmFPType *grad);

private:
    typedef daal::internal::Dnn<algorithmFPType, cpu> dnn;
    typedef daal::internal::DnnLayout<algorithmFPType, cpu> xDnnLayout;

    dnnPrimitive_t maxPoolPrim = NULL;

    size_t *inputSize     = NULL;
    TArray<size_t, cpu> inputSizePtr;

    size_t *inputStrides  = NULL;
    TArray<size_t, cpu> inputStridesPtr;

    size_t *outputSize    = NULL;
    TArray<size_t, cpu> outputSizePtr;

    size_t *outputStrides = NULL;
    TArray<size_t, cpu> outputStridesPtr;

    xDnnLayout ltUserInput;
    xDnnLayout ltUserOutput;
};

} // internal
} // backward
} // maximum_pooling2d
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
