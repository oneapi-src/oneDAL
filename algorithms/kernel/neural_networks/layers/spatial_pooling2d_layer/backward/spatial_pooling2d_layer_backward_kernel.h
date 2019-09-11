/* file: spatial_pooling2d_layer_backward_kernel.h */
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

//++
//  Declaration of template function that calculate backward pooling layer relults.
//--


#ifndef __SPATIAL_POOLING2D_LAYER_BACKWARD_KERNEL_H__
#define __SPATIAL_POOLING2D_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/spatial_pooling2d/spatial_pooling2d_layer_backward_types.h"
#include "spatial_pooling2d_layer_internal_types.h"
#include "kernel.h"
#include "tensor.h"
#include "maximum_pooling2d_layer_backward.h"
#include "neural_networks/layers/spatial_pooling2d/spatial_stochastic_pooling2d_layer.h"
#include "neural_networks/layers/spatial_pooling2d/spatial_maximum_pooling2d_layer.h"
#include "neural_networks/layers/spatial_pooling2d/spatial_average_pooling2d_layer.h"

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
namespace spatial_pooling2d
{
namespace backward
{
namespace internal
{

using namespace spatial_pooling2d::internal;

/**
 *  \brief Kernel for forward pooling layer results computation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT PoolingKernel : public Kernel
{};

template<typename algorithmFPType, CpuType cpu>
class DAAL_EXPORT PoolingKernel<algorithmFPType, maximum, cpu> : public Kernel
{
public:
    virtual services::Status compute(const Tensor &inputGradientTensor,
                                           Tensor &gradientTensor,
                                     const Tensor &selectedPosTensor,
                                     const spatial_maximum_pooling2d::Parameter &parameter);
};

template<typename algorithmFPType, CpuType cpu>
class DAAL_EXPORT PoolingKernel<algorithmFPType, stochastic, cpu> : public Kernel
{
public:
    virtual services::Status compute(const Tensor &inputGradientTensor,
                                           Tensor &gradientTensor,
                                     const Tensor &selectedPosTensor,
                                     const spatial_stochastic_pooling2d::Parameter &parameter);
};

template<typename algorithmFPType, CpuType cpu>
class DAAL_EXPORT PoolingKernel<algorithmFPType, average, cpu> : public Kernel
{
public:
    virtual services::Status compute(const Tensor &inputGradientTensor,
                                           Tensor &gradientTensor,
                                     const spatial_average_pooling2d::Parameter &parameter);
};

} // internal
} // backward
} // spatial_pooling2d
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
