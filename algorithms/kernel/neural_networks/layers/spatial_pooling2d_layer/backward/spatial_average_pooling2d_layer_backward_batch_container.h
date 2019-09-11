/* file: spatial_average_pooling2d_layer_backward_batch_container.h */
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

/*
//++
//  Implementation of backward pooling layer container.
//--
*/

#ifndef __SPATIAL_PYRAMID_AVERAGE_POOLING2D_LAYER_BACKWARD_BATCH_CONTAINER_H__
#define __SPATIAL_PYRAMID_AVERAGE_POOLING2D_LAYER_BACKWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/spatial_pooling2d/spatial_average_pooling2d_layer.h"
#include "spatial_pooling2d_layer_backward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace spatial_average_pooling2d
{
namespace backward
{
namespace interface1
{
template<typename algorithmFPType, CpuType cpu>
BatchContainer<algorithmFPType, defaultDense, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(spatial_pooling2d::backward::internal::PoolingKernel, algorithmFPType, spatial_pooling2d::internal::average);
}

template<typename algorithmFPType, CpuType cpu>
BatchContainer<algorithmFPType, defaultDense, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, CpuType cpu>
services::Status BatchContainer<algorithmFPType, defaultDense, cpu>::compute()
{
    spatial_average_pooling2d::backward::Input *input = static_cast<spatial_average_pooling2d::backward::Input *>(_in);
    spatial_average_pooling2d::backward::Result *result = static_cast<spatial_average_pooling2d::backward::Result *>(_res);

    Tensor *inputGradTensor = input->get(layers::backward::inputGradient).get();
    Tensor *gradTensor = result->get(layers::backward::gradient).get();

    spatial_average_pooling2d::Parameter *parameter = static_cast<spatial_average_pooling2d::Parameter *>(_par);
    if (!parameter->propagateGradient) { return services::Status(); }

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, spatial_pooling2d::backward::internal::PoolingKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, spatial_pooling2d::internal::average),   \
                       compute, *inputGradTensor, *gradTensor, *parameter);
}
} // namespace interface1
} // namespace backward

} // namespace spatial_average_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
