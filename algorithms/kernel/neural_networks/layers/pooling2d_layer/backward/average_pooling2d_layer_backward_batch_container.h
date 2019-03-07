/* file: average_pooling2d_layer_backward_batch_container.h */
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

#ifndef __AVERAGE_POOLING2D_LAYER_BACKWARD_BATCH_CONTAINER_H__
#define __AVERAGE_POOLING2D_LAYER_BACKWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/pooling2d/average_pooling2d_layer.h"
#include "average_pooling2d_layer_backward_kernel.h"

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
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::PoolingKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::setupCompute()
{
    average_pooling2d::backward::Input *input = static_cast<average_pooling2d::backward::Input *>(_in);
    average_pooling2d::backward::Result *result = static_cast<average_pooling2d::backward::Result *>(_res);

    daal::services::Environment::env &env = *_env;

    const services::Collection<size_t>& inDimsFull  = input->get(layers::backward::inputGradient)->getDimensions();
    const services::Collection<size_t>& outDimsFull = result->get(layers::backward::gradient)->getDimensions();

    __DAAL_CALL_KERNEL(env, internal::PoolingKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), initialize, inDimsFull, outDimsFull);
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    average_pooling2d::backward::Input *input = static_cast<average_pooling2d::backward::Input *>(_in);
    average_pooling2d::backward::Result *result = static_cast<average_pooling2d::backward::Result *>(_res);

    average_pooling2d::Parameter *parameter = static_cast<average_pooling2d::Parameter *>(_par);
    if (!parameter->propagateGradient) { return services::Status(); }

    daal::services::Environment::env &env = *_env;

    Tensor *dataTensor = input->get(auxData).get();
    Tensor *inputGradTensor = input->get(layers::backward::inputGradient).get();
    Tensor *gradTensor  = result->get(layers::backward::gradient).get();

    __DAAL_CALL_KERNEL(env, internal::PoolingKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),   \
                       compute, *inputGradTensor, *parameter, *gradTensor, dataTensor);
}
} // namespace interface1
} // namespace backward

} // namespace average_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
