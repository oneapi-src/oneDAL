/* file: average_pooling2d_layer_forward_batch_container.h */
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
//  Implementation of forward pooling layer container.
//--
*/

#ifndef __AVERAGE_POOLING2D_LAYER_FORWARD_BATCH_CONTAINER_H__
#define __AVERAGE_POOLING2D_LAYER_FORWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/pooling2d/average_pooling2d_layer_forward.h"
#include "average_pooling2d_layer_forward_kernel.h"

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
    average_pooling2d::forward::Input *input = static_cast<average_pooling2d::forward::Input *>(_in);
    average_pooling2d::forward::Result *result = static_cast<average_pooling2d::forward::Result *>(_res);

    daal::services::Environment::env &env = *_env;

    const services::Collection<size_t>& inDimsFull  = input->get(layers::forward::data)->getDimensions();
    const services::Collection<size_t>& outDimsFull = result->get(layers::forward::value)->getDimensions();

    __DAAL_CALL_KERNEL(env, internal::PoolingKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), initialize, inDimsFull, outDimsFull);
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    average_pooling2d::forward::Input *input = static_cast<average_pooling2d::forward::Input *>(_in);
    average_pooling2d::forward::Result *result = static_cast<average_pooling2d::forward::Result *>(_res);

    average_pooling2d::Parameter *parameter = static_cast<average_pooling2d::Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    Tensor *dataTensor  = input->get(layers::forward::data).get();
    Tensor *valueTensor = result->get(layers::forward::value).get();

    __DAAL_CALL_KERNEL(env, internal::PoolingKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, *dataTensor, *parameter,
                       *valueTensor);
}
} // namespace interface1
} // namespace forward
} // namespace average_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
