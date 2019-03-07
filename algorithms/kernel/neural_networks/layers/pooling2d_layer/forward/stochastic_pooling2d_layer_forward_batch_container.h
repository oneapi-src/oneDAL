/* file: stochastic_pooling2d_layer_forward_batch_container.h */
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

#ifndef __STOCHASTIC_POOLING2D_LAYER_FORWARD_BATCH_CONTAINER_H__
#define __STOCHASTIC_POOLING2D_LAYER_FORWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/pooling2d/stochastic_pooling2d_layer_forward.h"
#include "stochastic_pooling2d_layer_forward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace stochastic_pooling2d
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
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    stochastic_pooling2d::forward::Input *input = static_cast<stochastic_pooling2d::forward::Input *>(_in);
    stochastic_pooling2d::forward::Result *result = static_cast<stochastic_pooling2d::forward::Result *>(_res);
    stochastic_pooling2d::Parameter *parameter = static_cast<stochastic_pooling2d::Parameter *>(_par);

    data_management::Tensor *dataTensor = input->get(layers::forward::data).get();
    data_management::Tensor *valueTensor = result->get(layers::forward::value).get();
    data_management::Tensor *selectedPosTensor = nullptr;
    if(parameter->predictionStage == false)
    {
        selectedPosTensor = result->get(auxSelectedIndices).get();
    }

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::PoolingKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, \
        *dataTensor, *valueTensor, selectedPosTensor, *parameter, *parameter->engine);
}
} // namespace interface1
} // namespace forward
} // namespace stochastic_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
