/* file: stochastic_pooling2d_layer_backward_batch_container.h */
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

#ifndef __STOCHASTIC_POOLING2D_LAYER_BACKWARD_BATCH_CONTAINER_H__
#define __STOCHASTIC_POOLING2D_LAYER_BACKWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/pooling2d/stochastic_pooling2d_layer.h"
#include "maximum_pooling2d_layer_backward_kernel.h"

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
namespace backward
{
namespace interface1
{
template<typename algorithmFPType, CpuType cpu>
BatchContainer<algorithmFPType, defaultDense, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(maximum_pooling2d::backward::internal::PoolingKernel, algorithmFPType, maximum_pooling2d::defaultDense);
}

template<typename algorithmFPType, CpuType cpu>
BatchContainer<algorithmFPType, defaultDense, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, CpuType cpu>
services::Status BatchContainer<algorithmFPType, defaultDense, cpu>::compute()
{
    stochastic_pooling2d::backward::Input *input = static_cast<stochastic_pooling2d::backward::Input *>(_in);
    stochastic_pooling2d::backward::Result *result = static_cast<stochastic_pooling2d::backward::Result *>(_res);

    Tensor *inputGradTensor = input->get(layers::backward::inputGradient).get();
    Tensor *selectedPosTensor = input->get(auxSelectedIndices).get();
    Tensor *gradTensor = result->get(layers::backward::gradient).get();

    stochastic_pooling2d::Parameter *parameter = static_cast<stochastic_pooling2d::Parameter *>(_par);
    if (!parameter->propagateGradient) { return services::Status(); }

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, maximum_pooling2d::backward::internal::PoolingKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, maximum_pooling2d::defaultDense),   \
                       compute, *inputGradTensor, *selectedPosTensor, *gradTensor, NULL, *parameter);
}
} // namespace interface1
} // namespace backward

} // namespace stochastic_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
