/* file: spatial_stochastic_pooling2d_layer_backward_batch_container.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of backward pooling layer container.
//--
*/

#ifndef __SPATIAL_PYRAMID_STOCHASTIC_POOLING2D_LAYER_BACKWARD_BATCH_CONTAINER_H__
#define __SPATIAL_PYRAMID_STOCHASTIC_POOLING2D_LAYER_BACKWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/spatial_pooling2d/spatial_stochastic_pooling2d_layer.h"
#include "spatial_pooling2d_layer_backward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace spatial_stochastic_pooling2d
{
namespace backward
{
namespace interface1
{
template<typename algorithmFPType, CpuType cpu>
BatchContainer<algorithmFPType, defaultDense, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(spatial_pooling2d::backward::internal::PoolingKernel, algorithmFPType, spatial_pooling2d::internal::stochastic);
}

template<typename algorithmFPType, CpuType cpu>
BatchContainer<algorithmFPType, defaultDense, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, CpuType cpu>
void BatchContainer<algorithmFPType, defaultDense, cpu>::compute()
{
    spatial_stochastic_pooling2d::backward::Input *input = static_cast<spatial_stochastic_pooling2d::backward::Input *>(_in);
    spatial_stochastic_pooling2d::backward::Result *result = static_cast<spatial_stochastic_pooling2d::backward::Result *>(_res);

    Tensor *inputGradTensor = input->get(layers::backward::inputGradient).get();
    Tensor *selectedPosTensor = input->get(auxSelectedIndices).get();
    NumericTable *inputDimensions = input->get(auxInputDimensions).get();
    Tensor *gradTensor = result->get(layers::backward::gradient).get();

    spatial_stochastic_pooling2d::Parameter *parameter = static_cast<spatial_stochastic_pooling2d::Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, spatial_pooling2d::backward::internal::PoolingKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, spatial_pooling2d::internal::stochastic),   \
                       compute, inputGradTensor, gradTensor, inputDimensions, selectedPosTensor, parameter);
}
} // namespace interface1
} // namespace backward

} // namespace spatial_stochastic_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
