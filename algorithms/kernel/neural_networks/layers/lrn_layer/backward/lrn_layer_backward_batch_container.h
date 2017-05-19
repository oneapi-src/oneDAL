/* file: lrn_layer_backward_batch_container.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of the backward local response normalization layer.
//--
*/

#ifndef __LRN_LAYER_BACKWARD_BATCH_CONTAINER_H__
#define __LRN_LAYER_BACKWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/lrn/lrn_layer.h"
#include "lrn_layer_backward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace lrn
{
namespace backward
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::LRNKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    lrn::backward::Input *input = static_cast<lrn::backward::Input *>(_in);
    lrn::backward::Result *result = static_cast<lrn::backward::Result *>(_res);

    lrn::Parameter *parameter = static_cast<lrn::Parameter *>(_par);
    if (!parameter->propagateGradient) { return services::Status(); }

    daal::services::Environment::env &env = *_env;

    Tensor *dataTensor          = input->get(layers::lrn::auxData).get();
    Tensor *sMinusBetaTensor    = input->get(lrn::auxSmBeta).get();
    Tensor *inputGradientTensor = input->get(layers::backward::inputGradient).get();
    Tensor *gradientTensor      = result->get(layers::backward::gradient).get();

    __DAAL_CALL_KERNEL(env, internal::LRNKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, dataTensor, sMinusBetaTensor, inputGradientTensor,
                                                                         gradientTensor, parameter);
}
} // namespace interface1
} // namespace backward

} // namespace lrn
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
