/* file: prelu_layer_backward_batch_container.h */
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
//  Implementation of prelu calculation functions.
//--
*/

#ifndef __PRELU_LAYER_BACKWARD_BATCH_CONTAINER_H__
#define __PRELU_LAYER_BACKWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/prelu/prelu_layer.h"
#include "prelu_layer_backward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace prelu
{
namespace backward
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::PReLUKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
void BatchContainer<algorithmFPType, method, cpu>::compute()
{
    prelu::backward::Input *input = static_cast<prelu::backward::Input *>(_in);
    prelu::backward::Result *result = static_cast<prelu::backward::Result *>(_res);

    prelu::Parameter *parameter = static_cast<prelu::Parameter *>(_par);;
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::PReLUKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, input, parameter,
                       result);
}
} // namespace interface1
} // namespace backward

} // namespace prelu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
