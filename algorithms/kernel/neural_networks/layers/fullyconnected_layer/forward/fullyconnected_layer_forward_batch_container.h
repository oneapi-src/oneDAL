/* file: fullyconnected_layer_forward_batch_container.h */
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
//  Implementation of fullyconnected calculation algorithm container.
//--
*/

#ifndef __FULLYCONNECTED_LAYER_FORWARD_BATCH_CONTAINER_H__
#define __FULLYCONNECTED_LAYER_FORWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/fullyconnected/fullyconnected_layer.h"
#include "fullyconnected_layer_forward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace fullyconnected
{
namespace forward
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::FullyconnectedKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
void BatchContainer<algorithmFPType, method, cpu>::compute()
{
    fullyconnected::forward::Input *input = static_cast<fullyconnected::forward::Input *>(_in);
    fullyconnected::forward::Result *result = static_cast<fullyconnected::forward::Result *>(_res);

    fullyconnected::Parameter *parameter = static_cast<fullyconnected::Parameter *>(_par);;
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::FullyconnectedKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, input, parameter, result);
}
} // namespace interface1
} // namespace forward

} // namespace fullyconnected
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
