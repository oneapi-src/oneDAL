/* file: logistic_layer_backward_batch_container.h */
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
//  Implementation of logistic function calculation algorithm container.
//--
*/

#ifndef __LOGISTIC_LAYER_BACKWARD_BATCH_CONTAINER_H__
#define __LOGISTIC_LAYER_BACKWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/logistic/logistic_layer.h"
#include "logistic_layer_backward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace logistic
{
namespace backward
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::LogisticKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    logistic::backward::Input *input   = static_cast<logistic::backward::Input *>(_in);
    logistic::backward::Result *result = static_cast<logistic::backward::Result *>(_res);

    const layers::Parameter *parameter = static_cast<const Parameter *>(_par);
    if (!parameter->propagateGradient) { return services::Status(); }

    daal::services::Environment::env &env = *_env;

    Tensor *inputTensor         = input->get(layers::backward::inputGradient).get();
    Tensor *resultTensor        = result->get(layers::backward::gradient).get();
    Tensor *forwardOutputTensor = input->get(logistic::auxValue).get();

    __DAAL_CALL_KERNEL(env, internal::LogisticKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, inputTensor,
                       resultTensor, forwardOutputTensor);
}
} // namespace interface1
} // namespace backward

} // namespace logistic
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
