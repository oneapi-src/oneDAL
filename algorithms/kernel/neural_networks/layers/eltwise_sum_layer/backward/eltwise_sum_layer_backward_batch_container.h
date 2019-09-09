/* file: eltwise_sum_layer_backward_batch_container.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of element-wise sum calculation algorithm container.
//--
*/

#ifndef __ELTWISE_SUM_LAYER_BACKWARD_BATCH_CONTAINER_H__
#define __ELTWISE_SUM_LAYER_BACKWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/eltwise_sum/eltwise_sum_layer.h"
#include "eltwise_sum_layer_backward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace eltwise_sum
{
namespace backward
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::EltwiseSumKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    eltwise_sum::backward::Input *input = static_cast<eltwise_sum::backward::Input *>(_in);
    eltwise_sum::backward::Result *result = static_cast<eltwise_sum::backward::Result *>(_res);

    const size_t nOutputs = input->getNumberOfCoefficients();

    TArray<Tensor *, cpu> outputTensorsBlock(nOutputs);
    Tensor **outputTensors = outputTensorsBlock.get();
    if (!outputTensors) { return services::Status(services::ErrorMemoryAllocationFailed); }

    for (size_t i = 0; i < nOutputs; i++)
    {
        outputTensors[i] = result->get(layers::backward::resultLayerData, i).get();
    }

    Tensor *inputGradient = input->get(layers::backward::inputGradient).get();
    Tensor *coefficients  = input->get(eltwise_sum::auxCoefficients).get();

    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::EltwiseSumKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
        inputGradient, coefficients, outputTensors, nOutputs);
}
} // namespace interface1
} // namespace backward
} // namespace eltwise_sum
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
