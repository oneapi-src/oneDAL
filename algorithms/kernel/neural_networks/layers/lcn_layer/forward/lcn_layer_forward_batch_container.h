/* file: lcn_layer_forward_batch_container.h */
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
//  Implementation of local contrast normalization algorithm container.
//--
*/

#ifndef __LCN_LAYER_FORWARD_BATCH_CONTAINER_H__
#define __LCN_LAYER_FORWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/lcn/lcn_layer.h"
#include "lcn_layer_forward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace lcn
{
namespace forward
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::LCNKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
void BatchContainer<algorithmFPType, method, cpu>::compute()
{
    lcn::forward::Input *input = static_cast<lcn::forward::Input *>(_in);
    lcn::forward::Result *result = static_cast<lcn::forward::Result *>(_res);

    lcn::Parameter *parameter = static_cast<lcn::Parameter *>(_par);;
    daal::services::Environment::env &env = *_env;

    Tensor *inputTensor        = input->get(layers::forward::data).get();
    Tensor *resultTensor       = result->get(layers::forward::value).get();
    Tensor *centeredDataTensor = result->get(lcn::auxCenteredData).get();
    Tensor *sigmaTensor        = result->get(lcn::auxSigma).get();
    Tensor *cTensor            = result->get(lcn::auxC).get();
    Tensor *invMaxTensor       = result->get(lcn::auxInvMax).get();
    Tensor *kernelTensor       = parameter->kernel.get();

    internal::LCNTask<algorithmFPType, method, cpu> task(inputTensor, resultTensor, centeredDataTensor, sigmaTensor, cTensor, invMaxTensor, parameter, kernelTensor);

    __DAAL_CALL_KERNEL(env, internal::LCNKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, task, parameter);
}

} // namespace interface1
} // namespace forward

} // namespace lcn
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
