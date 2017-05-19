/* file: fullyconnected_layer_backward_batch_container.h */
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
//  Implementation of fullyconnected calculation algorithm container.
//--
*/

#ifndef __FULLYCONNECTED_LAYER_BACKWARD_BATCH_CONTAINER_H__
#define __FULLYCONNECTED_LAYER_BACKWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/fullyconnected/fullyconnected_layer.h"
#include "fullyconnected_layer_backward_kernel.h"

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
namespace backward
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
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    fullyconnected::backward::Input *input = static_cast<fullyconnected::backward::Input *>(_in);
    fullyconnected::backward::Result *result = static_cast<fullyconnected::backward::Result *>(_res);

    fullyconnected::Parameter *parameter = static_cast<fullyconnected::Parameter *>(_par);;
    daal::services::Environment::env &env = *_env;

    Tensor *inGradTensor  = input->get(layers::backward::inputGradient).get();
    Tensor *xTensor       = input->get(fullyconnected::auxData).get();
    Tensor *wTensor       = input->get(fullyconnected::auxWeights).get();
    Tensor *wDerTensor    = result->get(layers::backward::weightDerivatives).get();
    Tensor *bDerTensor    = result->get(layers::backward::biasDerivatives).get();
    Tensor *resultTensor  = result->get(layers::backward::gradient).get();

    __DAAL_CALL_KERNEL(env, internal::FullyconnectedKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, inGradTensor, xTensor,
                                                                                    wTensor, wDerTensor, bDerTensor, resultTensor , parameter);
}
} // namespace interface1
} // namespace backward
} // namespace fullyconnected
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
