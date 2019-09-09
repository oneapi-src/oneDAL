/* file: elu_layer_forward_batch_container.h */
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
//  Implementation of elu calculation algorithm container.
//--
*/

#ifndef __ELU_LAYER_FORWARD_BATCH_CONTAINER_H__
#define __ELU_LAYER_FORWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/elu/elu_layer.h"
#include "elu_layer_forward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace elu
{
namespace forward
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::ELUKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    elu::Parameter *parameter = static_cast<elu::Parameter *>(_par);
    elu::forward::Input *input = static_cast<elu::forward::Input *>(_in);
    elu::forward::Result *result = static_cast<elu::forward::Result *>(_res);

    daal::services::Environment::env &env = *_env;

    Tensor *dataTensor                 = input->get(layers::forward::data).get();
    Tensor *valueTensor                = result->get(layers::forward::value).get();
    Tensor *auxIntermediateValueTensor = result->get(layers::elu::auxIntermediateValue).get();

    __DAAL_CALL_KERNEL(env, internal::ELUKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
        *parameter, *dataTensor, *valueTensor, auxIntermediateValueTensor);
}
} // namespace interface1
} // namespace forward

} // namespace elu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
