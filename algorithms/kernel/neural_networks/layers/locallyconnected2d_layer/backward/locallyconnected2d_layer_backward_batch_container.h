/* file: locallyconnected2d_layer_backward_batch_container.h */
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
//  Implementation of locallyconnected2d calculation algorithm container.
//--
*/

#ifndef __LOCALLYCONNECTED2D_LAYER_BACKWARD_BATCH_CONTAINER_H__
#define __LOCALLYCONNECTED2D_LAYER_BACKWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/locallyconnected2d/locallyconnected2d_layer.h"
#include "locallyconnected2d_layer_backward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace locallyconnected2d
{
namespace backward
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::LocallyConnected2dKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
void BatchContainer<algorithmFPType, method, cpu>::compute()
{
    locallyconnected2d::backward::Input *input = static_cast<locallyconnected2d::backward::Input *>(_in);
    locallyconnected2d::backward::Result *result = static_cast<locallyconnected2d::backward::Result *>(_res);

    locallyconnected2d::Parameter *parameter = static_cast<locallyconnected2d::Parameter *>(_par);;
    daal::services::Environment::env &env = *_env;

    LayerData *layerData     = input->get(layers::backward::inputFromForward).get();
    Tensor *inGradTensor     = input->get(layers::backward::inputGradient).get();
    Tensor *gradientTensor   = result->get(layers::backward::gradient).get();
    Tensor *wDerTensor       = result->get(layers::backward::weightDerivatives).get();
    Tensor *bDerTensor       = result->get(layers::backward::biasDerivatives).get();
    Tensor *auxDataTensor    = staticPointerCast<Tensor, SerializationIface>((*layerData)[locallyconnected2d::auxData]).get();
    Tensor *auxWeightsTensor = staticPointerCast<Tensor, SerializationIface>((*layerData)[locallyconnected2d::auxWeights]).get();

    __DAAL_CALL_KERNEL(env, internal::LocallyConnected2dKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, inGradTensor, gradientTensor,
                                auxDataTensor, auxWeightsTensor, wDerTensor, bDerTensor, parameter);
}
} // namespace interface1
} // namespace backward

} // namespace locallyconnected2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
