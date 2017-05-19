/* file: transposed_conv2d_layer_backward_batch_container.h */
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
//  Implementation of transposed convolution 2d calculation algorithm container.
//--
*/

#ifndef __TRANSPOSED_CONV2D_LAYER_BACKWARD_BATCH_CONTAINER_H__
#define __TRANSPOSED_CONV2D_LAYER_BACKWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/transposed_conv2d/transposed_conv2d_layer.h"
#include "transposed_conv2d_layer_backward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace transposed_conv2d
{
namespace backward
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::TransposedConv2dKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    transposed_conv2d::backward::Input *input = static_cast<transposed_conv2d::backward::Input *>(_in);
    transposed_conv2d::backward::Result *result = static_cast<transposed_conv2d::backward::Result *>(_res);

    transposed_conv2d::Parameter *parameter = static_cast<transposed_conv2d::Parameter *>(_par);;
    daal::services::Environment::env &env = *_env;

    Tensor *inGradTensor  = input->get(layers::backward::inputGradient).get();
    LayerData* layerData  = input->get(layers::backward::inputFromForward).get();
    Tensor *wDerTensor    = result->get(layers::backward::weightDerivatives).get();
    Tensor *bDerTensor    = result->get(layers::backward::biasDerivatives).get();
    Tensor *resultTensor  = result->get(layers::backward::gradient).get();

    Tensor *xTensor = static_cast<Tensor *>(((*layerData)[transposed_conv2d::auxData]).get());
    Tensor *wTensor = static_cast<Tensor *>(((*layerData)[transposed_conv2d::auxWeights]).get());

    __DAAL_CALL_KERNEL(env, internal::TransposedConv2dKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, inGradTensor, xTensor, wTensor,
                                                                                   parameter, wDerTensor, bDerTensor, resultTensor);
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::setupCompute()
{
    return services::Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::resetCompute()
{
    return services::Status();
}

} // namespace interface1
} // namespace backward

} // namespace transposed_conv2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
