/* file: convolution2d_layer_forward_batch_container.h */
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
//  Implementation of convolution2d calculation algorithm container.
//--
*/

#ifndef __CONVOLUTION2D_LAYER_FORWARD_BATCH_CONTAINER_H__
#define __CONVOLUTION2D_LAYER_FORWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/convolution2d/convolution2d_layer.h"
#include "convolution2d_layer_forward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace convolution2d
{
namespace forward
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::Convolution2dKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
void BatchContainer<algorithmFPType, method, cpu>::compute()
{
    convolution2d::forward::Input *input = static_cast<convolution2d::forward::Input *>(_in);
    convolution2d::forward::Result *result = static_cast<convolution2d::forward::Result *>(_res);

    convolution2d::Parameter *parameter = static_cast<convolution2d::Parameter *>(_par);;
    daal::services::Environment::env &env = *_env;

    Tensor* inputTensor  = input->get(layers::forward::data).get();
    Tensor* wTensor      = input->get(layers::forward::weights).get();
    Tensor* bTensor      = input->get(layers::forward::biases).get();
    Tensor* resultTensor = result->get(layers::forward::value).get();

    __DAAL_CALL_KERNEL(env, internal::Convolution2dKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, inputTensor, wTensor, bTensor, parameter, resultTensor);
}
} // namespace interface1
} // namespace forward

} // namespace convolution2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
