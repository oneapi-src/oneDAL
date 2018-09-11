/* file: smoothrelu_layer_backward_batch_container.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
// Implementation of the backward smooth rectifier linear unit (smooth relu) layer
//--
*/

#ifndef __SMOOTHRELU_LAYER_BACKWARD_BATCH_CONTAINER_H__
#define __SMOOTHRELU_LAYER_BACKWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/smoothrelu/smoothrelu_layer.h"
#include "smoothrelu_layer_backward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace smoothrelu
{
namespace backward
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::SmoothReLUKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    smoothrelu::backward::Input *input = static_cast<smoothrelu::backward::Input *>(_in);
    smoothrelu::backward::Result *result = static_cast<smoothrelu::backward::Result *>(_res);

    const Parameter *param = static_cast<const Parameter *>(_par);
    if (!param->propagateGradient) { return services::Status(); }

    daal::services::Environment::env &env = *_env;

    Tensor *inputTensor        = input->get(layers::backward::inputGradient).get();
    Tensor *forwardValueTensor = input->get(smoothrelu::auxData).get();
    Tensor *resultTensor       = result->get(layers::backward::gradient).get();

    __DAAL_CALL_KERNEL(env, internal::SmoothReLUKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, *inputTensor, *forwardValueTensor, *resultTensor);
}
} // namespace interface1
} // namespace backward

} // namespace smoothrelu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
