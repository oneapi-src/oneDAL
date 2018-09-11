/* file: relu_layer_backward_batch_container.h */
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
//  Implementation of relu calculation algorithm container.
//--
*/

#ifndef __RELU_LAYER_BACKWARD_BATCH_CONTAINER_H__
#define __RELU_LAYER_BACKWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/relu/relu_layer.h"
#include "relu_layer_backward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace relu
{
namespace backward
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::ReLUKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    const layers::Parameter *par = static_cast<const layers::Parameter *>(_par);
    if (!par->propagateGradient) { return services::Status(); }

    relu::backward::Input *input = static_cast<relu::backward::Input *>(_in);
    relu::backward::Result *result = static_cast<relu::backward::Result *>(_res);

    daal::services::Environment::env &env = *_env;

    Tensor *inputGradientTensor = input->get(layers::backward::inputGradient).get();
    Tensor *forwardDataTensor   = input->get(relu::auxData).get();
    Tensor *resultTensor        = result->get(layers::backward::gradient).get();

    __DAAL_CALL_KERNEL(env, internal::ReLUKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, *inputGradientTensor, *forwardDataTensor, *resultTensor);
}
} // namespace interface1
} // namespace backward
} // namespace relu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
