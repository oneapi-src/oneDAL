/* file: dropout_layer_backward_batch_container.h */
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
//  Implementation of the backward dropout layer
//--
*/

#ifndef __DROPOUT_LAYER_BACKWARD_BATCH_CONTAINER_H__
#define __DROPOUT_LAYER_BACKWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/dropout/dropout_layer.h"
#include "dropout_layer_backward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace dropout
{
namespace backward
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::DropoutKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    dropout::backward::Input *input = static_cast<dropout::backward::Input *>(_in);
    dropout::backward::Result *result = static_cast<dropout::backward::Result *>(_res);

    dropout::Parameter *parameter = static_cast<dropout::Parameter *>(_par);
    if (!parameter->propagateGradient) { return services::Status(); }

    daal::services::Environment::env &env = *_env;

    Tensor *inputGradientTable = input->get(layers::backward::inputGradient).get();
    Tensor *maskTable          = input->get(dropout::auxRetainMask).get();
    Tensor *resultTable        = result->get(layers::backward::gradient).get();

    __DAAL_CALL_KERNEL(env, internal::DropoutKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, *inputGradientTable, *maskTable, *resultTable);
}
} // namespace interface1
} // namespace backward

} // namespace dropout
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
