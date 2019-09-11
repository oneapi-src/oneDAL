/* file: fullyconnected_layer_backward_batch_container.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

    fullyconnected::Parameter *parameter = static_cast<fullyconnected::Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    Tensor *inGradTensor  = input->get(layers::backward::inputGradient).get();
    Tensor *xTensor       = input->get(fullyconnected::auxData).get();
    Tensor *wTensor       = input->get(fullyconnected::auxWeights).get();
    Tensor *wDerTensor    = result->get(layers::backward::weightDerivatives).get();
    Tensor *bDerTensor    = result->get(layers::backward::biasDerivatives).get();
    Tensor *resultTensor  = result->get(layers::backward::gradient).get();

    __DAAL_CALL_KERNEL(env, internal::FullyconnectedKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, *inGradTensor, *xTensor,
                                                                                    *wTensor, *wDerTensor, *bDerTensor, *resultTensor, *parameter);
}
} // namespace interface1
} // namespace backward
} // namespace fullyconnected
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
