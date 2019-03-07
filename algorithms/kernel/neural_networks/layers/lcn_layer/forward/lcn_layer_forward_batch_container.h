/* file: lcn_layer_forward_batch_container.h */
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
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    lcn::forward::Input *input = static_cast<lcn::forward::Input *>(_in);
    lcn::forward::Result *result = static_cast<lcn::forward::Result *>(_res);

    lcn::Parameter *parameter = static_cast<lcn::Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    Tensor *inputTensor        = input->get(layers::forward::data).get();
    Tensor *resultTensor       = result->get(layers::forward::value).get();
    Tensor *centeredDataTensor = result->get(lcn::auxCenteredData).get();
    Tensor *sigmaTensor        = result->get(lcn::auxSigma).get();
    Tensor *cTensor            = result->get(lcn::auxC).get();
    Tensor *invMaxTensor       = result->get(lcn::auxInvMax).get();
    Tensor *kernelTensor       = parameter->kernel.get();

    __DAAL_CALL_KERNEL(env, internal::LCNKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, *inputTensor, *sigmaTensor, *cTensor, *resultTensor,
        *centeredDataTensor, *invMaxTensor, *parameter, *kernelTensor);
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::setupCompute()
{
    lcn::forward::Input *input = static_cast<lcn::forward::Input *>(_in);
    lcn::forward::Result *result = static_cast<lcn::forward::Result *>(_res);

    lcn::Parameter *parameter = static_cast<lcn::Parameter *>(_par);
    if (!parameter->propagateGradient) { return services::Status(); }

    daal::services::Environment::env &env = *_env;

    Tensor *inputTensor        = input->get(layers::forward::data).get();
    Tensor *resultTensor       = result->get(layers::forward::value).get();
    Tensor *centeredDataTensor = result->get(lcn::auxCenteredData).get();
    Tensor *sigmaTensor        = result->get(lcn::auxSigma).get();
    Tensor *cTensor            = result->get(lcn::auxC).get();
    Tensor *invMaxTensor       = result->get(lcn::auxInvMax).get();
    Tensor *kernelTensor       = parameter->kernel.get();

    __DAAL_CALL_KERNEL(env, internal::LCNKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), initialize, *inputTensor, *cTensor, *invMaxTensor,
                       *parameter, *kernelTensor);
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::resetCompute()
{
    __DAAL_CALL_KERNEL(env, internal::LCNKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), reset);
}

} // namespace interface1
} // namespace forward
} // namespace lcn
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
