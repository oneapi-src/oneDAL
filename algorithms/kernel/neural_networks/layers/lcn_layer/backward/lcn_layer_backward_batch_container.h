/* file: lcn_layer_backward_batch_container.h */
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
//  Implementation of local contrast normalization calculation algorithm container.
//--
*/

#ifndef __CONVOLUTION2D_LAYER_BACKWARD_BATCH_CONTAINER_H__
#define __CONVOLUTION2D_LAYER_BACKWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/lcn/lcn_layer.h"
#include "lcn_layer_backward_kernel.h"

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
namespace backward
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
    lcn::backward::Input *input = static_cast<lcn::backward::Input *>(_in);
    lcn::backward::Result *result = static_cast<lcn::backward::Result *>(_res);

    lcn::Parameter *parameter = static_cast<lcn::Parameter *>(_par);
    if (!parameter->propagateGradient) { return services::Status(); }

    daal::services::Environment::env &env = *_env;

    LayerData *layerData          = input->get(layers::backward::inputFromForward).get();
    Tensor *inGradTensor          = input->get(layers::backward::inputGradient).get();
    Tensor *gradientTensor        = result->get(layers::backward::gradient).get();
    Tensor *auxCenteredDataTensor = staticPointerCast<Tensor, SerializationIface>((*layerData)[lcn::auxCenteredData]).get();
    Tensor *auxSigmaTensor        = staticPointerCast<Tensor, SerializationIface>((*layerData)[lcn::auxSigma]).get();
    Tensor *auxCTensor            = staticPointerCast<Tensor, SerializationIface>((*layerData)[lcn::auxC]).get();
    Tensor *auxInvMaxTensor       = staticPointerCast<Tensor, SerializationIface>((*layerData)[lcn::auxInvMax]).get();
    Tensor *kernelTensor          = parameter->kernel.get();

    __DAAL_CALL_KERNEL(env, internal::LCNKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, *auxCenteredDataTensor, *auxSigmaTensor,
                       *auxCTensor, *auxInvMaxTensor, *kernelTensor, *inGradTensor, *gradientTensor, *parameter);
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::setupCompute()
{
    lcn::backward::Input *input = static_cast<lcn::backward::Input *>(_in);
    lcn::backward::Result *result = static_cast<lcn::backward::Result *>(_res);

    lcn::Parameter *parameter = static_cast<lcn::Parameter *>(_par);
    if (!parameter->propagateGradient) { return services::Status(); }

    daal::services::Environment::env &env = *_env;

    LayerData *layerData          = input->get(layers::backward::inputFromForward).get();
    Tensor *auxCenteredDataTensor = staticPointerCast<Tensor, SerializationIface>((*layerData)[lcn::auxCenteredData]).get();
    Tensor *auxSigmaTensor        = staticPointerCast<Tensor, SerializationIface>((*layerData)[lcn::auxSigma]).get();
    Tensor *auxCTensor            = staticPointerCast<Tensor, SerializationIface>((*layerData)[lcn::auxC]).get();
    Tensor *kernelTensor          = parameter->kernel.get();

    __DAAL_CALL_KERNEL(env, internal::LCNKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), initialize, *auxCenteredDataTensor,
                       *auxSigmaTensor, *auxCTensor, *kernelTensor, *parameter);
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::resetCompute()
{
    __DAAL_CALL_KERNEL(env, internal::LCNKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), reset);
}

} // namespace interface1
} // namespace backward
} // namespace lcn
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
