/* file: locallyconnected2d_layer_backward_batch_container.h */
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
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    locallyconnected2d::backward::Input *input = static_cast<locallyconnected2d::backward::Input *>(_in);
    locallyconnected2d::backward::Result *result = static_cast<locallyconnected2d::backward::Result *>(_res);

    locallyconnected2d::Parameter *parameter = static_cast<locallyconnected2d::Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    LayerData *layerData     = input->get(layers::backward::inputFromForward).get();
    Tensor *inGradTensor     = input->get(layers::backward::inputGradient).get();
    Tensor *gradientTensor   = result->get(layers::backward::gradient).get();
    Tensor *wDerTensor       = result->get(layers::backward::weightDerivatives).get();
    Tensor *bDerTensor       = result->get(layers::backward::biasDerivatives).get();
    Tensor *auxDataTensor    = staticPointerCast<Tensor, SerializationIface>((*layerData)[locallyconnected2d::auxData]).get();
    Tensor *auxWeightsTensor = staticPointerCast<Tensor, SerializationIface>((*layerData)[locallyconnected2d::auxWeights]).get();

    __DAAL_CALL_KERNEL(env, internal::LocallyConnected2dKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, *inGradTensor, *gradientTensor,
                                *auxDataTensor, *auxWeightsTensor, *wDerTensor, *bDerTensor, *parameter);
}
} // namespace interface1
} // namespace backward

} // namespace locallyconnected2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
