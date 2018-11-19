/* file: convolution2d_layer_backward_batch_container.h */
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
//  Implementation of convolution2d calculation algorithm container.
//--
*/

#ifndef __CONVOLUTION2D_LAYER_BACKWARD_BATCH_CONTAINER_H__
#define __CONVOLUTION2D_LAYER_BACKWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/convolution2d/convolution2d_layer.h"
#include "convolution2d_layer_backward_kernel.h"

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
namespace backward
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
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    convolution2d::backward::Input *input = static_cast<convolution2d::backward::Input *>(_in);
    convolution2d::backward::Result *result = static_cast<convolution2d::backward::Result *>(_res);

    convolution2d::Parameter *parameter = static_cast<convolution2d::Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    Tensor *inGradTensor  = input->get(layers::backward::inputGradient).get();
    LayerData* layerData  = input->get(layers::backward::inputFromForward).get();
    Tensor *wDerTensor    = result->get(layers::backward::weightDerivatives).get();
    Tensor *bDerTensor    = result->get(layers::backward::biasDerivatives).get();
    Tensor *resultTensor  = result->get(layers::backward::gradient).get();

    Tensor *xTensor = static_cast<Tensor *>(((*layerData)[convolution2d::auxData]).get());
    Tensor *wTensor = static_cast<Tensor *>(((*layerData)[convolution2d::auxWeights]).get());

    __DAAL_CALL_KERNEL(env, internal::Convolution2dKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, inGradTensor, xTensor, wTensor,
                                                                                   *parameter, wDerTensor, bDerTensor, resultTensor);
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::setupCompute()
{
    __DAAL_CALL_KERNEL(env, internal::Convolution2dKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), initialize);
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::resetCompute()
{
    __DAAL_CALL_KERNEL(env, internal::Convolution2dKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), reset);
}

} // namespace interface1
} // namespace backward

} // namespace convolution2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
