/* file: transposed_conv2d_layer_backward_batch_container.h */
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

    transposed_conv2d::Parameter *parameter = static_cast<transposed_conv2d::Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    Tensor *inGradTensor  = input->get(layers::backward::inputGradient).get();
    Tensor *xTensor       = input->get(transposed_conv2d::auxData).get();
    Tensor *wTensor       = input->get(transposed_conv2d::auxWeights).get();
    Tensor *wDerTensor    = result->get(layers::backward::weightDerivatives).get();
    Tensor *bDerTensor    = result->get(layers::backward::biasDerivatives).get();
    Tensor *resultTensor  = result->get(layers::backward::gradient).get();

    __DAAL_CALL_KERNEL(env, internal::TransposedConv2dKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       *inGradTensor, *xTensor, *wTensor, *parameter, *wDerTensor, *bDerTensor, *resultTensor);
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
