/* file: transposed_conv2d_layer_forward_batch_container.h */
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
//  Implementation of transposed convolution 2d calculation algorithm container.
//--
*/

#ifndef __TRANSPOSED_CONV2D_LAYER_FORWARD_BATCH_CONTAINER_H__
#define __TRANSPOSED_CONV2D_LAYER_FORWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/transposed_conv2d/transposed_conv2d_layer.h"
#include "transposed_conv2d_layer_forward_kernel.h"

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
namespace forward
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
services::Status BatchContainer<algorithmFPType, method, cpu>::setupCompute()
{
    return completeInput();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    transposed_conv2d::forward::Input *input = static_cast<transposed_conv2d::forward::Input *>(_in);
    transposed_conv2d::forward::Result *result = static_cast<transposed_conv2d::forward::Result *>(_res);

    transposed_conv2d::Parameter *parameter = static_cast<transposed_conv2d::Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    Tensor *inputTensor  = input->get(layers::forward::data).get();
    Tensor *wTensor      = input->get(layers::forward::weights).get();
    Tensor *bTensor      = input->get(layers::forward::biases).get();
    Tensor *resultTensor = result->get(layers::forward::value).get();

    __DAAL_CALL_KERNEL(env, internal::TransposedConv2dKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       *inputTensor, *wTensor, *bTensor, *parameter, *resultTensor);
}
} // namespace interface1
} // namespace forward

} // namespace transposed_conv2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
