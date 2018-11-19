/* file: locallyconnected2d_layer_forward_batch_container.h */
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

#ifndef __LOCALLYCONNECTED2D_LAYER_FORWARD_BATCH_CONTAINER_H__
#define __LOCALLYCONNECTED2D_LAYER_FORWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/locallyconnected2d/locallyconnected2d_layer.h"
#include "locallyconnected2d_layer_forward_kernel.h"

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
namespace forward
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
services::Status BatchContainer<algorithmFPType, method, cpu>::setupCompute()
{
    return completeInput();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    locallyconnected2d::forward::Input *input = static_cast<locallyconnected2d::forward::Input *>(_in);
    locallyconnected2d::forward::Result *result = static_cast<locallyconnected2d::forward::Result *>(_res);

    locallyconnected2d::Parameter *parameter = static_cast<locallyconnected2d::Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    Tensor *inputTensor        = input->get(layers::forward::data).get();
    Tensor *weightsTensor      = input->get(layers::forward::weights).get();
    Tensor *biasesTensor       = input->get(layers::forward::biases).get();
    Tensor *valueTensor        = result->get(layers::forward::value).get();

    __DAAL_CALL_KERNEL(env, internal::LocallyConnected2dKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
        *inputTensor, *weightsTensor, *biasesTensor, *valueTensor, *parameter);
}
} // namespace interface1
} // namespace forward

} // namespace locallyconnected2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
