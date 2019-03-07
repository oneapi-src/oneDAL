/* file: elu_layer_forward_batch_container.h */
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
//  Implementation of elu calculation algorithm container.
//--
*/

#ifndef __ELU_LAYER_FORWARD_BATCH_CONTAINER_H__
#define __ELU_LAYER_FORWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/elu/elu_layer.h"
#include "elu_layer_forward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace elu
{
namespace forward
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::ELUKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    elu::Parameter *parameter = static_cast<elu::Parameter *>(_par);
    elu::forward::Input *input = static_cast<elu::forward::Input *>(_in);
    elu::forward::Result *result = static_cast<elu::forward::Result *>(_res);

    daal::services::Environment::env &env = *_env;

    Tensor *dataTensor                 = input->get(layers::forward::data).get();
    Tensor *valueTensor                = result->get(layers::forward::value).get();
    Tensor *auxIntermediateValueTensor = result->get(layers::elu::auxIntermediateValue).get();

    __DAAL_CALL_KERNEL(env, internal::ELUKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
        *parameter, *dataTensor, *valueTensor, auxIntermediateValueTensor);
}
} // namespace interface1
} // namespace forward

} // namespace elu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
