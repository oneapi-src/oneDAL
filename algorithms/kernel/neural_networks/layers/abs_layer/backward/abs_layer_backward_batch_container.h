/* file: abs_layer_backward_batch_container.h */
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
//  Implementation of abs layer container.
//--
*/

#ifndef __ABS_LAYER_BACKWARD_BATCH_CONTAINER_H__
#define __ABS_LAYER_BACKWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/abs/abs_layer.h"
#include "abs_layer_backward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace abs
{
namespace backward
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::AbsKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    abs::backward::Input *input = static_cast<abs::backward::Input *>(_in);
    abs::backward::Result *result = static_cast<abs::backward::Result *>(_res);

    const layers::Parameter *par = static_cast<const layers::Parameter *>(_par);
    if (!par->propagateGradient) { return services::Status(); }

    daal::services::Environment::env &env = *_env;

    Tensor *inputTensor  = input->get(layers::backward::inputGradient).get();
    Tensor *dataTensor   = input->get(abs::auxData).get();
    Tensor *resultTensor = result->get(layers::backward::gradient).get();

    __DAAL_CALL_KERNEL(env, internal::AbsKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, *inputTensor, *dataTensor, *resultTensor);
}
} // namespace interface1
} // namespace backward

} // namespace abs
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
