/* file: lrn_layer_backward_batch_container.h */
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
//  Implementation of the backward local response normalization layer.
//--
*/

#ifndef __LRN_LAYER_BACKWARD_BATCH_CONTAINER_H__
#define __LRN_LAYER_BACKWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/lrn/lrn_layer.h"
#include "lrn_layer_backward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace lrn
{
namespace backward
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::LRNKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    lrn::backward::Input *input = static_cast<lrn::backward::Input *>(_in);
    lrn::backward::Result *result = static_cast<lrn::backward::Result *>(_res);

    lrn::Parameter *parameter = static_cast<lrn::Parameter *>(_par);
    if (!parameter->propagateGradient) { return services::Status(); }

    daal::services::Environment::env &env = *_env;

    Tensor *dataTensor          = input->get(layers::lrn::auxData).get();
    Tensor *sMinusBetaTensor    = input->get(lrn::auxSmBeta).get();
    Tensor *inputGradientTensor = input->get(layers::backward::inputGradient).get();
    Tensor *gradientTensor      = result->get(layers::backward::gradient).get();

    __DAAL_CALL_KERNEL(env, internal::LRNKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, *dataTensor, *sMinusBetaTensor, *inputGradientTensor,
                                                                         *gradientTensor, *parameter);
}
} // namespace interface1
} // namespace backward

} // namespace lrn
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
