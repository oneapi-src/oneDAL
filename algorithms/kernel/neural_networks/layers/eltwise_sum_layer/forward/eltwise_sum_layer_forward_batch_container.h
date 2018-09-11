/* file: eltwise_sum_layer_forward_batch_container.h */
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
//  Implementation of element-wise sum calculation algorithm container.
//--
*/

#ifndef __ELTWISE_SUM_LAYER_FORWARD_BATCH_CONTAINER_H__
#define __ELTWISE_SUM_LAYER_FORWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/eltwise_sum/eltwise_sum_layer.h"
#include "eltwise_sum_layer_forward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace eltwise_sum
{
namespace forward
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::EltwiseSumKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    using namespace daal::internal;

    eltwise_sum::forward::Input *input = static_cast<eltwise_sum::forward::Input *>(_in);
    eltwise_sum::forward::Result *result = static_cast<eltwise_sum::forward::Result *>(_res);

    const size_t nInputs = input->get(layers::forward::inputLayerData)->size();

    TArray<Tensor *, cpu> inputBlock(nInputs);
    Tensor **inputTensors = inputBlock.get();
    if (!inputTensors) { return services::Status(services::ErrorMemoryAllocationFailed); }

    for (size_t i = 0; i < nInputs; i++)
    {
        inputTensors[i] = input->get(layers::forward::inputLayerData, i).get();
    }

    Tensor *coefficients               = input->get(eltwise_sum::forward::coefficients).get();
    Tensor *value                      = result->get(layers::forward::value).get();
    Tensor *auxCoefficients            = result->get(eltwise_sum::auxCoefficients).get();
    NumericTable *numberOfCoefficients = result->get(eltwise_sum::auxNumberOfCoefficients).get();

    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::EltwiseSumKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
        compute, inputTensors, value, coefficients, auxCoefficients, numberOfCoefficients, nInputs);
}
} // namespace interface1
} // namespace forward
} // namespace eltwise_sum
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
