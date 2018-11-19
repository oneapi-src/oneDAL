/* file: split_layer_backward_batch_container.h */
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
//  Implementation of split layer container.
//--
*/

#ifndef __SPLIT_LAYER_BACKWARD_BATCH_CONTAINER_H__
#define __SPLIT_LAYER_BACKWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/split/split_layer.h"
#include "split_layer_backward_kernel.h"
#include "service_numeric_table.h"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace split
{
namespace backward
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::SplitKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    split::backward::Input *input = static_cast<split::backward::Input *>(_in);
    split::backward::Result *result = static_cast<split::backward::Result *>(_res);

    split::Parameter *parameter = static_cast<split::Parameter *>(_par);
    if (!parameter->propagateGradient) { return services::Status(); }

    daal::services::Environment::env &env = *_env;

    Tensor *resultTable = result->get(layers::backward::gradient).get();
    const size_t nInputs = parameter->nInputs;

    TArray<Tensor *, cpu> inputBlock(nInputs);
    Tensor **inputTensors = inputBlock.get();
    DAAL_CHECK_MALLOC(inputTensors);

    for(size_t i = 0; i < nInputs; i++)
    {
        inputTensors[i] = input->get(inputGradientCollection, i).get();
    }

    __DAAL_CALL_KERNEL(env, internal::SplitKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, inputTensors, resultTable, nInputs);
}
} // namespace interface1
} // namespace backward
} // namespace split
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
