/* file: concat_layer_forward_batch_container.h */
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
//  Implementation of concat layer container.
//--
*/

#ifndef __CONCAT_LAYER_FORWARD_BATCH_CONTAINER_H__
#define __CONCAT_LAYER_FORWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/concat/concat_layer.h"
#include "concat_layer_forward_kernel.h"
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
namespace concat
{
namespace forward
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::ConcatKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    concat::forward::Input *input = static_cast<concat::forward::Input *>(_in);
    concat::forward::Result *result = static_cast<concat::forward::Result *>(_res);

    concat::Parameter *parameter = static_cast<concat::Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    const size_t nInputs = input->get(layers::forward::inputLayerData)->size();

    Tensor *resultTensor  = result->get(layers::forward::value).get();

    TArray<Tensor *, cpu> inputBlock(nInputs);
    Tensor **inputTensors = inputBlock.get();
    DAAL_CHECK_MALLOC(inputTensors);

    for(size_t i = 0; i < nInputs; i++)
    {
        inputTensors[i] = input->get(layers::forward::inputLayerData, i).get();
    }

    __DAAL_CALL_KERNEL(env, internal::ConcatKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, nInputs, inputTensors, parameter,
                       resultTensor);
}
} // namespace interface1
} // namespace forward

} // namespace concat
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
