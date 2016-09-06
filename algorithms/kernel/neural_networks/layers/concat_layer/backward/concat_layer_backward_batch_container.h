/* file: concat_layer_backward_batch_container.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of concat layer container.
//--
*/

#ifndef __CONCAT_LAYER_BACKWARD_BATCH_CONTAINER_H__
#define __CONCAT_LAYER_BACKWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/concat/concat_layer.h"
#include "concat_layer_backward_kernel.h"
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
namespace backward
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
void BatchContainer<algorithmFPType, method, cpu>::compute()
{
    concat::backward::Input *input = static_cast<concat::backward::Input *>(_in);
    concat::backward::Result *result = static_cast<concat::backward::Result *>(_res);

    concat::Parameter *parameter = static_cast<concat::Parameter *>(_par);;
    daal::services::Environment::env &env = *_env;

    Tensor *inputTensor              = input->get(layers::backward::inputGradient).get();
    NumericTable *forwardOutputTable = input->get(layers::concat::auxInputDimensions).get();

    size_t nOutputs = forwardOutputTable->getNumberOfColumns();

    SmartPtr<cpu> resultBlock( nOutputs * sizeof(Tensor *));
    Tensor **resultTensors = (Tensor **)resultBlock.get();
    if (!resultTensors)  { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    for(size_t i = 0; i < nOutputs; i++)
    {
        resultTensors[i] = result->get(layers::backward::resultLayerData, i).get();
    }

    __DAAL_CALL_KERNEL(env, internal::ConcatKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, inputTensor, forwardOutputTable, parameter,
                       resultTensors);
}
} // namespace interface1
} // namespace backward

} // namespace concat
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
