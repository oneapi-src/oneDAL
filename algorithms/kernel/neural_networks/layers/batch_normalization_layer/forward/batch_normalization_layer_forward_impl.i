/* file: batch_normalization_layer_forward_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of forward batch normalization layer
//--
*/

#ifndef __BATCH_NORMALIZATION_LAYER_FORWARD_IMPL_I__
#define __BATCH_NORMALIZATION_LAYER_FORWARD_IMPL_I__

#include "service_math.h"
#include "threading.h"

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace batch_normalization
{
namespace forward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
Status BatchNormalizationKernel<algorithmFPType, method, cpu>::compute(
    const BatchNormalizationTaskDescriptor &descriptor)
{
    return _task->compute(descriptor);
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status BatchNormalizationKernel<algorithmFPType, method, cpu>::initialize(
    const BatchNormalizationTaskDescriptor &descriptor)
{
    _task.reset(new InternalBatchNormalizationTask()); DAAL_CHECK_MALLOC(_task.get());
    return _task->initialize(descriptor);
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status BatchNormalizationKernel<algorithmFPType, method, cpu>::reset()
{
    _task.reset();
    return Status();
}

} // namespace internal
} // namespace forward
} // namespace batch_normalization
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
