/* file: batch_normalization_layer_forward_kernel.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

//++
//  Declaration of template function that calculate forward batch normalization layer relults.
//--

#ifndef __BATCH_NORMALIZATION_LAYER_FORWARD_KERNEL_H__
#define __BATCH_NORMALIZATION_LAYER_FORWARD_KERNEL_H__

#include "algorithms/neural_networks/layers/batch_normalization/batch_normalization_layer_forward.h"
#include "algorithms/neural_networks/layers/batch_normalization/batch_normalization_layer_forward_types.h"

#include "algorithms/kernel/kernel.h"
#include "algorithms/threading/threading.h"

#include "externals/service_math.h"
#include "service/kernel/data_management/service_tensor.h"
#include "service/kernel/service_unique_ptr.h"
#include "service/kernel/data_management/service_numeric_table.h"

#include "algorithms/kernel/neural_networks/layers/batch_normalization_layer/forward/batch_normalization_layer_forward_task.h"
#include "algorithms/kernel/neural_networks/layers/batch_normalization_layer/forward/batch_normalization_layer_forward_task_descriptor.h"

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
/**
 *  \brief Kernel for forward batch normalization layer results computation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchNormalizationKernel : public Kernel
{
private:
    typedef CommonBatchNormalizationTask<algorithmFPType, method, cpu> InternalBatchNormalizationTask;

public:
    services::Status initialize(const BatchNormalizationTaskDescriptor & descriptor);
    services::Status compute(const BatchNormalizationTaskDescriptor & descriptor);
    services::Status reset();

private:
    UniquePtr<InternalBatchNormalizationTask, cpu> _task;
};

} // namespace internal
} // namespace forward
} // namespace batch_normalization
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
