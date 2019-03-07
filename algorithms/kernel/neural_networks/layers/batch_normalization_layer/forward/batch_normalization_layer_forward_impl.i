/* file: batch_normalization_layer_forward_impl.i */
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
