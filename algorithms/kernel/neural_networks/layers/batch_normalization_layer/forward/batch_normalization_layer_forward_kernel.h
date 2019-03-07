/* file: batch_normalization_layer_forward_kernel.h */
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

//++
//  Declaration of template function that calculate forward batch normalization layer relults.
//--

#ifndef __BATCH_NORMALIZATION_LAYER_FORWARD_KERNEL_H__
#define __BATCH_NORMALIZATION_LAYER_FORWARD_KERNEL_H__

#include "neural_networks/layers/batch_normalization/batch_normalization_layer_forward.h"
#include "neural_networks/layers/batch_normalization/batch_normalization_layer_forward_types.h"

#include "kernel.h"
#include "threading.h"

#include "service_math.h"
#include "service_tensor.h"
#include "service_unique_ptr.h"
#include "service_numeric_table.h"

#include "batch_normalization_layer_forward_task.h"
#include "batch_normalization_layer_forward_task_descriptor.h"

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
template<typename algorithmFPType, Method method, CpuType cpu>
class BatchNormalizationKernel : public Kernel
{
private:
    typedef CommonBatchNormalizationTask<algorithmFPType, method, cpu> InternalBatchNormalizationTask;

public:
    services::Status initialize(const BatchNormalizationTaskDescriptor &descriptor);
    services::Status compute(const BatchNormalizationTaskDescriptor &descriptor);
    services::Status reset();

private:
    UniquePtr<InternalBatchNormalizationTask, cpu> _task;
};

} // internal
} // forward
} // batch_normalization
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
