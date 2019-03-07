/* file: batch_normalization_layer_forward_batch_container.h */
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
//  Implementation of forward batch normalization layer container.
//--
*/

#ifndef __BATCH_NORMALIZATION_LAYER_FORWARD_BATCH_CONTAINER_H__
#define __BATCH_NORMALIZATION_LAYER_FORWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/batch_normalization/batch_normalization_layer_forward.h"
#include "batch_normalization_layer_forward_kernel.h"

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
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::BatchNormalizationKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::setupCompute()
{
    Input     *input     = static_cast<Input *>(_in);
    Result    *result    = static_cast<Result *>(_res);
    Parameter *parameter = static_cast<Parameter *>(_par);

    daal::services::Environment::env &env = *_env;
    services::Status s;
    DAAL_CHECK_STATUS(s, completeInput());
    __DAAL_CALL_KERNEL(env, internal::BatchNormalizationKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), initialize,
        internal::BatchNormalizationTaskDescriptor(input, result, parameter));
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::resetCompute()
{
    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::BatchNormalizationKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), reset);
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input     *input     = static_cast<Input *>(_in);
    Result    *result    = static_cast<Result *>(_res);
    Parameter *parameter = static_cast<Parameter *>(_par);

    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::BatchNormalizationKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
        internal::BatchNormalizationTaskDescriptor(input, result, parameter));
}

} // namespace interface1
} // namespace forward
} // namespace batch_normalization
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
