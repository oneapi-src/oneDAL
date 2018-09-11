/* file: neural_networks_training_batch_container.h */
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
//  Implementation of neural_networks calculation algorithm container.
//--
*/

#ifndef __NEURAL_NETWORKS_TRAINING_BATCH_CONTAINER_H__
#define __NEURAL_NETWORKS_TRAINING_BATCH_CONTAINER_H__

#include "neural_networks/neural_networks_training.h"
#include "neural_networks_types.h"
#include "neural_networks_training_types.h"
#include "neural_networks_training_feedforward_kernel.h"
#include "kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace training
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::interface1::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::TrainingKernelBatch, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    daal::services::Environment::env &env = *_env;

    Tensor* data = input->get(training::data).get();
    Model* nnModel = result->get(training::model).get();
    KeyValueDataCollectionPtr groundTruthCollectionPtr = input->get(training::groundTruthCollection);

    __DAAL_CALL_KERNEL(env, internal::TrainingKernelBatch,
        __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, data, nnModel, groundTruthCollectionPtr);
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::setupCompute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    Parameter *parameter = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    Tensor* data = input->get(training::data).get();
    Model* nnModel = result->get(training::model).get();
    KeyValueDataCollectionPtr groundTruthCollectionPtr = input->get(training::groundTruthCollection);

    __DAAL_CALL_KERNEL(env, internal::TrainingKernelBatch,
        __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), initialize, data, nnModel, groundTruthCollectionPtr, parameter);
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::resetCompute()
{
    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::TrainingKernelBatch, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), reset);
}

} // namespace interface1
} // namespace training
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
