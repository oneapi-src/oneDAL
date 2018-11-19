/* file: neural_networks_training_distr_container.h */
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

#ifndef __NEURAL_NETWORKS_TRAINING_DISTR_CONTAINER_H__
#define __NEURAL_NETWORKS_TRAINING_DISTR_CONTAINER_H__

#include "neural_networks/neural_networks_training_distributed.h"
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
/* TrainingKernelDistributed */
template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::interface1::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::TrainingKernelDistributed, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step1Local> *input = static_cast<DistributedInput<step1Local> *>(_in);
    PartialResult *partialResult = static_cast<PartialResult *>(_pres);
    Parameter *parameter = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    Tensor* data = input->get(training::data).get();
    Model* nnModel = input->get(training::inputModel).get();
    KeyValueDataCollectionPtr groundTruthCollectionPtr = input->get(training::groundTruthCollection);

    __DAAL_CALL_KERNEL(env, internal::TrainingKernelDistributed, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       data, nnModel, groundTruthCollectionPtr, partialResult, parameter);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::setupCompute()
{
    DistributedInput<step1Local> *input = static_cast<DistributedInput<step1Local> *>(_in);
    Parameter *parameter = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    Tensor* data = input->get(training::data).get();
    Model* nnModel = input->get(training::inputModel).get();
    KeyValueDataCollectionPtr groundTruthCollectionPtr = input->get(training::groundTruthCollection);

    __DAAL_CALL_KERNEL(env, internal::TrainingKernelDistributed, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), initialize,
                       data, nnModel, groundTruthCollectionPtr, parameter);
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::resetCompute()
{
    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::TrainingKernelDistributed, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), reset);
}

/* TrainingKernelDistributedStep2 */
template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::interface1::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::TrainingKernelDistributedStep2, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step2Master> *input = static_cast<DistributedInput<step2Master> *>(_in);
    DistributedPartialResult *partialResult = static_cast<DistributedPartialResult *>(_pres);
    Parameter *parameter = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    KeyValueDataCollection* collection = input->get(training::partialResults).get();
    Model* nnModel = partialResult->get(resultFromMaster)->get(training::model).get();

    __DAAL_CALL_KERNEL(env, internal::TrainingKernelDistributedStep2, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       collection, parameter, nnModel);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::setupCompute()
{
    return services::Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::resetCompute()
{
    return services::Status();
}

} // namespace interface1
} // namespace training
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
