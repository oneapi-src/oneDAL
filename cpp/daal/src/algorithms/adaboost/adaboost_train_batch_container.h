/* file: adaboost_train_batch_container.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of Ada Boost algorithm container -- a class that contains
//  Freund Ada Boost kernels for supported architectures.
//--
*/

#ifndef __ADABOOST_TRAIN_BATCH_CONTAINER_H__
#define __ADABOOST_TRAIN_BATCH_CONTAINER_H__

#include "algorithms/boosting/adaboost_training_batch.h"
#include "src/algorithms/adaboost/adaboost_train_kernel.h"
#include "src/algorithms/kernel.h"

namespace daal
{
namespace algorithms
{
namespace adaboost
{
namespace training
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::AdaBoostTrainKernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    classifier::training::Input * input = static_cast<classifier::training::Input *>(_in);
    adaboost::training::Result * result = static_cast<adaboost::training::Result *>(_res);

    NumericTablePtr a[2];
    a[0] = services::staticPointerCast<NumericTable>(input->get(classifier::training::data));
    a[1] = services::staticPointerCast<NumericTable>(input->get(classifier::training::labels));

    adaboost::Model * r                    = static_cast<adaboost::Model *>(result->get(classifier::training::model).get());
    NumericTable * weakLearnersErrorsTable = result->get(adaboost::training::weakLearnersErrors).get();
    DAAL_CHECK_MALLOC(_par)
    const adaboost::Parameter * par = static_cast<adaboost::Parameter *>(_par);

    daal::services::Environment::env & env = *_env;
    __DAAL_CALL_KERNEL(env, internal::AdaBoostTrainKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, a, r, weakLearnersErrorsTable,
                       par);
}
} // namespace training
} // namespace adaboost
} // namespace algorithms
} // namespace daal

#endif // __ADABOOST_TRAINING_BATCH_CONTAINER_H__
