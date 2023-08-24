/* file: brownboost_train_batch_container.h */
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
//  Implementation of Brown Boost algorithm container -- a class that contains
//  Freund Brown Boost kernels for supported architectures.
//--
*/

#ifndef __BROWNBOOST_TRAIN_BATCH_CONTAINER_H__
#define __BROWNBOOST_TRAIN_BATCH_CONTAINER_H__

#include "algorithms/boosting/brownboost_training_batch.h"
#include "src/algorithms/brownboost/brownboost_train_kernel.h"
#include "src/algorithms/kernel.h"

namespace daal
{
namespace algorithms
{
namespace brownboost
{
namespace training
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::BrownBoostTrainKernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    brownboost::training::Result * result = static_cast<brownboost::training::Result *>(_res);
    classifier::training::Input * input   = static_cast<classifier::training::Input *>(_in);

    size_t n = input->size();

    NumericTablePtr a[2];
    a[0]                        = services::staticPointerCast<NumericTable>(input->get(classifier::training::data));
    a[1]                        = services::staticPointerCast<NumericTable>(input->get(classifier::training::labels));
    brownboost::Model * r       = static_cast<brownboost::Model *>(result->get(classifier::training::model).get());
    brownboost::Parameter * par = static_cast<brownboost::Parameter *>(_par);

    daal::services::Environment::env & env = *_env;
    __DAAL_CALL_KERNEL(env, internal::BrownBoostTrainKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, n, a, r, par);
}

} // namespace training
} // namespace brownboost
} // namespace algorithms
} // namespace daal

#endif // __BROWNBOOST_TRAINING_BATCH_CONTAINER_H__
