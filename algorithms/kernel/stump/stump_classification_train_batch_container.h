/* file: stump_classification_train_batch_container.h */
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
//  Implementation of Decision Stump algorithm container -- a class that contains
//  Friedman Decision Stump kernels for supported architectures.
//--
*/

#ifndef __STUMP_CLASSIFICATION_TRAIN_BATCH_CONTAINER_H__
#define __STUMP_CLASSIFICATION_TRAIN_BATCH_CONTAINER_H__

#include "stump_classification_training_batch.h"
#include "stump_classification_train_kernel.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace classification
{
namespace training
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::StumpTrainKernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    classifier::training::Input * input              = static_cast<classifier::training::Input *>(_in);
    stump::classification::training::Result * result = static_cast<stump::classification::training::Result *>(_res);
    DAAL_CHECK_MALLOC(_par)
    const Parameter * par = static_cast<Parameter *>(_par);
    size_t n              = input->size();
    NumericTable * a[3];
    a[0]                             = static_cast<NumericTable *>(input->get(classifier::training::data).get());
    a[1]                             = static_cast<NumericTable *>(input->get(classifier::training::labels).get());
    a[2]                             = static_cast<NumericTable *>(input->get(classifier::training::weights).get());
    stump::classification::Model * r = static_cast<stump::classification::Model *>(result->get(classifier::training::model).get());

    daal::services::Environment::env & env = *_env;
    __DAAL_CALL_KERNEL(env, internal::StumpTrainKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, n, a, r, par);
}

} // namespace training
} // namespace classification
} // namespace stump
} // namespace algorithms
} // namespace daal

#endif
