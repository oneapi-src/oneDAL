/* file: logitboost_predict_batch_container.h */
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
//  Implementation of Logit Boost algorithm container -- a class
//  that contains fast Logit Boost prediction kernels
//  for supported architectures.
//--
*/

#ifndef __LOGITBOOST_PREDICT_BATCH_CONTAINER__
#define __LOGITBOOST_PREDICT_BATCH_CONTAINER__

#include "algorithms/boosting/logitboost_predict.h"
#include "src/algorithms/logitboost/logitboost_predict_dense_default_kernel.h"

namespace daal
{
namespace algorithms
{
namespace logitboost
{
namespace prediction
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::LogitBoostPredictKernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    classifier::prediction::Input * input   = static_cast<classifier::prediction::Input *>(_in);
    classifier::prediction::Result * result = static_cast<classifier::prediction::Result *>(_res);

    NumericTablePtr a           = input->get(classifier::prediction::data);
    logitboost::Model * m       = static_cast<logitboost::Model *>(input->get(classifier::prediction::model).get());
    NumericTable * r            = static_cast<NumericTable *>(result->get(classifier::prediction::prediction).get());
    logitboost::Parameter * par = static_cast<logitboost::Parameter *>(_par);

    daal::services::Environment::env & env = *_env;
    __DAAL_CALL_KERNEL(env, internal::LogitBoostPredictKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, a, m, r, par);
}
} // namespace prediction
} // namespace logitboost
} // namespace algorithms
} // namespace daal

#endif // __LOGITBOOST_PREDICT_CONTAINER__
