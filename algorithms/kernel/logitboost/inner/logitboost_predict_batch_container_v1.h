/* file: logitboost_predict_batch_container_v1.h */
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
//  Implementation of Logit Boost algorithm container -- a class
//  that contains fast Logit Boost prediction kernels
//  for supported architectures.
//--
*/

#ifndef __LOGITBOOST_PREDICT_BATCH_CONTAINER_V1__
#define __LOGITBOOST_PREDICT_BATCH_CONTAINER_V1__

#include "logitboost_predict.h"
#include "logitboost_predict_dense_default_kernel_v1.h"

namespace daal
{
namespace algorithms
{
namespace logitboost
{
namespace prediction
{
namespace interface1
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::I1LogitBoostPredictKernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    classifier::prediction::interface1::Input * input   = static_cast<classifier::prediction::interface1::Input *>(_in);
    classifier::prediction::interface1::Result * result = static_cast<classifier::prediction::interface1::Result *>(_res);

    NumericTablePtr a                       = input->get(classifier::prediction::data);
    logitboost::interface1::Model * m       = static_cast<logitboost::interface1::Model *>(input->get(classifier::prediction::model).get());
    NumericTable * r                        = static_cast<NumericTable *>(result->get(classifier::prediction::prediction).get());
    logitboost::interface1::Parameter * par = static_cast<logitboost::interface1::Parameter *>(_par);

    daal::services::Environment::env & env = *_env;
    __DAAL_CALL_KERNEL(env, internal::I1LogitBoostPredictKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, a, m, r, par);
}
} // namespace interface1
} // namespace prediction
} // namespace logitboost
} // namespace algorithms
} // namespace daal

#endif // __LOGITBOOST_PREDICT_CONTAINER__
