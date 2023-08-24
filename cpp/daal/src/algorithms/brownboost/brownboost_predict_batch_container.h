/* file: brownboost_predict_batch_container.h */
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
//  Implementation of Brown Boost prediction algorithm container --
//  a class that contains Fast Brown Boost kernels for supported architectures.
//--
*/

#include "src/algorithms/brownboost/brownboost_predict_kernel.h"

namespace daal
{
namespace algorithms
{
namespace brownboost
{
namespace prediction
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::BrownBoostPredictKernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    classifier::prediction::Result * result = static_cast<classifier::prediction::Result *>(_res);
    classifier::prediction::Input * input   = static_cast<classifier::prediction::Input *>(_in);

    NumericTablePtr a           = input->get(classifier::prediction::data);
    brownboost::Model * m       = static_cast<brownboost::Model *>(input->get(classifier::prediction::model).get());
    NumericTablePtr r           = result->get(classifier::prediction::prediction);
    brownboost::Parameter * par = static_cast<brownboost::Parameter *>(_par);

    daal::services::Environment::env & env = *_env;
    __DAAL_CALL_KERNEL(env, internal::BrownBoostPredictKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, a, m, r, par);
}

} // namespace prediction
} // namespace brownboost
} // namespace algorithms
} // namespace daal
