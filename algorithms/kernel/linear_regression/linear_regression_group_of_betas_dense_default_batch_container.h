/* file: linear_regression_group_of_betas_dense_default_batch_container.h */
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
//  Implementation of the container for the multi-class confusion matrix.
//--
*/

#ifndef __LINEAR_REGRESSION_GROUP_OF_BETAS_DENSE_DEFAULT_BATCH_CONTAINER_H__
#define __LINEAR_REGRESSION_GROUP_OF_BETAS_DENSE_DEFAULT_BATCH_CONTAINER_H__

#include "algorithms/linear_regression/linear_regression_group_of_betas_batch.h"
#include "linear_regression_group_of_betas_dense_default_batch_kernel.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace quality_metric
{
namespace group_of_betas
{
using namespace daal::data_management;

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::GroupOfBetasKernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input   = static_cast<Input *>(_in);
    Result * result = static_cast<Result *>(_res);
    Parameter * par = static_cast<Parameter *>(_par);

    NumericTable * out[] = { result->get(expectedMeans).get(), result->get(expectedVariance).get(),
                             result->get(regSS).get(),         result->get(resSS).get(),
                             result->get(tSS).get(),           result->get(determinationCoeff).get(),
                             result->get(fStatistics).get() };

    daal::services::Environment::env & env = *_env;
    __DAAL_CALL_KERNEL(env, internal::GroupOfBetasKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute,
                       input->get(expectedResponses).get(), input->get(predictedResponses).get(), input->get(predictedReducedModelResponses).get(),
                       par->numBeta, par->numBetaReducedModel, par->accuracyThreshold, out);
}

} // namespace group_of_betas
} // namespace quality_metric
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

#endif
