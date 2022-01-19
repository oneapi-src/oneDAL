/* file: df_regression_train_container.h */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
//  Implementation of decision forest container.
//--
*/

#ifndef __DF_REGRESSION_TRAIN_CONTAINER_H__
#define __DF_REGRESSION_TRAIN_CONTAINER_H__

#include "src/algorithms/kernel.h"
#include "algorithms/decision_forest/decision_forest_regression_training_types.h"
#include "algorithms/decision_forest/decision_forest_regression_training_batch.h"
#include "src/algorithms/dtrees/forest/regression/df_regression_train_kernel.h"
#include "src/algorithms/dtrees/forest/regression/oneapi/df_regression_train_hist_kernel_oneapi.h"
#include "src/algorithms/dtrees/forest/regression/df_regression_model_impl.h"
#include "src/services/service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
{
namespace training
{
namespace interface2
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    auto & context    = services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    if (method == hist && !deviceInfo.isCpu)
    {
        __DAAL_INITIALIZE_KERNELS_SYCL(internal::RegressionTrainBatchKernelOneAPI, algorithmFPType, method);
    }
    else
    {
        __DAAL_INITIALIZE_KERNELS(internal::RegressionTrainBatchKernel, algorithmFPType, method);
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    auto & context    = services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    Input * input   = static_cast<Input *>(_in);
    Result * result = static_cast<Result *>(_res);

    const NumericTable * const x = input->get(data).get();
    const NumericTable * const y = input->get(dependentVariable).get();
    const NumericTable * const w = input->get(weights).get();

    decision_forest::regression::Model * m = result->get(model).get();

    const decision_forest::regression::training::Parameter * par = static_cast<decision_forest::regression::training::Parameter *>(_par);
    daal::services::Environment::env & env                       = *_env;

    if (method == hist && !deviceInfo.isCpu)
    {
        __DAAL_CALL_KERNEL_SYCL(env, internal::RegressionTrainBatchKernelOneAPI, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                                daal::services::internal::hostApp(*input), x, y, *m, *result, *par);
    }
    else
    {
        __DAAL_CALL_KERNEL(env, internal::RegressionTrainBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                           daal::services::internal::hostApp(*input), x, y, w, *m, *result, *par);
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::setupCompute()
{
    Result * result                                          = static_cast<Result *>(_res);
    decision_forest::regression::Model * m                   = result->get(model).get();
    decision_forest::regression::internal::ModelImpl * pImpl = dynamic_cast<decision_forest::regression::internal::ModelImpl *>(m);
    DAAL_ASSERT(pImpl);
    pImpl->clear();
    return services::Status();
}
} // namespace interface2
} // namespace training
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
#endif
