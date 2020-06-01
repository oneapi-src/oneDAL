/* file: df_regression_train_container_v1.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

#ifndef __DF_REGRESSION_TRAIN_CONTAINER_V1_H__
#define __DF_REGRESSION_TRAIN_CONTAINER_V1_H__

#include "algorithms/kernel/kernel.h"
#include "algorithms/decision_forest/decision_forest_regression_training_types.h"
#include "algorithms/decision_forest/decision_forest_regression_training_batch.h"
#include "algorithms/kernel/dtrees/forest/regression/df_regression_train_kernel.h"
#include "algorithms/kernel/dtrees/forest/regression/oneapi/df_regression_train_hist_kernel_oneapi.h"
#include "algorithms/kernel/dtrees/forest/regression/df_regression_model_impl.h"
#include "service/kernel/service_algo_utils.h"

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
namespace interface1
{
DAAL_FORCEINLINE void convertParameter(interface1::Parameter & par1, interface2::Parameter & par2)
{
    par2.nTrees                      = par1.nTrees;
    par2.observationsPerTreeFraction = par1.observationsPerTreeFraction;
    par2.featuresPerNode             = par1.featuresPerNode;
    par2.maxTreeDepth                = par1.maxTreeDepth;
    par2.minObservationsInLeafNode   = par1.minObservationsInLeafNode;
    par2.seed                        = par1.seed;
    par2.engine                      = par1.engine;
    par2.impurityThreshold           = par1.impurityThreshold;
    par2.varImportance               = par1.varImportance;
    par2.resultsToCompute            = par1.resultsToCompute;
    par2.memorySavingMode            = par1.memorySavingMode;
    par2.bootstrap                   = par1.bootstrap;
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    auto & context    = services::Environment::getInstance()->getDefaultExecutionContext();
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
    auto & context    = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & deviceInfo = context.getInfoDevice();

    Input * input   = static_cast<Input *>(_in);
    Result * result = static_cast<Result *>(_res);

    const NumericTable * x = input->get(data).get();
    const NumericTable * y = input->get(dependentVariable).get();

    decision_forest::regression::Model * m = result->get(model).get();

    decision_forest::regression::training::interface1::Parameter * par =
        static_cast<decision_forest::regression::training::interface1::Parameter *>(_par);
    decision_forest::regression::training::interface2::Parameter par2;
    convertParameter(*par, par2);
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::RegressionTrainBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       daal::services::internal::hostApp(*input), x, y, *m, *result, par2);
    if (method == hist && !deviceInfo.isCpu)
    {
        __DAAL_CALL_KERNEL_SYCL(env, internal::RegressionTrainBatchKernelOneAPI, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                                daal::services::internal::hostApp(*input), x, y, *m, *result, par2);
    }
    else
    {
        __DAAL_CALL_KERNEL(env, internal::RegressionTrainBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                           daal::services::internal::hostApp(*input), x, y, *m, *result, par2);
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
} // namespace interface1
} // namespace training
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
#endif
