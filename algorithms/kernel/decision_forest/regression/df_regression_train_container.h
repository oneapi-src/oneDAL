/* file: df_regression_train_container.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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

#include "kernel.h"
#include "decision_forest_regression_training_types.h"
#include "decision_forest_regression_training_batch.h"
#include "df_regression_train_kernel.h"

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

/**
 *  \brief Initialize list of decision forest
 *  kernels with implementations for supported architectures
 */
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::RegressionTrainBatchKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

/**
 *  \brief Choose appropriate kernel to calculate decision forest model.
 *
 *  \param env[in]  Environment
 *  \param a[in]    Array of numeric tables contating input data
 *  \param r[out]   Resulting model
 *  \param par[in]  Decision forest algorithm parameters
 */
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    const NumericTable *x = input->get(data).get();
    const NumericTable *y = input->get(dependentVariable).get();

    decision_forest::regression::Model *m = result->get(model).get();

    const Parameter *par = static_cast<decision_forest::regression::training::Parameter*>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::RegressionTrainBatchKernel,
        __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, x, y, *m, *result, *par);
}

}
}
}
}
}
#endif
