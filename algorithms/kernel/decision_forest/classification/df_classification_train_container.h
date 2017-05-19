/* file: df_classification_train_container.h */
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

#ifndef __DF_CLASSIFICATION_TRAIN_CONTAINER_H__
#define __DF_CLASSIFICATION_TRAIN_CONTAINER_H__

#include "kernel.h"
#include "decision_forest_classification_training_types.h"
#include "decision_forest_classification_training_batch.h"
#include "df_classification_train_kernel.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
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
    __DAAL_INITIALIZE_KERNELS(internal::ClassificationTrainBatchKernel, algorithmFPType, method);
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
 *  \param par[in]  decision forest algorithm parameters
 */
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    classifier::training::Input *input = static_cast<classifier::training::Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    NumericTable *x = input->get(classifier::training::data).get();
    NumericTable *y = input->get(classifier::training::labels).get();

    decision_forest::classification::Model *m = result->get(classifier::training::model).get();

    const decision_forest::classification::training::Parameter *par =
        static_cast<decision_forest::classification::training::Parameter*>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::ClassificationTrainBatchKernel,
        __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, x, y, *m, *result, *par);
}

}
}
}
}
}

#endif
