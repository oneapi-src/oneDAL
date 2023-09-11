/* file: decision_tree_regression_train_container.h */
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
//  Implementation of K-Nearest Neighbors container.
//--
*/

#ifndef __DECISION_TREE_REGRESSION_TRAIN_CONTAINER_H__
#define __DECISION_TREE_REGRESSION_TRAIN_CONTAINER_H__

#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_shared_ptr.h"
#include "algorithms/decision_tree/decision_tree_regression_training_batch.h"
#include "src/algorithms/decision_tree/decision_tree_regression_train_kernel.h"
#include "src/algorithms/decision_tree/decision_tree_regression_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace regression
{
namespace training
{
using namespace daal::data_management;

/**
 *  \brief Initialize list of Decision tree kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::DecisionTreeTrainBatchKernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

/**
 *  \brief Choose appropriate kernel to calculate Decision tree model.
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    const decision_tree::regression::training::Input * const input = static_cast<decision_tree::regression::training::Input *>(_in);
    Result * const result                                          = static_cast<Result *>(_res);

    const NumericTableConstPtr x  = input->get(data);
    const NumericTableConstPtr y  = input->get(dependentVariables);
    const NumericTableConstPtr w  = input->get(weights);
    const NumericTableConstPtr px = input->get(dataForPruning);
    const NumericTableConstPtr py = input->get(dependentVariablesForPruning);

    const ModelPtr r = result->get(model);

    const daal::algorithms::Parameter * const par = _par;
    daal::services::Environment::env & env        = *_env;

    __DAAL_CALL_KERNEL(env, internal::DecisionTreeTrainBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, x.get(), y.get(),
                       w.get(), px.get(), py.get(), r.get(), par);
}
} // namespace training
} // namespace regression
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
