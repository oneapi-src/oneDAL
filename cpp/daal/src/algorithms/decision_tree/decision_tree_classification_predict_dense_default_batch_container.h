/* file: decision_tree_classification_predict_dense_default_batch_container.h */
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
//  Implementation of Decision tree algorithm container - a class that contains fast Decision tree prediction kernels for supported
//  architectures.
//--
*/

#include "algorithms/decision_tree/decision_tree_classification_predict.h"
#include "src/algorithms/decision_tree/decision_tree_classification_predict_dense_default_batch.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace classification
{
namespace prediction
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv) : PredictionContainerIface()
{
    __DAAL_INITIALIZE_KERNELS(internal::DecisionTreePredictKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    const classifier::prediction::Input * const input = static_cast<const classifier::prediction::Input *>(_in);
    classifier::prediction::Result * const result     = static_cast<classifier::prediction::Result *>(_res);
    classifier::Parameter * const parameter           = static_cast<classifier::Parameter *>(_par);

    const data_management::NumericTableConstPtr a = input->get(classifier::prediction::data);
    const classifier::ModelConstPtr m             = input->get(classifier::prediction::model);
    const data_management::NumericTablePtr r      = result->get(classifier::prediction::prediction);
    data_management::NumericTablePtr prob; // Used to prevent shared pointer release
    data_management::NumericTable * const p = ((parameter->resultsToEvaluate & classifier::computeClassProbabilities) != 0) ?
                                                  (prob = result->get(classifier::prediction::probabilities)).get() :
                                                  nullptr;

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::DecisionTreePredictKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, a.get(), m.get(), r.get(),
                       p, parameter->nClasses);
}
} // namespace prediction
} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal
