/* file: df_classification_predict_dense_default_batch_container.h */
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
//  Implementation of decision forest algorithm container -- a class
//  that contains fast decision forest prediction kernels
//  for supported architectures.
//--
*/

#include "algorithms/decision_forest/decision_forest_classification_predict.h"
#include "src/algorithms/dtrees/forest/classification/df_classification_predict_dense_default_batch.h"
#include "src/services/service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace prediction
{
namespace interface3
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv) : PredictionContainerIface()
{
    __DAAL_INITIALIZE_KERNELS(internal::PredictKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    const Input * const input                     = static_cast<Input *>(_in);
    classifier::prediction::Result * const result = static_cast<classifier::prediction::Result *>(_res);
    const decision_forest::classification::prediction::Parameter * const par =
        dynamic_cast<decision_forest::classification::prediction::Parameter *>(_par);
    if (par == NULL)
    {
        return services::Status(services::ErrorNullResult);
    }
    const decision_forest::classification::Model * const m =
        static_cast<decision_forest::classification::Model *>(input->get(classifier::prediction::model).get());

    const NumericTable * const a = static_cast<NumericTable *>(input->get(classifier::prediction::data).get());
    NumericTable * const r =
        ((par->resultsToEvaluate & classifier::ResultToComputeId::computeClassLabels) ? result->get(classifier::prediction::prediction).get() :
                                                                                        nullptr);
    NumericTable * const prob = ((par->resultsToEvaluate & classifier::ResultToComputeId::computeClassProbabilities) ?
                                     result->get(classifier::prediction::probabilities).get() :
                                     nullptr);

    const daal::services::Environment::env & env = *_env;

    const VotingMethod votingMethod = par->votingMethod;

    __DAAL_CALL_KERNEL(env, internal::PredictKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       daal::services::internal::hostApp(*const_cast<Input *>(input)), a, m, r, prob, par->nClasses, votingMethod);
}
} // namespace interface3
} // namespace prediction
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
