/* file: logistic_regression_predict_container_v1.h */
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
//  Implementation of logistic regression algorithm container -- a class
//  that contains fast logistic regression prediction kernels
//  for supported architectures.
//--
*/

#include "algorithms/logistic_regression/logistic_regression_predict.h"
#include "../logistic_regression_predict_kernel.h"
#include "service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace prediction
{
namespace interface1
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
    Input * input                                                = static_cast<Input *>(_in);
    logistic_regression::prediction::interface1::Result * result = static_cast<logistic_regression::prediction::interface1::Result *>(_res);

    NumericTable * a               = static_cast<NumericTable *>(input->get(classifier::prediction::data).get());
    logistic_regression::Model * m = static_cast<logistic_regression::Model *>(input->get(classifier::prediction::model).get());
    const logistic_regression::prediction::interface1::Parameter * par = static_cast<logistic_regression::prediction::interface1::Parameter *>(_par);

    NumericTable * r       = ((par->resultsToCompute & computeClassesLabels) ? result->get(classifier::prediction::prediction).get() : nullptr);
    NumericTable * prob    = ((par->resultsToCompute & computeClassesProbabilities) ? result->get(probabilities).get() : nullptr);
    NumericTable * logProb = ((par->resultsToCompute & computeClassesLogProbabilities) ? result->get(logProbabilities).get() : nullptr);

    daal::services::Environment::env & env = *_env;
    __DAAL_CALL_KERNEL(env, internal::PredictKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       daal::services::internal::hostApp(*input), a, m, par->nClasses, r, prob, logProb);
}
} // namespace interface1
} // namespace prediction
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal
