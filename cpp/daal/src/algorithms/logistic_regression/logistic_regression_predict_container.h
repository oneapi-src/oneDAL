/* file: logistic_regression_predict_container.h */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
#include "algorithms/classifier/classifier_model.h"
#include "src/services/service_algo_utils.h"
#include "src/algorithms/logistic_regression/logistic_regression_predict_kernel.h"
#include "src/algorithms/logistic_regression/oneapi/logistic_regression_predict_kernel_oneapi.h"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace prediction
{
namespace interface2
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv) : PredictionContainerIface()
{
    auto & context    = services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    if (deviceInfo.isCpu)
    {
        __DAAL_INITIALIZE_KERNELS(internal::PredictKernel, algorithmFPType, method);
    }
    else
    {
        __DAAL_INITIALIZE_KERNELS_SYCL(internal::PredictBatchKernelOneAPI, algorithmFPType, method);
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
    Input * input                           = static_cast<Input *>(_in);
    classifier::prediction::Result * result = static_cast<classifier::prediction::Result *>(_res);

    NumericTable * a                  = static_cast<NumericTable *>(input->get(classifier::prediction::data).get());
    logistic_regression::Model * m    = static_cast<logistic_regression::Model *>(input->get(classifier::prediction::model).get());
    const classifier::Parameter * par = static_cast<classifier::Parameter *>(_par);

    NumericTable * r = ((par->resultsToEvaluate & classifier::computeClassLabels) ? result->get(classifier::prediction::prediction).get() : nullptr);
    NumericTable * prob =
        ((par->resultsToEvaluate & classifier::computeClassProbabilities) ? result->get(classifier::prediction::probabilities).get() : nullptr);
    NumericTable * logProb =
        ((par->resultsToEvaluate & classifier::computeClassLogProbabilities) ? result->get(classifier::prediction::logProbabilities).get() : nullptr);

    daal::services::Environment::env & env = *_env;

    auto & context    = services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    if (deviceInfo.isCpu)
    {
        __DAAL_CALL_KERNEL(env, internal::PredictKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                           daal::services::internal::hostApp(*input), a, m, par->nClasses, r, prob, logProb);
    }
    else
    {
        __DAAL_CALL_KERNEL_SYCL(env, internal::PredictBatchKernelOneAPI, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                                daal::services::internal::hostApp(*input), a, m, par->nClasses, r, prob, logProb);
    }
}

} // namespace interface2

} // namespace prediction
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal
