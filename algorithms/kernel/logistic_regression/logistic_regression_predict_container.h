/* file: logistic_regression_predict_container.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of logistic regression algorithm container -- a class
//  that contains fast logistic regression prediction kernels
//  for supported architectures.
//--
*/

#include "algorithms/logistic_regression/logistic_regression_predict.h"
#include "logistic_regression_predict_kernel.h"
#include "service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace prediction
{

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv) : PredictionContainerIface()
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
    Input *input = static_cast<Input *>(_in);
    logistic_regression::prediction::Result *result = static_cast<logistic_regression::prediction::Result *>(_res);

    NumericTable *a = static_cast<NumericTable *>(input->get(classifier::prediction::data).get());
    logistic_regression::Model *m = static_cast<logistic_regression::Model *>(input->get(classifier::prediction::model).get());
    const logistic_regression::prediction::Parameter *par = static_cast<logistic_regression::prediction::Parameter*>(_par);

    NumericTable *r = ((par->resultsToCompute & computeClassesLabels) ? result->get(classifier::prediction::prediction).get() : nullptr);
    NumericTable *prob = ((par->resultsToCompute & computeClassesProbabilities) ? result->get(probabilities).get() : nullptr);
    NumericTable *logProb = ((par->resultsToCompute & computeClassesLogProbabilities) ? result->get(logProbabilities).get() : nullptr);

    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::PredictKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
        daal::services::internal::hostApp(*input), a, m, par->nClasses, r, prob, logProb);
}

}
}
}
} // namespace daal
