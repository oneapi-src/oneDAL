/* file: logistic_regression_predict_result_fpt_v1.cpp */
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
//  Implementation of the logistic regression algorithm interface
//--
*/

#include "algorithms/logistic_regression/logistic_regression_predict_types.h"

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
template <typename algorithmFPType>
DAAL_EXPORT services::Status interface1::Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                                          const int method)
{
    services::Status s;
    const logistic_regression::prediction::interface1::Parameter * prm = (const logistic_regression::prediction::interface1::Parameter *)parameter;
    const logistic_regression::prediction::Input * inp                 = static_cast<const logistic_regression::prediction::Input *>(input);
    const size_t nProb                                                 = (prm->nClasses == 2 ? 1 : prm->nClasses);
    if (prm->resultsToCompute & computeClassesLabels)
        s = classifier::prediction::interface1::Result::allocate<algorithmFPType>(input, parameter, method);
    if (s.ok() && (prm->resultsToCompute & computeClassesProbabilities))
        set(probabilities, data_management::HomogenNumericTable<algorithmFPType>::create(nProb, inp->getNumberOfRows(),
                                                                                         data_management::NumericTableIface::doAllocate, &s));
    if (s.ok() && (prm->resultsToCompute & computeClassesLogProbabilities))
        set(logProbabilities, data_management::HomogenNumericTable<algorithmFPType>::create(nProb, inp->getNumberOfRows(),
                                                                                            data_management::NumericTableIface::doAllocate, &s));
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                    const daal::algorithms::Parameter * parameter, const int method);

} // namespace interface1

} // namespace prediction
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal
