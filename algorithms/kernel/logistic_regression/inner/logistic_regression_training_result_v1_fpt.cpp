/* file: logistic_regression_training_result_v1_fpt.cpp */
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

#include "algorithms/logistic_regression/logistic_regression_training_types.h"
#include "../logistic_regression_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace training
{
namespace interface1
{
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    services::Status s;
    const classifier::training::Input * inp              = static_cast<const classifier::training::Input *>(input);
    const size_t nFeatures                               = inp->get(classifier::training::data)->getNumberOfColumns();
    const logistic_regression::training::Parameter * prm = (const logistic_regression::training::Parameter *)parameter;
    set(classifier::training::model,
        ModelPtr(new logistic_regression::internal::ModelImpl(nFeatures, prm->interceptFlag, prm->nClasses, algorithmFPType(0), &s)));
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                    const daal::algorithms::Parameter * parameter, const int method);
} // namespace interface1

} // namespace training
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal
