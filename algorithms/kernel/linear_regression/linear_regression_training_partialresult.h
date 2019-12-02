/* file: linear_regression_training_partialresult.h */
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
//  Implementation of the linear regression algorithm interface
//--
*/

#ifndef __LINEAR_REGRESSION_TRAINING_PARTIALRESULT_
#define __LINEAR_REGRESSION_TRAINING_PARTIALRESULT_

#include "algorithms/linear_regression/linear_regression_training_types.h"
#include "linear_regression_ne_model_impl.h"
#include "linear_regression_qr_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace training
{
/**
 * Allocates memory to store a partial result of linear regression model-based training
 * \param[in] input %Input object for the algorithm
 * \param[in] method Method of linear regression model-based training
 * \param[in] parameter %Parameter of linear regression model-based training
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status PartialResult::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                                     const int method)
{
    /* input object can be an instance of both Input and DistributedInput<step2Master> classes.
       Both classes have multiple inheritance with InputIface as a second base class.
       That's why we use dynamic_cast here. */
    const InputIface * const in = dynamic_cast<const InputIface *>(input);
    const Parameter & par       = *(static_cast<const Parameter *>(parameter));
    const algorithmFPType dummy = 1.0;
    services::Status s;
    if (method == qrDense)
    {
        set(partialModel, linear_regression::ModelPtr(new linear_regression::internal::ModelQRImpl(
                              in->getNumberOfFeatures(), in->getNumberOfDependentVariables(), par, dummy, s)));
    }
    else if (method == normEqDense)
    {
        set(partialModel, linear_regression::ModelPtr(new linear_regression::internal::ModelNormEqImpl(
                              in->getNumberOfFeatures(), in->getNumberOfDependentVariables(), par, dummy, s)));
    }

    return s;
}

template <typename algorithmFPType>
DAAL_EXPORT services::Status PartialResult::initialize(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                                       const int method)
{
    return get(partialModel)->initialize();
}

} // namespace training
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

#endif
