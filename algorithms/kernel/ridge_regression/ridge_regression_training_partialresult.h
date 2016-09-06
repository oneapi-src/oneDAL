/* file: ridge_regression_training_partialresult.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of the class defining the ridge regression model
//--
*/

#ifndef __RIDGE_REGRESSION_TRAINING_PARTIALRESULT_
#define __RIDGE_REGRESSION_TRAINING_PARTIALRESULT_

#include "algorithms/ridge_regression/ridge_regression_training_types.h"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace training
{

/**
 * Allocates memory to store a partial result of ridge regression model-based training
 * \param[in] input %Input object for the algorithm
 * \param[in] method Method of ridge regression model-based training
 * \param[in] parameter %Parameter of ridge regression model-based training
 */
template <typename algorithmFPType>
DAAL_EXPORT void PartialResult::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    if (method == normEqDense)
    {
        const algorithmFPType dummy = 1.0;
        set(partialModel, services::SharedPtr<daal::algorithms::ridge_regression::Model>(
                new ModelNormEq((static_cast<const InputIface *>(input))->getNFeatures(),
                                (static_cast<const InputIface *>(input))->getNDependentVariables(),
                                *(static_cast<const Parameter *>(parameter)), dummy)));
    }
}

} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif
