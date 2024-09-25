/* file: ridge_regression_training_result.h */
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
//  Implementation of the class defining the ridge regression model
//--
*/

#ifndef __RIDGE_REGRESSION_TRAINING_RESULT_H__
#define __RIDGE_REGRESSION_TRAINING_RESULT_H__

#include "algorithms/ridge_regression/ridge_regression_training_types.h"
#include "src/algorithms/ridge_regression/ridge_regression_ne_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace training
{
using namespace daal::services;

/**
 * Allocates memory to store the result of ridge regression model-based training
 * \param[in] input Pointer to an object containing the input data
 * \param[in] parameter %Parameter of ridge regression model-based training
 * \param[in] method Computation method for the algorithm
 */
template <typename algorithmFPType>
Status Result::allocate(const daal::algorithms::Input * input, const Parameter * parameter, const int method)
{
    const Input * const in = static_cast<const Input *>(input);

    Status s;
    if (method == normEqDense)
    {
        const algorithmFPType dummy = 1.0;
        set(model, ridge_regression::ModelPtr(new ridge_regression::internal::ModelNormEqImpl(
                       in->getNumberOfFeatures(), in->getNumberOfDependentVariables(), *parameter, dummy, s)));
    }

    return s;
}

// *
//  * Allocates memory to store the result of ridge regression model-based training
//  * \param[in] partialResult Pointer to an object containing the input data
//  * \param[in] method        Computation method of the algorithm
//  * \param[in] parameter     %Parameter of ridge regression model-based training

template <typename algorithmFPType>
Status Result::allocate(const daal::algorithms::PartialResult * partialResult, const Parameter * parameter, int method)
{
    const PartialResult * const partialRes = static_cast<const PartialResult *>(partialResult);

    Status s;
    if (method == normEqDense)
    {
        algorithmFPType dummy = 1.0;
        set(model, ridge_regression::ModelPtr(new ridge_regression::internal::ModelNormEqImpl(
                       partialRes->getNumberOfFeatures(), partialRes->getNumberOfDependentVariables(), *parameter, dummy, s)));
    }

    return s;
}

} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif
