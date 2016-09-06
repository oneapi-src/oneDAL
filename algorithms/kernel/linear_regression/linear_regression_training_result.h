/* file: linear_regression_training_result.h */
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
//  Implementation of the linear regression algorithm interface
//--
*/

#ifndef __LINEAR_REGRESSION_TRAINING_RESULT_
#define __LINEAR_REGRESSION_TRAINING_RESULT_

#include "algorithms/linear_regression/linear_regression_training_types.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace training
{

/**
 * Allocates memory to store the result of linear regression model-based training
 * \param[in] input Pointer to an object containing the input data
 * \param[in] method Computation method for the algorithm
 * \param[in] parameter %Parameter of linear regression model-based training
 */
template<typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, const Parameter *parameter, const int method)
{
    const Input *in = static_cast<const Input *>(input);

    if(method == qrDense)
    {
        algorithmFPType dummy = 1.0;
        set(model, services::SharedPtr<daal::algorithms::linear_regression::Model>(new ModelQR(in->getNFeatures(),
                                                                                               in->getNDependentVariables(),
                                                                                               *parameter, dummy)));
    }
    else if(method == normEqDense)
    {
        algorithmFPType dummy = 1.0;
        set(model, services::SharedPtr<daal::algorithms::linear_regression::Model>(new ModelNormEq(in->getNFeatures(),
                                                                                                   in->getNDependentVariables(),
                                                                                                   *parameter, dummy)));
    }
}

/**
 * Allocates memory to store the result of linear regression model-based training
 * \param[in] partialResult Pointer to an object containing the input data
 * \param[in] method        Computation method of the algorithm
 * \param[in] parameter     %Parameter of linear regression model-based training
 */
template<typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::PartialResult *partialResult, const Parameter *parameter, const int method)
{
    const PartialResult *partialRes = static_cast<const PartialResult *>(partialResult);

    if(method == qrDense)
    {
        algorithmFPType dummy = 1.0;
        set(model, services::SharedPtr<daal::algorithms::linear_regression::Model>(new ModelQR(partialRes->getNFeatures(),
                                                                                               partialRes->getNDependentVariables(),
                                                                                               *parameter, dummy)));
    }
    else if(method == normEqDense)
    {
        algorithmFPType dummy = 1.0;
        set(model, services::SharedPtr<daal::algorithms::linear_regression::Model>(new ModelNormEq(partialRes->getNFeatures(),
                                                                                                   partialRes->getNDependentVariables(),
                                                                                                   *parameter, dummy)));
    }
}

} // namespace training
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

#endif
