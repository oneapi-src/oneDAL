/* file: linear_regression_training_partialresult.h */
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

#ifndef __LINEAR_REGRESSION_TRAINING_PARTIALRESULT_
#define __LINEAR_REGRESSION_TRAINING_PARTIALRESULT_

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
 * Allocates memory to store a partial result of linear regression model-based training
 * \param[in] input %Input object for the algorithm
 * \param[in] method Method of linear regression model-based training
 * \param[in] parameter %Parameter of linear regression model-based training
 */
template <typename algorithmFPType>
DAAL_EXPORT void PartialResult::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    if(method == qrDense)
    {
        algorithmFPType dummy = 1.0;
        set(partialModel, services::SharedPtr<daal::algorithms::linear_regression::Model>(
                new ModelQR((static_cast<const InputIface *>(input))->getNFeatures(),
                            (static_cast<const InputIface *>(input))->getNDependentVariables(),
                            *(static_cast<const Parameter *>(parameter)), dummy)));
    }
    else if(method == normEqDense)
    {
        algorithmFPType dummy = 1.0;
        set(partialModel, services::SharedPtr<daal::algorithms::linear_regression::Model>(
                new ModelNormEq((static_cast<const InputIface *>(input))->getNFeatures(),
                                (static_cast<const InputIface *>(input))->getNDependentVariables(),
                                *(static_cast<const Parameter *>(parameter)), dummy)));
    }
}

} // namespace training
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

#endif
