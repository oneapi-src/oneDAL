/* file: ridge_regression_predict_batch.h */
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

#ifndef __RIDGE_REGRESSION_PREDICT_BATCH_
#define __RIDGE_REGRESSION_PREDICT_BATCH_

#include "algorithms/ridge_regression/ridge_regression_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace prediction
{

/**
 * Allocates memory to store a partial result of ridge regression model-based prediction
 * \param[in] input   %Input object
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Algorithm method
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method)
{
    const size_t nVectors = (static_cast<const Input *>(input))->get(data)->getNumberOfRows();
    const size_t nDependentVariables = (static_cast<const Input *>(input))->get(model)->getNumberOfResponses();

    Argument::set(prediction,
                      data_management::SerializationIfacePtr(
                      new data_management::HomogenNumericTable<algorithmFPType>
                      (nDependentVariables, nVectors, data_management::NumericTableIface::doAllocate)));
}

} // namespace prediction
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif
