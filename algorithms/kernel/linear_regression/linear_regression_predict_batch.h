/* file: linear_regression_predict_batch.h */
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

#ifndef __LINEAR_REGRESSION_PREDICT_BATCH_
#define __LINEAR_REGRESSION_PREDICT_BATCH_

#include "algorithms/linear_regression/linear_regression_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace prediction
{

/**
 * Allocates memory to store a partial result of linear regression model-based prediction
 * \param[in] input   %Input object
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Algorithm method
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
{
    size_t nVectors = (static_cast<const Input *>(input))->get(data)->getNumberOfRows();
    size_t nDependentVariables = (static_cast<const Input *>(input))->get(model)->getNumberOfResponses();

    Argument::set(prediction,
                  data_management::SerializationIfacePtr(
                      new data_management::HomogenNumericTable<algorithmFPType>
                      (nDependentVariables, nVectors, data_management::NumericTableIface::doAllocate)));
}

} // namespace prediction
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

#endif
