/* file: df_regression_predict_result.h */
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
//  Implementation of the decision forest algorithm interface
//--
*/

#ifndef __DF_REGRESSION_PREDICT_RESULT_H_
#define __DF_REGRESSION_PREDICT_RESULT_H_

#include "algorithms/decision_forest/decision_forest_regression_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
{
namespace prediction
{
/**
 * Allocates memory to store a partial result of decision forest model-based prediction
 * \param[in] input   %Input object
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Algorithm method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method)
{
    size_t nVectors = (static_cast<const Input *>(input))->get(data)->getNumberOfRows();
    services::Status st;
    set(prediction, data_management::HomogenNumericTable<algorithmFPType>::create(1, nVectors, data_management::NumericTableIface::doAllocate, &st));
    return st;
}

} // namespace prediction
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
