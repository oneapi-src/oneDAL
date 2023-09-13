/* file: classifier_predict_v1.cpp */
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
//  Implementation of classifier prediction methods.
//--
*/

#include "algorithms/classifier/classifier_predict_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace classifier
{
namespace prediction
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_CLASSIFIER_PREDICTION_RESULT_ID);
Result::Result() : daal::algorithms::Result(lastResultId + 1) {}
Result::Result(size_t n) : daal::algorithms::Result(n) {}

/**
 * Returns the prediction result of the classification algorithm
 * \param[in] id   Identifier of the prediction result, \ref ResultId
 * \return         Prediction result that corresponds to the given identifier
 */
data_management::NumericTablePtr Result::get(ResultId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the prediction result of the classification algorithm
 * \param[in] id    Identifier of the prediction result, \ref ResultId
 * \param[in] value Pointer to the prediction result
 */
void Result::set(ResultId id, const data_management::NumericTablePtr & value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of the Result object
 * \param[in] input     Pointer to the the input object
 * \param[in] parameter Pointer to the algorithm parameters
 * \param[in] method    Computation method
 */
services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    return checkImpl(input, parameter);
}

services::Status Result::checkImpl(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter) const
{
    size_t nRows                              = (static_cast<const InputIface *>(input))->getNumberOfRows();
    data_management::NumericTablePtr resTable = get(prediction);

    return data_management::checkNumericTable(resTable.get(), predictionStr(), data_management::packed_mask, 0, 1, nRows);
}
} // namespace interface1
} // namespace prediction
} // namespace classifier
} // namespace algorithms
} // namespace daal
