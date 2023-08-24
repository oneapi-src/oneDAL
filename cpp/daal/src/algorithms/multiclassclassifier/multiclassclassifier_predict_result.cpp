/* file: multiclassclassifier_predict_result.cpp */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  Implementation of multiclass prediction Result.
//--
*/

#include "algorithms/multi_class_classifier/multi_class_classifier_predict_types.h"
#include "src/services/daal_strings.h"
#include "src/services/serialization_utils.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace prediction
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_MULTICLASS_PREDICTION_RESULT_ID);

Result::Result() : classifier::prediction::Result(lastResultId + 1) {}

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
    services::Status s;
    DAAL_CHECK_STATUS_VAR(s);

    const size_t nRows          = (static_cast<const classifier::prediction::InputIface *>(input))->getNumberOfRows();
    const Parameter * const par = static_cast<const Parameter *>(parameter);
    DAAL_CHECK(par, services::ErrorNullParameterNotSupported);

    const size_t nClasses = par->nClasses;

    if (par->resultsToEvaluate & computeClassLabels)
        DAAL_CHECK_STATUS(s, data_management::checkNumericTable(get(prediction).get(), predictionStr(), data_management::packed_mask, 0, 1, nRows));

    if (par->resultsToEvaluate & computeDecisionFunction)
        DAAL_CHECK_STATUS(s, data_management::checkNumericTable(get(decisionFunction).get(), decisionFunctionStr(), data_management::packed_mask, 0,
                                                                nClasses * (nClasses - 1) / 2, nRows));

    return s;
}

} // namespace prediction
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal
