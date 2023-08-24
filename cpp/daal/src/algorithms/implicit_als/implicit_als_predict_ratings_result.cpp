/* file: implicit_als_predict_ratings_result.cpp */
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
//  Implementation of implicit als algorithm and types methods.
//--
*/

#include "algorithms/implicit_als/implicit_als_predict_ratings_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace prediction
{
namespace ratings
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_IMPLICIT_ALS_PREDICTION_RATINGS_RESULT_ID);
Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

/**
 * Returns the prediction result of the implicit ALS algorithm
 * \param[in] id   Identifier of the prediction result, \ref ResultId
 * \return         Prediction result that corresponds to the given identifier
 */
data_management::NumericTablePtr Result::get(ResultId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the prediction result of the implicit ALS algorithm
 * \param[in] id    Identifier of the prediction result, \ref ResultId
 * \param[in] ptr   Pointer to the prediction result
 */
void Result::set(ResultId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the result of the rating prediction stage of the implicit ALS algorithm
 * \param[in] input       %Input object for the algorithm
 * \param[in] parameter   %Parameter of the algorithm
 * \param[in] method      Computation method of the algorithm
 */
services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    const InputIface * algInput = static_cast<const InputIface *>(input);
    const size_t nUsers         = algInput->getNumberOfUsers();
    const size_t nItems         = algInput->getNumberOfItems();

    const int unexpectedLayouts = (int)packed_mask;
    return checkNumericTable(get(prediction).get(), predictionStr(), unexpectedLayouts, 0, nItems, nUsers);
}

} // namespace ratings
} // namespace prediction
} // namespace implicit_als
} // namespace algorithms
} // namespace daal
