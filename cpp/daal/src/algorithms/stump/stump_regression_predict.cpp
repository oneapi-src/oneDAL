/* file: stump_regression_predict.cpp */
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
//  Implementation of stump algorithm and types methods.
//--
*/

#include "algorithms/algorithm.h"
#include "src/services/serialization_utils.h"
#include "algorithms/stump/stump_regression_predict_types.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace regression
{
namespace prediction
{
using namespace daal::data_management;
using namespace daal::services;

__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_STUMP_REGRESSION_PREDICTION_RESULT_ID);

Input::Input() : algorithms::regression::prediction::Input(algorithms::regression::prediction::lastModelInputId + 1) {}
Input::Input(const Input & other) : daal::algorithms::regression::prediction::Input(other) {}

/**
 * Returns the input Numeric Table object in the prediction stage of the regression algorithm
 * \param[in] id    Identifier of the input NumericTable object
 * \return          %Input object that corresponds to the given identifier
 */
data_management::NumericTablePtr Input::get(daal::algorithms::regression::prediction::NumericTableInputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Returns the input Model object in the prediction stage of the Stump algorithm
 * \param[in] id    Identifier of the input Model object
 * \return          %Input object that corresponds to the given identifier
 */
stump::regression::ModelPtr Input::get(daal::algorithms::regression::prediction::ModelInputId id) const
{
    return services::staticPointerCast<stump::regression::Model, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the input NumericTable object in the prediction stage of the regression algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(daal::algorithms::regression::prediction::NumericTableInputId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets the input Model object in the prediction stage of the Stump algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(daal::algorithms::regression::prediction::ModelInputId id, const stump::regression::ModelPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the correctness of the input object
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
services::Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, daal::algorithms::regression::prediction::Input::check(parameter, method));

    stump::regression::ModelPtr m  = get(daal::algorithms::regression::prediction::model);
    const size_t modelSplitFeature = m->getSplitFeature();
    DAAL_CHECK(modelSplitFeature < get(daal::algorithms::regression::prediction::data)->getNumberOfColumns(),
               services::ErrorStumpIncorrectSplitFeature);
    return services::Status();
}

Result::Result() : algorithms::regression::prediction::Result(algorithms::regression::prediction::lastResultId + 1) {}

NumericTablePtr Result::get(ResultId id) const
{
    return algorithms::regression::prediction::Result::get(algorithms::regression::prediction::ResultId(id));
}

void Result::set(ResultId id, const NumericTablePtr & value)
{
    algorithms::regression::prediction::Result::set(algorithms::regression::prediction::ResultId(id), value);
}
Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, algorithms::regression::prediction::Result::check(input, par, method));
    DAAL_CHECK_EX(get(prediction)->getNumberOfColumns() == 1, services::ErrorIncorrectNumberOfColumns, services::ArgumentName, predictionStr());
    return s;
}

} // namespace prediction
} // namespace regression
} // namespace stump
} // namespace algorithms
} // namespace daal
