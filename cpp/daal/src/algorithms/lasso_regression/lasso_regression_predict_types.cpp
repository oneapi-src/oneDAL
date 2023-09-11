/* file: lasso_regression_predict_types.cpp */
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
//  Implementation of lasso regression algorithm classes.
//--
*/

#include "algorithms/lasso_regression/lasso_regression_predict_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"
#include "src/algorithms/lasso_regression/lasso_regression_model_impl.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace lasso_regression
{
namespace prediction
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_LASSO_REGRESSION_PREDICTION_RESULT_ID);

/** Default constructor */
Input::Input() : linear_model::prediction::Input(lastModelInputId + 1) {}
Input::Input(const Input & other) : linear_model::prediction::Input(other) {}

/**
 * Returns an input object for making lasso regression model-based prediction
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(NumericTableInputId id) const
{
    return linear_model::prediction::Input::get(linear_model::prediction::NumericTableInputId(id));
}

/**
 * Returns an input object for making lasso regression model-based prediction
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
lasso_regression::ModelPtr Input::get(ModelInputId id) const
{
    return lasso_regression::Model::cast(linear_model::prediction::Input::get(linear_model::prediction::ModelInputId(id)));
}

/**
 * Sets an input object for making lasso regression model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Input::set(NumericTableInputId id, const NumericTablePtr & value)
{
    linear_model::prediction::Input::set(linear_model::prediction::NumericTableInputId(id), value);
}

/**
 * Sets an input object for making lasso regression model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Input::set(ModelInputId id, const lasso_regression::ModelPtr & value)
{
    linear_model::prediction::Input::set(linear_model::prediction::ModelInputId(id), value);
}

Result::Result() : linear_model::prediction::Result(lastResultId + 1) {}

/**
 * Returns the result of lasso regression model-based prediction
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return linear_model::prediction::Result::get(linear_model::prediction::ResultId(id));
}

/**
 * Sets the result of lasso regression model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Result::set(ResultId id, const NumericTablePtr & value)
{
    linear_model::prediction::Result::set(linear_model::prediction::ResultId(id), value);
}

} // namespace prediction
} // namespace lasso_regression
} // namespace algorithms
} // namespace daal
