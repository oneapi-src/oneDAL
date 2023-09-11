/* file: decision_tree_regression_predict_batch.cpp */
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
//  Implementation of the interface for Decision tree model-based prediction
//--
*/

#include "algorithms/algorithm.h"
#include "src/services/serialization_utils.h"
#include "algorithms/decision_tree/decision_tree_regression_predict_types.h"
#include "src/algorithms/decision_tree/decision_tree_regression_model_impl.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace regression
{
namespace prediction
{
using namespace daal::data_management;
using namespace daal::services;

__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_DECISION_TREE_REGRESSION_PREDICTION_RESULT_ID);

Input::Input() : algorithms::regression::prediction::Input(lastModelInputId + 1) {}
Input::Input(const Input & other) : algorithms::regression::prediction::Input(other) {}

NumericTablePtr Input::get(NumericTableInputId id) const
{
    return algorithms::regression::prediction::Input::get(algorithms::regression::prediction::NumericTableInputId(id));
}

ModelPtr Input::get(ModelInputId id) const
{
    return staticPointerCast<decision_tree::regression::Model, data_management::SerializationIface>(Argument::get(id));
}

void Input::set(NumericTableInputId id, const data_management::NumericTablePtr & ptr)
{
    algorithms::regression::prediction::Input::set(algorithms::regression::prediction::NumericTableInputId(id), ptr);
}

void Input::set(ModelInputId id, const ModelPtr & value)
{
    algorithms::regression::prediction::Input::set(algorithms::regression::prediction::ModelInputId(id), value);
}

Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    return algorithms::regression::prediction::Input::check(parameter, method);
}

Result::Result() : algorithms::regression::prediction::Result(lastResultId + 1) {}

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
    DAAL_CHECK_EX(get(prediction)->getNumberOfColumns() == 1, ErrorIncorrectNumberOfColumns, ArgumentName, predictionStr());
    return s;
}

} // namespace prediction
} // namespace regression
} // namespace decision_tree
} // namespace algorithms
} // namespace daal
