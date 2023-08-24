/* file: logistic_regression_training_result.cpp */
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
//  Implementation of logistic regression algorithm classes.
//--
*/

#include "algorithms/logistic_regression/logistic_regression_training_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace training
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_LOGISTIC_REGRESSION_TRAINING_RESULT_ID);
Result::Result() : algorithms::classifier::training::Result(classifier::training::lastResultId + 1) {};

logistic_regression::ModelPtr Result::get(classifier::training::ResultId id) const
{
    return logistic_regression::Model::cast(algorithms::classifier::training::Result::get(id));
}

void Result::set(classifier::training::ResultId id, const logistic_regression::ModelPtr & value)
{
    algorithms::classifier::training::Result::set(id, value);
}

services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    return algorithms::classifier::training::Result::check(input, par, method);
}

Parameter::Parameter(size_t nClasses, const SolverPtr & solver)
    : classifier::Parameter(nClasses), interceptFlag(true), penaltyL1(0.), penaltyL2(0), optimizationSolver(solver)
{}

Status Parameter::check() const
{
    DAAL_CHECK_EX(penaltyL1 >= 0, services::ErrorIncorrectParameter, services::ParameterName, penaltyL1Str());
    DAAL_CHECK_EX(penaltyL2 >= 0, services::ErrorIncorrectParameter, services::ParameterName, penaltyL2Str());
    return services::Status();
}
} // namespace training
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal
