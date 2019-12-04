/* file: logistic_regression_training_result_v2.cpp */
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
//  Implementation of logistic regression algorithm classes.
//--
*/

#include "algorithms/logistic_regression/logistic_regression_training_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

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
namespace interface2
{
Parameter::Parameter(size_t nClasses, const SolverPtr & solver)
    : classifier::interface1::Parameter(nClasses), interceptFlag(true), penaltyL1(0.), penaltyL2(0), optimizationSolver(solver)
{}

Status Parameter::check() const
{
    DAAL_CHECK_EX(penaltyL1 >= 0, services::ErrorIncorrectParameter, services::ParameterName, penaltyL1Str());
    DAAL_CHECK_EX(penaltyL2 >= 0, services::ErrorIncorrectParameter, services::ParameterName, penaltyL2Str());
    return services::Status();
}
} // namespace interface2

} // namespace training
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal
