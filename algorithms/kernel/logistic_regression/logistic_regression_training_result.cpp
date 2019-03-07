/* file: logistic_regression_training_result.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_LOGISTIC_REGRESSION_TRAINING_RESULT_ID);
Result::Result() : algorithms::classifier::training::Result(classifier::training::lastResultId + 1) {};

logistic_regression::ModelPtr Result::get(classifier::training::ResultId id) const
{
    return logistic_regression::Model::cast(algorithms::classifier::training::Result::get(id));
}

void Result::set(classifier::training::ResultId id, const logistic_regression::ModelPtr &value)
{
    algorithms::classifier::training::Result::set(id, value);
}

services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    return algorithms::classifier::training::Result::check(input, par, method);
}

Parameter::Parameter(size_t nClasses, const SolverPtr& solver):
    classifier::Parameter(nClasses), interceptFlag(true), penaltyL1(0.), penaltyL2(0), optimizationSolver(solver)
{
}

Status Parameter::check() const
{
    DAAL_CHECK_EX(penaltyL1 >= 0, services::ErrorIncorrectParameter, services::ParameterName, penaltyL1Str());
    DAAL_CHECK_EX(penaltyL2 >= 0, services::ErrorIncorrectParameter, services::ParameterName, penaltyL2Str());
    return services::Status();
}


} // namespace interface1
} // namespace training
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal
