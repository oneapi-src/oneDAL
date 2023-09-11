/* file: linear_regression_training_result.cpp */
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
//  Implementation of linear regression algorithm classes.
//--
*/

#include "algorithms/linear_regression/linear_regression_training_types.h"
#include "src/services/serialization_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace training
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_LINEAR_REGRESSION_TRAINING_RESULT_ID);
Result::Result() : linear_model::training::Result(lastResultId + 1) {}

/**
 * Returns the result of linear regression model-based training
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
linear_regression::ModelPtr Result::get(ResultId id) const
{
    return linear_regression::Model::cast(linear_model::training::Result::get(linear_model::training::ResultId(id)));
}

/**
 * Sets the result of linear regression model-based training
 * \param[in] id      Identifier of the result
 * \param[in] value   Result
 */
void Result::set(ResultId id, const linear_regression::ModelPtr & value)
{
    linear_model::training::Result::set(linear_model::training::ResultId(id), value);
}

/**
 * Checks the result of linear regression model-based training
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 *
 * \return Status of computations
 */
services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, linear_model::training::Result::check(input, par, method));

    /* input object can be an instance of both Input and DistributedInput<step2Master> classes.
       Both classes have multiple inheritance with InputIface as a second base class.
       That's why we use dynamic_cast here. */
    const InputIface * in = dynamic_cast<const InputIface *>(input);
    DAAL_CHECK(in, ErrorNullInput);

    const size_t nBeta      = in->getNumberOfFeatures() + 1;
    const size_t nResponses = in->getNumberOfDependentVariables();

    const linear_regression::ModelPtr model = get(training::model);

    return linear_regression::checkModel(model.get(), *par, nBeta, nResponses, method);
}

/**
 * Checks the result of the linear regression model-based training
 * \param[in] pr      %PartialResult of the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 *
 * \return Status of computations
 */
services::Status Result::check(const daal::algorithms::PartialResult * pr, const daal::algorithms::Parameter * par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, ErrorIncorrectNumberOfOutputNumericTables);
    const PartialResult * partRes = static_cast<const PartialResult *>(pr);

    size_t nBeta      = partRes->getNumberOfFeatures() + 1;
    size_t nResponses = partRes->getNumberOfDependentVariables();

    linear_regression::ModelPtr model = get(training::model);

    return linear_regression::checkModel(model.get(), *par, nBeta, nResponses, method);
}

} // namespace training
} // namespace linear_regression
} // namespace algorithms
} // namespace daal
