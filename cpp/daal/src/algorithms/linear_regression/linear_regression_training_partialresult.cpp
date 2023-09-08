/* file: linear_regression_training_partialresult.cpp */
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
__DAAL_REGISTER_SERIALIZATION_CLASS(PartialResult, SERIALIZATION_LINEAR_REGRESSION_PARTIAL_RESULT_ID);
PartialResult::PartialResult() : linear_model::training::PartialResult(lastPartialResultID + 1) {};

/**
* Returns a partial result of linear regression model-based training
* \param[in] id    Identifier of the partial result
* \return          Partial result that corresponds to the given identifier
*/
linear_regression::ModelPtr PartialResult::get(PartialResultID id) const
{
    return linear_regression::Model::cast(Argument::get(id));
}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
size_t PartialResult::getNumberOfFeatures() const
{
    if (!get(partialModel)) return 0;
    return get(partialModel)->getNumberOfFeatures();
}

/**
* Returns the number of dependent variables
* \return Number of dependent variables
*/
size_t PartialResult::getNumberOfDependentVariables() const
{
    if (!get(partialModel)) return 0;
    return get(partialModel)->getNumberOfResponses();
}

/**
 * Sets an argument of the partial result
 * \param[in] id      Identifier of the argument
 * \param[in] value   Pointer to the argument
 */
void PartialResult::set(PartialResultID id, const linear_regression::ModelPtr & value)
{
    Argument::set(id, value);
}

/**
 * Checks a partial result of the linear regression algorithm
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 *
 * \return Status of computations
 */
services::Status PartialResult::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, ErrorIncorrectNumberOfOutputNumericTables);
    /* input object can be an instance of both Input and DistributedInput<step2Master> classes.
       Both classes have multiple inheritance with InputIface as a second base class.
       That's why we use dynamic_cast here. */
    const InputIface * in = dynamic_cast<const InputIface *>(input);
    DAAL_CHECK(in, ErrorNullInput);

    size_t nBeta      = in->getNumberOfFeatures() + 1;
    size_t nResponses = in->getNumberOfDependentVariables();

    linear_regression::ModelPtr partialModel = get(training::partialModel);

    return linear_regression::checkModel(partialModel.get(), *par, nBeta, nResponses, method);
}

/**
 * Checks a partial result of the linear regression algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 *
 * \return Status of computations
 */
services::Status PartialResult::check(const daal::algorithms::Parameter * par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, ErrorIncorrectNumberOfOutputNumericTables);

    linear_regression::ModelPtr partialModel = get(training::partialModel);
    DAAL_CHECK(partialModel, Error::create(ErrorNullPartialModel));

    size_t nBeta      = partialModel->getNumberOfBetas();
    size_t nResponses = partialModel->getNumberOfResponses();

    return linear_regression::checkModel(partialModel.get(), *par, nBeta, nResponses, method);
}

} // namespace training
} // namespace linear_regression
} // namespace algorithms
} // namespace daal
