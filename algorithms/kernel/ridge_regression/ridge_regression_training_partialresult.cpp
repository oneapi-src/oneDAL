/* file: ridge_regression_training_partialresult.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of ridge regression algorithm classes.
//--
*/

#include "algorithms/ridge_regression/ridge_regression_training_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace training
{
namespace interface1
{

PartialResult::PartialResult() : daal::algorithms::PartialResult(1) {};

/**
* Returns a partial result of ridge regression model-based training
* \param[in] id    Identifier of the partial result
* \return          Partial result that corresponds to the given identifier
*/
SharedPtr<daal::algorithms::ridge_regression::Model> PartialResult::get(PartialResultID id) const
{
    return staticPointerCast<daal::algorithms::ridge_regression::Model, SerializationIface>(Argument::get(id));
}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
size_t PartialResult::getNFeatures() const { return get(partialModel)->getNumberOfFeatures(); }

/**
* Returns the number of dependent variables
* \return Number of dependent variables
*/
size_t PartialResult::getNDependentVariables() const { return get(partialModel)->getNumberOfResponses(); }

/**
 * Sets an argument of the partial result
 * \param[in] id      Identifier of the argument
 * \param[in] value   Pointer to the argument
 */
void PartialResult::set(PartialResultID id, const SharedPtr<daal::algorithms::ridge_regression::Model> &value)
{
    Argument::set(id, value);
}

/**
 * Checks a partial result of the ridge regression algorithm
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
void PartialResult::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, ErrorIncorrectNumberOfOutputNumericTables);
    const InputIface *in = static_cast<const InputIface *>(input);

    SharedPtr<ridge_regression::Model> partialModel = get(training::partialModel);

    size_t coefdim = in->getNFeatures() + 1;
    size_t nrhs = in->getNDependentVariables();

    checkModel(partialModel.get(), par, this->_errors.get(), coefdim, nrhs, method);
}

/**
 * Checks a partial result of the ridge regression algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
void PartialResult::check(const daal::algorithms::Parameter * par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, ErrorIncorrectNumberOfOutputNumericTables);

    SharedPtr<ridge_regression::Model> partialModel = get(training::partialModel);
    DAAL_CHECK(partialModel, ErrorNullPartialModel);

    size_t coefdim = partialModel->getNumberOfBetas();
    size_t nrhs = partialModel->getNumberOfResponses();

    checkModel(partialModel.get(), par, this->_errors.get(), coefdim, nrhs, method);
}

} // namespace interface1
} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal
