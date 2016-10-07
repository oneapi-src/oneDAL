/* file: ridge_regression_training_result.cpp */
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

Result::Result() : daal::algorithms::Result(1) {}

/**
 * Returns the result of ridge regression model-based training
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
SharedPtr<daal::algorithms::ridge_regression::Model> Result::get(ResultId id) const
{
    return staticPointerCast<daal::algorithms::ridge_regression::Model, SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of ridge regression model-based training
 * \param[in] id      Identifier of the result
 * \param[in] value   Result
 */
void Result::set(ResultId id, const SharedPtr<daal::algorithms::ridge_regression::Model> & value)
{
    Argument::set(id, value);
}

/**
 * Checks the result of ridge regression model-based training
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
void Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, ErrorIncorrectNumberOfOutputNumericTables);
    const InputIface *in = static_cast<const InputIface *>(input);

    SharedPtr<ridge_regression::Model> model = get(training::model);

    size_t coefdim = in->getNFeatures() + 1;
    size_t nrhs = in->getNDependentVariables();

    checkModel(model.get(), par, this->_errors.get(), coefdim, nrhs, method);
}

/**
 * Checks the result of the ridge regression model-based training
 * \param[in] pr      %PartialResult of the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
void Result::check(const daal::algorithms::PartialResult * pr, const daal::algorithms::Parameter * par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, ErrorIncorrectNumberOfOutputNumericTables);
    const PartialResult *partRes = static_cast<const PartialResult *>(pr);

    SharedPtr<ridge_regression::Model> model = get(training::model);

    size_t coefdim = partRes->getNFeatures() + 1;
    size_t nrhs = partRes->getNDependentVariables();

    checkModel(model.get(), par, this->_errors.get(), coefdim, nrhs, method);
}

} // namespace interface1
} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal
