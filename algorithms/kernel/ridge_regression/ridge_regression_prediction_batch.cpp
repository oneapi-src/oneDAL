/* file: ridge_regression_prediction_batch.cpp */
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

#include "algorithms/ridge_regression/ridge_regression_predict_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace prediction
{
namespace interface1
{

/** Default constructor */
Input::Input() : daal::algorithms::Input(2) {}
/**
 * Returns an input object for making ridge regression model-based prediction
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(NumericTableInputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Returns an input object for making ridge regression model-based prediction
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
SharedPtr<ridge_regression::Model> Input::get(ModelInputId id) const
{
    return staticPointerCast<ridge_regression::Model, SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for making ridge regression model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Input::set(NumericTableInputId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

/**
 * Sets an input object for making ridge regression model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Input::set(ModelInputId id, const SharedPtr<ridge_regression::Model> & value)
{
    Argument::set(id, value);
}

/**
 * Checks an input object for making ridge regression model-based prediction
 */
void Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    DAAL_CHECK(Argument::size() == 2, ErrorIncorrectNumberOfInputNumericTables);
    NumericTablePtr dataTable = get(data);

    if(!checkNumericTable(dataTable.get(), this->_errors.get(), dataStr())) { return; }
    size_t coefdim = get(data)->getNumberOfColumns() + 1;

    const SharedPtr<ridge_regression::Model> model = get(prediction::model);
    DAAL_CHECK(model, ErrorNullModel);

    size_t nrhs = model->getNumberOfResponses();
    if(!checkNumericTable(model->getBeta().get(), this->_errors.get(), betaStr(), 0, 0, coefdim, nrhs)) { return; }

    size_t dimWithoutBeta = coefdim;
    if(!model->getInterceptFlag())
    {
        dimWithoutBeta--;
    }
}


Result::Result() : daal::algorithms::Result(1) {}

/**
 * Returns the result of ridge regression model-based prediction
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of ridge regression model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Result::set(ResultId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

/**
 * Checks the result of ridge regression model-based prediction
 * \param[in] input   %Input object
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
void Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, ErrorIncorrectNumberOfOutputNumericTables);
    NumericTablePtr predictionTable = get(prediction);

    const Input* in = static_cast<const Input *>(input);

    size_t nVectors = in->get(data)->getNumberOfRows();
    size_t nFeatures = in->get(model)->getNumberOfResponses();

    if(!checkNumericTable(predictionTable.get(), this->_errors.get(), predictionStr(), 0, 0, nFeatures, nVectors)) { return; }
}


} // namespace interface1
} // namespace prediction
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal
