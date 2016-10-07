/* file: linear_regression_training_input.cpp */
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
//  Implementation of linear regression algorithm classes.
//--
*/

#include "algorithms/linear_regression/linear_regression_training_types.h"

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
namespace interface1
{

/** Default constructor */
Input::Input() : InputIface(2) {}

/**
 * Returns an input object for linear regression model-based training
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for linear regression model-based training
 * \param[in] id      Identifier of the input object
 * \param[in] value   Pointer to the object
 */
void Input::set(InputId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
size_t Input::getNFeatures() const { return get(data)->getNumberOfColumns(); }

/**
* Returns the number of dependent variables
* \return Number of dependent variables
*/
size_t Input::getNDependentVariables() const { return get(dependentVariables)->getNumberOfColumns(); }

/**
* Checks an input object for the linear regression algorithm
* \param[in] par     Algorithm parameter
* \param[in] method  Computation method
*/

void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    DAAL_CHECK(Argument::size() == 2, ErrorIncorrectNumberOfInputNumericTables);

    NumericTablePtr dataTable = get(data);
    NumericTablePtr dependentVariableTable = get(dependentVariables);

    if(!checkNumericTable(dataTable.get(), this->_errors.get(), dataStr())) { return; }

    size_t nRowsInData = dataTable->getNumberOfRows();
    size_t nColumnsInData = dataTable->getNumberOfColumns();

    DAAL_CHECK(nRowsInData >= nColumnsInData, ErrorIncorrectNumberOfObservations);

    if(!checkNumericTable(dependentVariableTable.get(), this->_errors.get(), dependentVariableStr(), 0, 0, 0, nRowsInData)) { return; }
}

} // namespace interface1
} // namespace training
} // namespace linear_regression
} // namespace algorithms
} // namespace daal
