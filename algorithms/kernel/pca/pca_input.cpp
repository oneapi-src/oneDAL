/* file: pca_input.cpp */
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
//  Implementation of PCA algorithm interface.
//--
*/

#include "algorithms/pca/pca_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface1
{

Input::Input() : InputIface(1) {};

/**
* Returns the input object of the PCA algorithm
 * \param[in] id    Identifier of the input object
 * \return          Input object that corresponds to the given identifier
*/
NumericTablePtr Input::get(InputDatasetId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets input dataset for the PCA algorithm
 * \param[in] id      Identifier of the input object
 * \param[in] value   Pointer to the input object
 */
void Input::set(InputDatasetId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
    _isCorrelation = false;
}

/**
 * Sets input correlation matrix for the PCA algorithm
 * \param[in] id      Identifier of the input object
 * \param[in] value   Pointer to the input object
 */
void Input::set(InputCorrelationId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
    _isCorrelation = true;
}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
size_t Input::getNFeatures() const
{
    return (staticPointerCast<NumericTable, SerializationIface>(Argument::get(data)))->getNumberOfColumns();
}

/**
* Checks input algorithm parameters
* \param[in] par     Algorithm %parameter
* \param[in] method  Computation method
* \return Errors detected while checking
*/
void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, ErrorIncorrectNumberOfInputNumericTables);
    NumericTablePtr dataTable = get(data);
    if(_isCorrelation)
    {
        int unexpectedLayouts = (int)NumericTableIface::upperPackedTriangularMatrix | (int)NumericTableIface::lowerPackedTriangularMatrix;
        if(!checkNumericTable(dataTable.get(), this->_errors.get(), correlationStr(), unexpectedLayouts)) { return; }
        DAAL_CHECK_EX(dataTable->getNumberOfColumns() == dataTable->getNumberOfRows(), ErrorNumericTableIsNotSquare, ArgumentName, correlationStr());
    }
    else
    {
        if(!checkNumericTable(dataTable.get(), this->_errors.get(), dataStr())) { return; }
    }
}

} // namespace interface1
} // namespace pca
} // namespace algorithms
} // namespace daal
