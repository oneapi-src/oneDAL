/* file: smoothrelu.cpp */
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
//  Implementation of smoothrelu algorithm and types methods.
//--
*/

#include "smoothrelu_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace math
{
namespace smoothrelu
{
namespace interface1
{
Input::Input() : daal::algorithms::Input(1) {};

/**
 * Returns input NumericTable of the SmoothReLU algorithm
 * \param[in] id    Identifier of the input numeric table
 * \return          %Input numeric table that corresponds to the given identifier
 */
data_management::NumericTablePtr Input::get(InputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets input for the SmoothReLU algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks parameters of the SmoothReLU algorithm
 * \param[in] par     %Parameter of algorithm
 * \param[in] method  Computation method of the algorithm
 */
void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    if(Argument::size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

    data_management::NumericTablePtr inTable = get(data);
    if(!data_management::checkNumericTable(inTable.get(), this->_errors.get(), dataStr())) { return; }
}

Result::Result() : daal::algorithms::Result(1) {}

/**
 * Returns result of the SmoothReLU algorithm
 * \param[in] id   Identifier of the result
 * \return         Final result that corresponds to the given identifier
 */
data_management::NumericTablePtr Result::get(ResultId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the SmoothReLU algorithm
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Pointer to the result
 */
void Result::set(ResultId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the result of the smoothrelu algorithm
 * \param[in] in   %Input of algorithm
 * \param[in] par     %Parameter of algorithm
 * \param[in] method  Computation method of the algorithm
 */
void Result::check(const daal::algorithms::Input *in, const daal::algorithms::Parameter *par, int method) const
{
    if(Argument::size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }
    if(in == 0) { this->_errors->add(services::ErrorNullInput); return; }

    data_management::NumericTablePtr dataTable = (static_cast<const Input *>(in))->get(data);
    data_management::NumericTablePtr resultTable = get(value);

    if(!data_management::checkNumericTable(dataTable.get(), this->_errors.get(), dataStr())) { return; }

    size_t nDataRows = dataTable->getNumberOfRows();
    size_t nDataColumns = dataTable->getNumberOfColumns();

    int unexpectedLayouts = (int)data_management::NumericTableIface::upperPackedSymmetricMatrix |
                            (int)data_management::NumericTableIface::lowerPackedSymmetricMatrix |
                            (int)data_management::NumericTableIface::upperPackedTriangularMatrix |
                            (int)data_management::NumericTableIface::lowerPackedTriangularMatrix |
                            (int)data_management::NumericTableIface::csrArray;
    if(!data_management::checkNumericTable(resultTable.get(), this->_errors.get(), valueStr(), unexpectedLayouts, 0, nDataColumns, nDataRows)) { return; }
}

}// namespace interface1
}// namespace smoothrelu
}// namespace math
}// namespace algorithms
}// namespace daal
