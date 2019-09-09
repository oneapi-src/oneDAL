/* file: softmax.cpp */
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
//  Implementation of softmax algorithm and types methods.
//--
*/

#include "softmax_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace math
{
namespace softmax
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_SOFTMAX_RESULT_ID);
/** Default constructor */
Input::Input() : daal::algorithms::Input(lastInputId + 1) {};

/**
 * Returns an input object for the softmax function
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for the softmax function
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks an input object for the softmax function
 * \param[in] par     Function parameter
 * \param[in] method  Computation method
 */
Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, ErrorIncorrectNumberOfInputNumericTables);

    NumericTablePtr inTable = get(data);
    return checkNumericTable(inTable.get(), dataStr());
}

/** Default constructor */
Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

/**
 * Returns the result of the softmax function
 * \param[in] id   Identifier of the result
 * \return         Result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the softmax function
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Result
 */
void Result::set(ResultId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the result of the softmax function
 * \param[in] in   %Input of the softmax function
 * \param[in] par     %Parameter of the softmax function
 * \param[in] method  Computation method of the softmax function
 */
Status Result::check(const daal::algorithms::Input *in, const daal::algorithms::Parameter *par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, ErrorIncorrectNumberOfOutputNumericTables);
    DAAL_CHECK(in != 0, ErrorNullInput);

    NumericTablePtr dataTable = (static_cast<const Input *>(in))->get(data);
    NumericTablePtr resultTable = get(value);

    Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(dataTable.get(), dataStr()));

    const size_t nDataRows = dataTable->getNumberOfRows();
    const size_t nDataColumns = dataTable->getNumberOfColumns();

    const int unexpectedLayouts = (int)NumericTableIface::upperPackedSymmetricMatrix |
                                  (int)NumericTableIface::lowerPackedSymmetricMatrix |
                                  (int)NumericTableIface::upperPackedTriangularMatrix |
                                  (int)NumericTableIface::lowerPackedTriangularMatrix |
                                  (int)NumericTableIface::csrArray;
    return checkNumericTable(resultTable.get(), valueStr(), unexpectedLayouts, 0, nDataColumns, nDataRows);
}

}// namespace interface1
}// namespace softmax
}// namespace math
}// namespace algorithms
}// namespace daal
