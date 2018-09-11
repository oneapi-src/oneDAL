/* file: logistic.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
//  Implementation of logistic algorithm and types methods.
//--
*/

#include "logistic_types.h"
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
namespace logistic
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_LOGISTIC_RESULT_ID);
/** Default constructor */
Input::Input() : daal::algorithms::Input(lastInputId + 1) {};

/**
 * Returns an input object for the logistic function
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for the logistic function
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks an input object for the logistic function
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
 * Returns result of the logistic function
 * \param[in] id   Identifier of the result
 * \return         Result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the logistic function
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Result
 */
void Result::set(ResultId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the result of the logistic function
 * \param[in] in   %Input of the logistic function
 * \param[in] par     %Parameter of the logistic function
 * \param[in] method  Computation method of the the logistic function
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
}// namespace logistic
}// namespace math
}// namespace algorithms
}// namespace daal
