/* file: abs.cpp */
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
//  Implementation of abs algorithm and types methods.
//--
*/

#include "abs_types.h"
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
namespace abs
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_ABS_RESULT_ID);
/** Default constructor */
Input::Input() : daal::algorithms::Input(lastInputId + 1) {};

/**
 * Returns an input object for the absolute value function
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for the absolute value function
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks an input object for the absolute value function
 * \param[in] par     Function parameter
 * \param[in] method  Computation method
 */
Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, ErrorIncorrectNumberOfInputNumericTables);

    NumericTablePtr inTable = get(data);

    if(method == fastCSR)
    {
        const int expectedLayouts = (int)NumericTableIface::csrArray;
        return checkNumericTable(inTable.get(), dataStr(), 0, expectedLayouts);
    }
    else
    {
        return checkNumericTable(inTable.get(), dataStr());
    }
    return Status();
}

/** Default constructor */
Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

/**
 * Returns the result of the absolute value function
 * \param[in] id   Identifier of the result
 * \return         Result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the absolute value function
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Result
 */
void Result::set(ResultId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the result of the absolute value function
 * \param[in] in   %Input of the absolute value function
 * \param[in] par     %Parameter of the absolute value function
 * \param[in] method  Computation method of the absolute value function
 */
Status Result::check(const daal::algorithms::Input *in, const daal::algorithms::Parameter *par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, ErrorIncorrectNumberOfOutputNumericTables);
    DAAL_CHECK(in != 0, ErrorNullInput);

    NumericTablePtr dataTable = (static_cast<const Input *>(in))->get(data);
    NumericTablePtr resultTable = get(value);

    Status s;
    if(method == fastCSR)
    {
        const int expectedLayouts = (int)NumericTableIface::csrArray;
        DAAL_CHECK_STATUS(s, checkNumericTable(dataTable.get(), dataStr(), 0, expectedLayouts));

        const size_t nDataRows = dataTable->getNumberOfRows();
        const size_t nDataColumns = dataTable->getNumberOfColumns();

        DAAL_CHECK_STATUS(s, checkNumericTable(resultTable.get(), valueStr(), 0, expectedLayouts, nDataColumns, nDataRows));

        CSRNumericTableIfacePtr inputTable =
            dynamicPointerCast<CSRNumericTableIface, NumericTable>(dataTable);

        CSRNumericTableIfacePtr resTable =
            dynamicPointerCast<CSRNumericTableIface, NumericTable>(resultTable);

        const size_t inSize = inputTable->getDataSize();
        const size_t resSize = resTable->getDataSize();

        DAAL_CHECK(inSize == resSize, ErrorIncorrectSizeOfArray);
    }
    else
    {
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
    return s;
}

}// namespace interface1
}// namespace abs
}// namespace math
}// namespace algorithms
}// namespace daal
