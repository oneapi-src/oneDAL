/* file: smoothrelu.cpp */
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
//  Implementation of smoothrelu algorithm and types methods.
//--
*/

#include "smoothrelu_types.h"
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
namespace smoothrelu
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_SMOOTHRELU_RESULT_ID);
Input::Input() : daal::algorithms::Input(lastInputId + 1) {};

/**
 * Returns input NumericTable of the SmoothReLU algorithm
 * \param[in] id    Identifier of the input numeric table
 * \return          %Input numeric table that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets input for the SmoothReLU algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks parameters of the SmoothReLU algorithm
 * \param[in] par     %Parameter of algorithm
 * \param[in] method  Computation method of the algorithm
 */
Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, ErrorIncorrectNumberOfInputNumericTables);

    NumericTablePtr inTable = get(data);
    return checkNumericTable(inTable.get(), dataStr());
}

Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

/**
 * Returns result of the SmoothReLU algorithm
 * \param[in] id   Identifier of the result
 * \return         Final result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the SmoothReLU algorithm
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Pointer to the result
 */
void Result::set(ResultId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the result of the smoothrelu algorithm
 * \param[in] in   %Input of algorithm
 * \param[in] par     %Parameter of algorithm
 * \param[in] method  Computation method of the algorithm
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
}// namespace smoothrelu
}// namespace math
}// namespace algorithms
}// namespace daal
