/* file: low_order_moments_input.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of LowOrderMoments classes.
//--
*/

#include "algorithms/moments/low_order_moments_types.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{
namespace interface1
{

Input::Input() : InputIface(lastInputId + 1) {}
Input::Input(const Input& other) : InputIface(other){}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
services::Status Input::getNumberOfColumns(size_t& nCols) const
{
    NumericTablePtr ntPtr = NumericTable::cast(Argument::get(0));
    if(ntPtr)
    {
        nCols = ntPtr.get() ? ntPtr->getNumberOfColumns() : 0;
        return services::Status();
    }
    nCols = 0;
        return services::Status(ErrorNullNumericTable);
}

/**
 * Returns the input object for the low order %moments algorithm
 * \param[in] id    Identifier of the %input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets input object for the low order %moments algorithm
 * \param[in] id    Identifier of the %input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s;
    NumericTablePtr dataTable = get(data);
    int unexpectedLayouts = 0;
    if(method == fastCSR || method == singlePassCSR || method == sumCSR)
    {
        int expectedLayout = (int)NumericTableIface::csrArray;
        DAAL_CHECK_STATUS(s, checkNumericTable(dataTable.get(), dataStr(), 0, expectedLayout));
    }
    else
    {
        DAAL_CHECK_STATUS(s, checkNumericTable(dataTable.get(), dataStr()));
    }
    if(method == sumDense || method == sumCSR)
    {
        NumericTablePtr sum = dataTable->basicStatistics.get(NumericTableIface::sum);
        DAAL_CHECK_STATUS(s, checkNumericTable(sum.get(), basicStatisticsSumStr(), 0, 0, dataTable->getNumberOfColumns(), 1));
    }
    return s;
}

} // namespace interface1
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal
