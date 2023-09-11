/* file: low_order_moments_input.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of LowOrderMoments classes.
//--
*/

#include "algorithms/moments/low_order_moments_types.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{
Input::Input() : InputIface(lastInputId + 1) {}
Input::Input(const Input & other) : InputIface(other) {}
Input & Input::operator=(const Input & other)
{
    InputIface::operator=(other);
    return *this;
}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
services::Status Input::getNumberOfColumns(size_t & nCols) const
{
    NumericTablePtr ntPtr = NumericTable::cast(Argument::get(0));
    if (ntPtr)
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
void Input::set(InputId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

services::Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    services::Status s;
    NumericTablePtr dataTable = get(data);
    if (method == fastCSR || method == singlePassCSR || method == sumCSR)
    {
        int expectedLayout = (int)NumericTableIface::csrArray;
        DAAL_CHECK_STATUS(s, checkNumericTable(dataTable.get(), dataStr(), 0, expectedLayout));
    }
    else
    {
        DAAL_CHECK_STATUS(s, checkNumericTable(dataTable.get(), dataStr()));
    }
    if (method == sumDense || method == sumCSR)
    {
        NumericTablePtr sum = dataTable->basicStatistics.get(NumericTableIface::sum);
        DAAL_CHECK_STATUS(s, checkNumericTable(sum.get(), basicStatisticsSumStr(), 0, 0, dataTable->getNumberOfColumns(), 1));
    }
    return s;
}

} // namespace low_order_moments
} // namespace algorithms
} // namespace daal
