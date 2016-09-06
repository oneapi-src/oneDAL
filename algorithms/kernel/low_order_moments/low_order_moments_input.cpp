/* file: low_order_moments_input.cpp */
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
//  Implementation of LowOrderMoments classes.
//--
*/

#include "algorithms/moments/low_order_moments_types.h"

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

Input::Input() : InputIface(1) {}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
size_t Input::getNumberOfColumns() const
{
    NumericTablePtr ntPtr = NumericTable::cast(Argument::get(0));

    if(ntPtr)
    {
        return ntPtr->getNumberOfColumns();
    }
    else
    {
        this->_errors->add(ErrorNullNumericTable);
        return 0;
    }
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

void Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    NumericTablePtr dataTable = get(data);
    int unexpectedLayouts = 0;
    if(method == fastCSR || method == singlePassCSR || method == sumCSR)
    {
        int expectedLayout = (int)NumericTableIface::csrArray;
        if(!checkNumericTable(dataTable.get(), this->_errors.get(), dataStr(), 0, expectedLayout)) { return; }
    }
    else
    {
        if(!checkNumericTable(dataTable.get(), this->_errors.get(), dataStr())) { return; }
    }
    if(method == sumDense || method == sumCSR)
    {
        NumericTablePtr sum = dataTable->basicStatistics.get(NumericTableIface::sum);
        if(!checkNumericTable(sum.get(), this->_errors.get(), basicStatisticsSumStr(), 0, 0, dataTable->getNumberOfColumns(), 1)) { return; }
    }
}

} // namespace interface1
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal
