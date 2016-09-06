/* file: covariance_input.cpp */
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
//  Implementation of covariance algorithm and types methods.
//--
*/

#include "covariance_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace interface1
{

Input::Input() : InputIface(1)
    {}

/**
 * Returns number of columns in the input data set
 * \return Number of columns in the input data set
 */
size_t Input::getNumberOfFeatures() const
{
    NumericTablePtr ntPtr = NumericTable::cast(get(data));
    if(ntPtr)
    {
        return ntPtr->getNumberOfColumns();
    }
    else
    {
        this->_errors->add(ErrorIncorrectSizeOfInputNumericTable);
        return 0;
    }
}

/**
 * Returns the input object of the correlation or variance-covariance matrix algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the input object of the correlation or variance-covariance matrix algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks algorithm parameters
 * \param[in] parameter Pointer to the structure of algorithm parameters
 * \param[in] method    Computation method
 */
void Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    int expectedLayouts = 0;

    if (method == fastCSR || method == singlePassCSR || method == sumCSR)
    {
        expectedLayouts = (int)NumericTableIface::csrArray;
    }

    if (!checkNumericTable(get(data).get(), this->_errors.get(), dataStr(), 0, expectedLayouts)) { return; }

    if (method == sumDense || method == sumCSR)
    {
        size_t nFeatures = get(data)->getNumberOfColumns();

        if (!checkNumericTable(get(data)->basicStatistics.get(NumericTableIface::sum).get(),
            this->_errors.get(), sumStr(), 0, 0, nFeatures, 1)) { return; }
    }
}

}//namespace interface1

}//namespace covariance
}// namespace algorithms
}// namespace daal
