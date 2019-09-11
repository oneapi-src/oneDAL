/* file: covariance_input.cpp */
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
//  Implementation of covariance algorithm and types methods.
//--
*/

#include "covariance_types.h"
#include "daal_strings.h"

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

Input::Input() : InputIface(lastInputId + 1)
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
    return 0;
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
services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s;
    int expectedLayouts = 0;

    if (method == fastCSR || method == singlePassCSR || method == sumCSR)
    {
        expectedLayouts = (int)NumericTableIface::csrArray;
    }

    s |= checkNumericTable(get(data).get(), dataStr(), 0, expectedLayouts);
    if(!s) return s;

    if (method == sumDense || method == sumCSR)
    {
        size_t nFeatures = get(data)->getNumberOfColumns();

        s |= checkNumericTable(get(data)->basicStatistics.get(NumericTableIface::sum).get(), sumStr(), 0, 0, nFeatures, 1);
    }
    return s;
}

}//namespace interface1

}//namespace covariance
}// namespace algorithms
}// namespace daal
