/* file: dbscan_input_types.cpp */
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
//  Implementation of DBSCAN classes.
//--
*/

#include "algorithms/dbscan/dbscan_types.h"
#include "daal_defines.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace dbscan
{
namespace interface1
{

Input::Input() : daal::algorithms::Input(lastInputId + 1) {}

/**
* Returns an input object for the DBSCAN algorithm
* \param[in] id    Identifier of the input object
* \return          %Input object that corresponds to the given identifier
*/
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
* Sets an input object for the DBSCAN algorithm
* \param[in] id    Identifier of the input object
* \param[in] ptr   Pointer to the object
*/
void Input::set(InputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
* Checks input objects for the DBSCAN algorithm
* \param[in] par     Algorithm parameter
* \param[in] method  Computation method of the algorithm
*/
services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(data).get(), dataStr(), 0, 0));

    if (get(weights))
    {
        const size_t nRows = get(data)->getNumberOfRows();
        const int unexpectedLayouts = (int)packed_mask;

        DAAL_CHECK_STATUS(s, checkNumericTable(get(weights).get(), dataStr(), unexpectedLayouts, 0, 1, nRows));
    }
    return s;
}

} // namespace interface1
} // namespace dbscan
} // namespace algorithm
} // namespace daal
