/* file: svd_input.cpp */
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
//  Implementation of svd classes.
//--
*/

#include "algorithms/svd/svd_types.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace svd
{
namespace interface1
{

/** Default constructor */
Input::Input() : daal::algorithms::Input(lastInputId + 1) {}
Input::Input(const Input& other) : daal::algorithms::Input(other){}

/**
 * Returns input object of the SVD algorithm
 * \param[in] id    Identifier of the input object
 * \return          Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets input object for the SVD algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] value Pointer to the input object
 */
void Input::set(InputId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

Status Input::getNumberOfColumns(size_t *nFeatures) const
{
    if(!nFeatures)
        return Status(ErrorNullParameterNotSupported);

    NumericTablePtr dataTable = get(data);
    if(dataTable)
    {
        *nFeatures = dataTable->getNumberOfColumns();
    }
    else
    {
        return Status(Error::create(ErrorNullNumericTable, ArgumentName, dataStr()));
    }
    return Status();
}

Status Input::getNumberOfRows(size_t *nRows) const
{
    if(!nRows)
        return Status(ErrorNullParameterNotSupported);

    NumericTablePtr dataTable = get(data);
    if(dataTable)
    {
        *nRows = dataTable->getNumberOfRows();
    }
    else
    {
        return Status(Error::create(ErrorNullNumericTable, ArgumentName, dataStr()));
    }
    return Status();
}

/**
 * Checks parameters of the algorithm
 * \param[in] parameter Pointer to the parameters
 * \param[in] method Computation method
 */
Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    NumericTablePtr dataTable = get(data);
    Status s = checkNumericTable(dataTable.get(), dataStr());
    if(!s) { return s; }

    DAAL_CHECK_EX(dataTable->getNumberOfColumns() <= dataTable->getNumberOfRows(), ErrorIncorrectNumberOfRows, ArgumentName, dataStr());
    return Status();
}

} // namespace interface1
} // namespace svd
} // namespace algorithm
} // namespace daal
