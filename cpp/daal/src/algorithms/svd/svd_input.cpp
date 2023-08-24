/* file: svd_input.cpp */
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
//  Implementation of svd classes.
//--
*/

#include "algorithms/svd/svd_types.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace svd
{
/** Default constructor */
Input::Input() : daal::algorithms::Input(lastInputId + 1) {}
Input::Input(const Input & other) : daal::algorithms::Input(other) {}

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
void Input::set(InputId id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

Status Input::getNumberOfColumns(size_t * nFeatures) const
{
    if (!nFeatures) return Status(ErrorNullParameterNotSupported);

    NumericTablePtr dataTable = get(data);
    if (dataTable)
    {
        *nFeatures = dataTable->getNumberOfColumns();
    }
    else
    {
        return Status(Error::create(ErrorNullNumericTable, ArgumentName, dataStr()));
    }
    return Status();
}

Status Input::getNumberOfRows(size_t * nRows) const
{
    if (!nRows) return Status(ErrorNullParameterNotSupported);

    NumericTablePtr dataTable = get(data);
    if (dataTable)
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
Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    NumericTablePtr dataTable = get(data);
    return checkNumericTable(dataTable.get(), dataStr());
}

} // namespace svd
} // namespace algorithms
} // namespace daal
