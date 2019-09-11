/* file: engine_types.cpp */
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

//++
//  Implementation of initializer types.
//--

#include "engines/engine_types.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace interface1
{

Input::Input() : daal::algorithms::Input(1) {}

Input::Input(const Input& other) : daal::algorithms::Input(other) {}

data_management::NumericTablePtr Input::get(InputId id) const
{
    return data_management::NumericTable::cast(Argument::get(id));
}

void Input::set(InputId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, services::ErrorIncorrectNumberOfInputNumericTables);
    return data_management::checkNumericTable(get(tableToFill).get(), tableToFillStr());
}

Result::Result() : daal::algorithms::Result(1) {}

data_management::NumericTablePtr Result::get(ResultId id) const
{
    return data_management::NumericTable::cast(Argument::get(id));
}

void Result::set(ResultId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    DAAL_CHECK(Argument::size() == 1, services::ErrorIncorrectNumberOfInputNumericTables);
    const Input *algInput = static_cast<const Input *>(input);
    DAAL_CHECK(algInput, services::ErrorNullInput);
    return data_management::checkNumericTable(get(randomNumbers).get(), randomNumbersStr());
}

} // namespace interface1
} // namespace engines
} // namespace algorithms
} // namespace daal
