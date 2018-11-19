/* file: distribution_types.cpp */
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

//++
//  Implementation of distribution types.
//--

#include "distributions/distribution_types.h"
#include "daal_strings.h"
#include "mt19937_batch_impl.h"

namespace daal
{
namespace algorithms
{
namespace distributions
{
namespace interface1
{

ParameterBase::ParameterBase() : engine(engines::mt19937::Batch<>::create()) {}
ParameterBase::~ParameterBase() {}

Input::Input() : daal::algorithms::Input(1) {}
Input::~Input() {}

Input::Input(const Input& other) : daal::algorithms::Input(other){}

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

    const int expectedLayouts = (int)data_management::NumericTableIface::soa | (int)data_management::NumericTableIface::aos;
    return data_management::checkNumericTable(get(tableToFill).get(), tableToFillStr(), 0, expectedLayouts);
}

Result::Result() : daal::algorithms::Result(1) {}
Result::~Result() {}

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

    const int expectedLayouts = (int)data_management::NumericTableIface::soa | (int)data_management::NumericTableIface::aos;
    return data_management::checkNumericTable(get(randomNumbers).get(), randomNumbersStr(), 0, expectedLayouts);
}

} // namespace interface1
} // namespace distributions
} // namespace algorithms
} // namespace daal
