/* file: regression_training_input.cpp */
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
//  Implementation of the class defining the input objects
//  of the regression training algorithm
//--
*/

#include "algorithms/regression/regression_training_types.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace regression
{
namespace training
{
namespace interface1
{
using namespace daal::data_management;
using namespace daal::services;
Input::Input(size_t nElements) : daal::algorithms::Input(nElements)
{}
Input::Input(const Input& other) : daal::algorithms::Input(other)
{}

data_management::NumericTablePtr Input::get(InputId id) const
{
    return NumericTable::cast(Argument::get(id));
}

void Input::set(InputId id, const data_management::NumericTablePtr &value)
{
    Argument::set(id, value);
}

Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    const NumericTablePtr dataTable = get(data);
    const NumericTablePtr dependentVariableTable = get(dependentVariables);

    Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(dataTable.get(), dataStr()));

    size_t nRowsInData = dataTable->getNumberOfRows();

    return checkNumericTable(dependentVariableTable.get(), dependentVariableStr(), 0, 0, 0, nRowsInData);
}

}
}
}
}
}
