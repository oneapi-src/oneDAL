/* file: relu_fpt.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of relu algorithm and types methods.
//--
*/

#include "relu_types.h"
#include "service_numeric_table.h"

using namespace daal::services;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace math
{
namespace relu
{
namespace interface1
{
/**
 * Allocates memory to store the result of the rectified linear function
 * \param[in] input  %Input object for the rectified linear function
 * \param[in] par    %Parameter of the rectified linear function
 * \param[in] method Computation method of the rectified linear function
 */
template <typename algorithmFPType>
DAAL_EXPORT Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
{
    Status s;
    Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    DAAL_CHECK(algInput, ErrorNullInput);

    NumericTablePtr inputTable = algInput->get(data);
    DAAL_CHECK(inputTable.get(), ErrorNullInputNumericTable);

    if(method == fastCSR)
    {
        NumericTableIface::StorageLayout layout = inputTable->getDataLayout();
        DAAL_CHECK(layout == NumericTableIface::csrArray, ErrorIncorrectTypeOfInputNumericTable);

        CSRNumericTablePtr resTable;

        DAAL_CHECK_STATUS(s, createSparseTable<algorithmFPType>(algInput->get(data), resTable));
        Argument::set(value, staticPointerCast<SerializationIface, CSRNumericTable>(resTable));
    }
    else
    {
        set(value, HomogenNumericTable<algorithmFPType>::create(inputTable->getNumberOfColumns(), inputTable->getNumberOfRows(), NumericTable::doAllocate, &s));
    }
    return s;
}

template DAAL_EXPORT Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

}// namespace interface1
}// namespace relu
}// namespace math
}// namespace algorithms
}// namespace daal
