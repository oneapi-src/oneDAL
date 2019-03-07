/* file: relu_fpt.cpp */
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
