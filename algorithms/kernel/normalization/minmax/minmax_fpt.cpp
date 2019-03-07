/* file: minmax_fpt.cpp */
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
//  Implementation of minmax algorithm and types methods.
//--
*/

#include "minmax_types.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace minmax
{
namespace interface1
{
/**
 * Allocates memory to store the result of the minmax normalization algorithm
 * \param[in] input  %Input object for the minmax normalization algorithm
 * \param[in] par    %Parameter of the minmax normalization algorithm
 * \param[in] method Computation method of the minmax normalization algorithm
 */
template <typename algorithmFPType>
DAAL_EXPORT Status Result::allocate(const daal::algorithms::Input *input, int method)
{
    DAAL_CHECK(input, ErrorNullInput);

    const Input *algInput = static_cast<const Input *>(input);
    NumericTablePtr dataTable = algInput->get(data);

    Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(dataTable.get(), dataStr()));

    const size_t nRows = dataTable->getNumberOfRows();
    const size_t nColumns = dataTable->getNumberOfColumns();
    NumericTablePtr normalizedDataTable = HomogenNumericTable<algorithmFPType>::create(
                                                         nColumns, nRows, NumericTable::doAllocate, &s);
    DAAL_CHECK_STATUS_VAR(s);
    set(normalizedData, normalizedDataTable);
    return s;
}

template DAAL_EXPORT Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, int method);

}// namespace interface1
}// namespace minmax
}// namespace normalization
}// namespace algorithms
}// namespace daal
