/** file aos_numeric_table.cpp */
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

#include "aos_numeric_table.h"

namespace daal
{
namespace data_management
{
namespace interface1
{

AOSNumericTable::AOSNumericTable(size_t structSize, size_t ncol, size_t nrow): NumericTable(ncol, nrow)
{
    _layout     = aos;
    _structSize = structSize;

    initOffsets();
}

AOSNumericTable::AOSNumericTable(size_t structSize, size_t ncol, size_t nrow, services::Status &st): NumericTable(ncol, nrow, DictionaryIface::notEqual, st)//?
{
    _layout     = aos;
    _structSize = structSize;

    st |= initOffsets();
}

services::SharedPtr<AOSNumericTable> AOSNumericTable::create(size_t structSize, size_t ncol, size_t nrow,
                                                             services::Status *stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(AOSNumericTable, structSize, ncol, nrow);
}

}
}
}
