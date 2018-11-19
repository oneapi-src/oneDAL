/** file row_merged_numeric_table.cpp */
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

#include "row_merged_numeric_table.h"

namespace daal
{
namespace data_management
{
namespace interface1
{

RowMergedNumericTable::RowMergedNumericTable() : NumericTable(0, 0), _tables(new DataCollection) {}

RowMergedNumericTable::RowMergedNumericTable(NumericTablePtr table) : NumericTable(0, 0), _tables(new DataCollection)
{
    this->_status |= addNumericTable(table);
}

services::SharedPtr<RowMergedNumericTable> RowMergedNumericTable::create(services::Status *stat)
{
    DAAL_DEFAULT_CREATE_IMPL(RowMergedNumericTable);
}

services::SharedPtr<RowMergedNumericTable> RowMergedNumericTable::create(const NumericTablePtr &nestedTable,
                                                                         services::Status *stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(RowMergedNumericTable, nestedTable);
}

RowMergedNumericTable::RowMergedNumericTable(services::Status &st) : NumericTable(0, 0), _tables(new DataCollection)
{
    if (!_tables) { st.add(services::ErrorMemoryAllocationFailed); }
    this->_status |= st;
}

RowMergedNumericTable::RowMergedNumericTable(const NumericTablePtr &table, services::Status &st) :
    NumericTable(0, 0),
    _tables(new DataCollection)
{
    if (!_tables) { st.add(services::ErrorMemoryAllocationFailed); }
    st |= addNumericTable(table);
    this->_status |= st;
}

}
}
}
