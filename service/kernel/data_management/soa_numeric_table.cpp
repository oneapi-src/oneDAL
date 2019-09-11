/** file soa_numeric_table.cpp */
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

#include "soa_numeric_table.h"

namespace daal
{
namespace data_management
{
namespace interface1
{

SOANumericTable::SOANumericTable(size_t nColumns, size_t nRows, DictionaryIface::FeaturesEqual featuresEqual):
    NumericTable(nColumns, nRows, featuresEqual), _arrays(nColumns), _arraysInitialized(0), _partialMemStatus(notAllocated)
{
    _layout = soa;

    if( !resizePointersArray(nColumns) )
    {
        this->_status.add(services::ErrorMemoryAllocationFailed);
        return;
    }
}

services::SharedPtr<SOANumericTable> SOANumericTable::create(size_t nColumns, size_t nRows,
                                                             DictionaryIface::FeaturesEqual featuresEqual, services::Status *stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(SOANumericTable, nColumns, nRows, featuresEqual);
}

SOANumericTable::SOANumericTable( NumericTableDictionaryPtr ddict, size_t nRows, AllocationFlag memoryAllocationFlag):
        NumericTable(ddict), _arraysInitialized(0), _partialMemStatus(notAllocated)
{
    _layout = soa;
    this->_status |= setNumberOfRowsImpl( nRows );
    if( !resizePointersArray( getNumberOfColumns() ) )
    {
        this->_status.add(services::ErrorMemoryAllocationFailed);
        return;
    }
    if( memoryAllocationFlag == doAllocate )
    {
        this->_status |= allocateDataMemoryImpl();
    }
}

services::SharedPtr<SOANumericTable> SOANumericTable::create(NumericTableDictionaryPtr ddict, size_t nRows,
                                                             AllocationFlag memoryAllocationFlag,
                                                             services::Status *stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(SOANumericTable, ddict, nRows, memoryAllocationFlag);
}

SOANumericTable::SOANumericTable(size_t nColumns, size_t nRows, DictionaryIface::FeaturesEqual featuresEqual, services::Status &st):
        NumericTable(nColumns, nRows, featuresEqual, st), _arrays(nColumns), _arraysInitialized(0), _partialMemStatus(notAllocated)
{
    _layout = soa;
    if (!resizePointersArray(nColumns))
    {
        st.add(services::ErrorMemoryAllocationFailed);
        return;
    }
}

SOANumericTable::SOANumericTable(NumericTableDictionaryPtr ddict, size_t nRows, AllocationFlag memoryAllocationFlag, services::Status &st):
    NumericTable(ddict, st), _arraysInitialized(0), _partialMemStatus(notAllocated)
{
    _layout = soa;
    st |= setNumberOfRowsImpl( nRows );
    if( !resizePointersArray( getNumberOfColumns() ) )
    {
        st.add(services::ErrorMemoryAllocationFailed);
        return;
    }
    if( memoryAllocationFlag == doAllocate )
    {
        st |= allocateDataMemoryImpl();
    }
}

}
}
}
