/** file soa_numeric_table.cpp */
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

#include "soa_numeric_table.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
SOANumericTable::SOANumericTable(size_t nColumns, size_t nRows, DictionaryIface::FeaturesEqual featuresEqual)
    : NumericTable(nColumns, nRows, featuresEqual), _arrays(nColumns), _arraysInitialized(0), _partialMemStatus(notAllocated)
{
    _layout = soa;

    if (!resizePointersArray(nColumns))
    {
        this->_status.add(services::ErrorMemoryAllocationFailed);
        return;
    }
}

services::SharedPtr<SOANumericTable> SOANumericTable::create(size_t nColumns, size_t nRows, DictionaryIface::FeaturesEqual featuresEqual,
                                                             services::Status * stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(SOANumericTable, nColumns, nRows, featuresEqual);
}

SOANumericTable::SOANumericTable(NumericTableDictionaryPtr ddict, size_t nRows, AllocationFlag memoryAllocationFlag)
    : NumericTable(ddict), _arraysInitialized(0), _partialMemStatus(notAllocated)
{
    _layout = soa;
    this->_status |= setNumberOfRowsImpl(nRows);
    if (!resizePointersArray(getNumberOfColumns()))
    {
        this->_status.add(services::ErrorMemoryAllocationFailed);
        return;
    }
    if (memoryAllocationFlag == doAllocate)
    {
        this->_status |= allocateDataMemoryImpl();
    }
}

services::SharedPtr<SOANumericTable> SOANumericTable::create(NumericTableDictionaryPtr ddict, size_t nRows, AllocationFlag memoryAllocationFlag,
                                                             services::Status * stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(SOANumericTable, ddict, nRows, memoryAllocationFlag);
}

SOANumericTable::SOANumericTable(size_t nColumns, size_t nRows, DictionaryIface::FeaturesEqual featuresEqual, services::Status & st)
    : NumericTable(nColumns, nRows, featuresEqual, st), _arrays(nColumns), _arraysInitialized(0), _partialMemStatus(notAllocated)
{
    _layout = soa;
    if (!resizePointersArray(nColumns))
    {
        st.add(services::ErrorMemoryAllocationFailed);
        return;
    }
}

SOANumericTable::SOANumericTable(NumericTableDictionaryPtr ddict, size_t nRows, AllocationFlag memoryAllocationFlag, services::Status & st)
    : NumericTable(ddict, st), _arraysInitialized(0), _partialMemStatus(notAllocated)
{
    _layout = soa;
    st |= setNumberOfRowsImpl(nRows);
    if (!resizePointersArray(getNumberOfColumns()))
    {
        st.add(services::ErrorMemoryAllocationFailed);
        return;
    }
    if (memoryAllocationFlag == doAllocate)
    {
        st |= allocateDataMemoryImpl();
    }
}

} // namespace interface1
} // namespace data_management
} // namespace daal
