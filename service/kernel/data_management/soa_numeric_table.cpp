/** file soa_numeric_table.cpp */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

#include "data_management/data/soa_numeric_table.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
SOANumericTable::SOANumericTable(size_t nColumns, size_t nRows, DictionaryIface::FeaturesEqual featuresEqual)
    : NumericTable(nColumns, nRows, featuresEqual), _arrays(nColumns), _arraysInitialized(0), _partialMemStatus(notAllocated)
{
    _layout     = soa;
    _arrOffsets = NULL;
    _index      = 0;

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
    _layout     = soa;
    _arrOffsets = NULL;
    _index      = 0;
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
    _layout     = soa;
    _arrOffsets = NULL;
    _index      = 0;
    if (!resizePointersArray(nColumns))
    {
        st.add(services::ErrorMemoryAllocationFailed);
        return;
    }
}

SOANumericTable::SOANumericTable(NumericTableDictionaryPtr ddict, size_t nRows, AllocationFlag memoryAllocationFlag, services::Status & st)
    : NumericTable(ddict, st), _arraysInitialized(0), _partialMemStatus(notAllocated)
{
    _layout     = soa;
    _arrOffsets = NULL;
    _index      = 0;
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

services::Status SOANumericTable::searchMinPointer()
{
    size_t ncols = getNumberOfColumns();

    if (_arrOffsets) daal::services::daal_free(_arrOffsets);

    _arrOffsets = (DAAL_INT64 *)daal::services::daal_malloc(ncols * sizeof(DAAL_INT64));
    DAAL_CHECK_MALLOC(_arrOffsets)
    _index        = 0;
    char * ptrMin = (char *)_arrays[0].get();

    /* search index for min pointer */
    for (size_t i = 1; i < ncols; ++i)
    {
        if ((char *)_arrays[i].get() < ptrMin)
        {
            _index = i;
            ptrMin = (char *)_arrays[i].get();
        }
    }

    /* compute offsets */
    for (size_t i = 0; i < ncols; ++i)
    {
        char * pv      = (char *)(_arrays[i].get());
        _arrOffsets[i] = (DAAL_INT64)(pv - ptrMin);
        DAAL_ASSERT(_arrOffsets[i] >= 0)
    }

    return services::Status();
}

bool SOANumericTable::resizePointersArray(size_t nColumns)
{
    if (_arrays.size() >= nColumns)
    {
        size_t counter = 0;
        for (size_t i = 0; i < nColumns; i++)
        {
            counter += (_arrays[i] != 0);
        }
        _arraysInitialized = counter;

        if (_arraysInitialized == nColumns)
        {
            _memStatus = _partialMemStatus;
        }
        else
        {
            _memStatus = notAllocated;
        }

        return true;
    }
    _arrays.resize(nColumns);
    _memStatus = notAllocated;

    bool is_resized = _arrays.resize(nColumns);
    if (is_resized)
    {
        _memStatus = notAllocated;
    }

    if (_arrOffsets)
    {
        daal::services::daal_free(_arrOffsets);
        _arrOffsets = NULL;
        _index      = 0;
    }

    return is_resized;
}

services::Status SOANumericTable::setNumberOfColumnsImpl(size_t ncol)
{
    services::Status s;
    DAAL_CHECK_STATUS(s, NumericTable::setNumberOfColumnsImpl(ncol));

    if (!resizePointersArray(ncol))
    {
        return services::Status(services::ErrorMemoryAllocationFailed);
    }
    return s;
}

void SOANumericTable::freeDataMemoryImpl()
{
    _arrays.clear();
    _arrays.resize(_ddict->getNumberOfFeatures());
    _arraysInitialized = 0;

    _partialMemStatus = notAllocated;
    _memStatus        = notAllocated;
}

} // namespace interface1
} // namespace data_management
} // namespace daal
