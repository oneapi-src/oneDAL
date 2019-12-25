/* file: numeric_table_sycl_soa.h */
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

#ifndef __SYCL_SOA_NUMERIC_TABLE_H__
#define __SYCL_SOA_NUMERIC_TABLE_H__

#include "data_management/data/numeric_table_sycl.h"
#include "data_management/data/soa_numeric_table.h"
#include "oneapi/internal/buffer_utils.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
/**
 * @ingroup sycl
 * @{
 */

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__SYCLSOANUMERICTABLE"></a>
 *  \brief Class that provides methods to access data stored as a structure of arrays,
 *         where each (contiguous) array represents values corresponding to a specific feature.
 *         Each array is represented by SYCL* buffer.
 */
class DAAL_EXPORT SyclSOANumericTable : public SyclNumericTable
{
public:
    DECLARE_SERIALIZABLE_TAG();
    DECLARE_SERIALIZABLE_IMPL();

    /**
     *  Constructs an empty Numeric Table
     *  \param[in]  nColumns      Number of columns in the table
     *  \param[in]  nRows         Number of rows in the table
     *  \param[in]  featuresEqual Flag that makes all features in the NumericTableDictionary equal
     *  \param[out] stat          Status of the numeric table construction
     *  \return Empty numeric table
     */
    static services::SharedPtr<SyclSOANumericTable> create(size_t nColumns = 0, size_t nRows = 0,
                                                           DictionaryIface::FeaturesEqual featuresEqual = DictionaryIface::notEqual,
                                                           services::Status * stat                      = NULL)
    {
        DAAL_DEFAULT_CREATE_IMPL_EX(SyclSOANumericTable, nColumns, nRows, featuresEqual);
    }

    static services::SharedPtr<SyclSOANumericTable> create(NumericTableDictionaryPtr ddict, size_t nRows,
                                                           AllocationFlag memoryAllocationFlag = notAllocate, services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_IMPL_EX(SyclSOANumericTable, ddict, nRows, memoryAllocationFlag);
    }

    virtual ~SyclSOANumericTable() { freeDataMemoryImpl(); }

    /**
     *  Sets an array of values for a given feature
     *  \tparam T       Type of feature values
     *  \param[in]  bf  SYCL* buffer to the array of the T type that stores feature values
     *  \param[in]  idx Feature index
     */
    template <typename T>
    services::Status setArray(const services::Buffer<T> & bf, size_t idx)
    {
        if (_partialMemStatus != notAllocated && _partialMemStatus != userAllocated)
        {
            return services::Status(services::ErrorIncorrectNumberOfFeatures);
        }

        if (idx < getNumberOfColumns() && idx < _arrays.size())
        {
            _ddict->setFeature<T>(idx);

            if (_arrays[idx].empty() && bf)
            {
                _arraysInitialized++;
            }
            else if (!_arrays[idx].empty() && !bf)
            {
                _arraysInitialized--;
            }

            _arrays[idx] = oneapi::internal::UniversalBuffer(bf);

            if (isCpuTable())
            {
                return _cpuTable->setArray(bf.toHost(readOnly), idx);
            }
        }
        else
        {
            return services::Status(services::ErrorIncorrectNumberOfFeatures);
        }

        _partialMemStatus = userAllocated;

        if (_arraysInitialized == getNumberOfColumns())
        {
            _memStatus = userAllocated;
        }
        return services::Status();
    }

    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->getBlockOfRows(vector_idx, vector_num, rwflag, block);
        }

        return getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->getBlockOfRows(vector_idx, vector_num, rwflag, block);
        }

        return getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->getBlockOfRows(vector_idx, vector_num, rwflag, block);
        }

        return getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    services::Status releaseBlockOfRows(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->releaseBlockOfRows(block);
        }

        return releaseTBlock<double>(block);
    }
    services::Status releaseBlockOfRows(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->releaseBlockOfRows(block);
        }

        return releaseTBlock<float>(block);
    }
    services::Status releaseBlockOfRows(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->releaseBlockOfRows(block);
        }

        return releaseTBlock<int>(block);
    }

    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->getBlockOfColumnValues(feature_idx, vector_idx, value_num, rwflag, block);
        }

        return getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->getBlockOfColumnValues(feature_idx, vector_idx, value_num, rwflag, block);
        }

        return getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->getBlockOfColumnValues(feature_idx, vector_idx, value_num, rwflag, block);
        }

        return getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    services::Status releaseBlockOfColumnValues(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->releaseBlockOfColumnValues(block);
        }

        return releaseTFeature<double>(block);
    }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->releaseBlockOfColumnValues(block);
        }

        return releaseTFeature<float>(block);
    }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->releaseBlockOfColumnValues(block);
        }

        return releaseTFeature<int>(block);
    }

    virtual MemoryStatus getDataMemoryStatus() const DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->getDataMemoryStatus();
        }
        return NumericTable::getDataMemoryStatus();
    }

protected:
    explicit SyclSOANumericTable(size_t nColumns, size_t nRows, DictionaryIface::FeaturesEqual featuresEqual, services::Status & st)
        : SyclNumericTable(nColumns, nRows, featuresEqual), _arrays(nColumns), _arraysInitialized(0), _partialMemStatus(notAllocated)
    {
        _layout = soa;

        if (isCpuContext())
        {
            _cpuTable = SOANumericTable::create(nColumns, nRows, featuresEqual, &st);
        }
        else
        {
            if (!resizePointersArray(nColumns))
            {
                st.add(services::ErrorMemoryAllocationFailed);
                return;
            }
        }
    }

    explicit SyclSOANumericTable(NumericTableDictionaryPtr ddict, size_t nRows, AllocationFlag memoryAllocationFlag, services::Status & st)
        : SyclNumericTable(ddict, st), _arraysInitialized(0), _partialMemStatus(notAllocated)
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

    services::Status allocateArray(size_t idx, const NumericTableFeature & feature)
    {
        using namespace services;
        using namespace oneapi::internal;

        Status st;
        auto & context = oneapi::internal::getDefaultContext();
        size_t nrows   = getNumberOfRows();

        switch (feature.indexType)
        {
        case features::DAAL_INT8_U:
        {
            _arrays[idx] = context.allocate(TypeId::uint8, nrows, &st);
            break;
        }
        case features::DAAL_INT16_U:
        {
            _arrays[idx] = context.allocate(TypeId::uint16, nrows, &st);
            break;
        }
        case features::DAAL_INT32_U:
        {
            _arrays[idx] = context.allocate(TypeId::uint32, nrows, &st);
            break;
        }
        case features::DAAL_INT64_U:
        {
            _arrays[idx] = context.allocate(TypeId::uint64, nrows, &st);
            break;
        }

        case features::DAAL_INT8_S:
        {
            _arrays[idx] = context.allocate(TypeId::int8, nrows, &st);
            break;
        }
        case features::DAAL_INT16_S:
        {
            _arrays[idx] = context.allocate(TypeId::int16, nrows, &st);
            break;
        }
        case features::DAAL_INT32_S:
        {
            _arrays[idx] = context.allocate(TypeId::int32, nrows, &st);
            break;
        }
        case features::DAAL_INT64_S:
        {
            _arrays[idx] = context.allocate(TypeId::int64, nrows, &st);
            break;
        }

        case features::DAAL_FLOAT32:
        {
            _arrays[idx] = context.allocate(TypeId::float32, nrows, &st);
            break;
        }
        case features::DAAL_FLOAT64:
        {
            _arrays[idx] = context.allocate(TypeId::float64, nrows, &st);
            break;
        }

        default: st = Status(ErrorIncorrectParameter); break;
        }

        return st;
    }

    services::Status allocateDataMemoryImpl(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE
    {
        freeDataMemoryImpl();

        size_t ncol  = _ddict->getNumberOfFeatures();
        size_t nrows = getNumberOfRows();

        if (isCpuContext())
        {
            services::Status st;
            _cpuTable = SOANumericTable::create(_ddict, nrows, doAllocate, &st);
            return st;
        }
        else
        {
            if (ncol * nrows == 0)
            {
                if (nrows == 0)
                {
                    return services::Status(services::ErrorIncorrectNumberOfObservations);
                }
                else
                {
                    return services::Status(services::ErrorIncorrectNumberOfFeatures);
                }
            }

            for (size_t i = 0; i < ncol; i++)
            {
                NumericTableFeature f = (*_ddict)[i];
                if (f.typeSize != 0)
                {
                    DAAL_CHECK_STATUS_VAR(allocateArray(i, f));
                    _arraysInitialized++;
                }
                if (_arrays[i].empty())
                {
                    freeDataMemoryImpl();
                    return services::Status(services::ErrorMemoryAllocationFailed);
                }
            }

            if (_arraysInitialized > 0)
            {
                _partialMemStatus = internallyAllocated;
            }

            if (_arraysInitialized == ncol)
            {
                _memStatus = internallyAllocated;
            }
        }

        return services::Status();
    }

    bool resizePointersArray(size_t nColumns)
    {
        if (_arrays.size() >= nColumns)
        {
            size_t counter = 0;
            for (size_t i = 0; i < nColumns; i++)
            {
                counter += (_arrays[i].empty() != true);
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

        bool is_resized = _arrays.resize(nColumns);
        if (is_resized)
        {
            _memStatus = notAllocated;
        }

        return is_resized;
    }

    void freeDataMemoryImpl() DAAL_C11_OVERRIDE
    {
        _cpuTable.reset();
        _arrays.clear();
        _arrays.resize(_ddict->getNumberOfFeatures());
        _arraysInitialized = 0;

        _partialMemStatus = notAllocated;
        _memStatus        = notAllocated;
    }

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        using namespace oneapi::internal;

        NumericTable::serialImpl<Archive, onDeserialize>(arch);

        ReadWriteMode rwMode = readOnly;

        if (onDeserialize)
        {
            rwMode = readWrite;
            allocateDataMemoryImpl();
        }

        size_t ncol  = _ddict->getNumberOfFeatures();
        size_t nrows = getNumberOfRows();

        if (isCpuTable())
        {
            for (size_t i = 0; i < ncol; i++)
            {
                NumericTableFeature f = (*_ddict)[i];
                arch->set((char *)_cpuTable->getArray(i), nrows * f.typeSize);
            }
        }
        else
        {
            for (size_t i = 0; i < ncol; i++)
            {
                NumericTableFeature f = (*_ddict)[i];

                BufferHostReinterpreter<char> reinterpreter(_arrays[i], rwMode, nrows);
                TypeDispatcher::dispatch(_arrays[i].type(), reinterpreter);

                services::Status st;
                auto charPtr = reinterpreter.getResult(st);
                DAAL_CHECK_STATUS_VAR(st);

                arch->set(charPtr.get(), nrows * f.typeSize);
            }
        }

        return services::Status();
    }

private:
    template <typename T>
    services::Status getTBlock(size_t idx, size_t nrows, ReadWriteMode rwFlag, BlockDescriptor<T> & block)
    {
        using namespace oneapi::internal;

        size_t ncols = getNumberOfColumns();
        size_t nobs  = getNumberOfRows();
        block.setDetails(0, idx, rwFlag);

        if (idx >= nobs)
        {
            block.resizeBuffer(ncols, 0);
            return services::Status();
        }

        nrows = (idx + nrows < nobs) ? nrows : nobs - idx;

        if (!block.resizeBuffer(ncols, nrows))
        {
            return services::Status(services::ErrorMemoryAllocationFailed);
        }

        if (!(block.getRWFlag() & (int)readOnly))
        {
            return services::Status();
        }

        auto blockSharedPtr = block.getBlockSharedPtr();
        T * blockPtr        = blockSharedPtr.get();

        for (size_t j = 0; j < ncols; j++)
        {
            auto featureUniBuffer = _arrays[j];
            BufferConverterTo<T> converter(featureUniBuffer, idx, nrows);
            TypeDispatcher::dispatch(featureUniBuffer.type(), converter);

            services::Status st;
            auto buffer = converter.getResult(st);
            DAAL_CHECK_STATUS_VAR(st);

            auto colSharedPtr = buffer.toHost(readOnly, &st);
            DAAL_CHECK_STATUS_VAR(st);
            T * colPtr        = colSharedPtr.get();

            for (size_t i = 0; i < nrows; i++)
            {
                blockPtr[i * ncols + j] = colPtr[i];
            }
        }

        return services::Status();
    }

    template <typename T>
    services::Status releaseTBlock(BlockDescriptor<T> & block)
    {
        using namespace oneapi::internal;

        if (block.getRWFlag() & (int)writeOnly)
        {
            const size_t ncols = getNumberOfColumns();
            const size_t nrows = block.getNumberOfRows();
            services::Status st;

            auto blockBuffer    = block.getBuffer();
            auto blockSharedPtr = blockBuffer.toHost(readOnly, &st);
            DAAL_CHECK_STATUS_VAR(st);
            T * blockPtr        = blockSharedPtr.get();

            auto & context = getDefaultContext();
            auto tempColumn = context.allocate(TypeIds::id<T>(), nrows, &st);
            DAAL_CHECK_STATUS_VAR(st);

            for (size_t j = 0; j < ncols; j++)
            {
                {
                    auto tempColumnSharedPtr = tempColumn.template get<T>().toHost(readWrite, &st);
                    DAAL_CHECK_STATUS_VAR(st);
                    T * tempColumnPtr        = tempColumnSharedPtr.get();

                    for (size_t i = 0; i < nrows; i++)
                    {
                        tempColumnPtr[i] = blockPtr[i * ncols + j];
                    }
                }

                auto uniBuffer = _arrays[j];
                BufferConverterFrom<T> converter(tempColumn, uniBuffer, 0, nrows);
                TypeDispatcher::dispatch(uniBuffer.type(), converter);

                _arrays[j] = converter.getResult(st);
                DAAL_CHECK_STATUS_VAR(st);
            }
        }
        block.reset();
        return services::Status();
    }

    template <typename T>
    services::Status getTFeature(size_t feat_idx, size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> & block)
    {
        using namespace oneapi::internal;

        const size_t nobs  = getNumberOfRows();
        block.setDetails(feat_idx, idx, rwFlag);

        if (idx >= nobs)
        {
            block.resizeBuffer(1, 0);
            return services::Status(services::ErrorIncorrectIndex);
        }

        if (!(block.getRWFlag() & (int)readOnly))
        {
            return services::Status(services::ErrorIncorrectParameter);
        }

        nrows = (idx + nrows < nobs) ? nrows : nobs - idx;

        auto uniBuffer = _arrays[feat_idx];
        BufferConverterTo<T> converter(uniBuffer, idx, nrows);
        TypeDispatcher::dispatch(uniBuffer.type(), converter);
        services::Status st;

        auto buffer = converter.getResult(st);
        DAAL_CHECK_STATUS_VAR(st);
        block.setBuffer(buffer, 1, nrows);

        return services::Status();
    }

    template <typename T>
    services::Status releaseTFeature(BlockDescriptor<T> & block)
    {
        using namespace oneapi::internal;

        if (block.getRWFlag() & (int)writeOnly)
        {
            size_t feat_idx = block.getColumnsOffset();

            NumericTableFeature & f = (*_ddict)[feat_idx];

            if (features::internal::getIndexNumType<T>() != f.indexType)
            {
                auto uniBuffer = _arrays[feat_idx];
                BufferConverterFrom<T> converter(block.getBuffer(), uniBuffer, block.getRowsOffset(), block.getNumberOfRows());
                TypeDispatcher::dispatch(uniBuffer.type(), converter);

                services::Status st;
                _arrays[feat_idx] = converter.getResult(st);
                DAAL_CHECK_STATUS_VAR(st);
            }
        }
        block.reset();
        return services::Status();
    }

    inline bool isCpuTable() const { return (bool)_cpuTable; }

    static bool isCpuContext() { return oneapi::internal::getDefaultContext().getInfoDevice().isCpu; }

private:
    services::Collection<oneapi::internal::UniversalBuffer> _arrays;
    size_t _arraysInitialized;
    MemoryStatus _partialMemStatus;

    SOANumericTablePtr _cpuTable;
};

typedef services::SharedPtr<SyclSOANumericTable> SyclSOANumericTablePtr;
/** @} */
} // namespace interface1

using interface1::SyclSOANumericTable;
using interface1::SyclSOANumericTablePtr;

} // namespace data_management
} // namespace daal

#endif
