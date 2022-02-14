/* file: numeric_table_sycl_soa.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#include "data_management/data/internal/numeric_table_sycl.h"
#include "data_management/data/soa_numeric_table.h"
#include "services/internal/sycl/buffer_utils.h"

namespace daal
{
namespace data_management
{
namespace internal
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
    DECLARE_SERIALIZABLE_TAG()
    DECLARE_SERIALIZABLE_IMPL()

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
    services::Status setArray(const services::internal::Buffer<T> & bf, size_t idx)
    {
        if (_partialMemStatus != notAllocated && _partialMemStatus != userAllocated)
        {
            return services::throwIfPossible(services::ErrorIncorrectNumberOfFeatures);
        }

        if (idx >= getNumberOfColumns() || idx >= _arrays.size())
        {
            return services::throwIfPossible(services::ErrorIncorrectNumberOfFeatures);
        }

        if (getNumberOfRows() != bf.size())
        {
            return services::throwIfPossible(services::ErrorIncorrectParameter);
        }

        _ddict->setFeature<T>(idx);

        if (_arrays[idx].empty() && bf)
        {
            _arraysInitialized++;
        }
        else if (!_arrays[idx].empty() && !bf)
        {
            _arraysInitialized--;
        }

        _arrays[idx] = services::internal::sycl::UniversalBuffer(bf);

        if (isCpuTable())
        {
            services::Status status;
            auto hostPtr = bf.toHost(readOnly, status);
            DAAL_CHECK_STATUS_VAR(status);
            return _cpuTable->setArray(hostPtr, idx);
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
        return getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    services::Status releaseBlockOfRows(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTBlock<double>(block); }
    services::Status releaseBlockOfRows(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTBlock<float>(block); }
    services::Status releaseBlockOfRows(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTBlock<int>(block); }

    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    services::Status releaseBlockOfColumnValues(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTFeature<double>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTFeature<float>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTFeature<int>(block); }

    virtual MemoryStatus getDataMemoryStatus() const DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->getDataMemoryStatus();
        }
        return _memStatus;
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
                services::throwIfPossible(st);
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
            services::throwIfPossible(st);
            return;
        }
        if (memoryAllocationFlag == doAllocate)
        {
            st |= allocateDataMemoryImpl();
            return;
        }
    }

    services::Status allocateArray(size_t idx, const NumericTableFeature & feature)
    {
        using namespace services;
        using namespace services::internal::sycl;

        Status st;
        const size_t nrows = getNumberOfRows();

        if (idx >= _arrays.size())
        {
            return throwIfPossible(services::ErrorIncorrectNumberOfFeatures);
        }

        _arrays[idx] = allocateByNumericTableFeature(feature, nrows, st);
        services::throwIfPossible(st);
        return st;
    }

    services::Status allocateDataMemoryImpl(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(type == daal::dram);

        freeDataMemoryImpl();

        const size_t ncol  = _ddict->getNumberOfFeatures();
        const size_t nrows = getNumberOfRows();

        if (isCpuContext())
        {
            services::Status st;
            _cpuTable = SOANumericTable::create(_ddict, nrows, doAllocate, &st);
            return st;
        }
        else
        {
            auto status = checkSizeOverflow(nrows, ncol);
            if (!status) return services::throwIfPossible(status);

            if (ncol * nrows == 0)
            {
                if (nrows == 0)
                {
                    return services::throwIfPossible(services::ErrorIncorrectNumberOfObservations);
                }
                else
                {
                    return services::throwIfPossible(services::ErrorIncorrectNumberOfFeatures);
                }
            }

            for (size_t i = 0; i < ncol; i++)
            {
                NumericTableFeature f = (*_ddict)[i];
                if (f.typeSize != 0)
                {
                    status |= allocateArray(i, f);
                    DAAL_CHECK_STATUS_VAR(status);
                    _arraysInitialized++;
                }
                if (_arrays[i].empty())
                {
                    freeDataMemoryImpl();
                    status.add(services::ErrorMemoryAllocationFailed);
                    return services::throwIfPossible(status);
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
        _arrays            = services::Collection<services::internal::sycl::UniversalBuffer>(_ddict->getNumberOfFeatures());
        _arraysInitialized = 0;

        _partialMemStatus = notAllocated;
        _memStatus        = notAllocated;
    }

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        using namespace services::internal::sycl;

        auto status = NumericTable::serialImpl<Archive, onDeserialize>(arch);
        DAAL_CHECK_STATUS_VAR(status);

        ReadWriteMode rwMode = readOnly;

        if (onDeserialize)
        {
            rwMode = readWrite;
            status |= allocateDataMemoryImpl();
            DAAL_CHECK_STATUS_VAR(status);
        }

        const size_t ncol  = _ddict->getNumberOfFeatures();
        const size_t nrows = getNumberOfRows();

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
                TypeDispatcher::dispatch(_arrays[i].type(), reinterpreter, status);
                services::throwIfPossible(status);
                DAAL_CHECK_STATUS_VAR(status);

                auto charPtr = reinterpreter.getResult();
                arch->set(charPtr.get(), nrows * f.typeSize);
            }
        }

        return services::Status();
    }

private:
    static services::Status checkSizeOverflow(size_t nRows, size_t nCols)
    {
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows, nCols);
        return services::Status();
    }

    static services::Status checkOffsetOverflow(size_t size, size_t offset)
    {
        DAAL_OVERFLOW_CHECK_BY_ADDING(size_t, size, offset);
        return services::Status();
    }

    template <typename T>
    services::Status getTBlock(size_t idx, size_t nrows, ReadWriteMode rwFlag, BlockDescriptor<T> & block)
    {
        using namespace services::internal::sycl;

        if (isCpuTable())
        {
            return _cpuTable->getBlockOfRows(idx, nrows, rwFlag, block);
        }

        const size_t ncols = getNumberOfColumns();
        const size_t nobs  = getNumberOfRows();
        block.setDetails(0, idx, rwFlag);

        if (idx >= nobs)
        {
            if (!block.resizeBuffer(ncols, 0))
            {
                return services::throwIfPossible(services::ErrorMethodNotSupported);
            }
            return services::Status();
        }

        auto status = checkOffsetOverflow(nrows, idx);
        if (!status) return services::throwIfPossible(status);

        nrows = (idx + nrows < nobs) ? nrows : nobs - idx;

        if (!block.resizeBuffer(ncols, nrows))
        {
            return services::throwIfPossible(services::ErrorMemoryAllocationFailed);
        }

        if (!(block.getRWFlag() & (int)readOnly))
        {
            return services::Status();
        }

        auto blockSharedPtr = block.getBlockSharedPtr();
        T * blockPtr        = blockSharedPtr.get();

        DAAL_ASSERT(_arrays.size() == ncols);

        for (size_t j = 0; j < ncols; j++)
        {
            services::Status st;
            auto featureUniBuffer = _arrays[j];
            BufferConverterTo<T> converter(featureUniBuffer, idx, nrows);
            TypeDispatcher::dispatch(featureUniBuffer.type(), converter, st);
            services::throwIfPossible(st);
            DAAL_CHECK_STATUS_VAR(st);

            auto buffer = converter.getResult();
            DAAL_ASSERT(buffer.size() == nrows);

            auto colSharedPtr = buffer.toHost(readOnly, st);
            services::throwIfPossible(st);
            DAAL_CHECK_STATUS_VAR(st);
            T * colPtr = colSharedPtr.get();

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
        using namespace services::internal::sycl;

        if (isCpuTable())
        {
            return _cpuTable->releaseBlockOfRows(block);
        }

        if (block.getRWFlag() & (int)writeOnly)
        {
            const size_t ncols = getNumberOfColumns();
            const size_t nrows = block.getNumberOfRows();
            services::Status st;

            if (block.getNumberOfColumns() != ncols)
            {
                st.add(services::ErrorIncorrectParameter);
                return throwIfPossible(st);
            }

            auto blockBuffer    = block.getBuffer();
            auto blockSharedPtr = blockBuffer.toHost(readOnly, st);
            if (!st) return services::throwIfPossible(st);

            T * blockPtr = blockSharedPtr.get();

            auto & context  = services::internal::getDefaultContext();
            auto tempColumn = context.allocate(TypeIds::id<T>(), nrows, st);
            if (!st) return services::throwIfPossible(st);

            for (size_t j = 0; j < ncols; j++)
            {
                {
                    auto tempColumnSharedPtr = tempColumn.template get<T>().toHost(readWrite, st);
                    if (!st) return services::throwIfPossible(st);

                    T * tempColumnPtr = tempColumnSharedPtr.get();

                    for (size_t i = 0; i < nrows; i++)
                    {
                        tempColumnPtr[i] = blockPtr[i * ncols + j];
                    }
                }

                auto uniBuffer = _arrays[j];
                BufferConverterFrom<T> converter(tempColumn, uniBuffer, 0, nrows);
                TypeDispatcher::dispatch(uniBuffer.type(), converter, st);
                services::throwIfPossible(st);
                DAAL_CHECK_STATUS_VAR(st);

                _arrays[j] = converter.getResult();
            }
        }
        block.reset();
        return services::Status();
    }

    template <typename T>
    services::Status getTFeature(size_t feat_idx, size_t idx, size_t nrows, ReadWriteMode rwFlag, BlockDescriptor<T> & block)
    {
        using namespace services::internal::sycl;

        if (isCpuTable())
        {
            return _cpuTable->getBlockOfColumnValues(feat_idx, idx, nrows, rwFlag, block);
        }

        const size_t nobs  = getNumberOfRows();
        const size_t ncols = getNumberOfColumns();

        if (feat_idx >= ncols)
        {
            return services::throwIfPossible(services::ErrorIncorrectIndex);
        }

        block.setDetails(feat_idx, idx, rwFlag);

        if (idx >= nobs)
        {
            if (!block.resizeBuffer(1, 0))
            {
                return services::throwIfPossible(services::ErrorMethodNotSupported);
            }
            return services::Status();
        }

        auto st = checkOffsetOverflow(nrows, idx);
        if (!st) return services::throwIfPossible(st);

        nrows = (idx + nrows < nobs) ? nrows : nobs - idx;
        if (!(block.getRWFlag() & (int)readOnly))
        {
            if (!block.resizeBuffer(1, nrows))
            {
                return services::throwIfPossible(services::ErrorMemoryAllocationFailed);
            }
            return services::Status();
        }

        auto uniBuffer = _arrays[feat_idx];
        BufferConverterTo<T> converter(uniBuffer, idx, nrows);
        TypeDispatcher::dispatch(uniBuffer.type(), converter, st);
        services::throwIfPossible(st);
        DAAL_CHECK_STATUS_VAR(st);

        auto buffer = converter.getResult();
        block.setBuffer(buffer, 1, nrows);

        return st;
    }

    template <typename T>
    services::Status releaseTFeature(BlockDescriptor<T> & block)
    {
        using namespace services::internal::sycl;

        if (isCpuTable())
        {
            return _cpuTable->releaseBlockOfColumnValues(block);
        }

        if (block.getRWFlag() & (int)writeOnly)
        {
            const size_t feat_idx = block.getColumnsOffset();

            if (feat_idx >= getNumberOfColumns())
            {
                return services::throwIfPossible(services::ErrorIncorrectIndex);
            }

            NumericTableFeature & f = (*_ddict)[feat_idx];

            auto uniBuffer   = _arrays[feat_idx];
            auto blockBuffer = block.getBuffer();
            if ((features::internal::getIndexNumType<T>() != f.indexType) || (uniBuffer.get<T>() != blockBuffer))
            {
                services::Status st;

                auto uniBuffer = _arrays[feat_idx];
                BufferConverterFrom<T> converter(block.getBuffer(), uniBuffer, block.getRowsOffset(), block.getNumberOfRows());
                TypeDispatcher::dispatch(uniBuffer.type(), converter, st);
                services::throwIfPossible(st);
                DAAL_CHECK_STATUS_VAR(st);

                _arrays[feat_idx] = converter.getResult();
            }
        }
        block.reset();
        return services::Status();
    }

    inline bool isCpuTable() const { return (bool)_cpuTable; }

    static bool isCpuContext() { return services::internal::getDefaultContext().getInfoDevice().isCpu; }

private:
    services::Collection<services::internal::sycl::UniversalBuffer> _arrays;
    size_t _arraysInitialized;
    MemoryStatus _partialMemStatus;

    SOANumericTablePtr _cpuTable;
};

typedef services::SharedPtr<SyclSOANumericTable> SyclSOANumericTablePtr;
/** @} */
} // namespace interface1

using interface1::SyclSOANumericTable;
using interface1::SyclSOANumericTablePtr;

} // namespace internal
} // namespace data_management
} // namespace daal

#endif
