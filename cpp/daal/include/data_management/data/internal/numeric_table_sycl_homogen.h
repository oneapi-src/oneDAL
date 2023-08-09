/* file: numeric_table_sycl_homogen.h */
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

#ifndef __SYCL_HOMOGEN_NUMERIC_TABLE_H__
#define __SYCL_HOMOGEN_NUMERIC_TABLE_H__

#ifdef DAAL_SYCL_INTERFACE
    #include <sycl/sycl.hpp>
#endif

#include "data_management/data/internal/numeric_table_sycl.h"
#include "data_management/data/internal/conversion.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/internal/execution_context.h"

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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__SYCLHOMOGENNUMERICTABLE"></a>
 *  \brief Class that provides methods to access data stored as a one-dimentional SYCL* buffer.
 *  Table rows contain feature vectors, and columns contain values of individual features.
 *  \tparam DataType Defines the underlying data type that describes a Numeric Table
 */
template <typename DataType = DAAL_DATA_TYPE>
class DAAL_EXPORT SyclHomogenNumericTable : public SyclNumericTable
{
public:
    DECLARE_SERIALIZABLE_TAG()
    DECLARE_SERIALIZABLE_IMPL()

    DAAL_CAST_OPERATOR(SyclHomogenNumericTable)

public:
    /**
     *  Constructs a Numeric Table with buffer object
     *  \param[in]  buffer         Buffer with a homogeneous data set
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \param[out] stat           Status of the numeric table construction
     *  \return     Numeric table with user-allocated memory
     */
    static services::SharedPtr<SyclHomogenNumericTable<DataType> > create(const services::internal::Buffer<DataType> & buffer, size_t nColumns = 0,
                                                                          size_t nRows = 0, services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(SyclHomogenNumericTable, DataType, DictionaryIface::notEqual, buffer, nColumns, nRows);
    }

#ifdef DAAL_SYCL_INTERFACE_USM
    static services::SharedPtr<SyclHomogenNumericTable<DataType> > create(const services::SharedPtr<DataType> & usmData, size_t nColumns,
                                                                          size_t nRows, const ::sycl::queue & queue, services::Status * stat = NULL)
    {
        const size_t bufferSize = nColumns * nRows;

        // multiplication overflow check is done in the constructor.
        // its not a safety problem to postpone this check since services::internal::Buffer() constructor
        // do not perform any data allocations in case of input usm data - we can create it even with wrong bufferSize

        services::Status localStatus;
        services::internal::Buffer<DataType> buffer(usmData, bufferSize, queue, localStatus);
        services::internal::tryAssignStatusAndThrow(stat, localStatus);
        DAAL_CHECK_STATUS_RETURN_IF_FAIL(localStatus, services::SharedPtr<SyclHomogenNumericTable<DataType> >());

        return create(buffer, nColumns, nRows, stat);
    }
#endif

#ifdef DAAL_SYCL_INTERFACE_USM
    static services::SharedPtr<SyclHomogenNumericTable<DataType> > create(DataType * usmData, size_t nColumns, size_t nRows,
                                                                          const ::sycl::queue & queue, services::Status * stat = NULL)
    {
        const auto overflow_status = checkSizeOverflow(nRows, nColumns);
        if (!overflow_status)
        {
            services::throwIfPossible(overflow_status);
            DAAL_CHECK_COND_ERROR(stat, *stat, overflow_status);
            return services::SharedPtr<SyclHomogenNumericTable<DataType> >();
        }
        const size_t bufferSize = nColumns * nRows;

        services::Status localStatus;
        services::internal::Buffer<DataType> buffer(usmData, bufferSize, queue, localStatus);
        services::internal::tryAssignStatusAndThrow(stat, localStatus);
        DAAL_CHECK_STATUS_RETURN_IF_FAIL(localStatus, services::SharedPtr<SyclHomogenNumericTable<DataType> >());

        return create(buffer, nColumns, nRows, stat);
    }
#endif

    /**
     *  Constructs a Numeric Table
     *  \param[in]  nColumns              Number of columns in the table
     *  \param[in]  nRows                 Number of rows in the table
     *  \param[in]  memoryAllocationFlag  Flag that controls internal memory allocation for data in the numeric table
     *  \param[out] stat                  Status of the numeric table construction
     *  \return     Numeric table with user-allocated memory
     */
    static services::SharedPtr<SyclHomogenNumericTable<DataType> > create(size_t nColumns, size_t nRows, AllocationFlag memoryAllocationFlag,
                                                                          services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(SyclHomogenNumericTable, DataType, DictionaryIface::notEqual, nColumns, nRows, memoryAllocationFlag);
    }

    /**
     *  Constructs a Numeric Table with memory allocation controlled via a flag and fills the table with a constant
     *  \param[in]  nColumns                Number of columns in the table
     *  \param[in]  nRows                   Number of rows in the table
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \param[in]  constValue              Constant to initialize entries of the homogeneous numeric table
     *  \param[out] stat                    Status of the numeric table construction
     *  \return     Numeric table initialized with a constant
     */
    static services::SharedPtr<SyclHomogenNumericTable<DataType> > create(size_t nColumns, size_t nRows, AllocationFlag memoryAllocationFlag,
                                                                          const DataType & constValue, services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(SyclHomogenNumericTable, DataType, DictionaryIface::notEqual, nColumns, nRows, memoryAllocationFlag,
                                             constValue);
    }

    SyclHomogenNumericTable() : SyclNumericTable(0, 0, DictionaryIface::notEqual) {}

    ~SyclHomogenNumericTable() DAAL_C11_OVERRIDE
    {
        freeDataMemoryImpl();
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

    services::Status releaseBlockOfRows(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return releaseTBlock<double>(block);
    }

    services::Status releaseBlockOfRows(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return releaseTBlock<float>(block);
    }

    services::Status releaseBlockOfRows(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return releaseTBlock<int>(block);
    }

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

    services::Status releaseBlockOfColumnValues(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return releaseTFeature<double>(block);
    }

    services::Status releaseBlockOfColumnValues(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return releaseTFeature<float>(block);
    }

    services::Status releaseBlockOfColumnValues(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return releaseTFeature<int>(block);
    }

    services::Status assign(float value) DAAL_C11_OVERRIDE
    {
        return assignImpl<float>(value);
    }

    services::Status assign(double value) DAAL_C11_OVERRIDE
    {
        return assignImpl<double>(value);
    }

    services::Status assign(int value) DAAL_C11_OVERRIDE
    {
        return assignImpl<int>(value);
    }

protected:
    SyclHomogenNumericTable(DictionaryIface::FeaturesEqual featuresEqual, size_t nColumns, size_t nRows, services::Status & st)
        : SyclNumericTable(nColumns, nRows, featuresEqual, st)
    {
        _layout = NumericTableIface::aos;

        NumericTableFeature df;
        df.setType<DataType>();
        st |= _ddict->setAllFeatures(df);
        services::throwIfPossible(st);
    }

    SyclHomogenNumericTable(DictionaryIface::FeaturesEqual featuresEqual, const services::internal::Buffer<DataType> & buffer, size_t nColumns,
                            size_t nRows, services::Status & st)
        : SyclHomogenNumericTable(featuresEqual, nColumns, nRows, st)
    {
        st |= checkSizeOverflow(nRows, nColumns);
        services::throwIfPossible(st);

        if (nColumns * nRows > buffer.size())
        {
            st |= services::Error::create(services::ErrorIncorrectSizeOfArray, services::Row, "Buffer size is not enough to represent the table");
            services::throwIfPossible(st);
        }

        if (st)
        {
            _buffer    = buffer;
            _memStatus = userAllocated;
        }
    }

    SyclHomogenNumericTable(DictionaryIface::FeaturesEqual featuresEqual, size_t nColumns, size_t nRows,
                            NumericTable::AllocationFlag memoryAllocationFlag, services::Status & st)
        : SyclHomogenNumericTable(featuresEqual, nColumns, nRows, st)
    {
        if (memoryAllocationFlag == NumericTableIface::doAllocate)
        {
            st |= allocateDataMemoryImpl();
        }
    }

    SyclHomogenNumericTable(DictionaryIface::FeaturesEqual featuresEqual, size_t nColumns, size_t nRows,
                            NumericTable::AllocationFlag memoryAllocationFlag, const DataType & constValue, services::Status & st)
        : SyclHomogenNumericTable(featuresEqual, nColumns, nRows, memoryAllocationFlag, st)
    {
        st |= assignImpl<DataType>(constValue);
    }

    services::Status allocateDataMemoryImpl(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE
    {
        if (type != daal::dram)
        {
            return services::throwIfPossible(services::ErrorIncorrectParameter);
        }

        services::Status status;

        freeDataMemoryImpl();

        if (!getNumberOfRows() || !getNumberOfColumns())
        {
            return status;
        }

        if (isCpuContext())
        {
            status |= allocateDataMemoryOnCpu();
            DAAL_CHECK_STATUS_VAR(status);
        }
        else
        {
            status |= checkSizeOverflow(getNumberOfColumns(), getNumberOfRows());
            if (!status) return services::throwIfPossible(status);

            const size_t size = getNumberOfColumns() * getNumberOfRows();
            const auto universalBuffer =
                services::internal::getDefaultContext().allocate(services::internal::sycl::TypeIds::id<DataType>(), size, status);

            if (!status) return services::throwIfPossible(status);

            _buffer = universalBuffer.template get<DataType>();
        }

        _memStatus = internallyAllocated;
        return status;
    }

    void freeDataMemoryImpl() DAAL_C11_OVERRIDE
    {
        _buffer.reset();
        _cpuTable.reset();
        _memStatus = notAllocated;
    }

    services::Status setNumberOfColumnsImpl(size_t ncol) DAAL_C11_OVERRIDE
    {
        services::Status status;

        if (isCpuTable())
        {
            status |= _cpuTable->setNumberOfColumns(ncol);
            if (!status) return services::throwIfPossible(status);
        }

        if (_ddict->getNumberOfFeatures() != ncol)
        {
            status |= _ddict->resetDictionary();
            if (!status) return services::throwIfPossible(status);

            status |= _ddict->setNumberOfFeatures(ncol);
            if (!status) return services::throwIfPossible(status);

            NumericTableFeature df;
            df.setType<DataType>();
            status |= _ddict->setAllFeatures(df);
            if (!status) return services::throwIfPossible(status);
        }

        return status;
    }

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * archive)
    {
        auto st = NumericTable::serialImpl<Archive, onDeserialize>(archive);
        DAAL_CHECK_STATUS_VAR(st);

        if (onDeserialize)
        {
            st |= allocateDataMemoryImpl();
            DAAL_CHECK_STATUS_VAR(st);
        }

        const size_t size = getNumberOfColumns() * getNumberOfRows();
        // overflow checks done in constructors and allocateDataMemoryImpl() method

        if (isCpuTable())
        {
            archive->set(_cpuTable->getArray(), size);
        }
        else
        {
            const auto hostData = _buffer.toHost(onDeserialize ? data_management::writeOnly : data_management::readOnly, st);
            if (!st) return services::throwIfPossible(st);

            archive->set(hostData.get(), size);
        }

        return st;
    }

    template <typename T>
    services::Status assignImpl(T value)
    {
        services::Status status;

        if (_memStatus == notAllocated)
        {
            status |= services::Status(services::ErrorEmptyHomogenNumericTable);
            return services::throwIfPossible(status);
        }

        if (isCpuTable())
        {
            return _cpuTable->assign(value);
        }

        services::internal::getDefaultContext().fill(_buffer, (double)value, status);
        return services::throwIfPossible(status);
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

    template <typename T, typename U>
    struct BufferIO
    {
        static services::Status read(const services::internal::Buffer<U> & buffer, BlockDescriptor<T> & block, size_t nRows, size_t nCols)
        {
            DAAL_ASSERT(buffer.size() == nRows * nCols);
            services::Status status;

            if (!block.resizeBuffer(nCols, nRows))
            {
                return services::throwIfPossible(services::ErrorMemoryAllocationFailed);
            }

            auto hostPtr = buffer.toHost(data_management::readOnly, status);
            if (!status) return services::throwIfPossible(status);

            internal::VectorUpCast<U, T>()(nRows * nCols, hostPtr.get(), block.getBlockPtr());

            return status;
        }

        static services::Status write(services::internal::Buffer<U> buffer, const BlockDescriptor<T> & block, size_t nRows, size_t nCols)
        {
            services::Status status;

            DAAL_ASSERT(block.getNumberOfRows() == nRows);
            DAAL_ASSERT(block.getNumberOfColumns() == nCols);
            DAAL_ASSERT(buffer.size() == nRows * nCols);

            auto hostPtr = buffer.toHost(data_management::writeOnly, status);
            if (!status) return services::throwIfPossible(status);

            if (!block.getBlockPtr())
            {
                return services::throwIfPossible(services::ErrorNullPtr);
            }

            internal::VectorDownCast<T, U>()(nRows * nCols, block.getBlockPtr(), hostPtr.get());

            return status;
        }
    };

    template <typename T>
    struct BufferIO<T, T>
    {
        static services::Status read(const services::internal::Buffer<T> & buffer, BlockDescriptor<T> & block, size_t nRows, size_t nCols)
        {
            DAAL_ASSERT(buffer.size() == nRows * nCols);

            block.setBuffer(buffer, nCols, nRows);
            return services::Status();
        }

        static services::Status write(services::internal::Buffer<T> buffer, const BlockDescriptor<T> & block, size_t nRows, size_t nCols)
        {
            // The case when user calls block.setBuffer() on their side is not supported
            // SYCL have no API to check that two buffers or subbuffers point to the same memory.
            // Use of block.setBuffer() should be reviewed manually in the algorithms
            return services::Status();
        }
    };

    services::internal::Buffer<DataType> getSubBuffer(size_t rowOffset, size_t nRows, services::Status & st)
    {
        DAAL_ASSERT(rowOffset < getNumberOfRows());
        DAAL_ASSERT(nRows <= getNumberOfRows());

        const size_t nCols  = getNumberOfColumns();
        const size_t offset = rowOffset * nCols;
        const size_t size   = nRows * nCols;

        // Checks on offset+size correctness are done in getTBlock(), releaseTBlock() functions

        if (size == _buffer.size())
        {
            return _buffer;
        }
        services::internal::Buffer<DataType> subBuffer = _buffer.getSubBuffer(offset, size, st);
        services::throwIfPossible(st);

        return subBuffer;
    }

    template <typename T>
    services::Status getTBlock(size_t rowOffset, size_t nRowsBlockDesired, ReadWriteMode rwFlag, BlockDescriptor<T> & block)
    {
        if (isCpuTable())
        {
            return _cpuTable->getBlockOfRows(rowOffset, nRowsBlockDesired, rwFlag, block);
        }

        services::Status status;

        const size_t nRows = getNumberOfRows();
        const size_t nCols = getNumberOfColumns();
        block.setDetails(0, rowOffset, rwFlag);

        if (rowOffset >= nRows)
        {
            block.reset();
            return services::Status();
        }

        auto st = checkOffsetOverflow(nRowsBlockDesired, rowOffset);
        if (!st) return services::throwIfPossible(st);

        const size_t nRowsBlock = (rowOffset + nRowsBlockDesired < nRows) ? nRowsBlockDesired : nRows - rowOffset;

        auto subbuffer = getSubBuffer(rowOffset, nRowsBlock, st);
        DAAL_CHECK_STATUS_VAR(st);

        st |= BufferIO<T, DataType>::read(subbuffer, block, nRowsBlock, nCols);
        return st;
    }

    template <typename T>
    services::Status releaseTBlock(BlockDescriptor<T> & block)
    {
        if (isCpuTable())
        {
            return _cpuTable->releaseBlockOfRows(block);
        }

        services::Status status;

        if (block.getRWFlag() & (int)writeOnly)
        {
            const size_t nCols      = getNumberOfColumns();
            const size_t nRows      = getNumberOfRows();
            const size_t nRowsBlock = block.getNumberOfRows();
            const size_t rowOffset  = block.getRowsOffset();

            status |= checkOffsetOverflow(nRowsBlock, rowOffset);
            if (!status) return throwIfPossible(status);

            if ((nRowsBlock + rowOffset) > nRows || nCols != block.getNumberOfColumns())
            {
                return services::throwIfPossible(services::ErrorIncorrectParameter);
            }
            auto subbuffer = getSubBuffer(rowOffset, nRowsBlock, status);
            DAAL_CHECK_STATUS_VAR(status);

            status |= BufferIO<T, DataType>::write(subbuffer, block, nRowsBlock, nCols);
        }

        block.reset();
        return status;
    }

    template <typename T>
    services::Status getTFeature(size_t columnIndex, size_t rowOffset, size_t nRowsBlockDesired, ReadWriteMode rwFlag, BlockDescriptor<T> & block)
    {
        if (isCpuTable())
        {
            return _cpuTable->getBlockOfColumnValues(columnIndex, rowOffset, nRowsBlockDesired, rwFlag, block);
        }

        return services::throwIfPossible(services::ErrorMethodNotImplemented);
    }

    template <typename T>
    services::Status releaseTFeature(BlockDescriptor<T> & block)
    {
        if (isCpuTable())
        {
            return _cpuTable->releaseBlockOfColumnValues(block);
        }

        return services::throwIfPossible(services::ErrorMethodNotImplemented);
    }

    services::Status allocateDataMemoryOnCpu()
    {
        services::Status status;

        _cpuTable = HomogenNumericTable<DataType>::create(getNumberOfColumns(), getNumberOfRows(), NumericTableIface::doAllocate, &status);

        return status;
    }

    inline bool isCpuTable() const
    {
        return (bool)_cpuTable;
    }

    static bool isCpuContext()
    {
        return services::internal::getDefaultContext().getInfoDevice().isCpu;
    }

    services::internal::Buffer<DataType> _buffer;
    services::SharedPtr<HomogenNumericTable<DataType> > _cpuTable;
};
/** @} */

/**
 * Converts numeric table with arbitrary storage layout to SYCL homogen numeric table of the given type
 * \param[in]  src               Numeric table to be converted
 * \param[in]  st                Status of conversion
 * \return                       Pointer to SYCL homogen numeric table
 */
template <typename T>
inline daal::data_management::NumericTablePtr convertToSyclHomogen(NumericTable & src, services::Status & st)
{
    using namespace daal::services;

    size_t ncols = src.getNumberOfColumns();
    size_t nrows = src.getNumberOfRows();
    daal::data_management::NumericTablePtr emptyPtr;

    NumericTablePtr dst = SyclHomogenNumericTable<T>::create(ncols, nrows, NumericTableIface::doAllocate, &st);
    DAAL_CHECK_STATUS_RETURN_IF_FAIL(st, emptyPtr);
    BlockDescriptor<T> srcBlock;
    st |= src.getBlockOfRows(0, nrows, readOnly, srcBlock);
    DAAL_CHECK_STATUS_RETURN_IF_FAIL(st, emptyPtr);
    BlockDescriptor<T> dstBlock;
    st |= dst->getBlockOfRows(0, nrows, readOnly, dstBlock);
    DAAL_CHECK_STATUS_RETURN_IF_FAIL(st, emptyPtr);
    T * srcData      = srcBlock.getBlockPtr();
    auto hostDstData = dstBlock.getBuffer().toHost(writeOnly, st);
    DAAL_CHECK_STATUS_RETURN_IF_FAIL(st, emptyPtr);
    T * dstData = hostDstData.get();
    for (size_t i = 0; i < ncols * nrows; i++)
    {
        dstData[i] = srcData[i];
    }
    st |= src.releaseBlockOfRows(srcBlock);
    DAAL_CHECK_STATUS_RETURN_IF_FAIL(st, emptyPtr);
    st |= dst->releaseBlockOfRows(dstBlock);
    DAAL_CHECK_STATUS_RETURN_IF_FAIL(st, emptyPtr);
    return dst;
}

} // namespace interface1

using interface1::SyclHomogenNumericTable;
using interface1::convertToSyclHomogen;

} // namespace internal
} // namespace data_management
} // namespace daal

#endif
