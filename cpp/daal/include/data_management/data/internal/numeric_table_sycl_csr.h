/* file: numeric_table_sycl_csr.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

/*
//++
//  Implementation of a compressed sparse row (CSR) numeric table.
//--
*/

#ifndef __SYCL_CSR_NUMERIC_TABLE_H__
#define __SYCL_CSR_NUMERIC_TABLE_H__

#include "services/base.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/csr_numeric_table.h"
#include "data_management/data/data_serialize.h"
#include "data_management/data/internal/conversion.h"
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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__SYCLCSRNUMERICTABLE"></a>
 *  \brief Class that provides methods to access data stored in the CSR layout.
 *         Each array is represented by SYCL* buffer.
 */
class DAAL_EXPORT SyclCSRNumericTable : public SyclNumericTable, public CSRNumericTableIface
{
public:
    DECLARE_SERIALIZABLE_TAG()
    DECLARE_SERIALIZABLE_IMPL()

    DAAL_CAST_OPERATOR(SyclCSRNumericTable)

    /**
     *  Constructs SYCL CSR numeric table with user-allocated memory
     *  \tparam   DataType             Type of values in the Numeric Table
     *  \param[in]    bufferData       Buffer of values in the CSR layout. Let ptr_size denote the size of an array ptr
     *  \param[in]    bufferColIndices Buffer of column indices in the CSR layout. Values of indices are determined by the index base
     *  \param[in]    bufferRowOffsets Buffer of row indices in the CSR layout. Size of the array is nrow+1. The first element is 0/1
     *                                 in zero-/one-based indexing. The last element is ptr_size+0/1 in zero-/one-based indexing
     *  \param[in]    nColumns         Number of columns in the corresponding dense table
     *  \param[in]    nRows            Number of rows in the corresponding dense table
     *  \param[in]    indexing         Indexing scheme used to access data in the CSR layout
     *  \param[out]   stat             Status of the numeric table construction
     *  \return       SYCL CSR numeric table with user-allocated memory
     *  \note Present version of Intel(R) oneAPI Data Analytics Library supports 1-based indexing only
     */
    template <typename DataType>
    static services::SharedPtr<SyclCSRNumericTable> create(const services::internal::Buffer<DataType> & bufferData,
                                                           const services::internal::Buffer<size_t> & bufferColIndices,
                                                           const services::internal::Buffer<size_t> & bufferRowOffsets, size_t nColumns, size_t nRows,
                                                           CSRIndexing indexing = oneBased, services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_IMPL_EX(SyclCSRNumericTable, bufferData, bufferColIndices, bufferRowOffsets, nColumns, nRows, indexing);
    }

    virtual ~SyclCSRNumericTable() { freeDataMemoryImpl(); }

    virtual services::Status resize(size_t nrows) DAAL_C11_OVERRIDE { return setNumberOfRowsImpl(nrows); }

    /**
     *  Returns buffers to a data set stored in the CSR layout
     *  \param[out]    values      Buffer of values in the CSR layout
     *  \param[out]    colIndices  Buffer of column indices in the CSR layout
     *  \param[out]    rowOffsets  Buffer of row indices in the CSR layout
     */
    template <typename DataType>
    services::Status getArrays(services::internal::Buffer<DataType> & values, services::internal::Buffer<size_t> & colIndices,
                               services::internal::Buffer<size_t> & rowOffsets) const
    {
        values     = _values.get<DataType>();
        colIndices = _colIndices;
        rowOffsets = _rowOffsets;
        return services::Status();
    }
    /**
     *  Sets a buffers to a CSR data set
     *  \param[in]    values      Buffer of values in the CSR layout
     *  \param[in]    colIndices  Buffer of column indices in the CSR layout
     *  \param[in]    rowOffsets  Buffer of row indices in the CSR layout
     *  \param[in]    indexing    The indexing scheme for access to data in the CSR layout
     */
    template <typename DataType>
    services::Status setArrays(const services::internal::Buffer<DataType> & values, const services::internal::Buffer<size_t> & colIndices,
                               const services::internal::Buffer<size_t> & rowOffsets, CSRIndexing indexing = oneBased)
    {
        freeDataMemoryImpl();
        _values     = values;
        _colIndices = colIndices;
        _rowOffsets = rowOffsets;
        _indexing   = indexing;
        _dataSize   = values.size();

        if (values && colIndices && rowOffsets)
        {
            _memStatus = userAllocated;
        }
        return services::Status();
    }

    virtual services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag,
                                            BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->getBlockOfRows(vector_idx, vector_num, rwflag, block);
        }
        return getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    virtual services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag,
                                            BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->getBlockOfRows(vector_idx, vector_num, rwflag, block);
        }

        return getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    virtual services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag,
                                            BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->getBlockOfRows(vector_idx, vector_num, rwflag, block);
        }

        return getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    virtual services::Status releaseBlockOfRows(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->releaseBlockOfRows(block);
        }

        return releaseTBlock<double>(block);
    }
    virtual services::Status releaseBlockOfRows(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->releaseBlockOfRows(block);
        }

        return releaseTBlock<float>(block);
    }
    virtual services::Status releaseBlockOfRows(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->releaseBlockOfRows(block);
        }

        return releaseTBlock<int>(block);
    }

    virtual services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                                    BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->getBlockOfColumnValues(feature_idx, vector_idx, value_num, rwflag, block);
        }

        return getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    virtual services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                                    BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->getBlockOfColumnValues(feature_idx, vector_idx, value_num, rwflag, block);
        }

        return getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    virtual services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                                    BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->getBlockOfColumnValues(feature_idx, vector_idx, value_num, rwflag, block);
        }

        return getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    virtual services::Status releaseBlockOfColumnValues(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->releaseBlockOfColumnValues(block);
        }

        return releaseTFeature<double>(block);
    }
    virtual services::Status releaseBlockOfColumnValues(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->releaseBlockOfColumnValues(block);
        }

        return releaseTFeature<float>(block);
    }
    virtual services::Status releaseBlockOfColumnValues(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->releaseBlockOfColumnValues(block);
        }

        return releaseTFeature<int>(block);
    }

    virtual services::Status getSparseBlock(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag,
                                            CSRBlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->getSparseBlock(vector_idx, vector_num, rwflag, block);
        }

        return getSparseTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    virtual services::Status getSparseBlock(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag,
                                            CSRBlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->getSparseBlock(vector_idx, vector_num, rwflag, block);
        }

        return getSparseTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    virtual services::Status getSparseBlock(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag,
                                            CSRBlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->getSparseBlock(vector_idx, vector_num, rwflag, block);
        }

        return getSparseTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    virtual services::Status releaseSparseBlock(CSRBlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->releaseSparseBlock(block);
        }

        return releaseSparseTBlock<double>(block);
    }
    virtual services::Status releaseSparseBlock(CSRBlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->releaseSparseBlock(block);
        }

        return releaseSparseTBlock<float>(block);
    }
    virtual services::Status releaseSparseBlock(CSRBlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->releaseSparseBlock(block);
        }

        return releaseSparseTBlock<int>(block);
    }

    /**
     *  Allocates memory for a data set
     *  \param[in]    dataSize     Number of non-zero values
     *  \param[in]    type         Memory type
     */
    using daal::data_management::interface1::NumericTableIface::allocateDataMemory;

    services::Status allocateDataMemory(size_t dataSize, daal::MemType /*type*/ = daal::dram)
    {
        if (isCpuTable())
        {
            return _cpuTable->allocateDataMemory(dataSize);
        }

        using namespace services::internal::sycl;

        services::Status status;
        auto & context = services::internal::getDefaultContext();
        _dataSize      = dataSize;
        freeDataMemoryImpl();
        size_t nrow = getNumberOfRows();

        if (nrow == 0)
        {
            return services::Status(services::ErrorIncorrectNumberOfObservations);
        }

        const NumericTableFeature & f = (*_ddict)[0];
        _values                       = allocateByNumericTableFeature(f, dataSize, status);
        DAAL_CHECK_STATUS_VAR(status);
        _colIndicesU = context.allocate(services::internal::sycl::TypeIds::id<size_t>(), dataSize, status);
        DAAL_CHECK_STATUS_VAR(status);
        _rowOffsetsU = context.allocate(services::internal::sycl::TypeIds::id<size_t>(), (nrow + 1), status);
        DAAL_CHECK_STATUS_VAR(status);

        services::throwIfPossible(status);
        DAAL_CHECK_STATUS_VAR(status);

        _colIndices = _colIndicesU.template get<size_t>();
        _rowOffsets = _rowOffsetsU.template get<size_t>();
        DAAL_ASSERT(dataSize == _colIndices.size());

        _memStatus = internallyAllocated;
        services::throwIfPossible(status);
        return status;
    }

    /**
     * Returns the indexing scheme for access to data in the CSR layout
     * \return  CSR layout indexing
     */
    CSRIndexing getCSRIndexing() const { return _indexing; }

    /**
     * \copydoc NumericTableIface::check
     */
    virtual services::Status check(const char * description, bool checkDataAllocation = true) const DAAL_C11_OVERRIDE
    {
        services::Status s;
        if (_indexing != oneBased)
        {
            return services::Status(services::Error::create(services::ErrorUnsupportedCSRIndexing, services::ArgumentName, description));
        }

        return services::Status();
    }

protected:
    inline bool isCpuTable() const { return (bool)_cpuTable; }

    static bool isCpuContext() { return services::internal::getDefaultContext().getInfoDevice().isCpu; }

protected:
    NumericTableFeature _defaultFeature;
    CSRIndexing _indexing;
    size_t _dataSize;

    services::internal::sycl::UniversalBuffer _values;
    services::internal::sycl::UniversalBuffer _colIndicesU;
    services::internal::sycl::UniversalBuffer _rowOffsetsU;
    services::internal::Buffer<size_t> _colIndices;
    services::internal::Buffer<size_t> _rowOffsets;

    CSRNumericTablePtr _cpuTable;

    services::Status allocateDataMemoryImpl(daal::MemType /*type*/ = daal::dram) DAAL_C11_OVERRIDE
    {
        return services::Status(services::ErrorMethodNotSupported);
    }

    void freeDataMemoryImpl() DAAL_C11_OVERRIDE
    {
        _values = services::internal::sycl::UniversalBuffer();
        _colIndices.reset();
        _rowOffsets.reset();
        _memStatus = notAllocated;
    }

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * archive)
    {
        using namespace services::internal::sycl;
        services::Status status = SyclNumericTable::serialImpl<Archive, onDeserialize>(archive);

        size_t dataSize = 0;
        if (!onDeserialize)
        {
            dataSize = getDataSize();
        }
        archive->set(dataSize);

        if (onDeserialize)
        {
            if (isCpuTable())
            {
                DAAL_CHECK_STATUS(status, _cpuTable->allocateDataMemory(dataSize));
            }
            else
            {
                DAAL_CHECK_STATUS(status, allocateDataMemory(dataSize));
            }
        }

        size_t nfeat = getNumberOfColumns();
        size_t nobs  = getNumberOfRows();

        if (nfeat > 0)
        {
            NumericTableFeature & f = (*_ddict)[0];
            if (isCpuTable())
            {
                char * data         = NULL;
                size_t * colIndices = NULL;
                size_t * rowOffsets = NULL;

                _cpuTable->getArrays(&data, &colIndices, &rowOffsets);
                archive->set(data, dataSize * f.typeSize);
                archive->set(colIndices, dataSize);
                archive->set(rowOffsets, nobs + 1);
            }
            else
            {
                const auto accessMode = onDeserialize ? data_management::writeOnly : data_management::readOnly;

                services::SharedPtr<size_t> hostColIndices = _colIndices.toHost(accessMode, status);
                DAAL_CHECK_STATUS_VAR(status);
                services::SharedPtr<size_t> hostRowOffsets = _rowOffsets.toHost(accessMode, status);
                DAAL_CHECK_STATUS_VAR(status);

                BufferHostReinterpreter<char> reinterpreter(_values, accessMode, dataSize);
                TypeDispatcher::dispatch(_values.type(), reinterpreter, status);
                DAAL_CHECK_STATUS_VAR(status);
                services::SharedPtr<char> charPtr = reinterpreter.getResult();

                archive->set(charPtr.get(), dataSize * f.typeSize);
                archive->set(hostColIndices.get(), dataSize);
                archive->set(hostRowOffsets.get(), nobs + 1);
            }
        }
        return status;
    }

public:
    virtual size_t getDataSize() DAAL_C11_OVERRIDE
    {
        if (isCpuTable())
        {
            return _cpuTable->getDataSize();
        }

        return _dataSize;
    }

protected:
    template <typename DataType>
    SyclCSRNumericTable(const services::internal::Buffer<DataType> & bufferData, const services::internal::Buffer<size_t> & bufferColIndices,
                        const services::internal::Buffer<size_t> & bufferRowOffsets, size_t nColumns, size_t nRows, CSRIndexing indexing,
                        services::Status & st)
        : SyclNumericTable(nColumns, nRows, DictionaryIface::equal, st), _indexing(indexing)
    {
        _layout   = csrArray;
        _dataSize = bufferData.size();
        _defaultFeature.setType<DataType>();
        st |= _ddict->setAllFeatures(_defaultFeature);

        if (bufferData.size() != bufferColIndices.size())
        {
            st |= services::Error::create(services::ErrorIncorrectSizeOfArray);
            services::throwIfPossible(st);
            return;
        }

        if (bufferRowOffsets.size() != nRows + 1 && _dataSize)
        {
            st |= services::Error::create(services::ErrorIncorrectNumberOfRows);
            services::throwIfPossible(st);
            return;
        }

        if (isCpuContext())
        {
            if (!bufferData.size() && !bufferColIndices.size() && !bufferRowOffsets.size())
            {
                _cpuTable = CSRNumericTable::create<DataType>(NULL, NULL, NULL, nColumns, nRows, indexing, &st);
            }
            else
            {
                const services::SharedPtr<DataType> hostData     = bufferData.toHost(ReadWriteMode::readOnly, st);
                const services::SharedPtr<size_t> hostColIndices = bufferColIndices.toHost(ReadWriteMode::readOnly, st);
                const services::SharedPtr<size_t> hostRowOffsets = bufferRowOffsets.toHost(ReadWriteMode::readOnly, st);
                if (!st)
                {
                    services::throwIfPossible(st);
                    return;
                }

                _cpuTable = CSRNumericTable::create(hostData, hostColIndices, hostRowOffsets, nColumns, nRows, indexing, &st);
            }

            return;
        }
        if (_dataSize)
        {
            st |= setArrays(bufferData, bufferColIndices, bufferRowOffsets, indexing);
        }
    }

    template <typename T>
    services::Status getTBlock(size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> & block)
    {
        return services::throwIfPossible(services::ErrorMethodNotImplemented);
    }

    template <typename T>
    services::Status releaseTBlock(BlockDescriptor<T> & block)
    {
        return services::throwIfPossible(services::ErrorMethodNotImplemented);
    }

    template <typename T>
    services::Status getTFeature(size_t feat_idx, size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> & block)
    {
        return services::throwIfPossible(services::ErrorMethodNotImplemented);
    }

    template <typename T>
    services::Status releaseTFeature(BlockDescriptor<T> & block)
    {
        return services::throwIfPossible(services::ErrorMethodNotImplemented);
    }

    template <typename T>
    services::Status getSparseTBlock(size_t idx, size_t nrows, int rwFlag, CSRBlockDescriptor<T> & block)
    {
        using namespace services::internal::sycl;

        size_t ncols = getNumberOfColumns();
        size_t nobs  = getNumberOfRows();
        block.setDetails(ncols, idx, rwFlag);

        if (idx >= nobs)
        {
            block.resizeValuesBuffer(0);
            return services::Status();
        }

        nrows = (idx + nrows < nobs) ? nrows : nobs - idx;

        services::Status st;

        block.setRowIndicesBuffer(_rowOffsets);

        size_t offset   = 0;
        size_t datasize = _dataSize;
        if (idx == 0)
        {
            block.setRowIndicesBuffer(_rowOffsets);
        }
        else
        {
            services::internal::sycl::UniversalBuffer rowOffsetsNew =
                services::internal::getDefaultContext().allocate(services::internal::sycl::TypeIds::id<size_t>(), (nrows + 1), st);
            DAAL_CHECK_STATUS_VAR(st);
            services::SharedPtr<size_t> hostRowOffsetsNew = rowOffsetsNew.get<size_t>().toHost(ReadWriteMode::writeOnly, st);
            DAAL_CHECK_STATUS_VAR(st);
            const services::SharedPtr<size_t> hostRowOffsets = _rowOffsets.toHost(ReadWriteMode::readOnly, st);
            DAAL_CHECK_STATUS_VAR(st);

            size_t * rowOffsetsNewPtr    = hostRowOffsetsNew.get();
            const size_t * rowOffsetsPtr = hostRowOffsets.get();

            if (rowOffsetsNewPtr == NULL || rowOffsetsPtr == NULL)
            {
                return services::Status(services::ErrorNullPtr);
            }

            rowOffsetsNewPtr[0] = 1;
            for (size_t i = 0; i < nrows; ++i)
            {
                const size_t nNonZeroValuesInRow = rowOffsetsPtr[idx + i + 1] - rowOffsetsPtr[idx + i];
                rowOffsetsNewPtr[i + 1]          = rowOffsetsNewPtr[i] + nNonZeroValuesInRow;
            }
            offset   = rowOffsetsPtr[idx] - rowOffsetsPtr[0];
            datasize = rowOffsetsNewPtr[nrows + 1] - rowOffsetsNewPtr[1];
            block.setRowIndicesBuffer(rowOffsetsNew.get<size_t>());
        }

        BufferConverterTo<T> converter(_values, offset, datasize);
        TypeDispatcher::dispatch(_values.type(), converter, st);
        DAAL_CHECK_STATUS_VAR(st);

        services::internal::Buffer<T> valuesBuffer = converter.getResult();
        block.setValuesBuffer(valuesBuffer);
        block.setColumnIndicesBuffer(_colIndices.getSubBuffer(offset, datasize, st));

        return st;
    }

    template <typename T>
    services::Status releaseSparseTBlock(CSRBlockDescriptor<T> & block)
    {
        using namespace services::internal::sycl;

        if (block.getRWFlag() & (int)writeOnly)
        {
            NumericTableFeature & f = (*_ddict)[0];
            const int indexType     = f.indexType;

            if (data_management::features::DAAL_OTHER_T == indexType && features::internal::getIndexNumType<T>() != indexType)
            {
                block.reset();
                return services::Status(services::ErrorDataTypeNotSupported);
            }

            if (features::internal::getIndexNumType<T>() != indexType)
            {
                services::Status st;
                auto uniBuffer = _values;
                BufferConverterFrom<T> converter(block.getBlockValuesBuffer(), uniBuffer, block.getRowsOffset(), block.getNumberOfRows());
                TypeDispatcher::dispatch(uniBuffer.type(), converter, st);
                DAAL_CHECK_STATUS_VAR(st);

                _values = converter.getResult();
            }
        }
        block.reset();
        return services::Status();
    }

    virtual services::Status setNumberOfColumnsImpl(size_t ncol) DAAL_C11_OVERRIDE
    {
        _ddict->setNumberOfFeatures(ncol);
        _ddict->setAllFeatures(_defaultFeature);
        return services::Status();
    }
};
typedef services::SharedPtr<SyclCSRNumericTable> SyclCSRNumericTablePtr;
/** @} */
} // namespace interface1
using interface1::SyclCSRNumericTable;
using interface1::SyclCSRNumericTablePtr;

} // namespace internal
} // namespace data_management
} // namespace daal
#endif
