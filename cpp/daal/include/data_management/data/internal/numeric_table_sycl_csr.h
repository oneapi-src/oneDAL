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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__CSRNUMERICTABLE"></a>
 *  \brief Class that provides methods to access data stored in the CSR layout.
 */
class DAAL_EXPORT SyclCSRNumericTable : public SyclNumericTable, public CSRNumericTableIface
{
public:
    DECLARE_SERIALIZABLE_TAG()
    DECLARE_SERIALIZABLE_IMPL()

    DAAL_CAST_OPERATOR(SyclCSRNumericTable)

    /**
     *  Constructs CSR numeric table with user-allocated memory
     *  \tparam   DataType        Type of values in the Numeric Table
     *  \param[in]    ptr         Array of values in the CSR layout. Let ptr_size denote the size of an array ptr
     *  \param[in]    colIndices  Array of column indices in the CSR layout. Values of indices are determined by the index base
     *  \param[in]    rowOffsets  Array of row indices in the CSR layout. Size of the array is nrow+1. The first element is 0/1
     *                            in zero-/one-based indexing. The last element is ptr_size+0/1 in zero-/one-based indexing
     *  \param[in]    nColumns    Number of columns in the corresponding dense table
     *  \param[in]    nRows       Number of rows in the corresponding dense table
     *  \param[in]    indexing    Indexing scheme used to access data in the CSR layout
     *  \param[out]   stat        Status of the numeric table construction
     *  \return       CSR numeric table with user-allocated memory
     *  \note Present version of Intel(R) Data Analytics Acceleration Library supports 1-based indexing only
     */

    template <typename DataType>
    static services::SharedPtr<SyclCSRNumericTable> create(const services::internal::Buffer<DataType> & bufferData,
                                                           const services::internal::Buffer<size_t> & bufferColIndices,
                                                           const services::internal::Buffer<size_t> & bufferRowOffsets, size_t nColumns, size_t nRows,
                                                           CSRIndexing indexing = oneBased, services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_IMPL_EX(SyclCSRNumericTable, bufferData, bufferColIndices, bufferRowOffsets, nColumns, nRows, indexing);
    }

    /**
     *  Constructs CSR numeric table with user-allocated memory
     *  \param[in]    nColumns    Number of columns in the corresponding dense table
     *  \param[in]    nRows       Number of rows in the corresponding dense table
     *  \param[in]    indexing    Indexing scheme used to access data in the CSR layout
     *  \param[out]   stat        Status of the numeric table construction
     *  \return       CSR numeric table with user-allocated memory
     *  \note Present version of Intel(R) Data Analytics Acceleration Library supports 1-based indexing only
     */
    // template <typename DataType>
    // static services::SharedPtr<SyclCSRNumericTable> create(size_t nColumns, size_t nRows, size_t dataSize, AllocationFlag memoryAllocationFlag,
    //                                                        CSRIndexing indexing = oneBased, services::Status * stat = NULL)
    // {
    //     DAAL_DEFAULT_CREATE_IMPL_EX(SyclCSRNumericTable, DataType, nColumns, nRows, dataSize, memoryAllocationFlag, indexing);
    // }

    virtual ~SyclCSRNumericTable() { freeDataMemoryImpl(); }

    virtual services::Status resize(size_t nrows) DAAL_C11_OVERRIDE { return setNumberOfRowsImpl(nrows); }

    /**
     *  Returns  pointers to a data set stored in the CSR layout
     *  \param[out]    values         Array of values in the CSR layout
     *  \param[out]    colIndices  Array of column indices in the CSR layout
     *  \param[out]    rowOffsets  Array of row indices in the CSR layout
     */
    template <typename DataType>
    services::Status getArrays(services::internal::Buffer<DataType> & values, services::internal::Buffer<size_t> & colIndices,
                               services::internal::Buffer<size_t> & rowOffsets) const
    {
        if (values)
        {
            values = _values.get<DataType>();
        }
        if (colIndices)
        {
            colIndices = _colIndices;
        }
        if (rowOffsets)
        {
            rowOffsets = _rowOffsets;
        }
        return services::Status();
    }
    /**
     *  Sets a pointer to a CSR data set
     *  \param[in]    ptr         Array of values in the CSR layout
     *  \param[in]    colIndices  Array of column indices in the CSR layout
     *  \param[in]    rowOffsets  Array of row indices in the CSR layout
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

        if (nrow == 0) return services::Status(services::ErrorIncorrectNumberOfObservations);

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

        // _rowOffsets.get()[0] = ((_indexing == oneBased) ? 1 : 0);
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
        // DAAL_CHECK_STATUS(s, data_management::SyclNumericTable::check(description, checkDataAllocation));

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
                // services::SharedPtr<DataType> hostData = _values.get<DataType>().toHost(data_management::readOnly, status);
                // DAAL_CHECK_STATUS_VAR(status);
                services::SharedPtr<size_t> hostColIndices = _colIndices.toHost(data_management::readOnly, status);
                DAAL_CHECK_STATUS_VAR(status);
                services::SharedPtr<size_t> hostRowOffsets = _rowOffsets.toHost(data_management::readOnly, status);
                DAAL_CHECK_STATUS_VAR(status);

                // archive->set(hostData.get(), dataSize);
                archive->set(hostColIndices.get(), dataSize);
                archive->set(hostRowOffsets.get(), nobs + 1);
            }
        }
        return services::Status();
    }

public:
    virtual size_t getDataSize() DAAL_C11_OVERRIDE { return _dataSize; }

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

        DAAL_ASSERT(bufferData.size() == bufferColIndices.size());
        // DAAL_ASSERT(bufferRowOffsets.size() == nRows + 1);

        if (isCpuContext())
        {
            const auto hostData       = bufferData.toHost(ReadWriteMode::readOnly, st);
            const auto hostColIndices = bufferColIndices.toHost(ReadWriteMode::readOnly, st);
            const auto hostRowOffsets = bufferRowOffsets.toHost(ReadWriteMode::readOnly, st);
            _cpuTable                 = CSRNumericTable::create(hostData, hostColIndices, hostRowOffsets, nColumns, nRows, indexing, &st);
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
        services::throwIfPossible(services::ErrorMethodNotImplemented);
        return services::ErrorMethodNotImplemented;
    }

    template <typename T>
    services::Status releaseTBlock(BlockDescriptor<T> & block)
    {
        if (!(block.getRWFlag() & (int)writeOnly)) block.reset();
        return services::Status();
    }

    template <typename T>
    services::Status getTFeature(size_t feat_idx, size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> & block)
    {
        services::throwIfPossible(services::ErrorMethodNotImplemented);
        return services::ErrorMethodNotImplemented;
    }

    template <typename T>
    services::Status releaseTFeature(BlockDescriptor<T> & block)
    {
        services::throwIfPossible(services::ErrorMethodNotImplemented);
        return services::ErrorMethodNotImplemented;
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
        auto uniBuffer = _values;
        BufferConverterTo<T> converter(uniBuffer, 0, _dataSize);
        TypeDispatcher::dispatch(_values.type(), converter, st);
        DAAL_CHECK_STATUS_VAR(st);

        services::internal::Buffer<T> valuesBuffer = converter.getResult();
        // printf("valuesBuffer.size(): %lu; _dataSize: %lu\n", valuesBuffer.size(), _dataSize);
        block.setValuesBuffer(valuesBuffer);

        block.setColumnIndicesBuffer(_colIndices);
        block.setRowIndicesBuffer(_rowOffsets);

        // TODO idx!=0 for _rowOffsets
        if (idx != 0)
        {
            DAAL_ASSERT(false);
        }
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
