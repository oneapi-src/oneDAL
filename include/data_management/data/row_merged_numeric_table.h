/* file: row_merged_numeric_table.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of row merged numeric table.
//--
*/


#ifndef __ROW_MERGED_NUMERIC_TABLE_H__
#define __ROW_MERGED_NUMERIC_TABLE_H__

#include "data_management/data/numeric_table.h"
#include "services/daal_memory.h"
#include "services/daal_defines.h"

namespace daal
{
namespace data_management
{

namespace interface1
{
/**
 * @ingroup numeric_tables
 * @{
 */
/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__ROWMERGEDNUMERICTABLE"></a>
 *  \brief Class that provides methods to access a collection of numeric tables as if they are joined by rows
 */
class RowMergedNumericTable : public NumericTable
{
public:
    /**
     *  Constructor for an empty merge Numeric Table
     */
    RowMergedNumericTable() : NumericTable(0, 0), _tables(new DataCollection) {}

    /**
     *  Constructor for a Row Merged Numeric Table consisting of one table
     *  \param[in]  table       Pointer to the table
     */
    RowMergedNumericTable( NumericTablePtr table) : NumericTable(0, 0), _tables(new DataCollection)
    {
        addNumericTable(table);
    }

    /**
     *  Adds the table to the bottom of the Row Merged Numeric Table
     *  \param[in] table Pointer to the table
     */
    void addNumericTable(NumericTablePtr table)
    {
        if (table->getDataLayout() & csrArray)
        {
            this->_errors->add(services::ErrorIncorrectTypeOfInputNumericTable);
            return;
        }

        size_t ncols = getNumberOfColumns();
        size_t cols = table->getNumberOfColumns();

        if (ncols != 0 && ncols != cols)
        {
            this->_errors->add(services::ErrorIncorrectNumberOfFeatures);
            return;
        }

        _tables->push_back(table);

        if (ncols == 0)
        {
            setNumberOfColumns(cols);

            for (size_t i = 0; i < cols; i++)
            {
                NumericTableFeature &f = table->getDictionary()->operator[](i);
                _ddict->setFeature(f, i);
            }
        }

        size_t obs = table->getNumberOfRows();
        setNumberOfRows(_obsnum + obs);
    }

    void setNumberOfColumns(size_t ncols) DAAL_C11_OVERRIDE
    {
        for (size_t i = 0;i < _tables->size(); i++)
        {
            NumericTable* nt = (NumericTable*)(_tables->operator[](i).get());
            nt->setNumberOfColumns(ncols);
        }
        NumericTable::setNumberOfColumns(ncols);
    }

    void allocateDataMemory(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE
    {
        for (size_t i = 0;i < _tables->size(); i++)
        {
            NumericTable* nt = (NumericTable*)(_tables->operator[](i).get());
            nt->allocateDataMemory(type);
        }
    }

    void freeDataMemory() DAAL_C11_OVERRIDE
    {
        for (size_t i = 0;i < _tables->size(); i++)
        {
            NumericTable* nt = (NumericTable*)(_tables->operator[](i).get());
            nt->freeDataMemory();
        }
    }

    MemoryStatus getDataMemoryStatus() const DAAL_C11_OVERRIDE
    {
        if (_tables->size() == 0)
        {
            return notAllocated;
        }

        for (size_t i = 0;i < _tables->size(); i++)
        {
            NumericTable* nt = (NumericTable*)(_tables->operator[](i).get());
            if (nt->getDataMemoryStatus() == notAllocated)
            {
                return notAllocated;
            }
        }

        return internallyAllocated;
    }

    virtual int getSerializationTag() DAAL_C11_OVERRIDE
    {
        return SERIALIZATION_ROWMERGE_NT_ID;
    }

    void serializeImpl(InputDataArchive *archive) DAAL_C11_OVERRIDE
    {
        serialImpl<InputDataArchive, false>( archive );
    }

    void deserializeImpl(OutputDataArchive *archive) DAAL_C11_OVERRIDE
    {
        serialImpl<OutputDataArchive, true>( archive );
    }

    void getBlockOfRows(size_t vector_idx, size_t vector_num,
                          ReadWriteMode rwflag, BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    void getBlockOfRows(size_t vector_idx, size_t vector_num,
                          ReadWriteMode rwflag, BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    void getBlockOfRows(size_t vector_idx, size_t vector_num,
                          ReadWriteMode rwflag, BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    void releaseBlockOfRows(BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<double>(block);
    }
    void releaseBlockOfRows(BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<float>(block);
    }
    void releaseBlockOfRows(BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<int>(block);
    }

    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    void releaseBlockOfColumnValues(BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<double>(block);
    }
    void releaseBlockOfColumnValues(BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<float>(block);
    }
    void releaseBlockOfColumnValues(BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<int>(block);
    }

protected:
    template<typename Archive, bool onDeserialize>
    void serialImpl( Archive *arch )
    {
        NumericTable::serialImpl<Archive, onDeserialize>( arch );

        arch->setSharedPtrObj(_tables);
    }

private:
    template<typename T>
    void internal_inner_repack( size_t idx, size_t rows, size_t ncols, T *src, T *dst )
    {
        size_t i, j;

        for(i = 0; i < rows; i++)
        {
            for(j = 0; j < ncols; j++)
            {
                dst[(idx + i) * ncols + j] = src[i * ncols + j];
            }
        }
    }

    template<typename T>
    void internal_outer_repack( size_t idx, size_t rows, size_t ncols, T *src, T *dst )
    {
        size_t i, j;

        for(i = 0; i < rows; i++)
        {
            for(j = 0; j < ncols; j++)
            {
                dst[i * ncols + j] = src[(i + idx) * ncols + j];
            }
        }
    }

protected:
    template <typename T>
    void getTBlock( size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T>& block )
    {
        size_t ncols = getNumberOfColumns();
        size_t nobs = getNumberOfRows();
        block.setDetails( 0, idx, rwFlag );

        if (idx >= nobs)
        {
            block.resizeBuffer( ncols, 0 );
            return;
        }

        nrows = ( idx + nrows < nobs ) ? nrows : nobs - idx;

        if( !block.resizeBuffer( ncols, nrows ) )
        {
            this->_errors->add(services::ErrorMemoryAllocationFailed);
            return;
        }

        if( rwFlag & (int)readOnly )
        {
            size_t rows = 0;
            BlockDescriptor<T> innerBlock;
            for (size_t k = 0; k < _tables->size() && rows < idx + nrows; k++)
            {
                NumericTable* nt = (NumericTable*)(_tables->operator[](k).get());
                size_t lrows = nt->getNumberOfRows();

                if (rows + lrows > idx)
                {
                    size_t idxBegin = (rows < idx) ? idx : rows;
                    size_t idxEnd = (rows + lrows < idx + nrows) ? rows + lrows : idx + nrows;
                    nt->getBlockOfRows(idxBegin - rows, idxEnd - idxBegin, readOnly, innerBlock);

                    internal_inner_repack<T>( idxBegin - idx, idxEnd - idxBegin, ncols, innerBlock.getBlockPtr(), block.getBlockPtr());

                    nt->releaseBlockOfRows(innerBlock);
                }

                rows += lrows;
            }
        }
    }

    template <typename T>
    void releaseTBlock(BlockDescriptor<T>& block)
    {
        if(block.getRWFlag() & (int)writeOnly)
        {
            size_t ncols = getNumberOfColumns();
            size_t nrows = block.getNumberOfRows();
            size_t idx = block.getRowsOffset();
            size_t rows = 0;
            BlockDescriptor<T> innerBlock;
            for (size_t k = 0; k < _tables->size() && rows < idx + nrows; k++)
            {
                NumericTable* nt = (NumericTable*)(_tables->operator[](k).get());
                size_t lrows = nt->getNumberOfRows();

                if (rows + lrows > idx)
                {
                    size_t idxBegin = (rows < idx) ? idx : rows;
                    size_t idxEnd = (rows + lrows < idx + nrows) ? rows + lrows : idx + nrows;
                    nt->getBlockOfRows(idxBegin - rows, idxEnd - idxBegin, writeOnly, innerBlock);

                    internal_outer_repack<T>( idxBegin - idx, idxEnd - idxBegin, ncols, block.getBlockPtr(), innerBlock.getBlockPtr());

                    nt->releaseBlockOfRows(innerBlock);
                }

                rows += lrows;
            }
        }
        block.setDetails( 0, 0, 0 );
    }

    template <typename T>
    void getTFeature( size_t feat_idx, size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T>& block )
    {
        size_t ncols = getNumberOfColumns();
        size_t nobs = getNumberOfRows();
        block.setDetails( feat_idx, idx, rwFlag );

        if (idx >= nobs)
        {
            block.resizeBuffer( 1, 0 );
            return;
        }

        nrows = ( idx + nrows < nobs ) ? nrows : nobs - idx;
        if( !block.resizeBuffer( 1, nrows ) )
        {
            this->_errors->add(services::ErrorMemoryAllocationFailed);
            return;
        }

        if( rwFlag & (int)readOnly )
        {
            T* buffer = block.getBlockPtr();
            size_t rows = 0;
            for (size_t k = 0; k < _tables->size() && rows < idx + nrows; k++)
            {
                NumericTable* nt = (NumericTable*)(_tables->operator[](k).get());
                size_t lrows = nt->getNumberOfRows();

                if (rows + lrows > idx)
                {
                    size_t idxBegin = (rows < idx) ? idx : rows;
                    size_t idxEnd = (rows + lrows < idx + nrows) ? rows + lrows : idx + nrows;

                    BlockDescriptor<T> innerBlock;
                    nt->getBlockOfColumnValues(feat_idx, idxBegin - rows, idxEnd - idxBegin, readOnly, innerBlock);
                    T* location = innerBlock.getBlockPtr();
                    for (size_t i = idxBegin; i < idxEnd; i++)
                    {
                        buffer[i] = location[i - idxBegin];
                    }
                    nt->releaseBlockOfColumnValues(innerBlock);
                }

                rows += lrows;
            }
        }
    }

    template <typename T>
    void releaseTFeature( BlockDescriptor<T>& block )
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            size_t feat_idx = block.getColumnsOffset();
            size_t idx = block.getRowsOffset();
            size_t nrows = block.getNumberOfRows();
            size_t rows = 0;
            T* buffer = block.getBlockPtr();
            for (size_t k = 0; k < _tables->size() && rows < idx + nrows; k++)
            {
                NumericTable* nt = (NumericTable*)(_tables->operator[](k).get());
                size_t lrows = nt->getNumberOfRows();

                if (rows + lrows > idx)
                {
                    size_t idxBegin = (rows < idx) ? idx : rows;
                    size_t idxEnd = (rows + lrows < idx + nrows) ? rows + lrows : idx + nrows;

                    BlockDescriptor<T> innerBlock;
                    nt->getBlockOfColumnValues(feat_idx, idxBegin - rows, idxEnd - idxBegin, writeOnly, innerBlock);
                    T* location = innerBlock.getBlockPtr();
                    for (size_t i = idxBegin; i < idxEnd; i++)
                    {
                        location[i - idxBegin] = buffer[i];
                    }
                    nt->releaseBlockOfColumnValues(innerBlock);
                }

                rows += lrows;
            }
        }
        block.setDetails( 0, 0, 0 );
    }

protected:
    DataCollectionPtr _tables;
};
/** @} */
} // namespace interface1
using interface1::RowMergedNumericTable;

} // namespace data_management
} // namespace daal

#endif
