/* file: merged_numeric_table.h */
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
//  Implementation of merge numeric table.
//--
*/


#ifndef __MERGED_NUMERIC_TABLE_H__
#define __MERGED_NUMERIC_TABLE_H__

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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__MERGEDNUMERICTABLE"></a>
 *  \brief Class that provides methods to access a collection of numeric tables as if they are joined by columns
 */
class MergedNumericTable : public NumericTable
{
public:
    /**
     *  Constructor for an empty merge Numeric Table
     */
    MergedNumericTable() : NumericTable(0, 0), _tables(new DataCollection) {}

    /**
     *  Constructor for a merge Numeric Table consisting of one table
     *  \param[in]  table       Pointer to the table
     */
    MergedNumericTable( NumericTablePtr table) : NumericTable(0, 0), _tables(new DataCollection)
    {
        addNumericTable(table);
    }

    /**
     *  Constructor for a merge Numeric Table consisting of two tables
     *  \param[in]  first      Pointer to the first table
     *  \param[in]  second     Pointer to the second table
     */
    MergedNumericTable( NumericTablePtr first, NumericTablePtr second ):
        NumericTable(0, 0), _tables(new DataCollection)
    {
        addNumericTable(first);
        addNumericTable(second);
    }

    /**
     *  Adds the table to the right of the merge Numeric Table
     *  \param[in] table Pointer to the table
     */
    void addNumericTable(NumericTablePtr table)
    {
        if (table->getDataLayout() & csrArray)
        {
            this->_errors->add(services::ErrorIncorrectTypeOfInputNumericTable);
            return;
        }

        _tables->push_back(table);

        size_t ncols = getNumberOfColumns();
        size_t cols = table->getNumberOfColumns();

        setNumberOfColumns(ncols + cols);

        for (size_t i = 0; i < cols; i++)
        {
            NumericTableFeature &f = table->getDictionary()->operator[](i);
            _ddict->setFeature(f, ncols + i);
        }

        size_t obs = table->getNumberOfRows();

        if (obs != _obsnum)
        {
            if (obs < _obsnum || _tables->size() == 1)
            {
                _obsnum = obs;
            }
            setNumberOfRows(_obsnum);
        }
    }

    void setNumberOfRows(size_t nrow) DAAL_C11_OVERRIDE
    {
        for (size_t i = 0;i < _tables->size(); i++)
        {
            NumericTable* nt = (NumericTable*)(_tables->operator[](i).get());
            nt->setNumberOfRows(nrow);
        }
        _obsnum = nrow;
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

    virtual int getSerializationTag() DAAL_C11_OVERRIDE
    {
        return SERIALIZATION_MERGE_NT_ID;
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

    void allocateBasicStatistics() DAAL_C11_OVERRIDE
    {
        size_t ncols = getNumberOfColumns();
        size_t ntables = _tables->size();
        services::SharedPtr<MergedNumericTable> minMergeNT (new MergedNumericTable());
        services::SharedPtr<MergedNumericTable> maxMergeNT (new MergedNumericTable());
        services::SharedPtr<MergedNumericTable> sumMergeNT (new MergedNumericTable());
        services::SharedPtr<MergedNumericTable> sumSqMergeNT (new MergedNumericTable());
        for (size_t i = 0; i < ntables; i++) {
            NumericTable* nt = (NumericTable*)(_tables->operator[](i).get());
            nt->allocateBasicStatistics();
            minMergeNT->addNumericTable(nt->basicStatistics.get(NumericTable::minimum));
            maxMergeNT->addNumericTable(nt->basicStatistics.get(NumericTable::maximum));
            sumMergeNT->addNumericTable(nt->basicStatistics.get(NumericTable::sum));
            sumSqMergeNT->addNumericTable(nt->basicStatistics.get(NumericTable::sumSquares));
        }
        if (basicStatistics.get(NumericTable::minimum).get() == NULL ||
            basicStatistics.get(NumericTable::minimum)->getNumberOfColumns() != ncols)
        {
            basicStatistics.set(NumericTable::minimum, minMergeNT);
        }
        if (basicStatistics.get(NumericTable::maximum).get() == NULL ||
            basicStatistics.get(NumericTable::maximum)->getNumberOfColumns() != ncols)
        {
            basicStatistics.set(NumericTable::maximum, maxMergeNT);
        }
        if (basicStatistics.get(NumericTable::sum).get() == NULL ||
            basicStatistics.get(NumericTable::sum)->getNumberOfColumns() != ncols)
        {
            basicStatistics.set(NumericTable::sum, sumMergeNT);
        }
        if (basicStatistics.get(NumericTable::sumSquares).get() == NULL ||
            basicStatistics.get(NumericTable::sumSquares)->getNumberOfColumns() != ncols)
        {
            basicStatistics.set(NumericTable::sumSquares, sumSqMergeNT);
        }
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
    void internal_inner_repack( size_t pos, size_t cols, size_t rows, size_t ncols, T *src, T *dst )
    {
        size_t i, j;

        for(i = 0; i < rows; i++)
        {
            for(j = 0; j < cols; j++)
            {
                dst[i * ncols + j + pos] = src[i * cols + j];
            }
        }
    }

    template<typename T>
    void internal_outer_repack( size_t pos, size_t cols, size_t rows, size_t ncols, T *src, T *dst )
    {
        size_t i, j;

        for(i = 0; i < rows; i++)
        {
            for(j = 0; j < cols; j++)
            {
                dst[i * cols + j] = src[i * ncols + j + pos];
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
            size_t cols = 0;
            BlockDescriptor<T> innerBlock;
            for (size_t k = 0; k < _tables->size(); k++)
            {
                NumericTable* nt = (NumericTable*)(_tables->operator[](k).get());
                size_t lcols = nt->getNumberOfColumns();

                nt->getBlockOfRows(idx, nrows, readOnly, innerBlock);

                internal_inner_repack<T>( cols, lcols, nrows, ncols, innerBlock.getBlockPtr(), block.getBlockPtr());

                nt->releaseBlockOfRows(innerBlock);

                cols += lcols;
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
            size_t offset = block.getRowsOffset();
            size_t cols = 0;
            BlockDescriptor<T> innerBlock;
            for (size_t k = 0; k < _tables->size(); k++)
            {
                NumericTable* nt = (NumericTable*)(_tables->operator[](k).get());
                size_t lcols = nt->getNumberOfColumns();

                nt->getBlockOfRows(offset, nrows, writeOnly, innerBlock);

                internal_outer_repack<T>( cols, lcols, nrows, ncols, block.getBlockPtr(), innerBlock.getBlockPtr());

                nt->releaseBlockOfRows(innerBlock);

                cols += lcols;
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
            for (size_t k = 0; k < _tables->size(); k++)
            {
                NumericTable* nt = (NumericTable*)(_tables->operator[](k).get());
                size_t lcols = nt->getNumberOfColumns();

                if (lcols > feat_idx)
                {
                    BlockDescriptor<T> innerBlock;
                    nt->getBlockOfColumnValues(feat_idx, idx, nrows, readOnly, innerBlock);
                    T* location = innerBlock.getBlockPtr();
                    for (size_t i = 0; i < nrows; i++)
                    {
                        buffer[i] = location[i];
                    }
                    nt->releaseBlockOfColumnValues(innerBlock);
                    break;
                }

                feat_idx -= lcols;
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
            T* buffer = block.getBlockPtr();
            for (size_t k = 0; k < _tables->size(); k++)
            {
                NumericTable* nt = (NumericTable*)(_tables->operator[](k).get());
                size_t lcols = nt->getNumberOfColumns();

                if (lcols > feat_idx)
                {
                    BlockDescriptor<T> innerBlock;
                    nt->getBlockOfColumnValues(feat_idx, idx, nrows, writeOnly, innerBlock);
                    T* location = innerBlock.getBlockPtr();
                    for (size_t i = 0; i < nrows; i++)
                    {
                        location[i] = buffer[i];
                    }
                    nt->releaseBlockOfColumnValues(innerBlock);
                    break;
                }

                feat_idx -= lcols;
            }
        }
        block.setDetails( 0, 0, 0 );
    }

protected:
    DataCollectionPtr _tables;
};
/** @} */
} // namespace interface1
using interface1::MergedNumericTable;

} // namespace data_management
} // namespace daal

#endif
