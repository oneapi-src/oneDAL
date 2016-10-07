/* file: soa_numeric_table.h */
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
//  Implementation of a heterogeneous table stored as a structure of arrays.
//--
*/

#ifndef __SOA_NUMERIC_TABLE_H__
#define __SOA_NUMERIC_TABLE_H__

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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__SOANUMERICTABLE"></a>
 *  \brief Class that provides methods to access data stored as a structure of arrays,
 *         where each (contiguous) array represents values corresponding to a specific feature.
 */
class SOANumericTable : public NumericTable
{
public:

    /**
     *  Constructor for an empty Numeric Table
     *  \param[in]  nColumns    Number of columns in the table
     *  \param[in]  nRows       Number of rows in the table
     */
    SOANumericTable( size_t nColumns = 0, size_t nRows = 0 ): NumericTable(nColumns, nRows), _arrays(0)
    {
        _layout = soa;
        if( nColumns != 0 )
        {
            _arrays = (void **)daal::services::daal_malloc( sizeof(void *) * nColumns );

            if( _arrays == 0 )
            {
                this->_errors->add(services::ErrorMemoryAllocationFailed);
                return;
            }

            for(size_t i = 0; i < nColumns; i++)
            {
                _arrays[i] = 0;
            }
        }
    }

    /**
     *  Constructor for an empty Numeric Table with a predefined NumericTableDictionary
     *  \param[in]  ddict                 Pointer to the predefined NumericTableDictionary
     *  \param[in]  nRows                 Number of rows in the table
     *  \param[in]  memoryAllocationFlag  Flag that controls internal memory allocation for data in the numeric table
     */
    SOANumericTable( NumericTableDictionary *ddict, size_t nRows,
                     AllocationFlag memoryAllocationFlag = notAllocate ) : NumericTable(0, nRows), _arrays(0)
    {
        _layout = soa;
        setDictionary( ddict );
        if( memoryAllocationFlag == doAllocate ) { allocateDataMemory(); }
    }

    ~SOANumericTable()
    {
        freeDataMemory();

        if( _arrays != 0 )
        {
            daal::services::daal_free(_arrays);
        }
    }

    virtual int getSerializationTag() DAAL_C11_OVERRIDE
    {
        return SERIALIZATION_SOA_NT_ID;
    }

    /**
     *  Sets a pointer to an array of values for a given feature
     *  \tparam T       Type of feature values
     *  \param[in]  ptr Pointer to the array of the T type that stores feature values
     *  \param[in]  idx Feature index
     */
    template<typename T>
    void setArray(T *ptr, size_t idx)
    {
        if( _ddict == 0 )
        {
            _ddict = services::SharedPtr<NumericTableDictionary>(new NumericTableDictionary());
        }

        _ddict->setFeature<T>(idx);

        if( idx < _ddict->getNumberOfFeatures() )
        {
            _arrays[idx] = (void *)ptr;
        }
        else
        {
            this->_errors->add(services::ErrorIncorrectNumberOfFeatures);
            return;
        }

        _memStatus = userAllocated;
    }

    /**
     *  Returns a pointer to an array of values for a given feature
     *  \param[in]  idx Feature index
     *  \return Pointer to the array of values
     */
    void *getArray(size_t idx)
    {
        if( idx < _ddict->getNumberOfFeatures() )
        {
            return _arrays[idx];
        }
        else
        {
            this->_errors->add(services::ErrorIncorrectNumberOfFeatures);
            return (void *)0;
        }
    }

    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vector_idx, vector_num, rwflag, block);
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
        return getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
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

    void setDictionary( NumericTableDictionary *ddict ) DAAL_C11_OVERRIDE
    {
        NumericTable::setDictionary( ddict );
        if( this->_errors->size() != 0 ) { return; }

        if( _arrays != 0 )
        {
            daal::services::daal_free(_arrays);
        }

        size_t ncol = ddict->getNumberOfFeatures();

        if( ncol != 0 )
        {
            _arrays = (void **)daal::services::daal_malloc(sizeof(void *)*ncol);

            if( _arrays == 0 )
            {
                this->_errors->add(services::ErrorMemoryAllocationFailed);
                return;
            }

            for(size_t i = 0; i < ncol; i++)
            {
                _arrays[i] = 0;
            }
        }
    }

    void setNumberOfColumns(size_t ncol) DAAL_C11_OVERRIDE
    {
        NumericTable::setNumberOfColumns(ncol);
        if( this->_errors->size() != 0 ) { return; }

        if( _arrays != 0 )
        {
            daal::services::daal_free(_arrays);
            _arrays = NULL;
        }

        if( ncol != 0 )
        {
            _arrays = (void **)daal::services::daal_malloc(sizeof(void *)*ncol);

            if( _arrays == 0 )
            {
                this->_errors->add(services::ErrorMemoryAllocationFailed);
                return;
            }

            for(size_t i = 0; i < ncol; i++)
            {
                _arrays[i] = 0;
            }
        }
    }

    void allocateDataMemory(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE
    {
        freeDataMemory();

        size_t ncol = _ddict->getNumberOfFeatures();
        size_t nrows = getNumberOfRows();

        if( ncol * nrows == 0 )
        {
            if( nrows == 0 )
            {
                this->_errors->add(services::ErrorIncorrectNumberOfObservations);
                return;
            }
            else
            {
                this->_errors->add(services::ErrorIncorrectNumberOfFeatures);
                return;
            }
        }

        _memStatus = internallyAllocated;

        for(size_t i = 0; i < ncol; i++)
        {
            NumericTableFeature f = (*_ddict)[i];
            if( f.typeSize != 0 )
            {
                _arrays[i] = daal::services::daal_malloc( f.typeSize * nrows );
            }
            if( _arrays[i] == 0 )
            {
                freeDataMemory();
                return;
            }
        }
    }

    void freeDataMemory() DAAL_C11_OVERRIDE
    {
        if( getDataMemoryStatus() == internallyAllocated )
        {
            size_t ncol = _ddict->getNumberOfFeatures();
            for(size_t i = 0; i < ncol; i++)
            {
                if( _arrays[i] )
                {
                    daal::services::daal_free(_arrays[i]);
                    _arrays[i] = 0;
                }
            }
        }

        _memStatus = notAllocated;
    }

    /** \private */
    void serializeImpl  (InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<InputDataArchive, false>( arch );}

    /** \private */
    void deserializeImpl(OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<OutputDataArchive, true>( arch );}

    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl( Archive *arch )
    {
        NumericTable::serialImpl<Archive, onDeserialize>( arch );

        if( onDeserialize )
        {
            allocateDataMemory();
        }

        size_t ncol = _ddict->getNumberOfFeatures();
        size_t nrows = getNumberOfRows();

        for(size_t i = 0; i < ncol; i++)
        {
            NumericTableFeature f = (*_ddict)[i];
            void *ptr = getArray(i);

            arch->set( (char *)ptr, nrows * f.typeSize );
        }
    }

protected:
    void **_arrays;

private:

    template <typename T>
    void getTBlock( size_t idx, size_t nrows, ReadWriteMode rwFlag, BlockDescriptor<T>& block )
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

        if( !(block.getRWFlag() & (int)readOnly) ) return;

        T lbuf[32];

        size_t di = 32;

        T* buffer = block.getBlockPtr();

        for( size_t i = 0 ; i < nrows ; i += di )
        {
            if( i + di > nrows ) { di = nrows - i; }

            for( size_t j = 0 ; j < ncols ; j++ )
            {
                NumericTableFeature &f = (*_ddict)[j];

                char *ptr = (char *)_arrays[j] + (idx + i) * f.typeSize;

                data_feature_utils::vectorUpCast[f.indexType][data_feature_utils::getInternalNumType<T>()]
                ( di, ptr, lbuf );

                for( size_t ii = 0 ; ii < di; ii++ )
                {
                    buffer[ (i + ii)*ncols + j ] = lbuf[ii];
                }
            }
        }
    }

    template <typename T>
    void releaseTBlock( size_t idx, size_t nrows, T *buf, ReadWriteMode rwFlag )
    {
        if (rwFlag & (int)writeOnly)
        {
            size_t ncols = getNumberOfColumns();
            T lbuf[32];

            size_t i, ii, j;

            size_t di = 32;

            for( i = 0 ; i < nrows ; i += di )
            {
                if( i + di > nrows ) { di = nrows - i; }

                for( j = 0 ; j < ncols ; j++ )
                {
                    NumericTableFeature &f = (*_ddict)[j];

                    char *ptr = (char *)_arrays[j];
                    char *location = ptr + (idx + i) * f.typeSize;

                    for( ii = 0 ; ii < di; ii++ )
                    {
                        lbuf[ii] = buf[ (i + ii) * ncols + j ];
                    }

                    data_feature_utils::vectorDownCast[f.indexType][data_feature_utils::getInternalNumType<T>()]
                    ( di, lbuf, location );
                }
            }
        }
    }

    template <typename T>
    void releaseTBlock( BlockDescriptor<T>& block )
    {
        if(block.getRWFlag() & (int)writeOnly)
        {
            size_t ncols = getNumberOfColumns();
            size_t nrows = block.getNumberOfRows();
            size_t idx   = block.getRowsOffset();
            T lbuf[32];

            size_t di = 32;

            T* blockPtr = block.getBlockPtr();

            for( size_t i = 0 ; i < nrows ; i += di )
            {
                if( i + di > nrows ) { di = nrows - i; }

                for( size_t j = 0 ; j < ncols ; j++ )
                {
                    NumericTableFeature &f = (*_ddict)[j];

                    char *ptr = (char *)_arrays[j] + (idx + i) * f.typeSize;

                    for( size_t ii = 0 ; ii < di; ii++ )
                    {
                        lbuf[ii] = blockPtr[ (i + ii) * ncols + j ];
                    }

                    data_feature_utils::vectorDownCast[f.indexType][data_feature_utils::getInternalNumType<T>()]
                    ( di, lbuf, ptr );
                }
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

        char *ptr = (char *)_arrays[feat_idx];

        NumericTableFeature &f = (*_ddict)[feat_idx];

        if( data_feature_utils::getIndexNumType<T>() == f.indexType )
        {
            block.setPtr( (T *)ptr + idx, 1, nrows );
        }
        else
        {
            char *location = ptr + idx * f.typeSize;

            if( !block.resizeBuffer( 1, nrows ) )
            {
                this->_errors->add(services::ErrorMemoryAllocationFailed);
                return;
            }

            if( !(block.getRWFlag() & (int)readOnly) ) return;

            data_feature_utils::vectorUpCast[f.indexType][data_feature_utils::getInternalNumType<T>()]
            ( nrows, location, block.getBlockPtr() );
        }
    }

    template <typename T>
    void releaseTFeature( size_t feat_idx, size_t idx, size_t nrows, T *buf, ReadWriteMode rwFlag )
    {
        if (rwFlag & (int)writeOnly)
        {
            NumericTableFeature &f = (*_ddict)[feat_idx];

            if( data_feature_utils::getIndexNumType<T>() != f.indexType )
            {
                char *ptr      = (char *)_arrays[feat_idx];
                char *location = ptr + idx * f.typeSize;

                data_feature_utils::vectorDownCast[f.indexType][data_feature_utils::getInternalNumType<T>()]
                ( nrows, buf, location );
            }
        }
    }

    template <typename T>
    void releaseTFeature( BlockDescriptor<T>& block )
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            size_t feat_idx = block.getColumnsOffset();

            NumericTableFeature &f = (*_ddict)[feat_idx];

            if( data_feature_utils::getIndexNumType<T>() != f.indexType )
            {
                char *ptr = (char *)_arrays[feat_idx] + block.getRowsOffset() * f.typeSize;

                data_feature_utils::vectorDownCast[f.indexType][data_feature_utils::getInternalNumType<T>()]
                ( block.getNumberOfRows(), block.getBlockPtr(), ptr );
            }
        }
        block.setDetails( 0, 0, 0 );
    }
};
/** @} */
} // namespace interface1
using interface1::SOANumericTable;

}
} // namespace daal
#endif
