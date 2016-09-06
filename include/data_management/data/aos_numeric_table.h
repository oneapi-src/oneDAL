/* file: aos_numeric_table.h */
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

#ifndef __AOS_NUMERIC_TABLE_H__
#define __AOS_NUMERIC_TABLE_H__

namespace daal
{
namespace data_management
{

// Extended variant of the standard offsetof() macro  (not limited to only POD types)
/* Not sure if it's standard-compliant; most likely, it only works in certain environments.
   The constant 0x1000 (not NULL) is necessary to appease GCC. */
#define DAAL_STRUCT_MEMBER_OFFSET(class_name, member_name) \
    ((ptrdiff_t)&(reinterpret_cast<class_name*>(0x1000)->member_name) - 0x1000)

namespace interface1
{
/**
 * @ingroup numeric_tables
 * @{
 */
/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__AOSNUMERICTABLE"></a>
 *  \brief Class that provides methods to access data stored as a contiguous array
 *         of heterogeneous feature vectors, while each feature vector is represented
 *         by a data structure.
 *         Therefore, the data is represented as an array of structures.
 */
class AOSNumericTable : public NumericTable
{
public:

    /**
     *  Constructor for an empty Numeric Table with a predefined size of the structure that represents a feature vector
     *  \param[in]  structSize  Size of the structure that represents the feature vector
     *  \param[in]  ncol        Number of columns in the table
     *  \param[in]  nrow        Number of rows in the table
     */
    AOSNumericTable( size_t structSize = 0, size_t ncol = 0, size_t nrow = 0 ): NumericTable(ncol, nrow)
    {
        _ptr        = 0;
        _layout     = aos;
        _structSize = structSize;

        if( ncol > 0 )
        {
            _offsets = new size_t[ncol];
        }
        else
        {
            _offsets = 0;
        }
    }

    /**
     *  Constructor for a Numeric Table with user-allocated memory
     *  \param[in]  ptr     Pointer to a data set in the AOS format
     *  \param[in]  ncol    Number of columns in the table
     *  \param[in]  nrow    Number of rows in the table
     */
    template<typename StructDataType>
    AOSNumericTable( StructDataType *ptr, size_t ncol, size_t nrow = 0 ): NumericTable(ncol, nrow)
    {
        _ptr        = ptr;
        _layout     = aos;
        _structSize = sizeof(StructDataType);

        if( ncol > 0 )
        {
            _offsets = new size_t[ncol];
        }
        else
        {
            _offsets = 0;
        }
    }

    /** \private */
    ~AOSNumericTable()
    {
        if(_offsets)
        {
            delete[] _offsets;
        }
        freeDataMemory();
    }

    virtual int getSerializationTag() DAAL_C11_OVERRIDE
    {
        return SERIALIZATION_AOS_NT_ID;
    }

    /**
     *  Sets a pointer to an array of structures in a Numeric Table
     *  \param[in]  ptr Pointer to a data set in the AOS format
     *  \param[in]  obsnum Number of rows in the table
     */
    void setArray(void *const ptr, size_t obsnum = 0)
    {
        _ptr = ptr;
        _memStatus = userAllocated;
        setNumberOfRows( obsnum );
    }

    /**
     *  Returns a pointer to an array of structures in a Numeric Table
     *  \return Pointer to a data set in the AOS format
     */
    void *getArray()
    {
        return _ptr;
    }

    /**
     *  Sets a feature in an AOS Numeric Table
     *  \tparam T       Type of feature values
     *  \param[in]  idx Feature index
     *  \param[in]  offset Feature offset in the structure representing the feature vector
     */
    template<typename T>
    void setFeature(size_t idx, size_t offset)
    {
        if( _ddict.get() == NULL )
        {
            _ddict = services::SharedPtr<NumericTableDictionary>(new NumericTableDictionary());
        }

        _ddict->setFeature<T>(idx);

        _offsets[idx] = offset;
    }

    /**
     *  Sets an offset in an AOS Numeric Table
     *  \param[in] idx      Feature index
     *  \param[in] offset   Feature offset in the structure representing the feature vector
     */
    void setOffset(size_t idx, size_t offset)
    {
        _offsets[idx] = offset;
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

    void allocateDataMemory(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE
    {
        freeDataMemory();

        size_t size = _structSize * getNumberOfRows();

        if( size == 0 )
        {
            if( getNumberOfRows() == 0 )
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

        _ptr = daal::services::daal_malloc( size );

        if( _ptr == 0 )
        {
            this->_errors->add(services::ErrorMemoryAllocationFailed);
            return;
        }

        _memStatus = internallyAllocated;
    }

    void freeDataMemory() DAAL_C11_OVERRIDE
    {
        if( getDataMemoryStatus() == internallyAllocated )
        {
            daal::services::daal_free(_ptr);
        }

        _ptr = 0;
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

        size_t size = getNumberOfColumns() * getNumberOfRows();

        arch->set( (char *)_ptr, size * _structSize );
    }

protected:

    void   *_ptr;
    size_t  _structSize;
    size_t *_offsets;

private:

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

        if( !(rwFlag & (int)readOnly) ) return;

        char *ptr = (char *)_ptr + _structSize * idx;

        for( size_t j = 0 ; j < ncols ; j++ )
        {
            NumericTableFeature &f = (*_ddict)[j];

            char *location = ptr + _offsets[j];

            T* blockPtr = block.getBlockPtr();

            data_feature_utils::vectorStrideUpCast[f.indexType][data_feature_utils::getInternalNumType<T>()]
            ( nrows, location, _structSize, blockPtr + j, sizeof(T)*ncols );
        }
    }

    template <typename T>
    void releaseTBlock( size_t idx, size_t nrows, T *buf, int rwFlag )
    {
        if (rwFlag & (int)writeOnly)
        {
            size_t ncols = getNumberOfColumns();

            char *ptr = (char *)_ptr + _structSize * idx;

            size_t j;

            for( j = 0 ; j < ncols ; j++ )
            {
                NumericTableFeature &f = (*_ddict)[j];

                char *location = ptr + _offsets[j];

                data_feature_utils::vectorStrideDownCast[f.indexType][data_feature_utils::getInternalNumType<T>()]
                ( nrows, buf + j, sizeof(T)*ncols, location, _structSize );
            }
        }
    }

    template <typename T>
    void releaseTBlock( BlockDescriptor<T>& block )
    {
        if(block.getRWFlag() & (int)writeOnly)
        {
            size_t ncols = getNumberOfColumns();

            char *ptr = (char *)_ptr + _structSize * block.getRowsOffset();

            T* blockPtr = block.getBlockPtr();

            for( size_t j = 0 ; j < ncols ; j++ )
            {
                NumericTableFeature &f = (*_ddict)[j];

                char *location = ptr + _offsets[j];

                data_feature_utils::vectorStrideDownCast[f.indexType][data_feature_utils::getInternalNumType<T>()]
                ( block.getNumberOfRows(), blockPtr + j, sizeof(T)*ncols, location, _structSize );
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
        if( !(block.getRWFlag() & (int)readOnly) ) return;

        NumericTableFeature &f = (*_ddict)[feat_idx];

        char *ptr = (char *)_ptr + _structSize * idx + _offsets[feat_idx];

        data_feature_utils::vectorStrideUpCast[f.indexType][data_feature_utils::getInternalNumType<T>()]
        ( nrows, ptr, _structSize, block.getBlockPtr(), sizeof(T) );
    }

    template <typename T>
    void releaseTFeature( size_t feat_idx, size_t idx, size_t nrows, T *buf, int rwFlag )
    {
        if (rwFlag & (int)writeOnly)
        {
            NumericTableFeature &f = (*_ddict)[feat_idx];

            char *ptr      = (char *)_ptr + _structSize * idx;
            char *location = ptr + _offsets[feat_idx];

            data_feature_utils::vectorStrideDownCast[f.indexType][data_feature_utils::getInternalNumType<T>()]
            ( nrows, buf, sizeof(T), location, _structSize );
        }
    }

    template <typename T>
    void releaseTFeature( BlockDescriptor<T>& block )
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            size_t feat_idx = block.getColumnsOffset();

            NumericTableFeature &f = (*_ddict)[feat_idx];

            char *ptr = (char *)_ptr + _structSize * block.getRowsOffset() + _offsets[feat_idx];

            data_feature_utils::vectorStrideDownCast[f.indexType][data_feature_utils::getInternalNumType<T>()]
            ( block.getNumberOfRows(), block.getBlockPtr(), sizeof(T), ptr, _structSize );
        }
        block.setDetails( 0, 0, 0 );
    }
};
/** @} */
} // namespace interface1
using interface1::AOSNumericTable;

}
} // namespace daal
#endif
