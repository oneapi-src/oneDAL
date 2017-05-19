/* file: aos_numeric_table.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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

#include "data_management/data/data_serialize.h"

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
class DAAL_EXPORT AOSNumericTable : public NumericTable
{
public:
    DECLARE_SERIALIZABLE_TAG();

    /**
     *  Constructor for an empty Numeric Table with a predefined size of the structure that represents a feature vector
     *  \param[in]  structSize  Size of the structure that represents the feature vector
     *  \param[in]  ncol        Number of columns in the table
     *  \param[in]  nrow        Number of rows in the table
     */
    AOSNumericTable( size_t structSize = 0, size_t ncol = 0, size_t nrow = 0 ): NumericTable(ncol, nrow)
    {
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
    AOSNumericTable( services::SharedPtr<StructDataType> ptr, size_t ncol, size_t nrow = 0 ): NumericTable(ncol, nrow)
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

    /**
     *  Constructor for a Numeric Table with user-allocated memory
     *  \param[in]  ptr     Pointer to a data set in the AOS format
     *  \param[in]  ncol    Number of columns in the table
     *  \param[in]  nrow    Number of rows in the table
     */
    template<typename StructDataType>
    AOSNumericTable( StructDataType *ptr, size_t ncol, size_t nrow = 0 ): NumericTable(ncol, nrow)
    {
        _ptr        = services::SharedPtr<byte>((byte*)ptr, services::EmptyDeleter());
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
    virtual ~AOSNumericTable()
    {
        if(_offsets)
        {
            delete[] _offsets;
        }
        freeDataMemoryImpl();
    }

    /**
     *  Sets a pointer to an array of structures in a Numeric Table
     *  \param[in]  ptr Pointer to a data set in the AOS format
     *  \param[in]  obsnum Number of rows in the table
     */
    services::Status setArray(void *const ptr, size_t obsnum = 0)
    {
        _ptr = services::SharedPtr<byte>((byte*)ptr, services::EmptyDeleter());
        _memStatus = userAllocated;
        return setNumberOfRowsImpl( obsnum );
    }

    /**
     *  Sets a pointer to an array of structures in a Numeric Table
     *  \param[in]  ptr Pointer to a data set in the AOS format
     *  \param[in]  obsnum Number of rows in the table
     */
    services::Status setArray(const services::SharedPtr<byte>& ptr, size_t obsnum = 0)
    {
        _ptr = ptr;
        _memStatus = userAllocated;
        return setNumberOfRowsImpl( obsnum );
    }

    /**
     *  Returns a pointer to an array of structures in a Numeric Table
     *  \return Pointer to a data set in the AOS format
     */
    void *getArray()
    {
        return (void *)(_ptr.get());
    }

    /**
     *  Returns a pointer to an array of structures in a Numeric Table
     *  \return Pointer to a data set in the AOS format
     */
    const void *getArray() const
    {
        return (void *)(_ptr.get());
    }

    /**
     *  Returns a pointer to an array of structures in a Numeric Table
     *  \return Pointer to a data set in the AOS format
     */
    services::SharedPtr<byte> getArraySharedPtr()
    {
        return _ptr;
    }

    /**
     *  Sets a feature in an AOS Numeric Table
     *  \tparam     T              Type of feature values
     *  \param[in]  idx            Feature index
     *  \param[in]  offset         Feature offset in the structure representing the feature vector
     *  \param[in]  featureType    Feature type
     *  \param[in]  categoryNumber Number of categories for categorical features
     */
    template<typename T>
    services::Status setFeature(size_t idx, size_t offset, data_feature_utils::FeatureType featureType = data_feature_utils::DAAL_CONTINUOUS, size_t categoryNumber=0)
    {
        if( _ddict.get() == NULL )
        {
            _ddict = NumericTableDictionaryPtr(new NumericTableDictionary());
        }

        services::Status s = _ddict->setFeature<T>(idx);
        if(!s) return s;
        (*_ddict)[idx].featureType = featureType;
        (*_ddict)[idx].categoryNumber = categoryNumber;

        _offsets[idx] = offset;
        return s;
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

    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    services::Status releaseBlockOfRows(BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return releaseTBlock<double>(block);
    }
    services::Status releaseBlockOfRows(BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return releaseTBlock<float>(block);
    }
    services::Status releaseBlockOfRows(BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return releaseTBlock<int>(block);
    }

    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    services::Status releaseBlockOfColumnValues(BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return releaseTFeature<double>(block);
    }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return releaseTFeature<float>(block);
    }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return releaseTFeature<int>(block);
    }

    /** \private */
    void serializeImpl  (InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<InputDataArchive, false>( arch );}

    /** \private */
    void deserializeImpl(OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<OutputDataArchive, true>( arch );}

protected:
    services::SharedPtr<byte> _ptr;
    size_t  _structSize;
    size_t *_offsets;

    services::Status allocateDataMemoryImpl(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE
    {
        freeDataMemoryImpl();

        size_t size = _structSize * getNumberOfRows();

        if( size == 0 )
            return services::Status(getNumberOfRows() == 0 ? services::ErrorIncorrectNumberOfObservations :
                services::ErrorIncorrectNumberOfFeatures);

        _ptr = services::SharedPtr<byte>((byte *)daal::services::daal_malloc(size), services::ServiceDeleter());

        if(!_ptr)
            return services::Status(services::ErrorMemoryAllocationFailed);

        _memStatus = internallyAllocated;
        return services::Status();
    }

    void freeDataMemoryImpl() DAAL_C11_OVERRIDE
    {
        _ptr = services::SharedPtr<byte>();
        _memStatus = notAllocated;
    }

    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl( Archive *arch )
    {
        NumericTable::serialImpl<Archive, onDeserialize>( arch );
        arch->set(_structSize);

        if( onDeserialize )
        {
            allocateDataMemoryImpl();
        }

        size_t size = getNumberOfRows();

        arch->set( (char *)_ptr.get(), size * _structSize );
    }


private:

    template <typename T>
    services::Status getTBlock(size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T>& block)
    {
        size_t ncols = getNumberOfColumns();
        size_t nobs = getNumberOfRows();
        block.setDetails( 0, idx, rwFlag );

        if (idx >= nobs)
        {
            block.resizeBuffer( ncols, 0 );
            return services::Status();
        }

        nrows = ( idx + nrows < nobs ) ? nrows : nobs - idx;

        if( !block.resizeBuffer( ncols, nrows ) )
            return services::Status(services::ErrorMemoryAllocationFailed);

        if( !(rwFlag & (int)readOnly) )
            return services::Status();

        char *ptr = (char *)(_ptr.get()) + _structSize * idx;

        for( size_t j = 0 ; j < ncols ; j++ )
        {
            NumericTableFeature &f = (*_ddict)[j];

            char *location = ptr + _offsets[j];

            T* blockPtr = block.getBlockPtr();

            data_feature_utils::getVectorStrideUpCast(f.indexType, data_feature_utils::getInternalNumType<T>())
            ( nrows, location, _structSize, blockPtr + j, sizeof(T)*ncols );
        }
        return services::Status();
    }

    template <typename T>
    services::Status releaseTBlock( BlockDescriptor<T>& block )
    {
        if(block.getRWFlag() & (int)writeOnly)
        {
            size_t ncols = getNumberOfColumns();

            char *ptr = (char *)(_ptr.get()) + _structSize * block.getRowsOffset();

            T* blockPtr = block.getBlockPtr();

            for( size_t j = 0 ; j < ncols ; j++ )
            {
                NumericTableFeature &f = (*_ddict)[j];

                char *location = ptr + _offsets[j];

                data_feature_utils::getVectorStrideDownCast(f.indexType, data_feature_utils::getInternalNumType<T>())
                ( block.getNumberOfRows(), blockPtr + j, sizeof(T)*ncols, location, _structSize );
            }
        }
        block.reset();
        return services::Status();
    }

    template <typename T>
    services::Status getTFeature(size_t feat_idx, size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T>& block)
    {
        size_t ncols = getNumberOfColumns();
        size_t nobs = getNumberOfRows();
        block.setDetails( feat_idx, idx, rwFlag );

        if (idx >= nobs)
        {
            block.resizeBuffer( 1, 0 );
            return services::Status();
        }

        nrows = ( idx + nrows < nobs ) ? nrows : nobs - idx;

        if( !block.resizeBuffer( 1, nrows ) )
            return services::Status(services::ErrorMemoryAllocationFailed);

        if((block.getRWFlag() & (int)readOnly))
        {
            NumericTableFeature &f = (*_ddict)[feat_idx];
            char *ptr = (char *)(_ptr.get()) + _structSize * idx + _offsets[feat_idx];
            data_feature_utils::getVectorStrideUpCast(f.indexType, data_feature_utils::getInternalNumType<T>())
                (nrows, ptr, _structSize, block.getBlockPtr(), sizeof(T));
        }
        return services::Status();
    }

    template <typename T>
    services::Status releaseTFeature( BlockDescriptor<T>& block )
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            size_t feat_idx = block.getColumnsOffset();

            NumericTableFeature &f = (*_ddict)[feat_idx];

            char *ptr = (char *)(_ptr.get()) + _structSize * block.getRowsOffset() + _offsets[feat_idx];

            data_feature_utils::getVectorStrideDownCast(f.indexType, data_feature_utils::getInternalNumType<T>())
            ( block.getNumberOfRows(), block.getBlockPtr(), sizeof(T), ptr, _structSize );
        }
        block.reset();
        return services::Status();
    }
};
/** @} */
} // namespace interface1
using interface1::AOSNumericTable;

}
} // namespace daal
#endif
