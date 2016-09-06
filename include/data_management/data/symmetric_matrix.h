/* file: symmetric_matrix.h */
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
//  Declaration and implementation of a symmetric matrix.
//--
*/


#ifndef __SYMMETRIC_MATRIX_H__
#define __SYMMETRIC_MATRIX_H__

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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__PACKEDARRAYNUMERICTABLEIFACE"></a>
 *  \brief Abstract class that defines the interface of symmetric matrices stored as a one-dimensional array
 */
class PackedArrayNumericTableIface
{
public:
    virtual ~PackedArrayNumericTableIface()
    {}
    /**
     *  Gets the whole packed array of a requested data type
     *
     *  \param[in]  rwflag  Flag specifying read/write access to a block of feature vectors.
     *  \param[out] block   The block of feature values.
     *
     *  \return Actual number of feature vectors returned by the method.
     */
    virtual void getPackedArray(ReadWriteMode rwflag, BlockDescriptor<double> &block) = 0;

    /**
     *  Gets the whole packed array of a requested data type
     *
     *  \param[in]  rwflag  Flag specifying read/write access to a block of feature vectors.
     *  \param[out] block   The block of feature values.
     *
     *  \return Actual number of feature vectors returned by the method.
     */
    virtual void getPackedArray(ReadWriteMode rwflag, BlockDescriptor<float> &block) = 0;

    /**
     *  Gets the whole packed array of a requested data type
     *
     *  \param[in]  rwflag  Flag specifying read/write access to a block of feature vectors.
     *  \param[out] block   The block of feature values.
     *
     *  \return Actual number of feature vectors returned by the method.
     */
    virtual void getPackedArray(ReadWriteMode rwflag, BlockDescriptor<int> &block) = 0;

    /**
     *  Releases a packed array
     *  \param[in] block   The block of feature values.
     */
    virtual void releasePackedArray(BlockDescriptor<double> &block) = 0;

    /**
     *  Releases a packed array
     *  \param[in] block   The block of feature values.
     */
    virtual void releasePackedArray(BlockDescriptor<float> &block) = 0;

    /**
     *  Releases a packed array
     *  \param[in] block   The block of feature values.
     */
    virtual void releasePackedArray(BlockDescriptor<int> &block) = 0;
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__PACKEDSYMMETRICMATRIX"></a>
 *  \brief Class that provides methods to access symmetric matrices stored as a one-dimensional array.
 *  \tparam DataType Defines the underlying data type that describes the Numeric Table
 */
template<NumericTableIface::StorageLayout packedLayout, typename DataType = double>
class PackedSymmetricMatrix : public NumericTable, public PackedArrayNumericTableIface
{
public:
    /**
     *  Typedef that stores the datatype used for template instantiation
     */
    typedef DataType baseDataType;

public:
    /**
     *  Constructor for a Numeric Table with user-allocated memory
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[in]  nDim        Matrix dimension
     */
    PackedSymmetricMatrix( DataType *const ptr = 0, size_t nDim = 0 ):
        NumericTable( nDim, nDim ), _ptr(0)
    {
        _layout = packedLayout;
        setArray( ptr );

        NumericTableFeature df;
        df.setType<DataType>();
        _ddict->setAllFeatures(df);
    }

    /**
     *  Constructor for a Numeric Table with user-allocated memory and filling the table with a constant
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[in]  nDim        Matrix dimension
     *  \param[in]  constValue  Constant to initialize entries of the packed symmetric matrix
     */
    PackedSymmetricMatrix( DataType *const ptr, size_t nDim, const DataType &constValue ):
        NumericTable( nDim, nDim ), _ptr(0)
    {
        _layout = packedLayout;
        setArray( ptr );

        NumericTableFeature df;
        df.setType<DataType>();
        _ddict->setAllFeatures(df);

        assign( constValue );
    }

    /**
     *  Constructor for a Numeric Table with memory allocation controlled via a flag
     *  \param[in]  nDim                    Matrix dimension
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     */
    PackedSymmetricMatrix( size_t nDim, AllocationFlag memoryAllocationFlag ):
        NumericTable( nDim, nDim ), _ptr(0)
    {
        _layout = packedLayout;

        NumericTableFeature df;
        df.setType<DataType>();
        _ddict->setAllFeatures(df);

        if( memoryAllocationFlag == doAllocate ) { allocateDataMemory(); }
    }

    /**
     *  Constructor for a Numeric Table with memory allocation controlled via a flag and filling the table with a constant
     *  \param[in]  nDim                    Matrix dimension
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \param[in]  constValue              Constant to initialize entries of the packed symmetric matrix
     */
    PackedSymmetricMatrix( size_t nDim, NumericTable::AllocationFlag memoryAllocationFlag,
                           const DataType &constValue ):
        NumericTable( nDim, nDim ), _ptr(0)
    {
        _layout = packedLayout;

        NumericTableFeature df;
        df.setType<DataType>();
        _ddict->setAllFeatures(df);

        if( memoryAllocationFlag == doAllocate ) { allocateDataMemory(); }

        assign( constValue );
    }

    /** \private */
    virtual ~PackedSymmetricMatrix()
    {
        freeDataMemory();
    }

    virtual int getSerializationTag() DAAL_C11_OVERRIDE
    {
        return SERIALIZATION_PACKEDSYMMETRIC_NT_ID + 20 * (packedLayout == lowerPackedSymmetricMatrix) +
        data_feature_utils::getIndexNumType<DataType>();
    }

    /**
     *  Sets the number of columns in the Numeric Table
     *
     *  \param[in] nDim Matrix dimension
     */
    virtual void setNumberOfColumns(size_t nDim) DAAL_C11_OVERRIDE
    {
        if( _ddict->getNumberOfFeatures() != nDim )
        {
            _ddict->setNumberOfFeatures( nDim );

            NumericTableFeature df;
            df.setType<DataType>();
            _ddict->setAllFeatures(df);
        }

        _obsnum = nDim;
    }

    /**
     *  Sets the number of columns in the Numeric Table
     *
     *  \param[in] nDim Matrix dimension
     */
    virtual void setNumberOfRows(size_t nDim) DAAL_C11_OVERRIDE
    {
        return setNumberOfColumns( nDim );
    }

    /**
     *  Returns a pointer to a data set registered in the packed symmetric matrix
     *  \return Pointer to the data set
     */
    DataType *getArray() const
    {
        return _ptr;
    }

    /**
     *  Sets a pointer to a packed array
     *  \param[in] ptr Pointer to the data set in the packed format
     */
    void setArray( DataType *const ptr )
    {
        freeDataMemory();

        if( ptr == 0 )
        {
            this->_errors->add(services::ErrorEmptyHomogenNumericTable);
            return;
        }

        _ptr = ptr;
        _memStatus = userAllocated;
    }

    /**
     *  Fills a numeric table with a constant
     *  \param[in]  constValue  Constant to initialize entries of the packed symmetric matrix
     */
    void assign( const DataType &constValue )
    {
        if( _memStatus == notAllocated )
        {
            this->_errors->add(services::ErrorEmptyHomogenNumericTable);
            return;
        }

        size_t nDim = getNumberOfColumns();

        for( size_t i = 0; i < (nDim * (nDim + 1)) / 2 ; i++ )
        {
            _ptr[i] = constValue;
        }
    }

    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
        getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
        getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
        getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    void releaseBlockOfRows(BlockDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<double>(block);
    }
    void releaseBlockOfRows(BlockDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<float>(block);
    }
    void releaseBlockOfRows(BlockDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<int>(block);
    }

    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                ReadWriteMode rwflag, BlockDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
        getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                ReadWriteMode rwflag, BlockDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
        getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                ReadWriteMode rwflag, BlockDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
        getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    void releaseBlockOfColumnValues(BlockDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<double>(block);
    }
    void releaseBlockOfColumnValues(BlockDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<float>(block);
    }
    void releaseBlockOfColumnValues(BlockDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<int>(block);
    }


    void getPackedArray(ReadWriteMode rwflag, BlockDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
        getTPackedArray<double>(rwflag, block);
    }
    void getPackedArray(ReadWriteMode rwflag, BlockDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
        getTPackedArray<float>(rwflag, block);
    }
    void getPackedArray(ReadWriteMode rwflag, BlockDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
        getTPackedArray<int>(rwflag, block);
    }

    void releasePackedArray(BlockDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
        releaseTPackedArray<double>(block);
    }
    void releasePackedArray(BlockDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
        releaseTPackedArray<float>(block);
    }
    void releasePackedArray(BlockDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
        releaseTPackedArray<int>(block);
    }

    void allocateDataMemory(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE
    {
        freeDataMemory();

        size_t nDim = getNumberOfColumns();
        size_t size = (nDim * (nDim + 1)) / 2;

        if( size == 0 )
        {
            if( getNumberOfColumns() == 0 )
            {
                this->_errors->add(services::ErrorIncorrectNumberOfFeatures);
                return;
            }
            else
            {
                this->_errors->add(services::ErrorIncorrectNumberOfObservations);
                return;
            }
        }

        _ptr = (DataType *)daal::services::daal_malloc( size * sizeof(DataType) );

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

        size_t nDim = getNumberOfColumns();
        size_t size = (nDim * (nDim + 1)) / 2;

        arch->set( _ptr, size );
    }

protected:
    DataType *_ptr;

private:
    DataType *internal_getBlockOfRows( size_t idx )
    {
        size_t _featnum = _ddict->getNumberOfFeatures();
        return _ptr + _featnum * idx;
    }
    DataType *internal_getBlockOfRows( size_t idx, size_t feat_idx )
    {
        size_t _featnum = _ddict->getNumberOfFeatures();
        return _ptr + _featnum * idx + feat_idx;
    }

    template<typename T1, typename T2>
    void internal_repack( size_t p, size_t n, T1 *src, T2 *dst )
    {
        if( IsSameType<T1, T2>::value )
        {
            if( src != (T1 *)dst )
            {
                daal::services::daal_memcpy_s(dst, n * p * sizeof(T1), src, n * p * sizeof(T1));
            }
        }
        else
        {
            size_t i, j;

            for(i = 0; i < n; i++)
            {
                for(j = 0; j < p; j++)
                {
                    dst[i * p + j] = static_cast<T2>(src[i * p + j]);
                }
            }
        }
    }

    template<typename T1, typename T2>
    void internal_set_col_repack( size_t p, size_t n, T1 *src, T2 *dst )
    {
        size_t i;

        for(i = 0; i < n; i++)
        {
            dst[i * p] = static_cast<T2>(src[i]);
        }
    }

protected:
    baseDataType &getBaseValue( size_t dim, size_t rowIdx, size_t colIdx )
    {
        size_t rowStartOffset, colStartOffset;

        if( packedLayout == upperPackedSymmetricMatrix )
        {
            if( colIdx < rowIdx )
            {
                size_t tmp;
                tmp = colIdx;
                colIdx = rowIdx;
                rowIdx = tmp;
            }

            rowStartOffset = ((2 * dim - 1 * (rowIdx - 1)) * rowIdx) / 2; /* Arithmetic progression sum */
            colStartOffset = colIdx - rowIdx;
        }
        else /* here lowerPackedSymmetricMatrix is supposed */
        {
            if( colIdx > rowIdx )
            {
                size_t tmp;
                tmp = colIdx;
                colIdx = rowIdx;
                rowIdx = tmp;
            }

            rowStartOffset = ((2 + 1 * (rowIdx - 1)) * rowIdx) / 2; /* Arithmetic progression sum */
            colStartOffset = colIdx;

        }
        return *(_ptr + rowStartOffset + colStartOffset);
    }

    template <typename T>
    T getValue( size_t dim, size_t rowIdx, size_t colIdx )
    {
        return static_cast<T>( getBaseValue( dim, rowIdx, colIdx ) );
    }

    template <typename T>
    void setValue( size_t dim, size_t rowIdx, size_t colIdx, T value )
    {
        getBaseValue( dim, rowIdx, colIdx ) = static_cast<baseDataType>( value );
    }

    template <typename T>
    void getTBlock( size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> &block )
    {
        size_t nDim = getNumberOfColumns();
        block.setDetails( 0, idx, rwFlag );

        if (idx >= nDim)
        {
            block.resizeBuffer( nDim, 0 );
            return;
        }

        nrows = ( idx + nrows < nDim ) ? nrows : nDim - idx;

        if( !block.resizeBuffer( nDim, nrows ) )
        {
            this->_errors->add(services::ErrorMemoryAllocationFailed);
            return;
        }
        if( !(rwFlag & (int)readOnly) ) { return; }

        T *buffer = block.getBlockPtr();

        for( size_t iRow = 0; iRow < nrows; iRow++ )
        {
            for( size_t iCol = 0; iCol < nDim; iCol++ )
            {
                buffer[ iRow * nDim + iCol ] = getValue<T>( nDim, iRow + idx, iCol );
            }
        }
    }

    template <typename T>
    void releaseTBlock( BlockDescriptor<T> &block )
    {
        if(block.getRWFlag() & (int)writeOnly)
        {
            size_t nDim = getNumberOfColumns();
            size_t nrows = block.getNumberOfRows();
            size_t idx = block.getRowsOffset();
            T     *buffer = block.getBlockPtr();

            for( size_t iRow = 0; iRow < nrows; iRow++ )
            {
                for( size_t iCol = 0; iCol < nDim; iCol++ )
                {
                    setValue<T>( nDim, idx + iRow, iCol, buffer[ iRow * nDim + iCol ] );
                }
            }
        }
    }

    template <typename T>
    void getTFeature( size_t feat_idx, size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> &block )
    {
        size_t nDim = getNumberOfColumns();
        block.setDetails( feat_idx, idx, rwFlag );

        if (idx >= nDim)
        {
            block.resizeBuffer( nDim, 0 );
            return;
        }

        nrows = ( idx + nrows < nDim ) ? nrows : nDim - idx;

        if( !block.resizeBuffer( 1, nrows ) ) { return; }
        if( !(block.getRWFlag() & (int)readOnly) ) { return; }

        T *buffer = block.getBlockPtr();

        for( size_t iRow = 0; iRow < nrows; iRow++ )
        {
            buffer[ iRow ] = getValue<T>( nDim, iRow + idx, feat_idx );
        }
    }

    template <typename T>
    void releaseTFeature( BlockDescriptor<T> &block )
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            size_t nDim  = getNumberOfColumns();
            size_t nrows = block.getNumberOfRows();
            size_t idx = block.getRowsOffset();
            size_t feat_idx = block.getColumnsOffset();
            T     *buffer = block.getBlockPtr();

            for( size_t iRow = 0; iRow < nrows; iRow++ )
            {
                setValue<T>( nDim, iRow + idx, feat_idx, buffer[ iRow ] );
            }
        }
    }

    template <typename T>
    void getTPackedArray( int rwFlag, BlockDescriptor<T> &block )
    {
        size_t nDim = getNumberOfColumns();
        block.setDetails( 0, 0, rwFlag );

        size_t nSize = (nDim * (nDim + 1)) / 2;

        if( IsSameType<T, DataType>::value )
        {
            block.setPtr( (T *)_ptr, 1, nSize );
            return;
        }

        if( !block.resizeBuffer( 1, nSize ) ) { return; }

        if( !(rwFlag & (int)readOnly) ) { return; }

        T *buffer = block.getBlockPtr();
        for( size_t i = 0; i < nSize; i++ )
        {
            buffer[ i ] = static_cast<T>(*(_ptr + i));
        }
    }

    template <typename T>
    void releaseTPackedArray( BlockDescriptor<T> &block )
    {
        if( (block.getRWFlag() & (int)writeOnly) && !IsSameType<T, DataType>::value )
        {
            size_t nDim  = getNumberOfColumns();
            size_t nSize = (nDim * (nDim + 1)) / 2;
            T *buffer = block.getBlockPtr();

            for( size_t i = 0; i < nSize; i++ )
            {
                *(_ptr + i) = static_cast<baseDataType>(buffer[ i ]);
            }
        }
        block.setDetails( 0, 0, 0 );
    }
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__PACKEDTRIANGULARMATRIX"></a>
 *  \brief Class that provides methods to access a packed triangular matrix stored as a one-dimensional array.
 *  \tparam DataType Defines the underlying data type that describes the packed triangular matrix
 */
template<NumericTableIface::StorageLayout packedLayout, typename DataType = double>
class PackedTriangularMatrix : public NumericTable, public PackedArrayNumericTableIface
{
public:
    /**
     *  Typedef that stores the data type used for template instantiation
     */
    typedef DataType baseDataType;

public:
    /**
     *  Constructor for a Numeric Table with user-allocated memory
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[in]  nDim        Matrix dimension
     */
    PackedTriangularMatrix( DataType *const ptr = 0, size_t nDim = 0 ):
        NumericTable( nDim, nDim ), _ptr(0)
    {
        _layout = packedLayout;
        setArray( ptr );

        NumericTableFeature df;
        df.setType<DataType>();
        _ddict->setAllFeatures(df);
    }

    /**
     *  Constructor for a Numeric Table with user-allocated memory and filling the table with a constant
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[in]  nDim        Matrix dimension
     *  \param[in]  constValue  Constant to initialize entries of the packed symmetric matrix
     */
    PackedTriangularMatrix( DataType *const ptr, size_t nDim, const DataType &constValue ):
        NumericTable( nDim, nDim ), _ptr(0)
    {
        _layout = packedLayout;
        setArray( ptr );

        NumericTableFeature df;
        df.setType<DataType>();
        _ddict->setAllFeatures(df);

        assign( constValue );
    }

    /**
     *  Constructor for a Numeric Table with memory allocation controlled via a flag
     *  \param[in]  nDim                    Matrix dimension
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     */
    PackedTriangularMatrix( size_t nDim, AllocationFlag memoryAllocationFlag ):
        NumericTable( nDim, nDim ), _ptr(0)
    {
        _layout = packedLayout;

        NumericTableFeature df;
        df.setType<DataType>();
        _ddict->setAllFeatures(df);

        if( memoryAllocationFlag == doAllocate ) { allocateDataMemory(); }
    }

    /**
     *  Constructor for a Numeric Table with memory allocation controlled via a flag and filling the table with a constant
     *  \param[in]  nDim                    Matrix dimension
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \param[in]  constValue              Constant to initialize entries of the packed symmetric matrix
     */
    PackedTriangularMatrix( size_t nDim, NumericTable::AllocationFlag memoryAllocationFlag,
                            const DataType &constValue ):
        NumericTable( nDim, nDim ), _ptr(0)
    {
        _layout = packedLayout;

        NumericTableFeature df;
        df.setType<DataType>();

        _ddict->setAllFeatures(df);

        if( memoryAllocationFlag == doAllocate ) { allocateDataMemory(); }

        assign( constValue );
    }

    /** \private */
    virtual ~PackedTriangularMatrix()
    {
        freeDataMemory();
    }

    virtual int getSerializationTag() DAAL_C11_OVERRIDE
    {
        return SERIALIZATION_PACKEDTRIANGULAR_NT_ID + 20 * (packedLayout == lowerPackedSymmetricMatrix) +
               data_feature_utils::getIndexNumType<DataType>();
    }

    virtual void setNumberOfColumns(size_t nDim) DAAL_C11_OVERRIDE
    {
        if( _ddict->getNumberOfFeatures() != nDim )
        {
            _ddict->setNumberOfFeatures( nDim );

            NumericTableFeature df;
            df.setType<DataType>();
            _ddict->setAllFeatures(df);
        }

        _obsnum = nDim;
    }

    virtual void setNumberOfRows(size_t nDim) DAAL_C11_OVERRIDE
    {
        return setNumberOfColumns( nDim );
    }

    /**
     *  Returns a pointer to a data set registered in the packed symmetric matrix
     *  \return Pointer to the data set
     */
    DataType *getArray() const
    {
        return _ptr;
    }

    /**
     *  Sets a pointer to an array that stores a packed triangular matrix
     *  \param[in] ptr Pointer to the array that stores the packed triangular matrix
     */
    void setArray( DataType *const ptr )
    {
        freeDataMemory();

        if( ptr == 0 ) { this->_errors->add(services::ErrorEmptyHomogenNumericTable); return; }

        _ptr = ptr;
        _memStatus = userAllocated;
    }

    /**
     *  Fills a numeric table with a constant
     *  \param[in]  constValue  Constant to initialize entries of the packed symmetric matrix
     */
    void assign( const DataType &constValue )
    {
        if( _memStatus == notAllocated )
        {
            this->_errors->add(services::ErrorEmptyHomogenNumericTable);
            return;
        }

        size_t nDim = getNumberOfColumns();

        for( size_t i = 0; i < (nDim * (nDim + 1)) / 2 ; i++ )
        {
            _ptr[i] = constValue;
        }
    }

    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
        getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
        getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
        getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    void releaseBlockOfRows(BlockDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<double>(block);
    }
    void releaseBlockOfRows(BlockDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<float>(block);
    }
    void releaseBlockOfRows(BlockDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<int>(block);
    }

    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                ReadWriteMode rwflag, BlockDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
        getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                ReadWriteMode rwflag, BlockDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
        getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                ReadWriteMode rwflag, BlockDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
        getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    void releaseBlockOfColumnValues(BlockDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<double>(block);
    }
    void releaseBlockOfColumnValues(BlockDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<float>(block);
    }
    void releaseBlockOfColumnValues(BlockDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<int>(block);
    }

    void getPackedArray(ReadWriteMode rwflag, BlockDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
        getTPackedArray<double>(rwflag, block);
    }
    void getPackedArray(ReadWriteMode rwflag, BlockDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
        getTPackedArray<float>(rwflag, block);
    }
    void getPackedArray(ReadWriteMode rwflag, BlockDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
        getTPackedArray<int>(rwflag, block);
    }

    void releasePackedArray(BlockDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
        releaseTPackedArray<double>(block);
    }
    void releasePackedArray(BlockDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
        releaseTPackedArray<float>(block);
    }
    void releasePackedArray(BlockDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
        releaseTPackedArray<int>(block);
    }

    void allocateDataMemory(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE
    {
        freeDataMemory();

        size_t nDim = getNumberOfColumns();
        size_t size = (nDim * (nDim + 1)) / 2;

        if( size == 0 )
        {
            if( getNumberOfColumns() == 0 )
            {
                this->_errors->add(services::ErrorIncorrectNumberOfFeatures);
                return;
            }
            else
            {
                this->_errors->add(services::ErrorIncorrectNumberOfObservations);
                return;
            }
        }

        _ptr = (DataType *)daal::services::daal_malloc( size * sizeof(DataType) );

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

        size_t nDim = getNumberOfColumns();
        size_t size = (nDim * (nDim + 1)) / 2;

        arch->set( _ptr, size );
    }

protected:
    DataType *_ptr;

private:
    DataType *internal_getBlockOfRows( size_t idx )
    {
        size_t _featnum = _ddict->getNumberOfFeatures();
        return _ptr + _featnum * idx;
    }
    DataType *internal_getBlockOfRows( size_t idx, size_t feat_idx )
    {
        size_t _featnum = _ddict->getNumberOfFeatures();
        return _ptr + _featnum * idx + feat_idx;
    }

    template<typename T1, typename T2>
    void internal_repack( size_t p, size_t n, T1 *src, T2 *dst )
    {
        if( IsSameType<T1, T2>::value )
        {
            if( src != (T1 *)dst )
            {
                daal::services::daal_memcpy_s(dst, n * p * sizeof(T1), src, n * p * sizeof(T1));
            }
        }
        else
        {
            size_t i, j;

            for(i = 0; i < n; i++)
            {
                for(j = 0; j < p; j++)
                {
                    dst[i * p + j] = static_cast<T2>(src[i * p + j]);
                }
            }
        }
    }

    template<typename T1, typename T2>
    void internal_set_col_repack( size_t p, size_t n, T1 *src, T2 *dst )
    {
        size_t i;

        for(i = 0; i < n; i++)
        {
            dst[i * p] = static_cast<T2>(src[i]);
        }
    }

protected:
    baseDataType &getBaseValue( size_t dim, size_t rowIdx, size_t colIdx, baseDataType &zero )
    {
        size_t rowStartOffset, colStartOffset;

        if( packedLayout == upperPackedTriangularMatrix )
        {
            if( colIdx < rowIdx )
            {
                return zero;
            }

            rowStartOffset = ((2 * dim - 1 * (rowIdx - 1)) * rowIdx) / 2; /* Arithmetic progression sum */
            colStartOffset = colIdx - rowIdx;
        }
        else /* here lowerPackedTriangularMatrix is supposed */
        {
            if( colIdx > rowIdx )
            {
                return zero;
            }

            rowStartOffset = ((2 + 1 * (rowIdx - 1)) * rowIdx) / 2; /* Arithmetic progression sum */
            colStartOffset = colIdx;

        }
        return *(_ptr + rowStartOffset + colStartOffset);
    }

    template <typename T>
    T getValue( size_t dim, size_t rowIdx, size_t colIdx )
    {
        baseDataType zero = (baseDataType)0;
        return static_cast<T>( getBaseValue( dim, rowIdx, colIdx, zero ) );
    }

    template <typename T>
    void setValue( size_t dim, size_t rowIdx, size_t colIdx, T value )
    {
        baseDataType zero = (baseDataType)0;
        getBaseValue( dim, rowIdx, colIdx, zero ) = static_cast<baseDataType>( value );
    }

    template <typename T>
    void getTBlock( size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> &block )
    {
        size_t nDim = getNumberOfColumns();
        block.setDetails( 0, idx, rwFlag );

        if (idx >= nDim)
        {
            block.resizeBuffer( nDim, 0 );
            return;
        }

        nrows = ( idx + nrows < nDim ) ? nrows : nDim - idx;

        if( !block.resizeBuffer( nDim, nrows ) ) { return; }
        if( !(rwFlag & (int)readOnly) ) { return; }

        T *buffer = block.getBlockPtr();

        for( size_t iRow = 0; iRow < nrows; iRow++ )
        {
            for( size_t iCol = 0; iCol < nDim; iCol++ )
            {
                buffer[ iRow * nDim + iCol ] = getValue<T>( nDim, iRow + idx, iCol );
            }
        }
    }

    template <typename T>
    void releaseTBlock( BlockDescriptor<T> &block )
    {
        if(block.getRWFlag() & (int)writeOnly)
        {
            size_t nDim = getNumberOfColumns();
            size_t nrows = block.getNumberOfRows();
            size_t idx = block.getRowsOffset();
            T     *buffer = block.getBlockPtr();

            for( size_t iRow = 0; iRow < nrows; iRow++ )
            {
                for( size_t iCol = 0; iCol < nDim; iCol++ )
                {
                    setValue<T>( nDim, iRow + idx, iCol, buffer[ iRow * nDim + iCol ] );
                }
            }
        }
    }

    template <typename T>
    void getTFeature( size_t feat_idx, size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> &block )
    {
        size_t nDim = getNumberOfColumns();
        block.setDetails( feat_idx, idx, rwFlag );

        if (idx >= nDim)
        {
            block.resizeBuffer( nDim, 0 );
            return;
        }

        nrows = ( idx + nrows < nDim ) ? nrows : nDim - idx;

        if( !block.resizeBuffer( 1, nrows ) ) { return; }
        if( !(block.getRWFlag() & (int)readOnly) ) { return; }

        T *buffer = block.getBlockPtr();

        for( size_t iRow = 0; iRow < nrows; iRow++ )
        {
            buffer[ iRow ] = getValue<T>( nDim, iRow + idx, feat_idx );
        }
    }

    template <typename T>
    void releaseTFeature( BlockDescriptor<T> &block )
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            size_t nDim  = getNumberOfColumns();
            size_t nrows = block.getNumberOfRows();
            size_t idx = block.getRowsOffset();
            size_t feat_idx = block.getColumnsOffset();
            T     *buffer = block.getBlockPtr();

            for( size_t iRow = 0; iRow < nrows; iRow++ )
            {
                setValue<T>( nDim, iRow + idx, feat_idx, buffer[ iRow ] );
            }
        }
    }

    template <typename T>
    void getTPackedArray( int rwFlag, BlockDescriptor<T> &block )
    {
        size_t nDim = getNumberOfColumns();
        block.setDetails( 0, 0, rwFlag );

        size_t nSize = (nDim * (nDim + 1)) / 2;

        if( IsSameType<T, DataType>::value )
        {
            block.setPtr( (T *)_ptr, 1, nSize );
            return;
        }

        if( !block.resizeBuffer( 1, nSize ) ) { return; }

        if( !(rwFlag & (int)readOnly) ) { return; }

        T *buffer = block.getBlockPtr();
        for( size_t i = 0; i < nSize; i++ )
        {
            buffer[ i ] = static_cast<T>(*(_ptr + i));
        }
    }

    template <typename T>
    void releaseTPackedArray( BlockDescriptor<T> &block )
    {
        if( (block.getRWFlag() & (int)writeOnly) && !IsSameType<T, DataType>::value )
        {
            size_t nDim  = getNumberOfColumns();
            size_t nSize = (nDim * (nDim + 1)) / 2;
            T *buffer = block.getBlockPtr();

            for( size_t i = 0; i < nSize; i++ )
            {
                *(_ptr + i) = static_cast<baseDataType>(buffer[ i ]);
            }
        }
        block.setDetails( 0, 0, 0 );
    }
};
/** @} */
} // namespace interface1
using interface1::PackedArrayNumericTableIface;
using interface1::PackedSymmetricMatrix;
using interface1::PackedTriangularMatrix;

}
} // namespace daal
#endif
