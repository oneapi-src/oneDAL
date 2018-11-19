/* file: soa_numeric_table.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of a heterogeneous table stored as a structure of arrays.
//--
*/

#ifndef __SOA_NUMERIC_TABLE_H__
#define __SOA_NUMERIC_TABLE_H__

#include "data_management/data/numeric_table.h"
#include "data_management/data/internal/conversion.h"

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
class DAAL_EXPORT SOANumericTable : public NumericTable
{
public:
    DECLARE_SERIALIZABLE_TAG();
    DECLARE_SERIALIZABLE_IMPL();

    /**
     *  Constructor for an empty Numeric Table
     *  \param[in]  nColumns      Number of columns in the table
     *  \param[in]  nRows         Number of rows in the table
     *  \param[in]  featuresEqual Flag that makes all features in the NumericTableDictionary equal
     *  \DAAL_DEPRECATED_USE{ SOANumericTable::create }
     */
    SOANumericTable(size_t nColumns = 0, size_t nRows = 0, DictionaryIface::FeaturesEqual featuresEqual = DictionaryIface::notEqual);

    /**
     *  Constructs an empty Numeric Table
     *  \param[in]  nColumns      Number of columns in the table
     *  \param[in]  nRows         Number of rows in the table
     *  \param[in]  featuresEqual Flag that makes all features in the NumericTableDictionary equal
     *  \param[out] stat          Status of the numeric table construction
     *  \return Empty numeric table
     */
    static services::SharedPtr<SOANumericTable> create(size_t nColumns = 0, size_t nRows = 0,
                                                       DictionaryIface::FeaturesEqual featuresEqual = DictionaryIface::notEqual,
                                                       services::Status *stat = NULL);

    /**
     *  Constructor for an empty Numeric Table with a predefined NumericTableDictionary
     *  \param[in]  ddict                 Pointer to the predefined NumericTableDictionary
     *  \param[in]  nRows                 Number of rows in the table
     *  \param[in]  memoryAllocationFlag  Flag that controls internal memory allocation for data in the numeric table
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED SOANumericTable( NumericTableDictionary *ddict, size_t nRows, AllocationFlag memoryAllocationFlag = notAllocate ):
        NumericTable(NumericTableDictionaryPtr(ddict, services::EmptyDeleter())),
        _arraysInitialized(0), _partialMemStatus(notAllocated)
    {
        _layout = soa;
        this->_status |= setNumberOfRowsImpl( nRows );
        if( !resizePointersArray( getNumberOfColumns() ) )
        {
            this->_status.add(services::ErrorMemoryAllocationFailed);
            return;
        }
        if( memoryAllocationFlag == doAllocate )
        {
            this->_status |= allocateDataMemoryImpl();
        }
    }

    /**
     *  Constructor for an empty Numeric Table with a predefined NumericTableDictionary
     *  \param[in]  ddict                 Shared pointer to the predefined NumericTableDictionary
     *  \param[in]  nRows                 Number of rows in the table
     *  \param[in]  memoryAllocationFlag  Flag that controls internal memory allocation for data in the numeric table
     *  \DAAL_DEPRECATED_USE{ SOANumericTable::create }
     */
    SOANumericTable( NumericTableDictionaryPtr ddict, size_t nRows, AllocationFlag memoryAllocationFlag = notAllocate );

    /**
     *  Constructs an empty Numeric Table with a predefined NumericTableDictionary
     *  \param[in]  ddict                 Shared pointer to the predefined NumericTableDictionary
     *  \param[in]  nRows                 Number of rows in the table
     *  \param[in]  memoryAllocationFlag  Flag that controls internal memory allocation for data in the numeric table
     *  \param[out] stat                  Status of the numeric table construction
     *  \return     Numeric table with a predefined NumericTableDictionary
     */
    static services::SharedPtr<SOANumericTable> create(NumericTableDictionaryPtr ddict, size_t nRows,
                                                       AllocationFlag memoryAllocationFlag = notAllocate,
                                                       services::Status *stat = NULL);

    virtual ~SOANumericTable()
    {
        freeDataMemoryImpl();
    }

    /**
     *  Sets a pointer to an array of values for a given feature
     *  \tparam T       Type of feature values
     *  \param[in]  ptr Pointer to the array of the T type that stores feature values
     *  \param[in]  idx Feature index
     */
    template<typename T>
    services::Status setArray(const services::SharedPtr<T> &ptr, size_t idx)
    {
        if( _partialMemStatus != notAllocated && _partialMemStatus != userAllocated )
        {
            return services::Status(services::ErrorIncorrectNumberOfFeatures);
        }

        if( idx < getNumberOfColumns() && idx < _arrays.size() )
        {
            _ddict->setFeature<T>(idx);

            if( !_arrays[idx] && ptr )
            {
                _arraysInitialized++;
            }

            if( _arrays[idx] && !ptr )
            {
                _arraysInitialized--;
            }

            _arrays[idx] = services::reinterpretPointerCast<byte, T>(ptr);
        }
        else
        {
            return services::Status(services::ErrorIncorrectNumberOfFeatures);
        }

        _partialMemStatus = userAllocated;

        if(_arraysInitialized == getNumberOfColumns())
        {
            _memStatus = userAllocated;
        }
        return services::Status();
    }

    /**
    *  Sets a pointer to an array of values for a given feature
    *  \tparam T       Type of feature values
    *  \param[in]  ptr Pointer to the array of the T type that stores feature values
    *  \param[in]  idx Feature index
    */
    template<typename T>
    services::Status setArray(T *ptr, size_t idx)
    {
        return setArray(services::SharedPtr<T>(ptr, services::EmptyDeleter()), idx);
    }

    /**
     *  Returns a pointer to an array of values for a given feature
     *  \param[in]  idx Feature index
     *  \return Pointer to the array of values
     */
    services::SharedPtr<byte> getArraySharedPtr(size_t idx)
    {
        if( idx < _ddict->getNumberOfFeatures() )
        {
            return _arrays[idx];
        }
        else
        {
            this->_status.add(services::ErrorIncorrectNumberOfFeatures);
            return services::SharedPtr<byte>();
        }
    }

    /**
     *  Returns a pointer to an array of values for a given feature
     *  \param[in]  idx Feature index
     *  \return Pointer to the array of values
     */
    void *getArray(size_t idx)
    {
        return getArraySharedPtr(idx).get();
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

    DAAL_DEPRECATED_VIRTUAL services::Status setDictionary( NumericTableDictionary *ddict ) DAAL_C11_OVERRIDE
    {
        services::Status s;
        DAAL_CHECK_STATUS(s, NumericTable::setDictionary( ddict ));

        size_t ncol = ddict->getNumberOfFeatures();

        if( !resizePointersArray( ncol ) )
        {
            return services::Status(services::ErrorMemoryAllocationFailed);
        }
        return s;
    }

protected:

    SOANumericTable( size_t nColumns, size_t nRows, DictionaryIface::FeaturesEqual featuresEqual, services::Status &st );

    SOANumericTable( NumericTableDictionaryPtr ddict, size_t nRows, AllocationFlag memoryAllocationFlag, services::Status &st );

    services::Collection<services::SharedPtr<byte> > _arrays;
    size_t _arraysInitialized;
    MemoryStatus _partialMemStatus;

    bool resizePointersArray(size_t nColumns)
    {
        if( _arrays.size() >= nColumns )
        {
            size_t counter = 0;
            for(size_t i = 0; i < nColumns; i++)
            {
                counter += (_arrays[i] != 0);
            }
            _arraysInitialized = counter;

            if( _arraysInitialized == nColumns )
            {
                _memStatus = _partialMemStatus;
            }
            else
            {
                _memStatus = notAllocated;
            }

            return true;
        }
        _arrays.resize(nColumns);
        _memStatus = notAllocated;

        return true;
    }

    services::Status setNumberOfColumnsImpl(size_t ncol) DAAL_C11_OVERRIDE
    {
        services::Status s;
        DAAL_CHECK_STATUS(s, NumericTable::setNumberOfColumnsImpl(ncol));

        if( !resizePointersArray( ncol ) )
        {
            return services::Status(services::ErrorMemoryAllocationFailed);
        }
        return s;
    }

    services::Status allocateDataMemoryImpl(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE
    {
        freeDataMemoryImpl();

        size_t ncol = _ddict->getNumberOfFeatures();
        size_t nrows = getNumberOfRows();

        if( ncol * nrows == 0 )
        {
            if( nrows == 0 )
            {
                return services::Status(services::ErrorIncorrectNumberOfObservations);
            }
            else
            {
                return services::Status(services::ErrorIncorrectNumberOfFeatures);
            }
        }

        for(size_t i = 0; i < ncol; i++)
        {
            NumericTableFeature f = (*_ddict)[i];
            if( f.typeSize != 0 )
            {
                _arrays[i] = services::SharedPtr<byte>((byte *)daal::services::daal_malloc( f.typeSize * nrows ), services::ServiceDeleter());
                _arraysInitialized++;
            }
            if( !_arrays[i] )
            {
                freeDataMemoryImpl();
                return services::Status(services::ErrorMemoryAllocationFailed);
            }
        }

        if(_arraysInitialized > 0)
        {
            _partialMemStatus = internallyAllocated;
        }

        if(_arraysInitialized == ncol)
        {
            _memStatus = internallyAllocated;
        }
        return services::Status();
    }

    void freeDataMemoryImpl() DAAL_C11_OVERRIDE
    {
        _arrays.clear();
        _arrays.resize(_ddict->getNumberOfFeatures());
        _arraysInitialized = 0;

        _partialMemStatus = notAllocated;
        _memStatus = notAllocated;
    }

    template<typename Archive, bool onDeserialize>
    services::Status serialImpl( Archive *arch )
    {
        NumericTable::serialImpl<Archive, onDeserialize>( arch );

        if( onDeserialize )
        {
            allocateDataMemoryImpl();
        }

        size_t ncol = _ddict->getNumberOfFeatures();
        size_t nrows = getNumberOfRows();

        for(size_t i = 0; i < ncol; i++)
        {
            NumericTableFeature f = (*_ddict)[i];
            void *ptr = getArraySharedPtr(i).get();

            arch->set( (char *)ptr, nrows * f.typeSize );
        }

        return services::Status();
    }

private:

    template <typename T>
    services::Status getTBlock( size_t idx, size_t nrows, ReadWriteMode rwFlag, BlockDescriptor<T>& block )
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
        {
            return services::Status(services::ErrorMemoryAllocationFailed);
        }

        if( !(block.getRWFlag() & (int)readOnly) ) return services::Status();

        T lbuf[32];

        size_t di = 32;

        T *buffer = block.getBlockPtr();

        for( size_t i = 0 ; i < nrows ; i += di )
        {
            if( i + di > nrows ) { di = nrows - i; }

            for( size_t j = 0 ; j < ncols ; j++ )
            {
                NumericTableFeature &f = (*_ddict)[j];

                char *ptr = (char *)_arrays[j].get() + (idx + i) * f.typeSize;

                internal::getVectorUpCast(f.indexType, internal::getConversionDataType<T>())
                ( di, ptr, lbuf );

                for( size_t ii = 0 ; ii < di; ii++ )
                {
                    buffer[ (i + ii)*ncols + j ] = lbuf[ii];
                }
            }
        }
        return services::Status();
    }

    template <typename T>
    services::Status releaseTBlock( BlockDescriptor<T>& block )
    {
        if(block.getRWFlag() & (int)writeOnly)
        {
            size_t ncols = getNumberOfColumns();
            size_t nrows = block.getNumberOfRows();
            size_t idx   = block.getRowsOffset();
            T lbuf[32];

            size_t di = 32;

            T *blockPtr = block.getBlockPtr();

            for( size_t i = 0 ; i < nrows ; i += di )
            {
                if( i + di > nrows ) { di = nrows - i; }

                for( size_t j = 0 ; j < ncols ; j++ )
                {
                    NumericTableFeature &f = (*_ddict)[j];

                    char *ptr = (char *)_arrays[j].get() + (idx + i) * f.typeSize;

                    for( size_t ii = 0 ; ii < di; ii++ )
                    {
                        lbuf[ii] = blockPtr[ (i + ii) * ncols + j ];
                    }

                    internal::getVectorDownCast(f.indexType, internal::getConversionDataType<T>())
                    ( di, lbuf, ptr );
                }
            }
        }
        block.reset();
        return services::Status();
    }

    template <typename T>
    services::Status getTFeature( size_t feat_idx, size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T>& block )
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

        NumericTableFeature &f = (*_ddict)[feat_idx];

        if( features::internal::getIndexNumType<T>() == f.indexType )
        {
            block.setPtr(&(_arrays[feat_idx]), _arrays[feat_idx].get() + idx * f.typeSize , 1, nrows );
        }
        else
        {
            byte *location = _arrays[feat_idx].get() + idx * f.typeSize;

            if( !block.resizeBuffer( 1, nrows ) )
            {
                return services::Status(services::ErrorMemoryAllocationFailed);
            }

            if( !(block.getRWFlag() & (int)readOnly) ) return services::Status();

            internal::getVectorUpCast(f.indexType, internal::getConversionDataType<T>())
            ( nrows, location, block.getBlockPtr() );
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

            if( features::internal::getIndexNumType<T>() != f.indexType )
            {
                char *ptr = (char *)_arrays[feat_idx].get() + block.getRowsOffset() * f.typeSize;

                internal::getVectorDownCast(f.indexType, internal::getConversionDataType<T>())
                ( block.getNumberOfRows(), block.getBlockPtr(), ptr );
            }
        }
        block.reset();
        return services::Status();
    }
};
typedef services::SharedPtr<SOANumericTable> SOANumericTablePtr;
/** @} */
} // namespace interface1
using interface1::SOANumericTable;
using interface1::SOANumericTablePtr;

}
} // namespace daal
#endif
