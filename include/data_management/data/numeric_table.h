/* file: numeric_table.h */
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
//  Declaration and implementation of the base class for numeric tables.
//--
*/


#ifndef __NUMERIC_TABLE_H__
#define __NUMERIC_TABLE_H__

#include "services/base.h"
#include "services/daal_defines.h"
#include "services/daal_memory.h"
#include "services/error_handling.h"
#include "algorithms/algorithm_types.h"
#include "data_management/data/data_collection.h"
#include "data_management/data/data_dictionary.h"

#include "data_management/data/numeric_types.h"

namespace daal
{
/** \brief Contains classes that implement data management functionality, including NumericTables, DataSources, and Compression */
namespace data_management
{

namespace interface1
{
/**
 * @ingroup numeric_tables
 * @{
 */
class NumericTable;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__BLOCKDESCRIPTOR"></a>
 *  \brief %Base class that manages buffer memory for read/write operations required by numeric tables.
 */
template<typename DataType = double>
class DAAL_EXPORT BlockDescriptor
{
public:
    /** \private */
    BlockDescriptor() : _ptr(0), _buffer(0), _capacity(0), _ncols(0), _nrows(0), _colsOffset(0), _rowsOffset(0), _rwFlag(0), _aux_ptr(0) {}

    /** \private */
    ~BlockDescriptor() { freeBuffer(); }

    /**
     *   Gets a pointer to the buffer
     *  \return Pointer to the block
     */
    inline DataType *getBlockPtr() const { return _ptr; }

    /**
     *  Returns the number of columns in the block
     *  \return Number of columns
     */
    inline size_t getNumberOfColumns() const { return _ncols; }

    /**
     *  Returns the number of rows in the block
     *  \return Number of rows
     */
    inline size_t getNumberOfRows() const { return _nrows; }

public:
    /**
     *  \param[in] ptr      Pointer to the buffer
     *  \param[in] nColumns Number of columns
     *  \param[in] nRows    Number of rows
     */
    inline void setPtr( DataType *ptr, size_t nColumns, size_t nRows )
    {
        _ptr   = ptr;
        _ncols = nColumns;
        _nrows = nRows;
    }

    /**
     *  \param[in] nColumns      Number of columns
     *  \param[in] nRows         Number of rows
     *  \param[in] auxMemorySize Memory size
     */
    inline bool resizeBuffer( size_t nColumns, size_t nRows, size_t auxMemorySize = 0 )
    {
        _ncols = nColumns;
        _nrows = nRows;

        size_t newSize = nColumns * nRows * sizeof(DataType) + auxMemorySize;

        if ( newSize  > _capacity )
        {
            freeBuffer();
            _buffer = (DataType *)daal::services::daal_malloc(newSize);
            if ( _buffer != 0 )
            {
                _capacity = newSize;
            }
            else
            {
                return false;
            }

        }

        _ptr = _buffer;
        if(!auxMemorySize)
        {
            _aux_ptr = 0;
        }
        else
        {
            _aux_ptr = _buffer + nColumns * nRows;
        }

        return true;
    }

    inline void setDetails( size_t columnIdx, size_t rowIdx, int rwFlag )
    {
        _colsOffset = columnIdx;
        _rowsOffset = rowIdx;
        _rwFlag     = rwFlag;
    }

    inline size_t getColumnsOffset() const { return _colsOffset; }
    inline size_t getRowsOffset() const { return _rowsOffset; }
    inline size_t getRWFlag() const { return _rwFlag; }
    inline void  *getAdditionalBufferPtr() const { return _aux_ptr; }

protected:
    /**
     *  Frees the buffer
     */
    void freeBuffer()
    {
        if ( _capacity )
        {
            daal::services::daal_free( _buffer );
        }
        _buffer = 0;
        _capacity = 0;
    }

private:
    DataType *_ptr;      /*<! Pointer to the buffer */
    size_t    _nrows;    /*<! Buffer size in bytes */
    size_t    _ncols;    /*<! Buffer size in bytes */

    size_t _colsOffset;    /*<! Buffer size in bytes */
    size_t _rowsOffset;    /*<! Buffer size in bytes */
    int    _rwFlag;        /*<! Buffer size in bytes */

    void *_aux_ptr;

    DataType *_buffer;   /*<! Pointer to the buffer */
    size_t    _capacity; /*<! Buffer size in bytes */
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__NUMERICTABLEIFACE"></a>
 *  \brief Abstract interface class for a data management component responsible for representation of data in the numeric format.
 *  This class declares the most general methods for data access.
 */
class NumericTableIface
{
public:
    virtual ~NumericTableIface()
    {}
    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__MEMORYSTATUS"></a>
     * \brief Enumeration to specify the status of memory related to the Numeric Table
     */
    enum MemoryStatus
    {
        notAllocated,       /*!< No memory allocated */
        userAllocated,      /*!< Memory allocated on user side */
        internallyAllocated /*!< Memory allocated and managed by NumericTable */
    };

    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__ALLOCATIONFLAG"></a>
     * \brief Enumeration to specify whether the Numeric Table must allocate memory
     */
    enum AllocationFlag
    {
        notAllocate,    /*!< Memory will not be allocated by NumericTable */
        doAllocate      /*!< Memory will be allocated by NumericTable when needed */
    };

    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__BASICSTATISTICSID"></a>
     * \brief Enumeration to specify estimates of basic statistics stored
     */
    enum BasicStatisticsId
    {
        minimum    = 0, /*!< Minimum estimate */
        maximum    = 1, /*!< Maximum estimate */
        sum        = 2, /*!< Sum estimate */
        sumSquares = 3  /*!< Sum squares estimate */
    };

    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__FEATUREBASICSTATISTICS"></a>
     * \brief Enumeration to specify feature-specific estimates of basic statistics stored
     */
    enum FeatureBasicStatistics
    {
        counters /*!< Counters estimate */
    };

    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__NORMALIZATIONTYPE"></a>
     * \brief Enumeration to specify types of normalization
     */
    enum NormalizationType
    {
        nonNormalized           = 0, /*!< Default: non-normalized */
        standardScoreNormalized = 1  /*!< Standard score normalization (mean=0, variance=1) */
    };

    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__STORAGELAYOUT"></a>
     * \brief Storage layouts that may need to be supported
     */
    enum StorageLayout
    {
        soa                         = 1,    // 1
        aos                         = 2,    // 2
        csrArray                    = 1 << 4,
        upperPackedSymmetricMatrix  = 1 << 8,
        lowerPackedSymmetricMatrix  = 2 << 8,
        upperPackedTriangularMatrix = 1 << 7,
        lowerPackedTriangularMatrix = 4 << 8,

        layout_unknown      = 0x80000000 // the last bit set
    };

    /**
     *  Sets a data dictionary in the Numeric Table
     *  \param[in] ddict Pointer to the data dictionary
     */
    virtual void setDictionary( NumericTableDictionary *ddict ) = 0;

    /**
     *  Returns a pointer to a data dictionary
     *  \return Pointer to the data dictionary
     */
    virtual NumericTableDictionary *getDictionary() const = 0;

    /**
     *  Returns a shared pointer to a data dictionary
     *  \return Shared pointer to the data dictionary
     */
    virtual services::SharedPtr<NumericTableDictionary> getDictionarySharedPtr() const = 0;

    /**
     *  Resets a data dictionary for the Numeric Table
     */
    virtual void resetDictionary() = 0;

    /**
     *  Returns the type of a given feature
     *  \param[in] feature_idx Feature index
     *  \return Feature type
     */
    virtual data_feature_utils::FeatureType getFeatureType(size_t feature_idx) const = 0;

    /**
     *  Returns the number of categories for a given feature
     *  \param[in] feature_idx Feature index
     *  \return Number of categories
     */
    virtual size_t getNumberOfCategories(size_t feature_idx) const = 0;

    /**
     *  Returns a data layout used in the Numeric Table
     *  \return Data layout
     */
    virtual StorageLayout getDataLayout() const = 0;

    /**
     *  Sets the number of columns in the Numeric Table
     *
     *  \param[in] ncol Number of columns
     */
    virtual void setNumberOfColumns(size_t ncol) = 0;

    /**
     *  Sets the number of rows in the Numeric Table
     *
     *  \param[in] nrow Number of rows
     */
    virtual void setNumberOfRows(size_t nrow) = 0;

    /**
     *  Allocates memory for a data set
     */
    virtual void allocateDataMemory(daal::MemType type = daal::dram) = 0;

    /**
     *  Deallocates the memory allocated for a data set
     */
    virtual void freeDataMemory() = 0;

    /**
     *  Allocates Numeric Tables for basic statistics
     */
    virtual void allocateBasicStatistics() = 0;

    /**
     * Checks the correctness of this numeric table
     * \param[in] errors        Pointer to the collection of errors
     * \param[in] description   Additional information about error
     * \return                  Check status: True if the table satisfies the requirements, false otherwise.
     */
    virtual bool check(services::ErrorCollection *errors, const char *description) const = 0;

};
} // namespace interface1
using interface1::BlockDescriptor;
using interface1::NumericTableIface;

const int packed_mask = (int)NumericTableIface::csrArray                   |
                        (int)NumericTableIface::upperPackedSymmetricMatrix |
                        (int)NumericTableIface::lowerPackedSymmetricMatrix |
                        (int)NumericTableIface::upperPackedTriangularMatrix |
                        (int)NumericTableIface::lowerPackedTriangularMatrix;

namespace interface1
{
/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DENSENUMERICTABLEIFACE"></a>
 *  \brief Abstract interface class for a data management component responsible for accessing data in the numeric format.
 *  This class declares specific methods to access data in a dense homogeneous form.
 */
class DenseNumericTableIface
{
public:
    virtual ~DenseNumericTableIface()
    {}
    /**
     *  Gets a block of rows from a table.
     *
     *  \param[in] vector_idx Index of the first row to include into the block.
     *  \param[in] vector_num Number of rows in the block.
     *  \param[in] rwflag     Flag specifying read/write access to the block of feature vectors.
     *  \param[out] block     The block of feature vectors.
     *
     *  \return Actual number of feature vectors returned by the method.
     */
    virtual void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double> &block) = 0;

    /**
     *  Gets a block of rows from a table.
     *
     *  \param[in] vector_idx Index of the first row to include into the block.
     *  \param[in] vector_num Number of rows in the block.
     *  \param[in] rwflag     Flag specifying read/write access to the block of feature vectors.
     *  \param[out] block     The block of feature vectors.
     *
     *  \return Actual number of feature vectors returned by the method.
     */
    virtual void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float> &block) = 0;

    /**
     *  Gets a block of rows from a table.
     *
     *  \param[in] vector_idx Index of the first row to include into the block.
     *  \param[in] vector_num Number of rows in the block.
     *  \param[in] rwflag     Flag specifying read/write access to the block of feature vectors.
     *  \param[out] block     The block of feature vectors.
     *
     *  \return Actual number of feature vectors returned by the method.
     */
    virtual void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int> &block) = 0;

    /**
     *  Releases a block of rows.
     *  \param[in] block      The block of rows.
     */
    virtual void releaseBlockOfRows(BlockDescriptor<double> &block) = 0;

    /**
     *  Releases a block of rows.
     *  \param[in] block      The block of rows.
     */
    virtual void releaseBlockOfRows(BlockDescriptor<float> &block) = 0;

    /**
     *  Releases a block of rows.
     *  \param[in] block      The block of rows.
     */
    virtual void releaseBlockOfRows(BlockDescriptor<int> &block) = 0;

    /**
     *  Gets a block of values for a given feature.
     *
     *  \param[in] feature_idx Feature index.
     *  \param[in] vector_idx  Index of the first feature vector to include into the block.
     *  \param[in] value_num   Number of feature values in the block.
     *  \param[in] rwflag      Flag specifying read/write access to the block of feature values.
     *  \param[out] block      The block of feature values.
     *
     *  \return Actual number of feature values returned by the method.
     */
    virtual void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                        ReadWriteMode rwflag, BlockDescriptor<double> &block) = 0;

    /**
     *  Gets a block of values for a given feature.
     *
     *  \param[in] feature_idx Feature index.
     *  \param[in] vector_idx  Index of the first feature vector to include into the block.
     *  \param[in] value_num   Number of feature values in the block.
     *  \param[in] rwflag      Flag specifying read/write access to the block of feature values.
     *  \param[out] block      The block of feature values.
     *
     *  \return Actual number of feature values returned by the method.
     */
    virtual void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                        ReadWriteMode rwflag, BlockDescriptor<float> &block) = 0;

    /**
     *  Gets a block of values for a given feature.
     *
     *  \param[in] feature_idx Feature index.
     *  \param[in] vector_idx  Index of the first feature vector to include into the block.
     *  \param[in] value_num   Number of feature values in the block.
     *  \param[in] rwflag      Flag specifying read/write access to the block of feature values.
     *  \param[out] block      The block of feature values.
     *
     *  \return Actual number of feature values returned by the method.
     */
    virtual void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                        ReadWriteMode rwflag, BlockDescriptor<int> &block) = 0;

    /**
     *  Releases a block of values for a given feature.
     *  \param[in] block       The block of feature values.
     */
    virtual void releaseBlockOfColumnValues(BlockDescriptor<double> &block) = 0;

    /**
     *  Releases a block of values for a given feature.
     *  \param[in] block       The block of feature values.
     */
    virtual void releaseBlockOfColumnValues(BlockDescriptor<float> &block) = 0;

    /**
     *  Releases a block of values for a given feature.
     *  \param[in] block       The block of feature values.
     */
    virtual void releaseBlockOfColumnValues(BlockDescriptor<int> &block) = 0;
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__NUMERICTABLE"></a>
 *  \brief Class for a data management component responsible for representation of data in the numeric format.
 *  This class implements the most general methods for data access.
 */
class DAAL_EXPORT NumericTable : public SerializationIface, public NumericTableIface, public DenseNumericTableIface
{
public:
    DAAL_CAST_OPERATOR(NumericTable);

    /** \private */
    NumericTable( NumericTableDictionary *ddict ) : _errors(new services::KernelErrorCollection())
    {
        _obsnum       = 0;
        _ddict        = services::SharedPtr<NumericTableDictionary>(ddict, services::EmptyDeleter<NumericTableDictionary>());
        _layout       = layout_unknown;
        _memStatus    = notAllocated;
        _normalizationFlag = NumericTable::nonNormalized;
    }

    /** \private */
    NumericTable( services::SharedPtr<NumericTableDictionary> ddict ) : _errors(new services::KernelErrorCollection())
    {
        _obsnum       = 0;
        _ddict        = ddict;
        _layout       = layout_unknown;
        _memStatus    = notAllocated;
        _normalizationFlag = NumericTable::nonNormalized;
    }

    /** \private */
    NumericTable( size_t featnum, size_t obsnum, DictionaryIface::FeaturesEqual featuresEqual = DictionaryIface::notEqual ) : _errors(new services::KernelErrorCollection())
    {
        _obsnum       = obsnum;
        _ddict        = services::SharedPtr<NumericTableDictionary>(new NumericTableDictionary(featnum, featuresEqual));
        _layout       = layout_unknown;
        _memStatus    = notAllocated;
        _normalizationFlag = NumericTable::nonNormalized;
    }

    /** \private */
    virtual ~NumericTable() {}

    virtual void setDictionary( NumericTableDictionary *ddict ) DAAL_C11_OVERRIDE
    {
        _ddict = services::SharedPtr<NumericTableDictionary>(ddict, services::EmptyDeleter<NumericTableDictionary>());
    }

    virtual NumericTableDictionary *getDictionary() const DAAL_C11_OVERRIDE { return _ddict.get(); }

    virtual services::SharedPtr<NumericTableDictionary> getDictionarySharedPtr() const DAAL_C11_OVERRIDE { return _ddict; }

    virtual void resetDictionary() DAAL_C11_OVERRIDE {}

    /**
     *  Returns the number of columns in the Numeric Table
     *  \return Number of columns
     */
    size_t getNumberOfColumns() const
    {
        return _ddict->getNumberOfFeatures();
    }

    /**
     *  Returns the number of rows in the Numeric Table
     *  \return Number of rows
     */
    size_t getNumberOfRows() const { return _obsnum; }

    virtual void setNumberOfColumns(size_t ncol) DAAL_C11_OVERRIDE
    {
        _ddict->setNumberOfFeatures(ncol);
    }

    virtual void setNumberOfRows(size_t nrow) DAAL_C11_OVERRIDE
    {
        _obsnum = nrow;
    }

    StorageLayout getDataLayout() const DAAL_C11_OVERRIDE
    {
        return _layout;
    }

    data_feature_utils::FeatureType getFeatureType(size_t feature_idx) const DAAL_C11_OVERRIDE
    {
        if ( _ddict.get() != NULL && _ddict->getNumberOfFeatures() > feature_idx )
        {
            const NumericTableFeature &f = (*_ddict)[feature_idx];
            return f.featureType;
        }
        else
        {
            /* If no dictionary was set, all features are considered numeric */
            return data_feature_utils::DAAL_CONTINUOUS;
        }
    }

    size_t getNumberOfCategories(size_t feature_idx) const DAAL_C11_OVERRIDE
    {
        if ( _ddict.get() != NULL && _ddict->getNumberOfFeatures() > feature_idx &&
             getFeatureType(feature_idx) != data_feature_utils::DAAL_CONTINUOUS )
        {
            return 2; /* Support binary */
        }
        else
        {
            /* If no dictionary was set, all features are considered numeric */
            return -1;
        }
    }

    /**
     *  Gets the status of the memory used by a data set connected with a Numeric Table
     */
    virtual MemoryStatus getDataMemoryStatus() const { return _memStatus; }

    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl( Archive *arch )
    {
        arch->setSharedPtrObj( _ddict );

        arch->set( _obsnum );

        if( onDeserialize )
        {
            _memStatus = notAllocated;
        }

        arch->set( _layout );
    }

    /**
     *  Checks if dataset stored in the numeric table is normalized, according to the given normalization flag
     *  \param[in] flag Normalization flag to check
     *  \return Check result
     */
    bool isNormalized(NormalizationType flag)
    {
        return (_normalizationFlag == flag);
    }

    /**
     *  Sets the normalization flag for dataset stored in the numeric table
     *  \param[in] flag Normalization flag
     *  \return Previous value of the normalization flag
     */
    NormalizationType setNormalizationFlag(NormalizationType flag)
    {
        NormalizationType oldValue = _normalizationFlag;
        _normalizationFlag = flag;
        return oldValue;
    }

    /**
     *  Returns errors during the computation
     *  \return Errors during the computation
     */
    services::SharedPtr<services::KernelErrorCollection> getErrors()
    {
        return _errors;
    }

    /**
     *  Allocates Numeric Tables for basic statistics
     */
    virtual void allocateBasicStatistics() DAAL_C11_OVERRIDE;

    /**
     * Checks the correctness of this numeric table
     * \param[in] errors        Pointer to the collection of errors
     * \param[in] description   Additional information about error
     * \return                  Check status: True if the table satisfies the requirements, false otherwise.
     */
    virtual bool check(services::ErrorCollection *errors, const char *description) const DAAL_C11_OVERRIDE
    {
        if (getDataMemoryStatus() == notAllocated)
        {
            services::SharedPtr<services::Error> error = services::SharedPtr<services::Error>(new services::Error(services::ErrorNullNumericTable));
            error->addStringDetail(services::ArgumentName, description);
            errors->add(error);
            return false;
        }

        if (getNumberOfColumns() == 0)
        {
            services::SharedPtr<services::Error> error = services::SharedPtr<services::Error>(new services::Error(services::ErrorIncorrectNumberOfColumns));
            error->addStringDetail(services::ArgumentName, description);
            errors->add(error);
            return false;
        }
        if (getNumberOfRows() == 0)
        {
            services::SharedPtr<services::Error> error = services::SharedPtr<services::Error>(new services::Error(services::ErrorIncorrectNumberOfRows));
            error->addStringDetail(services::ArgumentName, description);
            errors->add(error);
            return false;
        }

        return true;
    }

public:
    /**
     *  <a name="DAAL-CLASS-DATA_MANAGEMENT__BASICSTATISTICSDATACOLLECTION"></a>
     *  \brief Basic statistics for each column of original Numeric Table
     */
    class BasicStatisticsDataCollection : public algorithms::Argument
    {
    public:
        BasicStatisticsDataCollection() : algorithms::Argument(4) {}

        services::SharedPtr<NumericTable> get(BasicStatisticsId id)
        {
            return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
        }

        void set(BasicStatisticsId id, const services::SharedPtr<NumericTable> &value)
        {
            Argument::set(id, value);
        }
    };

    BasicStatisticsDataCollection basicStatistics; /** Basic statistics container */

protected:
    services::SharedPtr<NumericTableDictionary> _ddict;

    size_t _obsnum;

    MemoryStatus  _memStatus;
    StorageLayout _layout;

    NormalizationType _normalizationFlag;

    services::SharedPtr<services::KernelErrorCollection> _errors;
};
typedef services::SharedPtr<NumericTable> NumericTablePtr;
typedef services::SharedPtr<const NumericTable> NumericTableConstPtr;
/** @} */

} // namespace interface1
using interface1::DenseNumericTableIface;
using interface1::NumericTable;
using interface1::NumericTablePtr;
using interface1::NumericTableConstPtr;

/**
 * Checks the correctness of this numeric table
 * \param[in]  nt                The numeric table to check
 * \param[out] errors            The collection of errors
 * \param[in]  description       Additional information about error
 * \param[in]  unexpectedLayouts The bit mask of invalid layouts for this numeric table.
 * \param[in]  expectedLayouts   The bit mask of valid layouts for this numeric table.
 * \param[in]  nColumns          Required number of columns.
 *                               nColumns = 0 means that required number of columns is not specified.
 * \param[in]  nRows             Required number of rows.
 *                               nRows = 0 means that required number of rows is not specified.
 * \return                       Check status: True if the table satisfies the requirements, false otherwise.
 */
DAAL_EXPORT bool checkNumericTable(const NumericTable *nt, services::ErrorCollection *errors, const char *description,
                                   const int unexpectedLayouts = 0, const int expectedLayouts = 0, size_t nColumns = 0, size_t nRows = 0);
/**
 * Converts numeric table with arbitrary storage layout to homogen numeric table of the given type
 * \param[in]  src               Pointer to numeric table
 * \param[in]  type              Type of result numeric table memory
 * \return                       Pointer to homogen numeric table
 */
template<typename DataType>
DAAL_EXPORT daal::services::SharedPtr<daal::data_management::NumericTable> convertToHomogen(NumericTable& src, daal::MemType type = daal::dram);
}
} // namespace daal
#endif
