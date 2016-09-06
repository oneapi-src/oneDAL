/* file: data_source.h */
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
//  Declaration and implementation of the base data source class.
//--
*/

#ifndef __DATA_SOURCE_H__
#define __DATA_SOURCE_H__


#include "data_management/data/data_dictionary.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/aos_numeric_table.h"
#include "data_management/data/soa_numeric_table.h"

#include "data_management/data_source/data_source_utils.h"

namespace daal
{
namespace data_management
{

namespace interface1
{
/**
 * @ingroup data_sources
 * @{
 */
/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATASOURCEIFACE"></a>
 *  \brief Abstract interface class that defines the interface for a data management component responsible for
 *  representation of data in the raw format. This class declares the most generic methods for data access.
 */
class DataSourceIface
{
public:
    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__DATASOURCESTATUS"></a>
     * \brief Specifies the status of the Data Source
     */
    enum DataSourceStatus
    {
        readyForLoad   = 1, /*!< Data is ready to be loaded via loadDataBlock() function */
        waitingForRows = 2, /*!< No data is available, but it may be ready in future */
        endOfData      = 3, /*!< No data is available */
        notReady       = 4  /*!< DataSource not ready for loading */
    };

    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__DICTIONARYCREATIONFLAG"></a>
     * \brief Specifies whether a Data %Dictionary is created from the context of a Data Source
     */
    enum DictionaryCreationFlag
    {
        notDictionaryFromContext = 1, /*!< Do not create dictionary automatically */
        doDictionaryFromContext  = 2  /*!< Do create dictionary when needed */
    };

    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__NUMERICTABLEALLOCATIONFLAG"></a>
     * \brief Specifies whether a Numeric Table is allocated inside of the Data Source object
     */
    enum NumericTableAllocationFlag
    {
        notAllocateNumericTable = 1, /*!< Do not allocate Numeric Table automatically */
        doAllocateNumericTable  = 2  /*!< Do allocate Numeric Table when needed */
    };

public:
    /**
     *  Returns a pointer to a data dictionary
     *  \return Pointer to the Data %Dictionary
     */
    virtual DataSourceDictionary *getDictionary() = 0;

    /**
     *  Sets a predefined Data %Dictionary
     */
    virtual void setDictionary(DataSourceDictionary *dict) = 0;

    /**
     *  Creates a Data Dictionary by extracting information from a Data Source
     */
    virtual void createDictionaryFromContext() = 0;

    /**
     *  Returns the status of a Data Source
     *  \return Status of the Data Source
     */
    virtual DataSourceStatus getStatus() = 0;

    /**
     *  Returns the number of columns in a Data Source
     *  \return Number of columns
     */
    virtual size_t getNumberOfColumns() = 0;

    /**
     *  Returns the number of rows available in a Data Source
     *  \return Number of rows
     */
    virtual size_t getNumberOfAvailableRows() = 0;

    /**
     *  Allocates a Numeric Table associated with a Data Source
     */
    virtual void allocateNumericTable() = 0;

    /**
     *  Returns a pointer to a Numeric Table associated with a Data Source
     *  \return Pointer to the Numeric Table
     */
    virtual NumericTablePtr &getNumericTable() = 0;

    /**
     *  Returns a pointer to a Numeric Table associated with a Data Source
     */
    virtual void freeNumericTable() = 0;

    /**
     *  Loads a data block of a specified size into an internally allocated Numeric Table
     *  \param[in] maxRows Maximum number of rows to load from a Data Source into the Numeric Table
     */
    virtual size_t loadDataBlock(size_t maxRows) = 0;

    /**
     *  Loads a data block of a specified size into an internally allocated Numeric Table
     *  \param[in] maxRows   Maximum number of rows to load from a Data Source into the Numeric Table
     *  \param[in] rowOffset Write data starting from rowOffset row
     *  \param[in] fullRows  Maximum number of rows to allocate in the Numeric Table
     */
    virtual size_t loadDataBlock(size_t maxRows, size_t rowOffset, size_t fullRows) = 0;

    /**
     *  Loads a data block of a specified size into a provided Numeric Table
     *  \param[in] maxRows Maximum number of rows to load from a Data Source into the Numeric Table
     *  \param[in] nt      Pointer to the Numeric Table
     */
    virtual size_t loadDataBlock(size_t maxRows, NumericTable *nt) = 0;

    /**
     *  Loads a data block of a specified size into an internally allocated Numeric Table
     *  \param[in] maxRows   Maximum number of rows to load from a Data Source into the Numeric Table
     *  \param[in] rowOffset Write data starting from rowOffset row
     *  \param[in] fullRows  Maximum number of rows to allocate in the Numeric Table
     *  \param[in] nt        Pointer to the Numeric Table
     */
    virtual size_t loadDataBlock(size_t maxRows, size_t rowOffset, size_t fullRows, NumericTable *nt) = 0;

    /**
     *  Loads a data block into an internally allocated Numeric Table
     */
    virtual size_t loadDataBlock() = 0;

    /**
     *  Loads a data block into a provided Numeric Table
     *  \param[in] nt      Pointer to the Numeric Table
     */
    virtual size_t loadDataBlock(NumericTable *nt) = 0;
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATASOURCE"></a>
 *  \brief Implements the abstract DataSourceIface interface
 */
class DataSource : public DataSourceIface
{
public:
    DataSource() : _dict(NULL), _errors(new services::ErrorCollection()), _initialMaxRows(10) {}

    virtual ~DataSource() {}

    DataSourceDictionary *getDictionary() DAAL_C11_OVERRIDE
    {
        checkDictionary();
        if( this->_errors->size() != 0 ) { return 0; }
        return _dict;
    }

    void setDictionary( DataSourceDictionary *dict ) DAAL_C11_OVERRIDE
    {
        if( _dict != NULL )
        {
            this->_errors->add(services::ErrorDictionaryAlreadyAvailable);
            return;
        }
        _dict = dict;
    }

    void createDictionaryFromContext() DAAL_C11_OVERRIDE
    {
        this->_errors->add(services::ErrorMethodNotSupported);
        return;
    }

    size_t loadDataBlock(size_t maxRows) DAAL_C11_OVERRIDE
    {
        this->_errors->add(services::ErrorMethodNotSupported);
        return 0;
    }

    size_t loadDataBlock(size_t maxRows, NumericTable *nt) DAAL_C11_OVERRIDE
    {
        this->_errors->add(services::ErrorMethodNotSupported);
        return 0;
    }

    size_t loadDataBlock(size_t maxRows, size_t rowOffset, size_t fullRows) DAAL_C11_OVERRIDE
    {
        this->_errors->add(services::ErrorMethodNotSupported);
        return 0;
    }

    size_t loadDataBlock(size_t maxRows, size_t rowOffset, size_t fullRows, NumericTable *nt) DAAL_C11_OVERRIDE
    {
        this->_errors->add(services::ErrorMethodNotSupported);
        return 0;
    }

    size_t loadDataBlock() DAAL_C11_OVERRIDE
    {
        this->_errors->add(services::ErrorMethodNotSupported);
        return 0;
    }

    size_t loadDataBlock(NumericTable *nt) DAAL_C11_OVERRIDE
    {
        this->_errors->add(services::ErrorMethodNotSupported);
        return 0;
    }

    NumericTablePtr &getNumericTable() DAAL_C11_OVERRIDE
    {
        checkNumericTable();

        return _spnt;
    }

    size_t getNumberOfColumns() DAAL_C11_OVERRIDE
    {
        checkDictionary();
        if( this->_errors->size() != 0 ) { return 0; }

        return _dict->getNumberOfFeatures();
    }

    /**
     * Returns errors during the computation
     * \return Errors during the computation
     */
    services::SharedPtr<services::ErrorCollection> getErrors()
    {
        services::ErrorCollectionPtr err(new services::ErrorCollection(*_errors));
        if(_spnt.get())
            err->add(_spnt->getErrors());
        if(_dict)
            err->add(_dict->getErrors());
        return err;
    }

protected:
    DataSourceDictionary    *_dict;
    NumericTablePtr _spnt;

    NumericTableAllocationFlag _autoNumericTableFlag;
    DictionaryCreationFlag     _autoDictionaryFlag;
    services::SharedPtr<services::ErrorCollection> _errors;

    size_t _initialMaxRows;

    /**
     * Checks a Numeric Table
     */
    void checkNumericTable()
    {
        if( _spnt.get() == NULL )
        {
            if( _autoNumericTableFlag == notAllocateNumericTable )
            {
                this->_errors->add(services::ErrorNumericTableNotAllocated);
                return;
            }

            allocateNumericTable();
        }
    }

    /**
     * Checks a Data Dictionary
     */
    void checkDictionary()
    {
        if( _dict == 0 )
        {
            if( _autoDictionaryFlag == notDictionaryFromContext )
            {
                this->_errors->add(services::ErrorDictionaryNotAvailable);
                return;
            }

            createDictionaryFromContext();
        }
    }

    /**
     *  Allocates a Numeric Table that corresponds to the template type
     *
     *  \tparam NumericTableType - Numeric Table type.
     *
     *  \param   nt      - Pointer to the allocated Numeric Table
     */
    template<typename NumericTableType> void allocateNumericTableImpl(NumericTableType **nt);

    /**
     *  Allocates a homogeneous Numeric Table that corresponds to the template type
     *
     *  \tparam FPType - Type of the homogeneous Numeric Table
     *
     *  \param   nt      - Pointer to the allocated Numeric Table
     */
    template<typename FPType> void allocateNumericTableImpl(HomogenNumericTable<FPType> **nt);

    size_t getStructureSize()
    {
        size_t structureSize = 0;
        size_t nFeatures = _dict->getNumberOfFeatures();
        for(size_t i = 0; i < nFeatures; i++)
        {
            data_feature_utils::IndexNumType indexNumType = (*_dict)[i].ntFeature.indexType;
            structureSize += (*_dict)[i].ntFeature.typeSize;
        }
        return structureSize;
    }

    void setNumericTableDictionary(NumericTable *nt)
    {
        services::SharedPtr<NumericTableDictionary> ntDict = nt->getDictionarySharedPtr();

        size_t nFeatures = _dict->getNumberOfFeatures();

        for(size_t i = 0; i < nFeatures; i++)
        {
            (*ntDict)[i] = (*_dict)[i].ntFeature;
        }
    }
};

template<typename NumericTableType>
inline void DataSource::allocateNumericTableImpl(NumericTableType **nt)
{
    *nt = 0;
}

template<>
inline void DataSource::allocateNumericTableImpl(AOSNumericTable **nt)
{
    size_t nFeatures = _dict->getNumberOfFeatures();
    size_t structureSize = getStructureSize();
    *nt = new AOSNumericTable(structureSize, nFeatures, 0);
    setNumericTableDictionary(*nt);
}

template<>
inline void DataSource::allocateNumericTableImpl(SOANumericTable **nt)
{
    *nt = 0;
}

template<typename FPType>
inline void DataSource::allocateNumericTableImpl(HomogenNumericTable<FPType> **nt)
{
    size_t nFeatures = _dict->getNumberOfFeatures();
    *nt = new HomogenNumericTable<FPType>(nFeatures, 0, NumericTableIface::notAllocate);
    setNumericTableDictionary(*nt);
}


/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATASOURCETEMPLATE"></a>
 *  \brief Implements the abstract DataSourceIface interface
 */
template< typename _numericTableType, typename _summaryStatisticsType = double >
class DataSourceTemplate : public DataSource
{
public:
    typedef _numericTableType numericTableType;

public:
    DataSourceTemplate( NumericTableAllocationFlag doAllocateNumericTable,
                        DictionaryCreationFlag doCreateDictionaryFromContext ) : DataSource()
    {
        DataSource::_autoNumericTableFlag = doAllocateNumericTable;
        DataSource::_autoDictionaryFlag   = doCreateDictionaryFromContext;
    }

    virtual ~DataSourceTemplate() {}

    void allocateNumericTable() DAAL_C11_OVERRIDE
    {
        if( _spnt.get() != NULL ) { this->_errors->add(services::ErrorNumericTableAlreadyAllocated); return; }

        checkDictionary();
        if( this->_errors->size() != 0 ) { return; }

        NumericTable *nt;

        allocateNumericTableImpl( (numericTableType **)&nt );
        _spnt = NumericTablePtr(nt);

        HomogenNumericTable<_summaryStatisticsType> *ssNt = 0;

        allocateNumericTableImpl( &ssNt );
        _spnt->basicStatistics.set(NumericTable::minimum  , NumericTablePtr(ssNt));

        allocateNumericTableImpl( &ssNt );
        _spnt->basicStatistics.set(NumericTable::maximum  , NumericTablePtr(ssNt));

        allocateNumericTableImpl( &ssNt );
        _spnt->basicStatistics.set(NumericTable::sum      , NumericTablePtr(ssNt));

        allocateNumericTableImpl( &ssNt );
        _spnt->basicStatistics.set(NumericTable::sumSquares, NumericTablePtr(ssNt));

    }

    void freeNumericTable() DAAL_C11_OVERRIDE
    {
        _spnt = NumericTablePtr();
    }

    void resizeNumericTableImpl( size_t linesToLoad, NumericTable* nt)
    {
        if( nt == NULL ) { this->_errors->add(services::ErrorNullInputNumericTable); return; }

        if( _dict == NULL ) { this->_errors->add(services::ErrorDictionaryNotAvailable); return; }

        size_t nFeatures = _dict->getNumberOfFeatures();
        bool needAllocate = false;

        if (nt->getNumberOfColumns() < nFeatures) {
            nt->setNumberOfColumns(nFeatures);
            needAllocate = true;
        }

        if( nt->getNumberOfRows() < linesToLoad )
        {
            nt->setNumberOfRows(linesToLoad);
            needAllocate = true;
        }

        if (needAllocate)
        {
            nt->allocateDataMemory();
        }

        size_t nCols = nt->getNumberOfColumns();

        nt->allocateBasicStatistics();

        NumericTablePtr ntMin   = nt->basicStatistics.get(NumericTable::minimum   );
        NumericTablePtr ntMax   = nt->basicStatistics.get(NumericTable::maximum   );
        NumericTablePtr ntSum   = nt->basicStatistics.get(NumericTable::sum       );
        NumericTablePtr ntSumSq = nt->basicStatistics.get(NumericTable::sumSquares);

        if( ntMin->getNumberOfColumns() != nCols || ntMin->getNumberOfRows() != 1 )
        {
            ntMin->setNumberOfColumns(nCols);
            ntMin->setNumberOfRows(1);
            ntMin->allocateDataMemory();
        }

        if( ntMax->getNumberOfColumns() != nCols || ntMax->getNumberOfRows() != 1 )
        {
            ntMax->setNumberOfColumns(nCols);
            ntMax->setNumberOfRows(1);
            ntMax->allocateDataMemory();
        }

        if( ntSum->getNumberOfColumns() != nCols || ntSum->getNumberOfRows() != 1 )
        {
            ntSum->setNumberOfColumns(nCols);
            ntSum->setNumberOfRows(1);
            ntSum->allocateDataMemory();
        }

        if( ntSumSq->getNumberOfColumns() != nCols || ntSumSq->getNumberOfRows() != 1 )
        {
            ntSumSq->setNumberOfColumns(nCols);
            ntSumSq->setNumberOfRows(1);
            ntSumSq->allocateDataMemory();
        }

    }

    void updateStatistics( size_t ntRowIndex, NumericTable *nt)
    {
        if( nt == NULL ) { this->_errors->add(services::ErrorNullInputNumericTable); return; }

        NumericTablePtr ntMin   = nt->basicStatistics.get(NumericTable::minimum   );
        NumericTablePtr ntMax   = nt->basicStatistics.get(NumericTable::maximum   );
        NumericTablePtr ntSum   = nt->basicStatistics.get(NumericTable::sum       );
        NumericTablePtr ntSumSq = nt->basicStatistics.get(NumericTable::sumSquares);

        BlockDescriptor<_summaryStatisticsType> blockMin;
        BlockDescriptor<_summaryStatisticsType> blockMax;
        BlockDescriptor<_summaryStatisticsType> blockSum;
        BlockDescriptor<_summaryStatisticsType> blockSumSq;

        ntMin->getBlockOfRows(0, 1, readWrite, blockMin);
        ntMax->getBlockOfRows(0, 1, readWrite, blockMax);
        ntSum->getBlockOfRows(0, 1, readWrite, blockSum);
        ntSumSq->getBlockOfRows(0, 1, readWrite, blockSumSq);

        _summaryStatisticsType *minimum    = blockMin.getBlockPtr();
        _summaryStatisticsType *maximum    = blockMax.getBlockPtr();
        _summaryStatisticsType *sum        = blockSum.getBlockPtr();
        _summaryStatisticsType *sumSquares = blockSumSq.getBlockPtr();

        if( minimum == NULL || maximum == NULL || sum == NULL || sumSquares == NULL )
        {
            this->_errors->add(services::ErrorIncorrectInputNumericTable);
            return;
        }

        size_t nCols = nt->getNumberOfColumns();

        BlockDescriptor<_summaryStatisticsType> block;
        nt->getBlockOfRows( ntRowIndex, 1, readOnly, block );
        _summaryStatisticsType *row = block.getBlockPtr();

        if( ntRowIndex != 0 )
        {
            for( size_t i = 0; i < nCols; i++ )
            {
                if( minimum[i] > row[i] ) { minimum[i] = row[i]; }
                if( maximum[i] < row[i] ) { maximum[i] = row[i]; }
                sum[i]   += row[i];
                sumSquares[i] += row[i] * row[i];
            }
        }
        else
        {
            for( size_t i = 0; i < nCols; i++ )
            {
                minimum[i]    = row[i];
                maximum[i]    = row[i];
                sum[i]        = row[i];
                sumSquares[i] = row[i] * row[i];
            }
        }

        nt->releaseBlockOfRows( block );
        ntMin->releaseBlockOfRows( blockMin );
        ntMax->releaseBlockOfRows( blockMax );
        ntSum->releaseBlockOfRows( blockSum );
        ntSumSq->releaseBlockOfRows( blockSumSq );
    }

    void combineSingleStatistics ( NumericTable *ntSrc, NumericTable *ntDst, bool wasEmpty, NumericTable::BasicStatisticsId id) {
        if( ntSrc == NULL || ntDst == NULL ) { this->_errors->add(services::ErrorNullInputNumericTable); return; }

        NumericTablePtr ntSrcStat = ntSrc->basicStatistics.get(id);
        NumericTablePtr ntDstStat = ntDst->basicStatistics.get(id);

        BlockDescriptor<_summaryStatisticsType> blockSrc;
        BlockDescriptor<_summaryStatisticsType> blockDst;

        ntSrcStat->getBlockOfRows(0, 1, readOnly, blockSrc);
        ntDstStat->getBlockOfRows(0, 1, readWrite, blockDst);

        _summaryStatisticsType *src = blockSrc.getBlockPtr();
        _summaryStatisticsType *dst = blockDst.getBlockPtr();

        if( src == NULL || dst == NULL )
        {
            this->_errors->add(services::ErrorIncorrectInputNumericTable);
            return;
        }

        size_t nColsSrc = ntSrc->getNumberOfColumns();
        size_t nCols    = ntDst->getNumberOfColumns();

        if (nCols != nColsSrc) {
            this->_errors->add(services::ErrorIncorrectInputNumericTable);
            return;
        }

        if( wasEmpty )
        {
            for( size_t i = 0; i < nCols; i++ )
            {
                dst[i] = src[i];
            }
        }
        else
        {
            if (id == NumericTable::minimum)
            {
                for( size_t i = 0; i < nCols; i++ )
                {
                    if (dst[i] > src[i])
                    {
                        dst[i] = src[i];
                    }
                }
            }
            else
            if (id == NumericTable::maximum)
            {
                for( size_t i = 0; i < nCols; i++ )
                {
                    if (dst[i] < src[i])
                    {
                        dst[i] = src[i];
                    }
                }
            }
            else
            if (id == NumericTable::sum)
            {
                for( size_t i = 0; i < nCols; i++ )
                {
                    dst[i] += src[i];
                }
            }
            else
            if (id == NumericTable::sumSquares)
            {
                for( size_t i = 0; i < nCols; i++ )
                {
                    dst[i] += src[i];
                }
            }
        }

        ntSrcStat->releaseBlockOfRows( blockSrc );
        ntDstStat->releaseBlockOfRows( blockDst );
    }

    void combineStatistics( NumericTable *ntSrc, NumericTable *ntDst, bool wasEmpty)
    {
        if( ntSrc == NULL || ntDst == NULL) { this->_errors->add(services::ErrorNullInputNumericTable); return; }

        combineSingleStatistics (ntSrc, ntDst, wasEmpty, NumericTable::minimum);
        combineSingleStatistics (ntSrc, ntDst, wasEmpty, NumericTable::maximum);
        combineSingleStatistics (ntSrc, ntDst, wasEmpty, NumericTable::sum);
        combineSingleStatistics (ntSrc, ntDst, wasEmpty, NumericTable::sumSquares);
    }
};
/** @} */
} // namespace interface1
using interface1::DataSourceIface;
using interface1::DataSource;
using interface1::DataSourceTemplate;

}
}
#endif
