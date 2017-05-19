/* file: string_data_source.h */
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
//  Implementation of the string data source class.
//--
*/

#ifndef __STRING_DATA_SOURCE_H__
#define __STRING_DATA_SOURCE_H__

#include "services/daal_memory.h"
#include "data_management/data_source/data_source.h"
#include "data_management/data/data_dictionary.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"

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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__STRINGDATASOURCE"></a>
 *  \brief Specifies methods to access data stored in byte arrays in the C-string format
 *  \tparam _featureManager     FeatureManager used to get numeric data from file strings
 */
template< typename _featureManager, typename _summaryStatisticsType = DAAL_SUMMARY_STATISTICS_TYPE >
class StringDataSource : public DataSourceTemplate<data_management::HomogenNumericTable<DAAL_DATA_TYPE>, _summaryStatisticsType>
{
public:
    using DataSourceIface::NumericTableAllocationFlag;
    using DataSourceIface::DictionaryCreationFlag;
    using DataSourceIface::DataSourceStatus;

    using DataSource::checkDictionary;
    using DataSource::checkNumericTable;
    using DataSource::freeNumericTable;
    using DataSource::_dict;
    using DataSource::_initialMaxRows;

    /**
     *  Typedef that stores the Feature Manager data type
     */
    typedef _featureManager FeatureManager;

protected:
    typedef data_management::HomogenNumericTable<DAAL_DATA_TYPE> DefaultNumericTableType;

    FeatureManager featureManager;

public:
    /**
     *  Main constructor for a Data Source
     *  \param[in]  data                            Byte array in the C-string format
     *  \param[in]  doAllocateNumericTable          Flag that specifies whether a Numeric Table
     *                                              associated with a File Data Source is allocated inside the Data Source
     *  \param[in]  doCreateDictionaryFromContext   Flag that specifies whether a Data Dictionary
     *                                              is created from the context of the File Data Source
     *  \param[in]  initialMaxRows                  Initial value of maximum number of rows in Numeric Table allocated in loadDataBlock() method
     */
    StringDataSource( const byte *data,
                      DataSourceIface::NumericTableAllocationFlag doAllocateNumericTable    = DataSource::notAllocateNumericTable,
                      DataSourceIface::DictionaryCreationFlag doCreateDictionaryFromContext = DataSource::notDictionaryFromContext,
                      size_t initialMaxRows = 10):
        DataSourceTemplate<DefaultNumericTableType, _summaryStatisticsType>(doAllocateNumericTable, doCreateDictionaryFromContext)
    {
        _rawLineBufferLen = 1024;
        _rawLineBuffer    = (char *)daal::services::daal_malloc( _rawLineBufferLen );
        if( _rawLineBuffer == 0 )
            this->_status.add(services::throwIfPossible(services::Status(services::ErrorMemoryAllocationFailed)));

        setData( data );

        _contextDictFlag = false;

        _initialMaxRows = initialMaxRows;
    }

    ~StringDataSource()
    {
        daal::services::daal_free( _rawLineBuffer );
        DataSourceTemplate<DefaultNumericTableType, _summaryStatisticsType>::freeNumericTable();
        if( _contextDictFlag )
        {
            delete _dict;
        }
    }

    /**
     *  Sets a new string as a source for data
     *  \param[in]  data  Byte array in the C-string format
     */
    void setData( const byte *data )
    {
        _stringBufferPos = 0;
        _stringBuffer    = (char *)data;
    }

    /**
     *  Gets data source string data
     *  \return  Byte array in the C-string format
     */
    const byte *getData()
    {
        return (const byte *)(_stringBuffer);
    }

    /**
     *  Resets a data source string
     */
    void resetData()
    {
        _stringBufferPos = 0;
    }

    /**
     *  Returns a Feature Manager associated with a String Data Source
     *  \return Feature Manager associated with the String Data Source
     */
    FeatureManager &getFeatureManager()
    {
        return featureManager;
    }

public:
    size_t loadDataBlock(size_t maxRows) DAAL_C11_OVERRIDE
    {
        services::Status s = checkDictionary();
        s.add(checkNumericTable());
        if(!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }
        return loadDataBlock(maxRows, this->DataSource::_spnt.get());
    }

    size_t loadDataBlock(size_t maxRows, NumericTable *nt) DAAL_C11_OVERRIDE
    {
        services::Status s = checkDictionary();
        if(!nt)
            s.add(services::Status(services::ErrorNullInputNumericTable));
        if(!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }


        s = DataSourceTemplate<DefaultNumericTableType, _summaryStatisticsType>::resizeNumericTableImpl( maxRows, nt );
        if(!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }

        nt->setNormalizationFlag(NumericTable::nonNormalized);

        size_t j;
        for(j = 0; j < maxRows && !iseof(); j++)
        {
            s = readLine();
            if(!s || !_rawLineLength)
                break;
            featureManager.parseRowIn(_rawLineBuffer, _rawLineLength, _dict, nt, j);

            DataSourceTemplate<DefaultNumericTableType, _summaryStatisticsType>::updateStatistics( j, nt );
        }

        nt->resize( j );
        if(!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }

        NumericTableDictionaryPtr ntDict = nt->getDictionarySharedPtr();
        size_t nFeatures = _dict->getNumberOfFeatures();
        ntDict->setNumberOfFeatures(nFeatures);
        for (size_t i = 0; i < nFeatures; i++)
        {
            ntDict->setFeature((*_dict)[i].ntFeature, i);
        }

        return j;
    }

    size_t loadDataBlock(size_t maxRows, size_t rowOffset, size_t fullRows) DAAL_C11_OVERRIDE
    {
        services::Status s = checkDictionary();
        if(s)
            s.add(checkNumericTable());
        if(!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }
        return loadDataBlock(maxRows, rowOffset, fullRows, this->DataSource::_spnt.get());
    }

    size_t loadDataBlock(size_t maxRows, size_t rowOffset, size_t fullRows, NumericTable *nt) DAAL_C11_OVERRIDE
    {
        services::Status s = checkDictionary();
        if(!nt)
            s.add(services::Status(services::ErrorNullInputNumericTable));
        if(!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }

        s = DataSourceTemplate<DefaultNumericTableType, _summaryStatisticsType>::resizeNumericTableImpl(fullRows, nt);
        if(!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }

        nt->setNormalizationFlag(NumericTable::nonNormalized);

        size_t j;

        for( j = 0; j < maxRows && !iseof() ; j++ )
        {
            s = readLine();
            if(!s || !_rawLineLength)
                break;
            featureManager.parseRowIn(_rawLineBuffer, _rawLineLength, _dict, nt, rowOffset + j);

            DataSourceTemplate<DefaultNumericTableType, _summaryStatisticsType>::updateStatistics( rowOffset + j, nt );
        }

        NumericTableDictionaryPtr ntDict = nt->getDictionarySharedPtr();
        size_t nFeatures = _dict->getNumberOfFeatures();
        ntDict->setNumberOfFeatures(nFeatures);
        for (size_t i = 0; i < nFeatures; i++)
        {
            ntDict->setFeature((*_dict)[i].ntFeature, i);
        }

        return fullRows;
    }

    size_t loadDataBlock() DAAL_C11_OVERRIDE
    {
        services::Status s = checkDictionary();
        s.add(checkNumericTable());
        if(!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }
        return loadDataBlock(this->DataSource::_spnt.get());
    }

    size_t loadDataBlock(NumericTable* nt) DAAL_C11_OVERRIDE
    {
        services::Status s = checkDictionary();
        if(!nt)
            s.add(services::Status(services::ErrorNullInputNumericTable));
        if(!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }

        size_t maxRows = (_initialMaxRows > 0 ? _initialMaxRows : 10);
        size_t nrows = 0;
        size_t ncols = _dict->getNumberOfFeatures();

        DataCollection tables;

        for(;; maxRows *= 2)
        {
            NumericTable *ntCurrent = new HomogenNumericTable<DAAL_DATA_TYPE>(ncols, maxRows, NumericTableIface::doAllocate);
            if (ntCurrent == NULL)
            {
                this->_status.add(services::throwIfPossible(services::Status(services::ErrorNumericTableNotAllocated)));
                break;
            }
            tables.push_back(NumericTablePtr(ntCurrent));
            size_t rows = loadDataBlock(maxRows, ntCurrent);
            nrows += rows;
            if (rows < maxRows)
                break;
        }

        s = DataSourceTemplate<DefaultNumericTableType, _summaryStatisticsType>::resizeNumericTableImpl( nrows, nt );
        if(!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }
        nt->setNormalizationFlag(NumericTable::nonNormalized);

        BlockDescriptor<DAAL_DATA_TYPE> blockCurrent, block;

        size_t pos = 0;

        for (size_t i = 0; i < tables.size(); i++)
        {
            NumericTable *ntCurrent = (NumericTable*)(tables[i].get());
            size_t rows = ntCurrent->getNumberOfRows();

            if(!rows)
                continue;

            ntCurrent->getBlockOfRows(0, rows, readOnly, blockCurrent);
            nt->getBlockOfRows(pos, rows, writeOnly, block);

            services::daal_memcpy_s(block.getBlockPtr(), rows * ncols * sizeof(DAAL_DATA_TYPE), blockCurrent.getBlockPtr(), rows * ncols * sizeof(DAAL_DATA_TYPE));

            ntCurrent->releaseBlockOfRows(blockCurrent);
            nt->releaseBlockOfRows(block);

            DataSourceTemplate<DefaultNumericTableType, _summaryStatisticsType>::combineStatistics( ntCurrent, nt, pos == 0);
            pos += rows;
        }

        NumericTableDictionaryPtr ntDict = nt->getDictionarySharedPtr();
        size_t nFeatures = _dict->getNumberOfFeatures();
        ntDict->setNumberOfFeatures(nFeatures);
        for (size_t i = 0; i < nFeatures; i++)
        {
            ntDict->setFeature((*_dict)[i].ntFeature, i);
        }

        return nrows;
    }

    services::Status createDictionaryFromContext() DAAL_C11_OVERRIDE
    {
        if(_dict)
        return services::throwIfPossible(services::Status(services::ErrorDictionaryAlreadyAvailable));

        _contextDictFlag = true;
        _dict = new DataSourceDictionary();

        services::Status s = readLine();
        if(!s)
        {
            delete _dict;
            _dict = NULL;
            return services::throwIfPossible(s);
        }

        featureManager.parseRowAsDictionary( _rawLineBuffer, _rawLineLength, _dict );

        _stringBufferPos = 0;
        return services::Status();
    }

    DataSourceIface::DataSourceStatus getStatus() DAAL_C11_OVERRIDE
    {
        return (iseof() ? DataSourceIface::endOfData : DataSourceIface::readyForLoad);
    }

    size_t getNumberOfAvailableRows() DAAL_C11_OVERRIDE
    {
        return 0;
    }

protected:
    bool enlargeBuffer()
    {
        int newRawLineBufferLen = _rawLineBufferLen * 2;
        char* newRawLineBuffer = (char *)daal::services::daal_malloc( newRawLineBufferLen );
        if( newRawLineBuffer == 0 )
            return false;
        daal::services::daal_memcpy_s(newRawLineBuffer, newRawLineBufferLen, _rawLineBuffer, _rawLineBufferLen);
        daal::services::daal_free( _rawLineBuffer );
        _rawLineBuffer = newRawLineBuffer;
        _rawLineBufferLen = newRawLineBufferLen;
        return true;
    }

    int readLine(char *buffer, int count)
    {
        int pos = 0;
        while (pos + 1 < count)
        {
            if (_stringBuffer[_stringBufferPos] != '\0')
            {
                buffer[pos] = _stringBuffer[_stringBufferPos];
                pos++;
                _stringBufferPos++;
                if (buffer[pos - 1] == '\n') break;
            }
            else
            {
                break;
            }
        }
        buffer[pos] = '\0';
        return pos;
    }

    inline bool iseof() const
    {
        return (_stringBuffer[_stringBufferPos] == '\0');
    }

    services::Status readLine()
    {
        _rawLineLength = 0;
        while(!iseof())
        {
            const int readLen = readLine (_rawLineBuffer + _rawLineLength, (int)(_rawLineBufferLen - _rawLineLength));
            if (readLen <= 0)
            {
                _rawLineLength = 0;
                return services::Status();
            }
            _rawLineLength += readLen;
            if (_rawLineBuffer[_rawLineLength - 1] == '\n' || _rawLineBuffer[_rawLineLength - 1] == '\r')
            {
                while (_rawLineLength > 0 && (_rawLineBuffer[_rawLineLength - 1] == '\n' || _rawLineBuffer[_rawLineLength - 1] == '\r'))
                {
                    _rawLineLength--;
                }
                _rawLineBuffer[_rawLineLength] = '\0';
                return services::Status();
            }
            if(!enlargeBuffer())
                return services::Status(services::ErrorMemoryAllocationFailed);
        }
        return services::Status();
    }

private:

    char  *_rawLineBuffer;
    int    _rawLineBufferLen;
    size_t _rawLineLength;

    char  *_stringBuffer;
    size_t _stringBufferPos;

    bool  _contextDictFlag;
};
/** @} */
} // namespace interface1
using interface1::StringDataSource;

}
}
#endif
