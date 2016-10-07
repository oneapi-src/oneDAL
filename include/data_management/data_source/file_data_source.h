/* file: file_data_source.h */
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
//  Implementation of the file data source class.
//--
*/

#ifndef __FILE_DATA_SOURCE_H__
#define __FILE_DATA_SOURCE_H__

#include <cstdio>
#include <cstring>
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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__FILEDATASOURCE"></a>
 *  \brief Specifies methods to access data stored in files
 *  \tparam _featureManager     FeatureManager to use to get numeric data from file strings
 */
template< typename _featureManager, typename _summaryStatisticsType = double >
class FileDataSource : public DataSourceTemplate<data_management::HomogenNumericTable<double>, _summaryStatisticsType>
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
     *  Typedef that stores the parser datatype
     */
    typedef _featureManager FeatureManager;

protected:
    typedef data_management::HomogenNumericTable<double> DefaultNumericTableType;

    FeatureManager featureManager;

public:
    /**
     *  Main constructor for a Data Source
     *  \param[in]  fileName                        Name of the file that stores data
     *  \param[in]  doAllocateNumericTable          Flag that specifies whether a Numeric Table
     *                                              associated with a File Data Source is allocated inside the Data Source
     *  \param[in]  doCreateDictionaryFromContext   Flag that specifies whether a Data %Dictionary
     *                                              is created from the context of the File Data Source
     *  \param[in]  initialMaxRows                  Initial value of maximum number of rows in Numeric Table allocated in loadDataBlock() method
     */
    FileDataSource( const std::string &fileName,
                    DataSourceIface::NumericTableAllocationFlag doAllocateNumericTable    = DataSource::notAllocateNumericTable,
                    DataSourceIface::DictionaryCreationFlag doCreateDictionaryFromContext = DataSource::notDictionaryFromContext,
                    size_t initialMaxRows = 10):
        DataSourceTemplate<DefaultNumericTableType, _summaryStatisticsType>(doAllocateNumericTable, doCreateDictionaryFromContext)
    {
        _fileName = fileName;

        _rawLineBufferLen = 1024;
        _rawLineBuffer    = (char *)daal::services::daal_malloc( _rawLineBufferLen );

        _fileBufferLen = 1048576;
        _fileBufferPos = _fileBufferLen;
        _fileBuffer    = (char *)daal::services::daal_malloc( _fileBufferLen );

    #if (defined(_MSC_VER)&&(_MSC_VER >= 1400))
        errno_t error;
        error = fopen_s( &_file, fileName.c_str(), "r" );
        if( error != 0)
        {
            this->_errors->add(services::ErrorOnFileOpen);
        }
    #else
        _file = fopen( fileName.c_str(), "r" );
    #endif

        if( !_file )
        {
            this->_errors->add(services::ErrorOnFileOpen);
        }

        _contextDictFlag      = false;

        _initialMaxRows = initialMaxRows;
    }

    ~FileDataSource()
    {
        if (_file)
        {
            fclose(_file);
        }
        daal::services::daal_free( _rawLineBuffer );
        daal::services::daal_free( _fileBuffer );
        DataSourceTemplate<DefaultNumericTableType, _summaryStatisticsType>::freeNumericTable();
        if( _contextDictFlag )
        {
            delete _dict;
        }
    }

    /**
     *  Returns a feature manager associated with a File Data Source
     *  \return Feature manager associated with the File Data Source
     */
    FeatureManager &getFeatureManager()
    {
        return featureManager;
    }

public:
    size_t loadDataBlock(size_t maxRows) DAAL_C11_OVERRIDE
    {
        checkDictionary();
        if( this->_errors->size() != 0 ) { return 0; }

        checkNumericTable();
        if( this->_errors->size() != 0 ) { return 0; }

        return loadDataBlock(maxRows, this->DataSource::_spnt.get());
    }

    size_t loadDataBlock(size_t maxRows, NumericTable* nt) DAAL_C11_OVERRIDE
    {
        checkDictionary();
        if( this->_errors->size() != 0 ) { return 0; }

        if( nt == NULL ) { this->_errors->add(services::ErrorNullInputNumericTable); return 0; }

        size_t j;

        DataSourceTemplate<DefaultNumericTableType, _summaryStatisticsType>::resizeNumericTableImpl( maxRows, nt );

        nt->setNormalizationFlag(NumericTable::nonNormalized);

        for( j = 0; j < maxRows; j++ )
        {
            readLine();
            if (_rawLineLength == 0) { break; }
            if(this->_errors->size() != 0) { break; }
            featureManager.parseRowIn( _rawLineBuffer, _rawLineLength, _dict, nt, j );

            DataSourceTemplate<DefaultNumericTableType, _summaryStatisticsType>::updateStatistics( j, nt );
        }

        nt->setNumberOfRows( j );

        NumericTableDictionary *ntDict = nt->getDictionary();
        size_t nFeatures = _dict->getNumberOfFeatures();
        ntDict->setNumberOfFeatures(nFeatures);
        for (size_t i = 0; i < nFeatures; i++)
        {
            ntDict->setFeature((*_dict)[i].ntFeature, i);
        }

        return j;
    }

    size_t loadDataBlock() DAAL_C11_OVERRIDE
    {
        checkDictionary();
        if( this->_errors->size() != 0 ) { return 0; }

        checkNumericTable();
        if( this->_errors->size() != 0 ) { return 0; }

        return loadDataBlock(this->DataSource::_spnt.get());
    }

    size_t loadDataBlock(NumericTable* nt) DAAL_C11_OVERRIDE
    {
        checkDictionary();
        if( this->_errors->size() != 0 ) { return 0; }

        if( nt == NULL ) { this->_errors->add(services::ErrorNullInputNumericTable); return 0; }

        size_t maxRows = (_initialMaxRows > 0 ? _initialMaxRows : 10);
        size_t nrows = 0;
        size_t ncols = _dict->getNumberOfFeatures();

        DataCollection tables;

        for( ; ; )
        {
            NumericTable *ntCurrent = new HomogenNumericTable<double>(ncols, maxRows, NumericTableIface::doAllocate);
            if (ntCurrent == NULL)
            {
                this->_errors->add(services::ErrorNumericTableNotAllocated);
                break;
            }
            tables.push_back(NumericTablePtr(ntCurrent));
            size_t rows = loadDataBlock(maxRows, ntCurrent);
            nrows += rows;
            if (rows < maxRows) { break; }
            maxRows *= 2;
        }

        DataSourceTemplate<DefaultNumericTableType, _summaryStatisticsType>::resizeNumericTableImpl( nrows, nt );
        nt->setNormalizationFlag(NumericTable::nonNormalized);

        BlockDescriptor<double> blockCurrent, block;

        size_t pos = 0;

        for (size_t i = 0; i < tables.size(); i++) {
            NumericTable *ntCurrent = (NumericTable*)(tables[i].get());
            size_t rows = ntCurrent->getNumberOfRows();

            if (rows == 0) { continue; }

            ntCurrent->getBlockOfRows(0, rows, readOnly, blockCurrent);
            nt->getBlockOfRows(pos, rows, writeOnly, block);

            services::daal_memcpy_s(block.getBlockPtr(), rows * ncols * sizeof(double), blockCurrent.getBlockPtr(), rows * ncols * sizeof(double));

            ntCurrent->releaseBlockOfRows(blockCurrent);
            nt->releaseBlockOfRows(block);

            DataSourceTemplate<DefaultNumericTableType, _summaryStatisticsType>::combineStatistics( ntCurrent, nt, pos == 0);
            pos += rows;
        }

        NumericTableDictionary *ntDict = nt->getDictionary();
        size_t nFeatures = _dict->getNumberOfFeatures();
        ntDict->setNumberOfFeatures(nFeatures);
        for (size_t i = 0; i < nFeatures; i++)
        {
            ntDict->setFeature((*_dict)[i].ntFeature, i);
        }

        return nrows;
    }

    void createDictionaryFromContext() DAAL_C11_OVERRIDE
    {
        if( _dict != NULL )
        {
            this->_errors->add(services::ErrorDictionaryAlreadyAvailable);
            return;
        }

        _contextDictFlag = true;
        _dict = new DataSourceDictionary();

        readLine();
        if( this->_errors->size() != 0 ) { return; }

        featureManager.parseRowAsDictionary( _rawLineBuffer, _rawLineLength, _dict );

        if( this->_errors->size() != 0 )
        {
            delete _dict;
            _dict = NULL;
        }

        fseek(_file, 0, SEEK_SET);

        _fileBufferPos = _fileBufferLen;
    }

    DataSourceIface::DataSourceStatus getStatus() DAAL_C11_OVERRIDE
    {
        if( iseof () )
        {
            return DataSourceIface::endOfData;
        }
        else
        {
            return DataSourceIface::readyForLoad;
        }
    }

    size_t getNumberOfAvailableRows() DAAL_C11_OVERRIDE
    {
        return 0;
    }

protected:
    void enlargeBuffer()
    {
        int newRawLineBufferLen = _rawLineBufferLen * 2;
        char* newRawLineBuffer = (char *)daal::services::daal_malloc( newRawLineBufferLen );
        if( newRawLineBuffer == 0 )
        {
            this->_errors->add(services::ErrorMemoryAllocationFailed);
            return;
        }
        daal::services::daal_memcpy_s(newRawLineBuffer, newRawLineBufferLen, _rawLineBuffer, _rawLineBufferLen);
        daal::services::daal_free( _rawLineBuffer );
        _rawLineBuffer = newRawLineBuffer;
        _rawLineBufferLen = newRawLineBufferLen;
    }

    inline bool iseof()
    {
        if ((_fileBufferPos == _fileBufferLen || _fileBuffer[_fileBufferPos] == '\0') && feof(_file))
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    int readLine(char *buffer, int count)
    {
        int pos = 0;
        while (pos + 1 < count)
        {
            if (_fileBufferPos < _fileBufferLen && _fileBuffer[_fileBufferPos] != '\0')
            {
                buffer[pos] = _fileBuffer[_fileBufferPos];
                pos++;
                _fileBufferPos++;
                if (buffer[pos - 1] == '\n') break;
            }
            else
            {
                if (iseof ()) break;
                _fileBufferPos = 0;
                int readLen;
                readLen = (int)fread(_fileBuffer, 1, _fileBufferLen, _file);
                if (readLen < _fileBufferLen)
                {
                    _fileBuffer[readLen] = '\0';
                }
                if (ferror(_file))
                {
                    this->_errors->add(services::ErrorOnFileRead);
                    break;
                }
            }
        }
        buffer[pos] = '\0';
        return pos;
    }

    void readLine()
    {
        _rawLineLength = 0;
        while (true) {
            if (iseof ()) { return; }
            int readLen = readLine (_rawLineBuffer + _rawLineLength, _rawLineBufferLen - _rawLineLength);
            if(this->_errors->size() != 0) { return; }
            if (readLen == -1) {
                _rawLineLength = 0;
                return;
            }
            _rawLineLength += readLen;
            if (_rawLineBuffer[_rawLineLength - 1] == '\n') {
                _rawLineBuffer[_rawLineLength - 1] = '\0';
                _rawLineLength--;
                return;
            }
            enlargeBuffer();
            if(this->_errors->size() != 0) { return; }
        }
    }

protected:
    std::string  _fileName;

    FILE *_file;

    char *_rawLineBuffer;
    int   _rawLineBufferLen;
    int   _rawLineLength;

    char *_fileBuffer;
    int   _fileBufferLen;
    int   _fileBufferPos;

    bool _contextDictFlag;
};
/** @} */
} // namespace interface1
using interface1::FileDataSource;

}
}
#endif
