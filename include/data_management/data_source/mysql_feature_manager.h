/* file: mysql_feature_manager.h */
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
//  Implementation of the MYSQL data source class.
//--
*/
#ifndef __MYSQL_FEATURE_MANAGER_H__
#define __MYSQL_FEATURE_MANAGER_H__

#include <sstream>
#include "services/daal_memory.h"
#include "data_management/data_source/data_source.h"
#include "data_management/data/data_dictionary.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <sql.h>
#include <sqltypes.h>
#include <sqlext.h>

using namespace std;

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
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MYSQLFEATUREMANAGER"></a>
 * \brief Contains MySQL-specific commands
 */
class MySQLFeatureManager
{
public:
    MySQLFeatureManager() : _errors(new services::ErrorCollection()) {}

    /**
     *  Executes an SQL statement from an ODBC statement handle and writes it to a Numeric Table
     *
     *  \param[in]   hdlStmt ODBC statement handle that contains an SQL query
     *  \param[out]  nt      Numeric Table to store query results
     *  \param[in]   maxRows Maximum number of rows that can be read
     */
    DataSourceIface::DataSourceStatus statementResultsNumericTable(SQLHSTMT hdlStmt, NumericTable *nt, size_t maxRows);

    /**
     *  Creates a data dictionary from an ODBC statement handle
     *
     *  \param[in]   hdlStmt ODBC statement handle that contains an SQL query
     *  \param[out]  dict    Dictionary to be created
     */
    void createDictionary(SQLHSTMT hdlStmt, DataSourceDictionary *dict)
    {
        SQLSMALLINT nFeatures = 0;
        SQLRETURN ret = SQLNumResultCols(hdlStmt, &nFeatures);

        dict->setNumberOfFeatures( nFeatures );

        SQLLEN sqlType;
        SQLLEN sqlIsUnsigned;
        SQLLEN sqlTypeLength;
        for (int i = 0 ; i < nFeatures; i++)
        {
            SQLSMALLINT bufferLenUsed;
            int bufferSize = 64;
            char label[64];
            ret = SQLColAttributes(hdlStmt, (SQLUSMALLINT)i + 1, SQL_DESC_UNSIGNED, NULL, 0, NULL, &sqlIsUnsigned);
            if (!SQL_SUCCEEDED(ret)) { _errors->add(services::ErrorODBC); return; }

            ret = SQLColAttributes(hdlStmt, (SQLUSMALLINT)i + 1, SQL_DESC_TYPE, NULL, 0, NULL, &sqlType);
            if (!SQL_SUCCEEDED(ret)) { _errors->add(services::ErrorODBC); return; }

            ret = SQLColAttributes(hdlStmt, (SQLUSMALLINT)i + 1, SQL_DESC_OCTET_LENGTH, NULL, 0, NULL, &sqlTypeLength);
            if (!SQL_SUCCEEDED(ret)) { _errors->add(services::ErrorODBC); return; }

            ret = SQLColAttributes(hdlStmt, (SQLUSMALLINT)i + 1, SQL_DESC_NAME , (SQLPOINTER)label, (SQLSMALLINT)bufferSize, &bufferLenUsed, NULL);
            if (!SQL_SUCCEEDED(ret)) { _errors->add(services::ErrorODBC); return; }

            sqlTypeLength *= 8;

            DataSourceFeature &feature = (*dict)[i];

            feature.setFeatureName(label);

            if (isToDouble(sqlType))
            {
                feature.setType<double>();
            }
            else if (isToFloat(sqlType))
            {
                feature.setType<float>();
            }
            else if (isToInt(sqlType))
            {
                if (sqlTypeLength <= 8)
                {
                    (sqlIsUnsigned == SQL_TRUE) ? feature.setType<unsigned char>() : feature.setType<char>();
                }
                else if (sqlTypeLength <= 16)
                {
                    (sqlIsUnsigned == SQL_TRUE) ? feature.setType<unsigned short>() : feature.setType<short>();
                }
                else if (sqlTypeLength <= 32)
                {
                    (sqlIsUnsigned == SQL_TRUE) ? feature.setType<unsigned int>() : feature.setType<int>();
                }
                else if (sqlTypeLength <= 64)
                {
                    (sqlIsUnsigned == SQL_TRUE) ? feature.setType<DAAL_UINT64>() : feature.setType<DAAL_INT64>();
                }
                else if (sqlType == SQL_BIGINT)
                {
                    (sqlIsUnsigned == SQL_TRUE) ? feature.setType<DAAL_UINT64>() : feature.setType<DAAL_INT64>();
                }
            }
        }
    }

    /**
     *  Limits the SELECT query.
     *
     *  \param[in]   query         Query to be limited
     *  \param[in]   idx_last_read Index of the first row to read
     *  \param[in]   maxRows       Maximum number of rows to read
     *
     *  \return Full limited query with the ';' symbol at the end
     */
    std::string setLimitQuery(std::string &query, size_t idx_last_read, size_t maxRows)
    {
        std::stringstream ss;
        ss << query << " LIMIT " << idx_last_read << ", " << maxRows << ";";
        return ss.str();
    }

    services::SharedPtr<services::ErrorCollection> getErrors()
    {
        return _errors;
    }

private:
    services::SharedPtr<services::ErrorCollection> _errors;

    size_t      getStrictureSize(NumericTableDictionary *dict);
    size_t      typeSize(data_feature_utils::IndexNumType indexNumType);
    SQLSMALLINT getTargetType(data_feature_utils::IndexNumType indexNumType);

    bool isToDouble(int identifier)
    {
        const int arraySize = 4;
        int SQLTypesToDouble[arraySize] = {SQL_NUMERIC, SQL_DECIMAL, SQL_DOUBLE, SQL_FLOAT};
        return isContain(identifier, SQLTypesToDouble, arraySize);
    }
    bool isToFloat(int identifier)
    {
        const int arraySize = 1;
        int SQLTypesToFloat[arraySize]  = {SQL_REAL};
        return isContain(identifier, SQLTypesToFloat, arraySize);
    }
    bool isToInt(int identifier)
    {
        const int arraySize = 6;
        int SQLTypesToInt[arraySize] = {SQL_INTEGER, SQL_SMALLINT, SQL_TINYINT, SQL_BIGINT, SQL_BIT, SQL_BINARY};
        return isContain(identifier, SQLTypesToInt, arraySize);
    }
    bool isContain(int identifier, int array[], int arraySize)
    {
        for (int i = 0; i < arraySize; i++)
        {
            if (array[i] == identifier)
            {
                return true;
            }
        }
        return false;
    }
};

DataSourceIface::DataSourceStatus MySQLFeatureManager::statementResultsNumericTable(SQLHSTMT hdlStmt, NumericTable *nt, size_t maxRows)
{
    SQLRETURN ret;
    size_t nFeatures = nt->getNumberOfColumns();
    nt->setNumberOfRows(maxRows);
    services::SharedPtr<NumericTableDictionary> dict = nt->getDictionarySharedPtr();
    data_feature_utils::IndexNumType indexNumType = data_feature_utils::getIndexNumType<double>();

    SQLLEN *bindInd     = (SQLLEN *)daal::services::daal_malloc(sizeof(SQLLEN) * nFeatures);
    double *fetchBuffer = (double *)daal::services::daal_malloc(sizeof(double) * nFeatures);
    for (int j = 0; j < nFeatures; j++)
    {
        if (indexNumType != data_feature_utils::DAAL_OTHER_T)
        {
            ret = SQLBindCol(hdlStmt, j + 1, getTargetType(indexNumType), (SQLPOINTER)&fetchBuffer[j], 0, &bindInd[j]);
            if (!SQL_SUCCEEDED(ret)) { _errors->add(services::ErrorODBC); return DataSource::notReady; }
        }
        else
        {
            bindInd[j] = 0;
        }
    }
    size_t read = 0;

    BlockDescriptor<double> block;
    nt->getBlockOfRows(0, maxRows, writeOnly, block);
    double *ntBuffer = block.getBlockPtr();

    while (SQL_SUCCEEDED(ret = SQLFetchScroll(hdlStmt, SQL_FETCH_NEXT, 1)))
    {
        for (int j = 0; j < nFeatures; j++)
        {
            if (bindInd[j] == SQL_NULL_DATA)
            {
                ntBuffer[read * nFeatures + j] = 0.0;
                continue;
            }
            indexNumType = (*dict)[j].indexType;
            if (indexNumType != data_feature_utils::DAAL_OTHER_T)
            {
                ntBuffer[read * nFeatures + j] = *((double *) & (fetchBuffer[j]));
            }
            else
            {
                ntBuffer[read * nFeatures + j] = 0.0;
            }
        }
        read++;
    }
    nt->setNumberOfRows(read);
    nt->releaseBlockOfRows(block);

    DataSourceIface::DataSourceStatus status = DataSourceIface::readyForLoad;
    if (ret != SQL_NO_DATA)
    {
        if (!SQL_SUCCEEDED(ret))
        {
            status = DataSourceIface::notReady;
            _errors->add(services::ErrorODBC);
        }
    }
    else
    {
        if (read < maxRows)
        {
            status = DataSourceIface::endOfData;
        }
    }
    daal::services::daal_free(fetchBuffer);
    daal::services::daal_free(bindInd);
    return status;
}

size_t MySQLFeatureManager::getStrictureSize(NumericTableDictionary *dict)
{
    size_t structureSize = 0;
    size_t nFeatures = dict->getNumberOfFeatures();
    for (int i = 0; i < nFeatures; i++)
    {
        data_feature_utils::IndexNumType indexNumType = (*dict)[i].indexType;
        structureSize += typeSize(indexNumType);
    }
    return structureSize;
}

size_t MySQLFeatureManager::typeSize(data_feature_utils::IndexNumType indexNumType)
{
    if      (indexNumType == data_feature_utils::DAAL_FLOAT32) { return 4; }
    else if (indexNumType == data_feature_utils::DAAL_FLOAT64) { return 8; }
    else if (indexNumType == data_feature_utils::DAAL_INT32_S) { return 4; }
    else if (indexNumType == data_feature_utils::DAAL_INT32_U) { return 4; }
    else if (indexNumType == data_feature_utils::DAAL_INT64_S) { return 8; }
    else if (indexNumType == data_feature_utils::DAAL_INT64_U) { return 8; }
    else if (indexNumType == data_feature_utils::DAAL_INT8_S)  { return 1; }
    else if (indexNumType == data_feature_utils::DAAL_INT8_U)  { return 1; }
    else if (indexNumType == data_feature_utils::DAAL_INT16_S) { return 2; }
    else if (indexNumType == data_feature_utils::DAAL_INT16_U) { return 2; }
    else /*indexNumType == data_feature_utils::DAAL_OTHER_T)*/ { return 4; }
}

SQLSMALLINT MySQLFeatureManager::getTargetType(data_feature_utils::IndexNumType indexNumType)
{
    if      (indexNumType == data_feature_utils::DAAL_FLOAT32) { return SQL_C_FLOAT; }
    else if (indexNumType == data_feature_utils::DAAL_FLOAT64) { return SQL_C_DOUBLE; }
    else if (indexNumType == data_feature_utils::DAAL_INT32_S) { return SQL_C_SLONG; }
    else if (indexNumType == data_feature_utils::DAAL_INT32_U) { return SQL_C_ULONG; }
    else if (indexNumType == data_feature_utils::DAAL_INT64_S) { return SQL_C_SBIGINT; }
    else if (indexNumType == data_feature_utils::DAAL_INT64_U) { return SQL_C_UBIGINT; }
    else if (indexNumType == data_feature_utils::DAAL_INT8_S)  { return SQL_C_STINYINT; }
    else if (indexNumType == data_feature_utils::DAAL_INT8_U)  { return SQL_C_UTINYINT; }
    else if (indexNumType == data_feature_utils::DAAL_INT16_S) { return SQL_C_SSHORT; }
    else if (indexNumType == data_feature_utils::DAAL_INT16_U) { return SQL_C_USHORT; }
    else /*indexNumType == data_feature_utils::DAAL_OTHER_T)*/ { return SQL_C_SLONG; }
}
/** @} */
} // namespace interface1
using interface1::MySQLFeatureManager;

}
}
#endif
