/* file: sql_feature_utils.h */
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

#ifndef __DATA_MANAGEMENT_DATA_SOURCE_INTERNAL_SQL_FEATURE_UTILS_H__
#define __DATA_MANAGEMENT_DATA_SOURCE_INTERNAL_SQL_FEATURE_UTILS_H__

#include <string>
#include "services/collection.h"
#include "services/internal/buffer.h"
#include "services/internal/utilities.h"

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

#include <sql.h>
#include <sqlext.h>
#include <sqltypes.h>

namespace daal
{
namespace data_management
{
namespace internal
{

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__INTERNAL__SQLFEATUREINFO"></a>
 * \brief Class that holds auxiliary information about single SQL column
 */
class SQLFeatureInfo : public Base
{
public:
    SQLSMALLINT sqlType;
    SQLLEN sqlOctetLength;
    services::String columnName;
    bool isSigned;

    SQLFeatureInfo() :
        sqlType(SQL_UNKNOWN_TYPE),
        sqlOctetLength(0),
        isSigned(false) { }

    explicit SQLFeatureInfo(const services::String &columnName,
                            SQLSMALLINT sqlType,
                            SQLLEN sqlOctetLength,
                            bool isSigned) :
        columnName(columnName),
        sqlType(sqlType),
        sqlOctetLength(sqlOctetLength),
        isSigned(isSigned) { }
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__INTERNAL__SQLFEATURESINFO"></a>
 * \brief Class that holds auxiliary information about multiple SQL columns
 */
class SQLFeaturesInfo : public Base
{
public:
    services::Status add(const SQLFeatureInfo &featureInfo)
    {
        _featuresInfo.safe_push_back(featureInfo);
        return services::Status();
    }

    const SQLFeatureInfo &get(size_t index) const
    {
        DAAL_ASSERT( index < _featuresInfo.size() );
        return _featuresInfo[index];
    }

    const SQLFeatureInfo &operator[](size_t index) const
    {
        return get(index);
    }

    size_t getNumberOfFeatures() const
    {
        return _featuresInfo.size();
    }
private:
    services::Collection<SQLFeatureInfo> _featuresInfo;
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__INTERNAL__SQLFETCHBUFFERFRAGMENT"></a>
 * \brief Represents fragment of SQL fetch buffer
 */
class SQLFetchBufferFragment : public Base
{
public:
    SQLFetchBufferFragment() :
        _rawFetchBuffer(NULL),
        _bufferSize(0),
        _actualDataSize(NULL) { }

    explicit SQLFetchBufferFragment(char *rawFetchBuffer,
                                    SQLLEN bufferSize,
                                    SQLLEN *actualDataSize) :
        _rawFetchBuffer(rawFetchBuffer),
        _bufferSize(bufferSize),
        _actualDataSize(actualDataSize) { }

    char *getBuffer() const
    {
        return _rawFetchBuffer;
    }

    SQLLEN getBufferSize() const
    {
        return _bufferSize;
    }

    SQLLEN getActualDataSize() const
    {
        return *_actualDataSize;
    }

    services::BufferView<char> view() const
    {
        return services::BufferView<char>(_rawFetchBuffer, *_actualDataSize);
    }

private:
    char *_rawFetchBuffer;
    SQLLEN _bufferSize;
    SQLLEN *_actualDataSize;
};

template<typename FloatingPointType>
inline SQLSMALLINT getSQLTypeForFloatingType();

template<>
inline SQLSMALLINT getSQLTypeForFloatingType<double>()
{ return SQL_C_DOUBLE; }

template<>
inline SQLSMALLINT getSQLTypeForFloatingType<float>()
{ return SQL_C_FLOAT; }

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__INTERNAL__SQLFETCHMODE"></a>
 * \brief Mode of fetching data from SQL table
 */
class SQLFetchMode
{
public:
    enum Value
    {
        useNativeSQLTypes,
        castToFloatingPointType
    };

    static SQLSMALLINT getTargetType(Value fetchMode)
    {
        switch (fetchMode)
        {
            case useNativeSQLTypes:
                return SQL_C_DEFAULT;

            case castToFloatingPointType:
                return getSQLTypeForFloatingType<DAAL_DATA_TYPE>();
        }
        return SQL_C_DEFAULT;
    }

private:
    SQLFetchMode();
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__INTERNAL__SQLFETCHBUFFER"></a>
 * \brief Class hold buffer for fetching data from SQL table,
 *        simplifies binding of SQL table columns
 */
class SQLFetchBuffer : public Base
{
public:
    static services::SharedPtr<SQLFetchBuffer> create(const SQLFeaturesInfo &featuresInfo,
                                                      const SQLFetchMode::Value &mode,
                                                      services::Status *status = NULL)
    {
        return services::internal::wrapSharedAndTryThrow(
            new SQLFetchBuffer(featuresInfo, mode, status), status);
    }

    size_t getNumberOfFeatures() const
    {
        DAAL_ASSERT( _bufferOffsets.size() > 0 );
        return _bufferOffsets.size() - 1;
    }

    char *getBufferForFeature(size_t featureIndex) const
    {
        DAAL_ASSERT( _bufferOffsets.size() > 0 );
        DAAL_ASSERT( featureIndex + 1 < _bufferOffsets.size() );
        return _buffer.offset( _bufferOffsets[featureIndex] );
    }

    SQLLEN *getActualDataSizeBufferForFeature(size_t featureIndex) const
    {
        DAAL_ASSERT( featureIndex < _actualDataSizes.size() );
        return _actualDataSizes.offset(featureIndex);
    }

    SQLLEN getBufferSizeForFeature(size_t featureIndex) const
    {
        DAAL_ASSERT( _bufferOffsets.size() > 0 );
        DAAL_ASSERT( featureIndex + 1 < _bufferOffsets.size() );
        const size_t begin = _bufferOffsets[featureIndex];
        const size_t end   = _bufferOffsets[featureIndex + 1];
        return (SQLLEN)(end - begin);
    }

    SQLLEN getActualDataSizeForFeature(size_t featureIndex) const
    {
        DAAL_ASSERT( featureIndex < _actualDataSizes.size() );
        return _actualDataSizes[featureIndex];
    }

    SQLFetchBufferFragment getFragment(size_t featureIndex) const
    {
        return SQLFetchBufferFragment( getBufferForFeature(featureIndex),
                                       getBufferSizeForFeature(featureIndex),
                                       getActualDataSizeBufferForFeature(featureIndex) );
    }

    void copyTo(const services::BufferView<DAAL_DATA_TYPE> &buffer) const
    {
        DAAL_ASSERT( _mode == SQLFetchMode::castToFloatingPointType );

        char *rawFetchBuffer = _buffer.data();
        DAAL_DATA_TYPE *targetBuffer = buffer.data();

        const size_t elementsToCopy = services::internal::minValue( buffer.size(),
                                                                    getNumberOfFeatures() );
        for (size_t i = 0; i < elementsToCopy; i++)
        {
            if (_actualDataSizes[i] == SQL_NULL_DATA)
            {
                targetBuffer[i] = DAAL_DATA_TYPE(0.0);
            }
            else
            {
                targetBuffer[i] = *( (DAAL_DATA_TYPE *)rawFetchBuffer );
            }

            rawFetchBuffer += sizeof(DAAL_DATA_TYPE);
        }
    }

private:
    SQLFetchBuffer(const SQLFetchBuffer &);
    SQLFetchBuffer &operator=(const SQLFetchBuffer &);

    explicit SQLFetchBuffer(const SQLFeaturesInfo &featuresInfo,
                            const SQLFetchMode::Value &mode,
                            services::Status *status = NULL) : _mode(mode)
    {
        services::internal::tryAssignStatusAndThrow(status, prepare(featuresInfo, mode));
    }

    services::Status prepare(const SQLFeaturesInfo &featuresInfo,
                             const SQLFetchMode::Value &mode)
    {
        services::Status status;

        const size_t numberOfFeatures = featuresInfo.getNumberOfFeatures();
        DAAL_CHECK_STATUS( status, _bufferOffsets.reallocate(numberOfFeatures + 1) );
        DAAL_CHECK_STATUS( status, _actualDataSizes.reallocate(numberOfFeatures) );

        _bufferOffsets[0] = 0;
        for (size_t i = 0; i < numberOfFeatures; i++)
        {
            const size_t bufferStride = (mode == SQLFetchMode::useNativeSQLTypes)
                                        ? featuresInfo[i].sqlOctetLength
                                        : sizeof(DAAL_DATA_TYPE);

            _bufferOffsets[i + 1] = _bufferOffsets[i] + bufferStride;
            _actualDataSizes[i] = 0;
        }

        const size_t bufferSize = _bufferOffsets[numberOfFeatures];
        DAAL_CHECK_STATUS( status, _buffer.reallocate(bufferSize) );

        return status;
    }

private:
    const SQLFetchMode::Value _mode;
    services::internal::Buffer<char> _buffer;
    services::internal::Buffer<SQLLEN> _bufferOffsets;
    services::internal::Buffer<SQLLEN> _actualDataSizes;
};
typedef services::SharedPtr<SQLFetchBuffer> SQLFetchBufferPtr;

} // namespace internal
} // namespace data_management
} // namespace daal

#endif
