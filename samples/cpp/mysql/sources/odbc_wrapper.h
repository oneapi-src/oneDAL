/* file: odbc_wrapper.h */
/*******************************************************************************
* Copyright 2017-2018 Intel Corporation.
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
*
* License:
* http://software.intel.com/en-us/articles/intel-sample-source-code-license-agr
* eement/
*******************************************************************************/

#ifndef __ODBC_WRAPPER_H__
#define __ODBC_WRAPPER_H__

#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>

#if defined(_WIN32) || defined(_WIN64)
#define NOMINMAX
#include <windows.h>
#endif

#include <sql.h>
#include <sqlext.h>
#include <sqltypes.h>

#include "daal.h"

namespace odbc_wrapper
{
namespace utils
{

std::vector<std::string> split(const std::string &input, char delim)
{
    std::stringstream ss(input);
    std::vector<std::string> parts;

    std::string item;
    while (std::getline(ss, item, delim))
    {
        parts.push_back(item);
    }

    if (input[input.size() - 1] == delim)
    {
        parts.push_back(std::string());
    }

    return parts;
}

} // namespace utils

class Error
{
public:
    static const int SQL_STATE_SIZE = 6;

    Error() :
        _nativeError(0),
        _state(SQL_STATE_SIZE, '\0'),
        _message(SQL_MAX_MESSAGE_LENGTH, '\0') { }

    explicit Error(SQLINTEGER nativeError,
                   const std::string &state,
                   const std::string &message) :
        _nativeError(nativeError),
        _state(state),
        _message(message) { }

    const std::string &state() const
    {
        return _state;
    }

    const std::string &message() const
    {
        return _message;
    }

    SQLINTEGER nativeError() const
    {
        return _nativeError;
    }

    void shrinkMessageSize(size_t actualSize)
    {
        _message.resize(actualSize);
    }

    SQLINTEGER &nativeErrorRef()
    {
        return _nativeError;
    }

private:
    SQLINTEGER _nativeError;
    std::string _state;
    std::string _message;
};

class Exception : public std::exception
{
public:
    virtual ~Exception() throw() { }

    void add(const Error &e)
    {
        _errors.push_back(e);
    }

    virtual const char *what() const throw()
    {
        return top().message().c_str();
    }

    const Error &top() const
    {
        return _errors.front();
    }

    const std::vector<Error> &errors() const
    {
        return _errors;
    }
private:
    std::vector<Error> _errors;
};

inline Exception formatException(SQLSMALLINT handleType, SQLHANDLE handle)
{
    Exception ex;

    SQLRETURN ret = SQL_SUCCESS;
    for (SQLRETURN i = 1; ret != SQL_NO_DATA; i++)
    {
        Error e;

        SQLSMALLINT messageActualSize;
        ret = SQLGetDiagRec(handleType, handle, i,
                            (SQLCHAR *)e.state().c_str(), &e.nativeErrorRef(),
                            (SQLCHAR *)e.message().c_str(), (SQLSMALLINT)e.message().size(), &messageActualSize);
        if (SQL_SUCCEEDED(ret))
        {
            e.shrinkMessageSize(messageActualSize);
            ex.add(e);
        }
    }

    return ex;
}

inline SQLRETURN call(SQLSMALLINT handleType, SQLHANDLE handle, SQLRETURN code)
{
    if (!SQL_SUCCEEDED(code))
    { throw formatException(handleType, handle); }
    return code;
}

class ColumnAttributes
{
public:
    ColumnAttributes() :
        _hstmt(NULL),
        _column(0),
        _sqlType(SQL_UNKNOWN_TYPE),
        _octetLength(0) { }

    explicit ColumnAttributes(SQLHSTMT hstmt, SQLUSMALLINT column) :
        _hstmt(hstmt),
        _column(column),
        _sqlType(SQL_UNKNOWN_TYPE),
        _octetLength(0)
    {
        if (!hstmt)
        { throw std::invalid_argument("hstmt can't be nullptr"); }
    }

    const std::string &name() const
    {
        if (!_name.size())
        {
            const size_t maxNameLength = 2048;

            std::string &nameMutable = const_cast<std::string &>(_name);
            nameMutable.resize(maxNameLength);

            SQLSMALLINT nameSizeActual;
            call
            (
                SQL_HANDLE_STMT, _hstmt,
                SQLColAttributes(_hstmt, _column, SQL_DESC_NAME,
                                 (SQLPOINTER)nameMutable.c_str(),
                                 (SQLSMALLINT)nameMutable.size(),
                                 &nameSizeActual, NULL)
            );

            nameMutable.resize(nameSizeActual);
        }
        return _name;
    }

    SQLLEN octetLength() const
    {
        return extractAttribute<SQLLEN>(!_octetLength, SQL_DESC_OCTET_LENGTH, _octetLength);
    }

    SQLLEN type() const
    {
        return extractAttribute<SQLLEN>(_sqlType == SQL_UNKNOWN_TYPE, SQL_DESC_TYPE, _sqlType);
    }

private:
    template<typename T>
    T extractAttribute(bool condition, SQLUSMALLINT attribute, const T &value) const
    {
        if (condition)
        {
            SQLLEN valueMutable;
            call
            (
                SQL_HANDLE_STMT, _hstmt,
                SQLColAttributes(_hstmt, _column, attribute,
                                 NULL, 0, NULL, &valueMutable)
            );
            const_cast<T &>(value) = (T)valueMutable;
        }
        return value;
    }

    SQLHSTMT _hstmt;
    SQLUSMALLINT _column;

    std::string _name;
    SQLLEN _sqlType;
    SQLLEN _octetLength;
};

class FetchBuffer
{
public:
    FetchBuffer() { }

    void allocate(SQLHSTMT hstmt, const std::vector<ColumnAttributes> &attributes)
    {
        allocateBuffers(attributes);
        bindBuffers(hstmt);
    }

    const char *offset(size_t columnIndex) const
    {
        return &_buffer[ _bufferOffsets[columnIndex] ];
    }

    SQLLEN size(size_t columnIndex) const
    {
        return _bufferOffsets[columnIndex + 1] -
               _bufferOffsets[columnIndex];
    }

    SQLLEN actualSize(size_t columnIndex) const
    {
        return _bufferActualSizes[columnIndex];
    }

private:
    FetchBuffer(const FetchBuffer &fetchBuffer);

    void allocateBuffers(const std::vector<ColumnAttributes> &attributes)
    {
        const size_t columnsNumber = attributes.size();
        _bufferOffsets.resize(columnsNumber + 1);
        _bufferActualSizes.resize(columnsNumber);

        _bufferOffsets[0] = 0;
        for (size_t i = 0; i < columnsNumber; i++)
        {
            _bufferActualSizes[i] = 0;
            _bufferOffsets[i + 1] = _bufferOffsets[i] + attributes[i].octetLength();
        }

        const size_t fetchBufferSize = _bufferOffsets[columnsNumber];
        _buffer.resize(fetchBufferSize);
    }

    void bindBuffers(SQLHSTMT hstmt)
    {
        for (size_t i = 0; i < _bufferActualSizes.size(); i++)
        {
            call
            (
                SQL_HANDLE_STMT, hstmt,
                SQLBindCol(hstmt, (SQLUSMALLINT)(i + 1), SQL_C_DEFAULT,
                           (SQLPOINTER)offset(i), size(i), &_bufferActualSizes[i])
            );
        }

    }

    char *offset(size_t columnIndex)
    {
        return &_buffer[ _bufferOffsets[columnIndex] ];
    }

    std::vector<char> _buffer;
    std::vector<SQLLEN> _bufferOffsets;
    std::vector<SQLLEN> _bufferActualSizes;
};

class Statement
{
private:
    class Impl
    {
    public:
        explicit Impl(SQLHDBC hdbc, const std::string &query) :
            _hstmt(NULL)
        {
            if (!hdbc)
            { throw std::invalid_argument("hdbc can't be nullptr"); }

            call( SQL_HANDLE_DBC, hdbc, SQLAllocHandle(SQL_HANDLE_STMT, hdbc, &_hstmt) );
            call( SQL_HANDLE_STMT, _hstmt, SQLExecDirect(_hstmt, (SQLCHAR *)query.c_str(), SQL_NTS) );

            createAttributes(_attributes);
            _fetchBuffer.allocate(_hstmt, _attributes);
        }

        ~Impl()
        {
            call( SQL_HANDLE_STMT, _hstmt, SQLFreeHandle(SQL_HANDLE_STMT, _hstmt) );
        }

        size_t columns() const
        {
            return _attributes.size();
        }

        const ColumnAttributes &attributes(size_t column) const
        {
            return _attributes[column];
        }

        const ColumnAttributes &attributes(const std::string &column) const
        {
            return attributes( toColumnIndex(column) );
        }

        template<typename T>
        T get(size_t column) const
        {
            if (sizeof(T) != getRawSize(column))
            {
                throw std::invalid_argument("Size of T is not equal to the size "
                                            "of data received from SQL table");
            }

            return *( (const T *)getRaw(column) );
        }

        template<typename T>
        T get(const std::string &column) const
        {
            return get<T>( toColumnIndex(column) );
        }

        bool fetch(bool throwExceptionIfNoData = false)
        {
            SQLRETURN ret = SQLFetchScroll(_hstmt, SQL_FETCH_NEXT, 0);
            if (SQL_SUCCEEDED(ret))
            {
                return true;
            }
            else
            {
                if (ret != SQL_NO_DATA || throwExceptionIfNoData)
                {
                    throw formatException( SQL_HANDLE_STMT, _hstmt );
                }
                return false;
            }

        }

    private:
        Impl(const Impl &);

        void createAttributes(std::vector<ColumnAttributes> &attributes)
        {
            const size_t columnsNumber = fetchColumnsNumber();
            attributes.resize(columnsNumber);

            for (size_t i = 0; i < columnsNumber; i++)
            {
                attributes[i] = ColumnAttributes(_hstmt, (SQLUSMALLINT)(i + 1));
                _attributesMap[ attributes[i].name() ] = i;
            }
        }

        size_t fetchColumnsNumber() const
        {
            SQLSMALLINT columnsNum;
            call( SQL_HANDLE_STMT, _hstmt, SQLNumResultCols(_hstmt, &columnsNum) );
            return (size_t)columnsNum;
        }

        size_t toColumnIndex(const std::string &column) const
        {
            typedef std::map<std::string, size_t> MapType;
            MapType &map = const_cast<MapType &>(_attributesMap);
            return map[column];
        }

        const char *getRaw(size_t column) const
        {
            return _fetchBuffer.offset(column);
        }

        size_t getRawSize(size_t column) const
        {
            return _fetchBuffer.actualSize(column);
        }

        SQLHSTMT _hstmt;
        FetchBuffer _fetchBuffer;

        std::vector<ColumnAttributes> _attributes;
        std::map<std::string, size_t> _attributesMap;
    };

public:
    explicit Statement(SQLHDBC hdbc, const std::string &query)
    {
        _impl.reset(new Impl(hdbc, query));
    }

    const ColumnAttributes &attributes(size_t column = 0) const
    {
        return _impl->attributes(column);
    }

    const ColumnAttributes &attributes(const std::string &column) const
    {
        return _impl->attributes(column);
    }

    template<typename T>
    T get(size_t column = 0) const
    {
        return _impl->get<T>(column);
    }

    template<typename T>
    T get(const std::string &column) const
    {
        return _impl->get<T>(column);
    }

    template<typename T>
    T first(size_t column = 0)
    {
        _impl->fetch(/* throwExceptionIfNoData = */ true);
        return _impl->get<T>(column);
    }

    bool fetch()
    {
        return _impl->fetch();
    }

private:
    daal::services::SharedPtr<Impl> _impl;
};

template<>
std::string Statement::Impl::get<std::string>(size_t column) const
{
    return std::string( getRaw(column), getRawSize(column) );
}

template<>
std::vector<char> Statement::Impl::get<std::vector<char> >(size_t column) const
{
    return std::vector<char>( getRaw(column), getRaw(column) + getRawSize(column) );
}

class Connection
{
private:
    class Impl
    {
    public:
        explicit Impl(const std::string &connectionString) :
            _henv(NULL),
            _hdbc(NULL)
        {
            call( SQL_HANDLE_ENV, _henv, SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &_henv) );
            call( SQL_HANDLE_ENV, _henv, SQLSetEnvAttr(_henv, SQL_ATTR_ODBC_VERSION, (SQLPOINTER) SQL_OV_ODBC3, SQL_IS_UINTEGER) );
            call( SQL_HANDLE_ENV, _henv, SQLAllocHandle(SQL_HANDLE_DBC, _henv, &_hdbc) );

            SQLSMALLINT outConnectionStringLength;
            call
            (
                SQL_HANDLE_DBC, _hdbc,
                SQLDriverConnect(_hdbc, SQL_NULL_HANDLE,
                                 (SQLCHAR *)connectionString.c_str(),
                                 (SQLSMALLINT)connectionString.size(),
                                 (SQLCHAR *)NULL,
                                 (SQLSMALLINT)0,
                                 &outConnectionStringLength,
                                 SQL_DRIVER_NOPROMPT)
            );
        }

        ~Impl()
        {
            call( SQL_HANDLE_DBC, _hdbc, SQLDisconnect(_hdbc) );
            call( SQL_HANDLE_DBC, _hdbc, SQLFreeHandle(SQL_HANDLE_DBC, _hdbc) );
            call( SQL_HANDLE_ENV, _henv, SQLFreeHandle(SQL_HANDLE_ENV, _henv) );
        }

        Statement execute(const std::string &query)
        {
            return Statement(_hdbc, query);
        }

    private:
        Impl(const Impl &);

        SQLHENV _henv;
        SQLHDBC _hdbc;
    };

public:
    explicit Connection(const std::string &connectionString)
    {
        _impl.reset(new Impl(connectionString));
    }

    Statement execute(const std::string &query)
    {
        return _impl->execute(query);
    }

    Statement execute(const std::string &query, const std::string &arg)
    {
        const std::vector<std::string> parts = utils::split(query, '?');
        if (parts.size() != 2)
        { throw std::invalid_argument("query must contain exactly one ? character"); }

        return execute(parts[0] + arg + parts[1]);
    }

    std::string id()
    {
        return execute("select connection_id()").first<std::string>();
    }

private:
    daal::services::SharedPtr<Impl> _impl;
};

} // namespace odbc_wrapper

#endif
