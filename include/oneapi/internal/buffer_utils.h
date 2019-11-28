/* file: buffer_utils.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#ifndef __DAAL_ONEAPI_INTERNAL_BUFFER_UTILS_H__
#define __DAAL_ONEAPI_INTERNAL_BUFFER_UTILS_H__

#include "oneapi/internal/utils.h"
#include "oneapi/internal/types_utils.h"
#include "data_management/data/internal/conversion.h"

namespace daal
{
namespace oneapi
{
namespace internal
{
namespace interface1
{
/** @ingroup oneapi_internal
 * @{
 */

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__BUFFERCONVERTERFROM"></a>
 *  \brief Converts UniversalBuffer from compile-time known type to
 *  runtime-known type
 */
template <typename DataType>
class BufferConverterFrom
{
public:
    BufferConverterFrom(const UniversalBuffer & src, UniversalBuffer & dest, size_t offset, size_t size)
        : _src(src), _dest(dest), _offset(offset), _size(size)
    {}

    UniversalBuffer getResult(services::Status & st)
    {
        st = _st;
        return _dest;
    }

    template <typename T>
    void operator()(Typelist<T>)
    {
        using namespace daal::data_management;
        using namespace daal::data_management::internal;

        _st = services::Status();

        auto srcBuffer  = _src.template get<DataType>();
        auto srcHostPtr = srcBuffer.toHost(readOnly);

        auto destBuffer    = _dest.template get<T>();
        auto destSubBuffer = destBuffer.getSubBuffer(_offset, _size);
        auto destHostPtr   = destSubBuffer.toHost(readWrite);

        VectorDownCast<DataType, T>()(_size, srcHostPtr.get(), destHostPtr.get());
    }

private:
    services::Status _st;

    UniversalBuffer _src;
    UniversalBuffer _dest;
    size_t _offset;
    size_t _size;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__BUFFERCONVERTERTO"></a>
 *  \brief Converts UniversalBuffer to compile-time known type from
 *  runtime-known type
 */
template <typename DataType>
class BufferConverterTo
{
public:
    BufferConverterTo(const UniversalBuffer & src, size_t offset, size_t size) : _src(src), _offset(offset), _size(size) {}

    services::Buffer<DataType> getResult(services::Status & st)
    {
        st = _st;
        return _dest;
    }

    template <typename T>
    void operator()(Typelist<T>)
    {
        using namespace daal::data_management;
        using namespace daal::data_management::internal;

        _st = services::Status();

        auto buffer      = _src.template get<T>();
        auto subbuffer   = buffer.getSubBuffer(_offset, _size);
        auto memoryBlock = subbuffer.toHost(readOnly);

        auto & context      = getDefaultContext();
        auto uniBufferBlock = context.allocate(TypeIds::id<DataType>(), _size, &_st);

        if (!_st)
        {
            return;
        }

        auto bufferBlock = uniBufferBlock.template get<DataType>();
        {
            auto bufferHostPtr = bufferBlock.toHost(readWrite);
            VectorUpCast<T, DataType>()(_size, memoryBlock.get(), bufferHostPtr.get());
        }
        _dest = bufferBlock;
    }

    void operator()(Typelist<DataType>)
    {
        _st = services::Status();

        auto buffer    = _src.template get<DataType>();
        auto subbuffer = buffer.getSubBuffer(_offset, _size);

        _dest = subbuffer;
    }

private:
    services::Status _st;

    UniversalBuffer _src;
    size_t _offset;
    size_t _size;

    services::Buffer<DataType> _dest;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__BUFFERHOSTREINTERPRETER"></a>
 *  \brief Reinterprets UniversalBuffer to host array of compile-time known type
 */
template <typename DataType>
class BufferHostReinterpreter
{
public:
    BufferHostReinterpreter(const UniversalBuffer & src, const data_management::ReadWriteMode & mode, size_t size)
        : _src(src), _mode(mode), _size(size)
    {}

    services::SharedPtr<DataType> getResult(services::Status & st)
    {
        st = _st;
        return _reinterpretedPtr;
    }

    template <typename T>
    void operator()(Typelist<T>)
    {
        auto buffer = _src.template get<T>();
        auto ptr    = buffer.toHost(_mode);

        _reinterpretedPtr = services::reinterpretPointerCast<DataType, T>(ptr);
    }

private:
    services::Status _st;

    UniversalBuffer _src;
    data_management::ReadWriteMode _mode;
    size_t _size;
    services::SharedPtr<DataType> _reinterpretedPtr;
};

/** @} */
} // namespace interface1

using interface1::BufferConverterFrom;
using interface1::BufferConverterTo;
using interface1::BufferHostReinterpreter;

} // namespace internal
} // namespace oneapi
} // namespace daal

#endif
