/* file: buffer_utils.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#ifndef __DAAL_SERVICES_INTERNAL_SYCL_BUFFER_UTILS_H__
#define __DAAL_SERVICES_INTERNAL_SYCL_BUFFER_UTILS_H__

#include "services/internal/execution_context.h"
#include "services/internal/sycl/types_utils.h"
#include "data_management/data/internal/conversion.h"

/// \cond INTERNAL
namespace daal
{
namespace services
{
namespace internal
{
namespace sycl
{
namespace interface1
{
template <typename DataType>
class BufferConverterFrom
{
public:
    BufferConverterFrom(const UniversalBuffer & src, UniversalBuffer & dest, size_t offset, size_t size)
        : _src(src), _dest(dest), _offset(offset), _size(size)
    {}

    UniversalBuffer getResult() { return _dest; }

    template <typename T>
    void operator()(Typelist<T>, Status & st)
    {
        using namespace daal::data_management;
        using namespace daal::data_management::internal;

        DAAL_ASSERT(!_src.empty());
        DAAL_ASSERT(!_dest.empty());
        DAAL_ASSERT_UNIVERSAL_BUFFER(_src, DataType, _size);
        DAAL_ASSERT_UNIVERSAL_BUFFER_TYPE(_dest, T);

        auto srcBuffer  = _src.template get<DataType>();
        auto srcHostPtr = srcBuffer.toHost(readOnly, st);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(st);

        auto destBuffer    = _dest.template get<T>();
        auto destSubBuffer = destBuffer.getSubBuffer(_offset, _size, st);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(st);

        auto destHostPtr = destSubBuffer.toHost(readWrite, st);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(st);

        VectorDownCast<DataType, T>()(_size, srcHostPtr.get(), destHostPtr.get());
    }

private:
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

    Buffer<DataType> getResult() { return _dest; }

    template <typename T>
    void operator()(Typelist<T>, Status & st)
    {
        using namespace daal::data_management;
        using namespace daal::data_management::internal;

        DAAL_ASSERT(!_src.empty());
        DAAL_ASSERT_UNIVERSAL_BUFFER_TYPE(_src, T);

        DAAL_ASSERT(_src.type() == TypeIds::id<T>());

        auto buffer = _src.template get<T>();

        auto subbuffer = buffer.getSubBuffer(_offset, _size, st);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(st);

        auto memoryBlock = subbuffer.toHost(readOnly, st);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(st);

        auto & context      = getDefaultContext();
        auto uniBufferBlock = context.allocate(TypeIds::id<DataType>(), _size, st);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(st);

        auto bufferBlock = uniBufferBlock.template get<DataType>();
        {
            auto bufferHostPtr = bufferBlock.toHost(readWrite, st);
            DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(st);
            VectorUpCast<T, DataType>()(_size, memoryBlock.get(), bufferHostPtr.get());
        }
        _dest = bufferBlock;
    }

    void operator()(Typelist<DataType>, Status & st)
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER_TYPE(_src, DataType);

        auto buffer    = _src.template get<DataType>();
        auto subbuffer = buffer.getSubBuffer(_offset, _size, st);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(st);
        _dest = subbuffer;
    }

private:
    UniversalBuffer _src;
    size_t _offset;
    size_t _size;

    Buffer<DataType> _dest;
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

    SharedPtr<DataType> getResult() { return _reinterpretedPtr; }

    template <typename T>
    void operator()(Typelist<T>, Status & st)
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(_src, T, _size);

        auto buffer = _src.template get<T>();
        auto ptr    = buffer.toHost(_mode, st);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(st);

        _reinterpretedPtr = reinterpretPointerCast<DataType, T>(ptr);
    }

private:
    UniversalBuffer _src;
    data_management::ReadWriteMode _mode;
    size_t _size;
    SharedPtr<DataType> _reinterpretedPtr;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__ALLOCATEBYNUMERICTABLEFEATURE"></a>
 *  \brief Allocate data by NumericTableFeature
 */
inline UniversalBuffer allocateByNumericTableFeature(const data_management::NumericTableFeature & feature, const size_t size,
                                                     services::Status & status)
{
    using namespace data_management;
    auto & context = services::internal::getDefaultContext();
    UniversalBuffer buffer;
    switch (feature.indexType)
    {
    case features::DAAL_INT8_U:
    {
        buffer = context.allocate(TypeId::uint8, size, status);
        break;
    }
    case features::DAAL_INT16_U:
    {
        buffer = context.allocate(TypeId::uint16, size, status);
        break;
    }
    case features::DAAL_INT32_U:
    {
        buffer = context.allocate(TypeId::uint32, size, status);
        break;
    }
    case features::DAAL_INT64_U:
    {
        buffer = context.allocate(TypeId::uint64, size, status);
        break;
    }

    case features::DAAL_INT8_S:
    {
        buffer = context.allocate(TypeId::int8, size, status);
        break;
    }
    case features::DAAL_INT16_S:
    {
        buffer = context.allocate(TypeId::int16, size, status);
        break;
    }
    case features::DAAL_INT32_S:
    {
        buffer = context.allocate(TypeId::int32, size, status);
        break;
    }
    case features::DAAL_INT64_S:
    {
        buffer = context.allocate(TypeId::int64, size, status);
        break;
    }

    case features::DAAL_FLOAT32:
    {
        buffer = context.allocate(TypeId::float32, size, status);
        break;
    }
    case features::DAAL_FLOAT64:
    {
        buffer = context.allocate(TypeId::float64, size, status);
        break;
    }

    default: status = services::Status(services::ErrorIncorrectParameter);
    }
    return buffer;
}

/** @} */
} // namespace interface1

using interface1::BufferConverterFrom;
using interface1::BufferConverterTo;
using interface1::BufferHostReinterpreter;
using interface1::allocateByNumericTableFeature;

} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal
/// \endcond

#endif
