/* file: types.h */
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

#ifndef __DAAL_SERVICES_INTERNAL_SYCL_TYPES_H__
#define __DAAL_SERVICES_INTERNAL_SYCL_TYPES_H__

#include <cstddef>
#include <stdint.h>

#include "services/daal_string.h"
#include "services/internal/any.h"
#include "services/internal/buffer.h"

#define DAAL_ASSERT_UNIVERSAL_BUFFER_TYPE(buffer, BufferType) DAAL_ASSERT((buffer).type() == TypeIds::id<BufferType>());

#define DAAL_ASSERT_UNIVERSAL_BUFFER(buffer, BufferType, bufferSize)             \
    {                                                                            \
        DAAL_ASSERT_UNIVERSAL_BUFFER_TYPE(buffer, BufferType)                    \
        DAAL_ASSERT((buffer).template get<BufferType>().size() >= (bufferSize)); \
    }

#define DAAL_ASSERT_UNIVERSAL_BUFFER2(buffer, bufferType1, bufferType2, bufferSize)                                                     \
    {                                                                                                                                   \
        DAAL_ASSERT(((buffer).type() == TypeIds::id<bufferType1>() && (buffer).template get<bufferType1>().size() >= (bufferSize))      \
                    || ((buffer).type() == TypeIds::id<bufferType2>() && (buffer).template get<bufferType2>().size() >= (bufferSize))); \
    }

namespace daal
{
namespace services
{
namespace internal
{
namespace sycl
{
/** @ingroup oneapi_internal
 * @{
 */

typedef ::int8_t int8_t;
typedef ::int16_t int16_t;
typedef ::int32_t int32_t;
typedef ::int64_t int64_t;
typedef ::uint8_t uint8_t;
typedef ::uint16_t uint16_t;
typedef ::uint32_t uint32_t;
typedef ::uint64_t uint64_t;
typedef float float32_t;
typedef double float64_t;

template <typename algorithmFPType>
inline String getKeyFPType()
{
    if (IsSameType<algorithmFPType, float>::value)
    {
        return String(" -D algorithmFPType=float -D algorithmFPType2=float2 -D algorithmFPType4=float4 -D FPTYPE_MAXVALUE=FLT_MAX");
    }
    if (IsSameType<algorithmFPType, double>::value)
    {
        return String(" -D algorithmFPType=double -D algorithmFPType2=double2  -D algorithmFPType4=double4 -D FPTYPE_MAXVALUE=DBL_MAX");
    }
    if (IsSameType<algorithmFPType, int32_t>::value)
    {
        return String(" -D algorithmFPType=int -D algorithmFPType2=int2  -D algorithmFPType4=int4 ");
    }
    if (IsSameType<algorithmFPType, uint32_t>::value)
    {
        return String(" -D algorithmFPType=uint -D algorithmFPType2=uint2  -D algorithmFPType4=uint4 ");
    }
    return String();
}

namespace interface1
{
/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__TYPEIDS"></a>
 *  \brief Mapping from standart types to enum values
 */
class TypeIds
{
public:
    enum Id
    {
        /* Signed integers */
        int8,
        int16,
        int32,
        int64,

        /* Unsigned integers */
        uint8,
        uint16,
        uint32,
        uint64,

        /* Floatin point */
        float32,
        float64,

        /* Other types */
        custom
    };

    template <typename T>
    static inline Id id();

private:
    TypeIds();
};
typedef TypeIds::Id TypeId;

namespace internal
{
template <typename T>
inline TypeId getTypeId()
{
    return TypeIds::custom;
}

#define DAAL_DECLARE_TYPE_ID_MAP(id_)                                  \
    template <>                                                        \
    inline TypeId getTypeId<daal::services::internal::sycl::id_##_t>() \
    {                                                                  \
        return TypeIds::id_;                                           \
    }

DAAL_DECLARE_TYPE_ID_MAP(int8)
DAAL_DECLARE_TYPE_ID_MAP(int16)
DAAL_DECLARE_TYPE_ID_MAP(int32)
DAAL_DECLARE_TYPE_ID_MAP(int64)
DAAL_DECLARE_TYPE_ID_MAP(uint8)
DAAL_DECLARE_TYPE_ID_MAP(uint16)
DAAL_DECLARE_TYPE_ID_MAP(uint32)
DAAL_DECLARE_TYPE_ID_MAP(uint64)
DAAL_DECLARE_TYPE_ID_MAP(float32)
DAAL_DECLARE_TYPE_ID_MAP(float64)

#undef DAAL_DECLARE_TYPE_ID_MAP

} // namespace internal

template <typename T>
inline TypeId TypeIds::id()
{
    return internal::getTypeId<T>();
}

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__EXECUTIONTARGETIDS"></a>
 *  \brief Enumeration of device types avaliable
 */
class ExecutionTargetIds
{
public:
    enum Id
    {
        host,
        device,
        unspecified
    };

private:
    ExecutionTargetIds();
};
typedef ExecutionTargetIds::Id ExecutionTargetId;

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__ACCESSMODEIDS"></a>
 *  \brief Access modes to kernel arguments
 */
class AccessModeIds
{
public:
    enum Id
    {
        read,
        write,
        readwrite
    };

private:
    AccessModeIds();
};
typedef AccessModeIds::Id AccessModeId;

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__UNIVERSALBUFFER"></a>
 *  \brief Non-templated wrapper for Buffer object
 */
class UniversalBuffer : public Base
{
public:
    UniversalBuffer() : _type(TypeIds::id<void>()) {}

    template <typename T>
    UniversalBuffer(const Buffer<T> & buffer) : _type(TypeIds::id<T>()), _anyBuffer(buffer)
    {}

    template <typename T>
    const Buffer<T> & get() const
    {
        return _anyBuffer.get<Buffer<T> >();
    }

    template <typename T>
    UniversalBuffer & operator=(const Buffer<T> & buffer)
    {
        _type      = TypeIds::id<T>();
        _anyBuffer = buffer;
        return *this;
    }

    TypeId type() const { return _type; }

    const Any & any() const { return _anyBuffer; }

    bool empty() const { return _anyBuffer.empty(); }

private:
    TypeId _type;
    Any _anyBuffer;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__LOCALBUFFER"></a>
 *  \brief Class representing local (device-only) buffer
 */
class LocalBuffer : public Base
{
public:
    LocalBuffer(TypeId id, size_t size) : _id(id), _size(size) {}

    TypeId type() const { return _id; }
    size_t size() const { return _size; }

private:
    TypeId _id;
    size_t _size;
};

} // namespace interface1

using interface1::TypeId;
using interface1::TypeIds;
using interface1::ExecutionTargetId;
using interface1::ExecutionTargetIds;
using interface1::AccessModeId;
using interface1::AccessModeIds;
using interface1::UniversalBuffer;
using interface1::LocalBuffer;

/** @} */
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal

#endif
