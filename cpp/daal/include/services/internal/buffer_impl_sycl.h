/* file: buffer_impl_sycl.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

#ifndef __DAAL_SERVICES_INTERNAL_BUFFER_SYCL_H__
#define __DAAL_SERVICES_INTERNAL_BUFFER_SYCL_H__

#ifndef DAAL_SYCL_INTERFACE
    #error "DAAL_SYCL_INTERFACE must be defined to include this file"
#endif

#include <CL/sycl.hpp>

#include "services/internal/any.h"
#include "services/internal/buffer_impl.h"
#include "services/internal/sycl/error_handling_sycl.h"

namespace daal
{
namespace services
{
namespace internal
{
/** @ingroup services_internal
 * @{
 */

template <typename T>
inline cl::sycl::buffer<T, 1> createEmptySyclBuffer()
{
    return cl::sycl::buffer<T, 1>(nullptr, cl::sycl::range<1> { 0 });
}

#ifdef DAAL_SYCL_INTERFACE_USM
/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__USMBUFFER"></a>
 *  \brief BufferIface implementation based on USM
 */
template <typename T>
class UsmBuffer : public Base, public UsmBufferIface<T>
{
public:
    static UsmBuffer<T> * create(const SharedPtr<T> & data, size_t size, cl::sycl::usm::alloc allocType, Status & status)
    {
        if (!data && size != size_t(0))
        {
            status |= ErrorNullPtr;
            return nullptr;
        }
        const auto newBuffer = new UsmBuffer<T>(data, size, allocType);
        DAAL_CHECK_COND_ERROR(newBuffer, status, ErrorMemoryAllocationFailed);
        return newBuffer;
    }

    static UsmBuffer<T> * create(T * data, size_t size, cl::sycl::usm::alloc allocType, Status & status)
    {
        return create(SharedPtr<T> { data, EmptyDeleter() }, size, allocType, status);
    }

    size_t size() const DAAL_C11_OVERRIDE { return _size; }

    Status apply(BufferVisitor<T> & visitor) const DAAL_C11_OVERRIDE { return visitor(*this); }

    UsmBuffer<T> * getSubBuffer(size_t offset, size_t size, Status & status) const DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(offset + size <= _size);
        return create(SharedPtr<T>(_data, _data.get() + offset), size, _allocType, status);
    }

    SharedPtr<T> getHostRead(Status & status) const DAAL_C11_OVERRIDE { return getHostPtr(status); }

    SharedPtr<T> getHostWrite(Status & status) const DAAL_C11_OVERRIDE { return getHostPtr(status); }

    SharedPtr<T> getHostReadWrite(Status & status) const DAAL_C11_OVERRIDE { return getHostPtr(status); }

    const SharedPtr<T> & get() const DAAL_C11_OVERRIDE { return _data; }

    cl::sycl::usm::alloc getAllocType() const { return _allocType; }

private:
    UsmBuffer(const SharedPtr<T> & data, size_t size, cl::sycl::usm::alloc allocType) : _data(data), _size(size), _allocType(allocType) {}

    SharedPtr<T> getHostPtr(Status & status) const
    {
        using namespace cl::sycl::usm;
        if (_allocType == alloc::host || _allocType == alloc::shared)
        {
            return _data;
        }

        /* Note: `cl::sycl::get_pointer_info` is not implemented right now. With
         * the `get_pointer_info` logic shall be the following: If device is
         * host or CPU, return `_data`, otherwise throw exception. */
        status |= Error::create(ErrorAccessUSMPointerOnOtherDevice, Sycl, "Cannot access device pointer on host");

        return SharedPtr<T>();
    }

    SharedPtr<T> _data;
    size_t _size;
    cl::sycl::usm::alloc _allocType;
};
#endif

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__SYCLHOSTDELETER"></a>
 *  \brief Deleter for SharedPtr that owns host accessor for SYCL* buffer
 */
template <typename T, cl::sycl::access::mode mode>
class SyclHostDeleter : public Base
{
public:
    typedef cl::sycl::accessor<T, 1, mode, cl::sycl::access::target::host_buffer> HostAccessorType;

public:
    explicit SyclHostDeleter(const cl::sycl::buffer<T, 1> & buffer, HostAccessorType * accessor) : _buffer(buffer), _hostAccessor(accessor)
    {
        DAAL_ASSERT(_hostAccessor);
    }

    void operator()(const void * ptr)
    {
        if (!_hostAccessor)
        {
            DAAL_ASSERT(!"Potential attempt to delete host accessor twice");
        }

        DAAL_ASSERT(ptr == _hostAccessor->get_pointer());
        delete _hostAccessor;
        _hostAccessor = nullptr;
    }

private:
    cl::sycl::buffer<T, 1> _buffer;
    HostAccessorType * _hostAccessor;
};

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__nativeBuffer"></a>
 *  \brief BufferIface implementation based on SYCL* buffer
 */
template <typename T>
class SyclBuffer : public Base, public SyclBufferIface<T>
{
private:
    typedef cl::sycl::buffer<T, 1> BufferType;

public:
    static SyclBuffer<T> * create(size_t size, Status & status)
    {
        const auto newBuffer = new SyclBuffer<T>(size, status);
        DAAL_CHECK_COND_ERROR(newBuffer, status, ErrorMemoryAllocationFailed);
        return newBuffer;
    }

    static SyclBuffer<T> * create(const BufferType & syclBuffer, Status & status)
    {
        const auto newBuffer = new SyclBuffer<T>(syclBuffer, status);
        DAAL_CHECK_COND_ERROR(newBuffer, status, ErrorMemoryAllocationFailed);
        return newBuffer;
    }

    size_t size() const DAAL_C11_OVERRIDE { return _nativeBuffer.get_count(); }

    Status apply(BufferVisitor<T> & visitor) const DAAL_C11_OVERRIDE { return visitor(*this); }

    SyclBuffer<T> * getSubBuffer(size_t offset, size_t size, Status & status) const DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(offset + size <= this->size());

        BufferType & nativeBuffer = const_cast<BufferType &>(_nativeBuffer);
        if (offset == 0 && size == this->size())
        {
            return create(nativeBuffer, status);
        }

        const auto nativeBufferWithOffset = createNativeBuffer(status, nativeBuffer, cl::sycl::id<1>(offset), cl::sycl::range<1>(size));
        DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, nullptr);

        return create(nativeBufferWithOffset, status);
    }

    SharedPtr<T> getHostRead(Status & status) const DAAL_C11_OVERRIDE { return getHostPtr<cl::sycl::access::mode::read>(status); }

    SharedPtr<T> getHostWrite(Status & status) const DAAL_C11_OVERRIDE { return getHostPtr<cl::sycl::access::mode::write>(status); }

    SharedPtr<T> getHostReadWrite(Status & status) const DAAL_C11_OVERRIDE { return getHostPtr<cl::sycl::access::mode::read_write>(status); }

    const BufferType & get() const { return _nativeBuffer; }

private:
    explicit SyclBuffer(size_t size, Status & status) : _nativeBuffer(createNativeBuffer(status, size)) {}

    explicit SyclBuffer(const BufferType & nativeBuffer, Status & status) : _nativeBuffer(createNativeBuffer(status, nativeBuffer)) {}

    template <cl::sycl::access::mode mode>
    SharedPtr<T> getHostPtr(Status & status) const
    {
        using DeleterType  = SyclHostDeleter<T, mode>;
        using AccessorType = typename DeleterType::HostAccessorType;
        return internal::sycl::catchSyclExceptions(
            status,
            [&]() {
                auto * accessor = new AccessorType(const_cast<BufferType &>(_nativeBuffer));
                return SharedPtr<T>(accessor->get_pointer(), DeleterType(_nativeBuffer, accessor));
            },
            [&]() { return SharedPtr<T>(); });
    }

    template <typename... Args>
    static BufferType createNativeBuffer(Status & status, Args &&... args)
    {
        return internal::sycl::catchSyclExceptions(
            status, [&]() { return BufferType(std::forward<Args>(args)...); }, [&]() { return createEmptySyclBuffer<T>(); });
    }

    BufferType _nativeBuffer;
};

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__CONVERTTOSYCL"></a>
 *  \brief BufferVisitor that converters any buffer to SYCL* buffer
 */
template <typename T>
class ConvertToSycl : public BufferVisitor<T>
{
private:
    typedef cl::sycl::buffer<T, 1> SyclBufferType;

public:
    Status operator()(const HostBuffer<T> & buffer) DAAL_C11_OVERRIDE
    {
        Status status;
        _nativeBuffer = wrap(status, buffer.get(), buffer.size());
        return status;
    }

    Status operator()(const UsmBufferIface<T> & buffer) DAAL_C11_OVERRIDE
    {
        Status status;
        _nativeBuffer = wrap(status, buffer.get(), buffer.size(), true);
        return status;
    }

    Status operator()(const SyclBufferIface<T> & buffer) DAAL_C11_OVERRIDE
    {
        _nativeBuffer = static_cast<const SyclBuffer<T> &>(buffer).get();
        return Status();
    }

    const SyclBufferType & get() const { return _nativeBuffer.get<SyclBufferType>(); }

private:
    static SyclBufferType wrap(Status & status, const SharedPtr<T> & ptr, size_t size, bool useHostPtr = false)
    {
        return internal::sycl::catchSyclExceptions(
            status,
            [&]() {
                const auto bufferProperties =
                    (useHostPtr) ? cl::sycl::property_list { cl::sycl::property::buffer::use_host_ptr() } : cl::sycl::property_list {};

                return SyclBufferType(ptr.get(), cl::sycl::range<1>(size), bufferProperties);
            },
            [&]() { return createEmptySyclBuffer<T>(); });
    }

    Any _nativeBuffer;
};

#ifdef DAAL_SYCL_INTERFACE_USM
/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__CONVERTTOUSM"></a>
 *  \brief BufferVisitor that converters any buffer to USM pointer
 */
template <typename T>
class ConvertToUsm : public BufferVisitor<T>
{
public:
    Status operator()(const HostBuffer<T> & buffer) DAAL_C11_OVERRIDE
    {
        _data = buffer.get();
        return Status();
    }

    Status operator()(const UsmBufferIface<T> & buffer) DAAL_C11_OVERRIDE
    {
        _data = buffer.get();
        return Status();
    }

    Status operator()(const SyclBufferIface<T> & buffer) DAAL_C11_OVERRIDE
    {
        Status status;
        /* NOTE: Performance might be not quite satisfactory. If the SYCL* buffer
       * is a wrapper over pointer (e.g., was created using `use_host_ptr`
       * property), `getHostReadWrite` will not create overhead. Otherwise,
       * getting host pointer will result in graph synchronization and potential
       * data copy. */
        _data = buffer.getHostReadWrite(status);
        return status;
    }

    const SharedPtr<T> & get() const { return _data; }

private:
    SharedPtr<T> _data;
};
#endif

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__nativeBufferCONVERTER"></a>
 *  \brief Groups high-level conversion methods for SYCL* buffer and USM
 */
template <typename T>
class SyclBufferConverter
{
public:
    cl::sycl::buffer<T, 1> toSycl(const internal::BufferIface<T> & buffer, Status & status)
    {
        ConvertToSycl<T> action;
        status |= buffer.apply(action);
        DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, createEmptySyclBuffer<T>());
        return action.get();
    }

#ifdef DAAL_SYCL_INTERFACE_USM
    SharedPtr<T> toUSM(const internal::BufferIface<T> & buffer, Status & status)
    {
        ConvertToUsm<T> action;
        status |= buffer.apply(action);
        DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, SharedPtr<T>());
        return action.get();
    }
#endif
};

/** @} */

} // namespace internal
} // namespace services
} // namespace daal

#endif
