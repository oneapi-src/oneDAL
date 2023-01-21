/* file: buffer_impl_sycl.h */
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

#ifndef __DAAL_SERVICES_INTERNAL_BUFFER_SYCL_H__
#define __DAAL_SERVICES_INTERNAL_BUFFER_SYCL_H__

#ifndef DAAL_SYCL_INTERFACE
    #error "DAAL_SYCL_INTERFACE must be defined to include this file"
#endif

#include <sycl/sycl.hpp>

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
inline ::sycl::buffer<T, 1> createEmptySyclBuffer()
{
    return ::sycl::buffer<T, 1>(nullptr, ::sycl::range<1> { 0 });
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
    static UsmBuffer<T> * create(const SharedPtr<T> & data, size_t size, const ::sycl::queue & queue, Status & status)
    {
        if (!data && size != size_t(0))
        {
            status |= ErrorNullPtr;
            return nullptr;
        }
        const auto newBuffer = new UsmBuffer<T>(data, size, queue);
        DAAL_CHECK_COND_ERROR(newBuffer, status, ErrorMemoryAllocationFailed);
        return newBuffer;
    }

    static UsmBuffer<T> * create(T * data, size_t size, const ::sycl::queue & queue, Status & status)
    {
        return create(SharedPtr<T> { data, EmptyDeleter() }, size, queue, status);
    }

    size_t size() const DAAL_C11_OVERRIDE { return _size; }

    Status apply(BufferVisitor<T> & visitor) const DAAL_C11_OVERRIDE { return visitor(*this); }

    UsmBuffer<T> * getSubBuffer(size_t offset, size_t size, Status & status) const DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(offset + size <= _size);
        return create(SharedPtr<T>(_data, _data.get() + offset), size, _queue, status);
    }

    SharedPtr<T> getHostRead(Status & status) const DAAL_C11_OVERRIDE { return getHostPtr(true, false, status); }

    SharedPtr<T> getHostWrite(Status & status) const DAAL_C11_OVERRIDE { return getHostPtr(false, true, status); }

    SharedPtr<T> getHostReadWrite(Status & status) const DAAL_C11_OVERRIDE { return getHostPtr(true, true, status); }

    const SharedPtr<T> & get() const DAAL_C11_OVERRIDE { return _data; }

private:
    UsmBuffer(const SharedPtr<T> & data, size_t size, const ::sycl::queue & queue) : _data(data), _size(size), _queue(queue)
    {
        _allocType = ::sycl::get_pointer_type(data.get(), _queue.get_context());
        DAAL_ASSERT(_allocType != ::sycl::usm::alloc::unknown);
    }

    SharedPtr<T> getHostPtr(bool needCopyToHost, bool needSynchronize, Status & status) const
    {
        using namespace ::sycl::usm;
        if (_allocType == alloc::host || _allocType == alloc::shared)
        {
            return _data;
        }
        else if (_allocType == alloc::device)
        {
            auto host_ptr = SharedPtr<T>(::sycl::malloc_host<T>(_size, _queue), // TODO: use daal_malloc
                                         [q = this->_queue, data = this->_data, size = this->_size, needSynchronize](const void * hostData) mutable {
                                             if (needSynchronize)
                                             {
                                                 auto event = q.memcpy(data.get(), hostData, size * sizeof(T));
                                                 event.wait_and_throw();
                                             }
                                             ::sycl::free(const_cast<void *>(hostData), q);
                                         });
            if (!host_ptr)
            {
                status |= services::ErrorMemoryAllocationFailed;
                return host_ptr;
            }

            if (needCopyToHost)
            {
                status |= internal::sycl::catchSyclExceptions([&, q = this->_queue]() mutable {
                    auto event = q.memcpy(host_ptr.get(), _data.get(), _size * sizeof(T));
                    event.wait_and_throw();
                });
            }
            return host_ptr;
        }

        /* Note: `sycl::get_pointer_info` is not implemented right now. With
         * the `get_pointer_info` logic shall be the following: If device is
         * host or CPU, return `_data`, otherwise throw exception. */
        status |= Error::create(ErrorAccessUSMPointerOnOtherDevice, Sycl, "Cannot access unknown USM pointer on host");

        return SharedPtr<T>();
    }

    SharedPtr<T> _data;
    size_t _size;
    ::sycl::queue _queue;
    ::sycl::usm::alloc _allocType;
};
#endif

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__SYCLHOSTDELETER"></a>
 *  \brief Deleter for SharedPtr that owns host accessor for SYCL* buffer
 */
template <typename T, ::sycl::access::mode mode>
class SyclHostDeleter : public Base
{
public:
    typedef ::sycl::accessor<T, 1, mode, ::sycl::access::target::host_buffer> HostAccessorType;

public:
    explicit SyclHostDeleter(const ::sycl::buffer<T, 1> & buffer, HostAccessorType * accessor) : _buffer(buffer), _hostAccessor(accessor)
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
    ::sycl::buffer<T, 1> _buffer;
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
    typedef ::sycl::buffer<T, 1> BufferType;

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

        const auto nativeBufferWithOffset = createNativeBuffer(status, nativeBuffer, ::sycl::id<1>(offset), ::sycl::range<1>(size));
        DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, nullptr);

        return create(nativeBufferWithOffset, status);
    }

    SharedPtr<T> getHostRead(Status & status) const DAAL_C11_OVERRIDE { return getHostPtr< ::sycl::access::mode::read>(status); }

    SharedPtr<T> getHostWrite(Status & status) const DAAL_C11_OVERRIDE { return getHostPtr< ::sycl::access::mode::write>(status); }

    SharedPtr<T> getHostReadWrite(Status & status) const DAAL_C11_OVERRIDE { return getHostPtr< ::sycl::access::mode::read_write>(status); }

    const BufferType & get() const { return _nativeBuffer; }

private:
    explicit SyclBuffer(size_t size, Status & status) : _nativeBuffer(createNativeBuffer(status, size)) {}

    explicit SyclBuffer(const BufferType & nativeBuffer, Status & status) : _nativeBuffer(createNativeBuffer(status, nativeBuffer)) {}

    template < ::sycl::access::mode mode>
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
    typedef ::sycl::buffer<T, 1> SyclBufferType;

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
        auto hostPtr = buffer.getHostReadWrite(status);
        DAAL_CHECK_STATUS_VAR(status);

        _nativeBuffer = internal::sycl::catchSyclExceptions(
            status,
            [&]() {
                const auto bufferProperties = ::sycl::property_list { ::sycl::property::buffer::use_host_ptr() };

                return SyclBufferType(std::shared_ptr<T> { hostPtr.get(), [owner = hostPtr](T * ptr) {} }, ::sycl::range<1>(buffer.size()),
                                      bufferProperties);
            },
            [&]() { return createEmptySyclBuffer<T>(); });

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
                    (useHostPtr) ? ::sycl::property_list { ::sycl::property::buffer::use_host_ptr() } : ::sycl::property_list {};

                return SyclBufferType(ptr.get(), ::sycl::range<1>(size), bufferProperties);
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
    ConvertToUsm(::sycl::queue & queue, const data_management::ReadWriteMode & rwFlag) : _q(queue), _rwFlag(rwFlag) {}

    Status makeCopyToUSM(const SharedPtr<T> & hostData, size_t count)
    {
        Status st;
        // TODO: use malloc_device and queue.memcpy()
        auto usmData = ::sycl::malloc_shared<T>(count, _q);
        if (usmData == nullptr)
        {
            return services::ErrorMemoryAllocationFailed;
        }

        const size_t size = sizeof(T) * count;
        DAAL_ASSERT(size / sizeof(T) == count);

        if (_rwFlag & data_management::readOnly)
        {
            int result = daal_memcpy_s(usmData, size, hostData.get(), size);
            if (result)
            {
                return services::ErrorMemoryCopyFailedInternal;
            }
        }

        _data = SharedPtr<T>(usmData, [q = this->_q, rwFlag = this->_rwFlag, hostData, size](const void * data) mutable {
            if (rwFlag & data_management::writeOnly)
            {
                daal_memcpy_s(hostData.get(), size, data, size);
            }
            ::sycl::free(const_cast<void *>(data), q);
        });
        return st;
    }

    Status operator()(const HostBuffer<T> & buffer) DAAL_C11_OVERRIDE
    {
        auto hostData = buffer.get();
        return makeCopyToUSM(hostData, buffer.size());
    }

    Status operator()(const UsmBufferIface<T> & buffer) DAAL_C11_OVERRIDE
    {
        _data = buffer.get();
        return Status();
    }

    Status operator()(const SyclBufferIface<T> & buffer) DAAL_C11_OVERRIDE
    {
        Status st;
        auto hostData = buffer.getHostReadWrite(st);
        DAAL_CHECK_STATUS_VAR(st);
        return makeCopyToUSM(hostData, buffer.size());
    }

    const SharedPtr<T> & get() const { return _data; }

private:
    SharedPtr<T> _data;
    ::sycl::queue & _q;
    data_management::ReadWriteMode _rwFlag;
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
    ::sycl::buffer<T, 1> toSycl(const internal::BufferIface<T> & buffer, Status & status)
    {
        ConvertToSycl<T> action;
        status |= buffer.apply(action);
        DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, createEmptySyclBuffer<T>());
        return action.get();
    }

#ifdef DAAL_SYCL_INTERFACE_USM
    SharedPtr<T> toUSM(const internal::BufferIface<T> & buffer, ::sycl::queue & q, const data_management::ReadWriteMode & rwFlag, Status & status)
    {
        ConvertToUsm<T> action(q, rwFlag);
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
