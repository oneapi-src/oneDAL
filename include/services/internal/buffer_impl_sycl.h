/* file: buffer_impl_sycl.h */
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

#ifndef __DAAL_SERVICES_INTERNAL_BUFFER_SYCL_H__
#define __DAAL_SERVICES_INTERNAL_BUFFER_SYCL_H__

#include <CL/sycl.hpp>
#include "services/internal/any.h"
#include "services/internal/buffer_impl.h"

namespace daal
{
namespace services
{
namespace internal
{

/** @ingroup services_internal
 * @{
 */

#ifdef DAAL_SYCL_INTERFACE_USM
/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__USMBUFFER"></a>
 *  \brief BufferIface implementation based on USM
 */
template<typename T>
class UsmBuffer : public Base,
                  public UsmBufferIface<T>
{
public:
    UsmBuffer(const SharedPtr<T> &data, size_t size, cl::sycl::usm::alloc allocType) :
        _data(data), _size(size), _allocType(allocType) { }

    UsmBuffer(T *data, size_t size, cl::sycl::usm::alloc allocType) :
        _data(data, EmptyDeleter()), _size(size), _allocType(allocType) { }

    size_t size() const DAAL_C11_OVERRIDE
    { return _size; }

    void apply(BufferVisitor<T> &visitor) const DAAL_C11_OVERRIDE
    { visitor(*this); }

    UsmBuffer<T> *getSubBuffer(size_t offset, size_t size) const DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(offset + size <= _size);
        return new UsmBuffer<T>(SharedPtr<T>(_data, _data.get() + offset), size, _allocType);
    }

    SharedPtr<T> getHostRead(Status *status = nullptr) const DAAL_C11_OVERRIDE
    { return getHostPtr(status); }

    SharedPtr<T> getHostWrite(Status *status = nullptr) const DAAL_C11_OVERRIDE
    { return getHostPtr(status); }

    SharedPtr<T> getHostReadWrite(Status *status = nullptr) const DAAL_C11_OVERRIDE
    { return getHostPtr(status); }

    const SharedPtr<T> &get() const DAAL_C11_OVERRIDE
    { return _data; }

    cl::sycl::usm::alloc getAllocType() const
    { return _allocType; }

private:
    SharedPtr<T> getHostPtr(Status *status) const
    {
        using namespace cl::sycl::usm;
        if (_allocType == alloc::host || _allocType == alloc::shared)
        { return _data; }

        /* Note: `cl::sycl::get_pointer_info` is not implemented right now. With
         * the `get_pointer_info` logic shall be the following: If device is
         * host or CPU, return `_data`, otherwise throw exception. */
        const auto error = Error::create(ErrorAccessUSMPointerOnOtherDevice, Sycl,
                                         "Cannot access device pointer on host");
        tryAssignStatusAndThrow(status, error);

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
    typedef cl::sycl::accessor<T, 1, mode,
        cl::sycl::access::target::host_buffer> HostAccessorType;

public:
    explicit SyclHostDeleter(HostAccessorType *accessor)
        : _hostAccessor(accessor) { }

    void operator() (const void *ptr)
    {
        delete _hostAccessor;
        _hostAccessor = nullptr;
    }

private:
    HostAccessorType *_hostAccessor;
};

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__SYCLBUFFER"></a>
 *  \brief BufferIface implementation based on SYCL* buffer
 */
template<typename T>
class SyclBuffer : public Base,
                   public SyclBufferIface<T>
{
private:
    typedef cl::sycl::buffer<T, 1> BufferType;

public:
    explicit SyclBuffer(size_t size)
        : _syclBuffer(cl::sycl::range<1>(size)) { }

    explicit SyclBuffer(const BufferType &syclBuffer)
        : _syclBuffer(syclBuffer) { }

    size_t size() const DAAL_C11_OVERRIDE
    { return _syclBuffer.get_count(); }

    void apply(BufferVisitor<T> &visitor) const DAAL_C11_OVERRIDE
    { visitor(*this); }

    SyclBuffer<T> *getSubBuffer(size_t offset, size_t size) const DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(offset + size <= _size);

        /* Workaround: ComputeCpp does not provide constructor for SYCL* buffer
         * defined in standard 1.2.1:
         *      buffer(buffer<T, dimensions, AllocatorT> b,
         *             const id<dimensions> &baseIndex,
         *             const range<dimensions> &subRange);
         *
         * Instead, they provide the method that accepts the first argument via
         * non-const reference, so we remove cv-qualifier for _syclBuffer
        */
        BufferType &buffer = const_cast<BufferType &>(_syclBuffer);
        return new SyclBuffer<T>(BufferType(buffer, offset, size));
    }

    SharedPtr<T> getHostRead(Status *status = nullptr) const DAAL_C11_OVERRIDE
    { return getHostPtr<cl::sycl::access::mode::read>(); }

    SharedPtr<T> getHostWrite(Status *status = nullptr) const DAAL_C11_OVERRIDE
    { return getHostPtr<cl::sycl::access::mode::write>(); }

    SharedPtr<T> getHostReadWrite(Status *status = nullptr) const DAAL_C11_OVERRIDE
    { return getHostPtr<cl::sycl::access::mode::read_write>(); }

    const BufferType &get() const
    { return _syclBuffer; }

private:
    template <cl::sycl::access::mode mode>
    SharedPtr<T> getHostPtr() const
    {
        using DeleterType = SyclHostDeleter<T, mode>;
        using AccessorType = typename DeleterType::HostAccessorType;
        auto* accessor = new AccessorType(const_cast<BufferType&>(_syclBuffer));
        return SharedPtr<T>(accessor->get_pointer(), DeleterType(accessor));
    }

    BufferType _syclBuffer;
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
    void operator()(const HostBuffer<T> &buffer) DAAL_C11_OVERRIDE
    { _syclBuffer = wrap(buffer.get(), buffer.size()); }

    void operator()(const UsmBufferIface<T> &buffer) DAAL_C11_OVERRIDE
    { _syclBuffer = wrap(buffer.get(), buffer.size(), true); }

    void operator()(const SyclBufferIface<T> &buffer) DAAL_C11_OVERRIDE
    { _syclBuffer = static_cast<const SyclBuffer<T>&>(buffer).get(); }

    const SyclBufferType &get() const
    { return _syclBuffer.get<SyclBufferType>(); }

private:
    static SyclBufferType wrap(const SharedPtr<T> &ptr, size_t size, bool useHostPtr = false)
    {
        const auto bufferProperties = (useHostPtr)
            ? cl::sycl::property_list { cl::sycl::property::buffer::use_host_ptr() }
            : cl::sycl::property_list { };

        return SyclBufferType(ptr.get(), cl::sycl::range<1>(size), bufferProperties);
    }

    services::internal::Any _syclBuffer;
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
    void operator()(const HostBuffer<T> &buffer) DAAL_C11_OVERRIDE
    { _data = buffer.get(); }

    void operator()(const UsmBufferIface<T> &buffer) DAAL_C11_OVERRIDE
    { _data = buffer.get(); }

    void operator()(const SyclBufferIface<T> &buffer) DAAL_C11_OVERRIDE
    {
      /* NOTE: Performance might be not quite satisfactory. If the SYCL* buffer
       * is a wrapper over pointer (e.g., was created using `use_host_ptr`
       * property), `getHostReadWrite` will not create overhead. Otherwise,
       * getting host pointer will result in graph synchronization and potential
       * data copy. */
      // TODO: Report error or warning, if `buffer` does not has property `use_host_ptr`
      _data = buffer.getHostReadWrite();
    }

    const SharedPtr<T> &get() const
    { return _data; }

private:
    SharedPtr<T> _data;
};
#endif

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__SYCLBUFFERCONVERTER"></a>
 *  \brief Groups high-level conversion methods for SYCL* buffer and USM
 */
template<typename T>
class SyclBufferConverter
{
public:
    cl::sycl::buffer<T, 1> toSycl(const internal::BufferIface<T> &buffer)
    {
        ConvertToSycl<T> action;
        buffer.apply(action);
        return action.get();
    }

#ifdef DAAL_SYCL_INTERFACE_USM
    SharedPtr<T> toUSM(const internal::BufferIface<T> &buffer)
    {
        ConvertToUsm<T> action;
        buffer.apply(action);
        return action.get();
    }
#endif
};

/** @} */

} // namespace internal
} // namespace services
} // namespace daal

#endif
