/* file: buffer.h */
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

/*
//++
// Declaration and implementation of the shared pointer class.
//--
*/

#ifndef __DAAL_SERVICES_BUFFER_H__
#define __DAAL_SERVICES_BUFFER_H__

#include "services/daal_shared_ptr.h"
#include "services/internal/any.h"
#include "data_management/data/numeric_types.h"

#ifdef DAAL_SYCL_INTERFACE
#include <CL/sycl.hpp>
#endif

namespace daal
{
namespace services
{
namespace internal
{

/** @ingroup services_internal
 * @{
 */

template<typename T> class HostSharedPtrBufferImpl;
template<typename T> class SyclBufferImplIface;

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__BUFFERIMPLVISITOR"></a>
 *  \brief Visitor pattern implementation for Buffer class
 */
template<typename T>
class BufferImplVisitor : public Base
{
public:
    virtual void operator()(const HostSharedPtrBufferImpl<T> &bufferImpl) = 0;
    virtual void operator()(const SyclBufferImplIface<T> &bufferImpl) = 0;
};

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__BUFFERIMPLIFACE"></a>
 *  \brief Common Buffer interface
 */
class BufferImplIface
{
public:
    virtual ~BufferImplIface() { }
};

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__BUFFERIMPLTEMPLATEIFACE"></a>
 *  \brief Templated buffer interface
 */
template<typename T>
class BufferImplTemplateIface : public BufferImplIface
{
public:
    virtual size_t size() const = 0;
    virtual void apply(BufferImplVisitor<T> &visitor) const = 0;
    virtual BufferImplTemplateIface<T> *getSubBuffer(size_t offset, size_t size) const = 0;
    virtual SharedPtr< BufferImplTemplateIface<T> > allocateLike(size_t size) const = 0;
};

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__HOSTSHAREDPTRBUFFERIMPL"></a>
 *  \brief Implementation of Buffer as host shared pointer
 */
template<typename T>
class HostSharedPtrBufferImpl : public Base,
                                public BufferImplTemplateIface<T>
{
public:
    explicit HostSharedPtrBufferImpl(const SharedPtr<T> &data, size_t size)
        : _data(data), _size(size) { }

    explicit HostSharedPtrBufferImpl(T* data, size_t size)
        : _data(data, services::EmptyDeleter()), _size(size) { }

    size_t size() const DAAL_C11_OVERRIDE
    { return _size; }

    void apply(BufferImplVisitor<T> &visitor) const DAAL_C11_OVERRIDE
    { visitor(*this); }

    HostSharedPtrBufferImpl<T> *getSubBuffer(size_t offset, size_t size) const DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT( offset + size <= _size );
        return new HostSharedPtrBufferImpl<T>(SharedPtr<T>(_data, _data.get() + offset), size);
    }

    const SharedPtr<T> &get() const
    { return _data; }

    SharedPtr< BufferImplTemplateIface<T> > allocateLike(size_t size) const DAAL_C11_OVERRIDE
    {
        SharedPtr<T> data(new T[size]);
        return SharedPtr< HostSharedPtrBufferImpl<T> >(new HostSharedPtrBufferImpl<T>(data, size));
    }

private:
    SharedPtr<T> _data;
    size_t _size;
};

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__SYCLBUFFERIMPLIFACE"></a>
 *  \brief Interface of SYCL* buffer implementation
 */
template <typename T>
class SyclBufferImplIface : public BufferImplTemplateIface<T>
{
public:
    virtual SharedPtr<T> getReadHostSharedPtr() const = 0;
    virtual SharedPtr<T> getReadWriteHostSharedPtr() const = 0;
    virtual SharedPtr<T> getWriteHostSharedPtr() const = 0;
};

#ifdef DAAL_SYCL_INTERFACE

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__SYCLHOSTSHAREDPTRDELETER"></a>
 *  \brief RAII wrapper for host accessor to SYCL* buffer
 */
template <typename T, cl::sycl::access::mode mode>
class SyclHostSharedPtrDeleter
{
public:
    typedef cl::sycl::accessor<T, 1, mode,
                                     cl::sycl::access::target::host_buffer> HostAccessorType;

public:
    SyclHostSharedPtrDeleter(HostAccessorType* accessor)
        : _hostAccessor( accessor )
    {}

    void operator() (const void *ptr)
    {
        delete _hostAccessor;
    }

private:
    HostAccessorType* _hostAccessor;
};

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__SYCLBUFFERIMPL"></a>
 *  \brief Implementation of Buffer as SYCL* buffer
 */
template<typename T>
class SyclBufferImpl : public Base,
                       public SyclBufferImplIface<T>
{
private:
    typedef cl::sycl::buffer<T, 1> BufferType;

public:
    explicit SyclBufferImpl(size_t size) :
        _syclBuffer(cl::sycl::range<1>(size))
    { }

    explicit SyclBufferImpl(const BufferType &syclBuffer) :
        _syclBuffer(syclBuffer)
    { }

    size_t size() const DAAL_C11_OVERRIDE
    { return _syclBuffer.get_count(); }

    void apply(BufferImplVisitor<T> &visitor) const DAAL_C11_OVERRIDE
    { visitor(*this); }

    SyclBufferImpl<T> *getSubBuffer(size_t offset, size_t size) const DAAL_C11_OVERRIDE
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
        return new SyclBufferImpl<T>(BufferType(buffer, offset, size));
    }

    const BufferType &get() const
    {
        return _syclBuffer;
    }

    SharedPtr< BufferImplTemplateIface<T> > allocateLike(size_t size) const DAAL_C11_OVERRIDE
    {
        return SharedPtr< SyclBufferImpl<T> >(new SyclBufferImpl<T>(size));
    }

    virtual SharedPtr<T> getReadHostSharedPtr() const DAAL_C11_OVERRIDE
    {
        using DeleterType = SyclHostSharedPtrDeleter<T, cl::sycl::access::mode::read>;

        auto* accessor = new typename DeleterType::HostAccessorType(const_cast<BufferType&>(_syclBuffer));
        DeleterType deleter( accessor );

        return SharedPtr<T>(accessor->get_pointer(), deleter);
    }

    virtual SharedPtr<T> getReadWriteHostSharedPtr() const DAAL_C11_OVERRIDE
    {
        using DeleterType = SyclHostSharedPtrDeleter<T, cl::sycl::access::mode::read_write>;

        auto* accessor = new typename DeleterType::HostAccessorType(const_cast<BufferType&>(_syclBuffer));
        DeleterType deleter( accessor );

        return SharedPtr<T>(accessor->get_pointer(), deleter);
    }

    virtual SharedPtr<T> getWriteHostSharedPtr() const DAAL_C11_OVERRIDE
    {
        using DeleterType = SyclHostSharedPtrDeleter<T, cl::sycl::access::mode::write>;

        auto* accessor = new typename DeleterType::HostAccessorType(const_cast<BufferType&>(_syclBuffer));
        DeleterType deleter( accessor );

        return SharedPtr<T>(accessor->get_pointer(), deleter);
    }

private:
    BufferType _syclBuffer;
};

#else

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__SYCLBUFFERIMPL"></a>
 *  \brief Implementation of Buffer as SYCL* buffer
 */
template<typename T>
class SyclBufferImpl : public Base,
                       public SyclBufferImplIface<T>
{
public:
    size_t size() const DAAL_C11_OVERRIDE
    { return 0; }

    void apply(BufferImplVisitor<T> &visitor) const DAAL_C11_OVERRIDE { }

    virtual SharedPtr<T> getReadHostSharedPtr() const DAAL_C11_OVERRIDE
    {
        return SharedPtr<T>();
    }

    virtual SharedPtr<T> getReadWriteHostSharedPtr() const DAAL_C11_OVERRIDE
    {
        return SharedPtr<T>();
    }

    virtual SharedPtr<T> getWriteHostSharedPtr() const DAAL_C11_OVERRIDE
    {
        return SharedPtr<T>();
    }
};

#endif

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__CONVERTTOHOSTSHAREDPTR"></a>
 *  \brief Buffer converter to host shared pointer
 */
template<typename T>
class ConvertToHostSharedPtr: public BufferImplVisitor<T>
{
public:
    ConvertToHostSharedPtr(const data_management::ReadWriteMode& rwFlag)
        : _rwFlag(rwFlag)
    {}

    void operator()(const HostSharedPtrBufferImpl<T> &bufferImpl)
    {
        _hostSharedPtr = bufferImpl.get();
    }

    void operator()(const SyclBufferImplIface<T> &bufferImpl)
    {
        if (_rwFlag == data_management::readOnly)
        {
            _hostSharedPtr = bufferImpl.getReadHostSharedPtr();
        }
        else if (_rwFlag == data_management::readWrite)
        {
            _hostSharedPtr = bufferImpl.getReadWriteHostSharedPtr();
        }
        else if (_rwFlag == data_management::writeOnly)
        {
            _hostSharedPtr = bufferImpl.getWriteHostSharedPtr();
        }
        else
        {
            DAAL_ASSERT(!"Not implemented read-write mode");
        }
    }

    const SharedPtr<T> &getHostSharedPtr() const {
        return _hostSharedPtr;
    }

private:
    SharedPtr<T> _hostSharedPtr;
    data_management::ReadWriteMode _rwFlag;
};

#ifdef DAAL_SYCL_INTERFACE

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__CONVERTTOSYCL"></a>
 *  \brief Buffer converter to SYCL* buffer
 */
template<typename T>
class ConvertToSycl: public BufferImplVisitor<T>
{
private:
    typedef cl::sycl::buffer<T, 1> SyclBufferType;

public:
    void operator()(const HostSharedPtrBufferImpl<T> &bufferImpl)
    {
        _syclBuffer = wrap(bufferImpl.get().get(), bufferImpl.size());
    }

    void operator()(const SyclBufferImplIface<T> &bufferImpl)
    {
        _syclBuffer = static_cast<const SyclBufferImpl<T>&>(bufferImpl).get();
    }

    const SyclBufferType &getSyclBuffer() const {
        return _syclBuffer.get<SyclBufferType>();
    }

private:
    static SyclBufferType wrap(T *ptr, size_t size) {
        return SyclBufferType(ptr, cl::sycl::range<1>(size));
    }

    services::internal::Any _syclBuffer;
};
#endif

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__BUFFERCONVERTER"></a>
 *  \brief Buffer converter
 */
template<typename T>
class BufferConverter
{
public:
    static SharedPtr<T> toHostSharedPtr(const internal::BufferImplTemplateIface<T> &bufferImpl,
                                        const data_management::ReadWriteMode& rwMode)
    {
        ConvertToHostSharedPtr<T> action(rwMode);
        bufferImpl.apply(action);
        return action.getHostSharedPtr();
    }

#ifdef DAAL_SYCL_INTERFACE
    static cl::sycl::buffer<T, 1> toSycl(const internal::BufferImplTemplateIface<T> &bufferImpl)
    {
        ConvertToSycl<T> action;
        bufferImpl.apply(action);
        return action.getSyclBuffer();
    }
#endif
};

/** @} */
} // namespace internal

namespace interface1
{
/**
 * @ingroup sycl
 * @{
 */

/**
 *  <a name="DAAL-CLASS-SERVICES__BUFFER"></a>
 *  \brief Wrapper for a SYCL* buffer.
 *  Can hold data on the host side using shared pointer,
 *  or on host/device sides using SYCL* buffer.
 */
template<typename T>
class Buffer : public Base
{
public:
    /**
     *   Creates empty Buffer object
     */
    Buffer() { }

#ifdef DAAL_SYCL_INTERFACE
    /**
     *  Creates a Buffer object referencing a SYCL* buffer.
     *  Does not copy the data from the SYCL* buffer.
     */
    Buffer(const cl::sycl::buffer<T, 1> &buffer) :
        _impl(new internal::SyclBufferImpl<T>(buffer)) { }
#endif

    /**
     *   Creates a Buffer object from host-allocated raw pointer.
     *   Buffer does not own this pointer.
     */
    Buffer(T *data, size_t size) :
        _impl(new internal::HostSharedPtrBufferImpl<T>(data, size)) { }

    /**
     *   Creates a Buffer object referencing the shared pointer to the host-allocated data.
     */
    Buffer(const SharedPtr<T> &data, size_t size) :
        _impl(new internal::HostSharedPtrBufferImpl<T>(data, size)) { }

    /**
     *  Returns true if Buffer points to any data
     */
    operator bool() const
    {
        return _impl;
    }

    /**
     *  Returns true if Buffer is equal to \p other
     */
    bool operator==(const Buffer &other) const
    {
        return _impl.get() == other._impl.get();
    }

    /**
     *  Returns true if Buffer is not equal to \p other
     */
    bool operator!=(const Buffer &other) const
    {
        return _impl.get() != other._impl.get();
    }

    /**
     *  Converts data inside the buffer to the host side.
     *  \param[in] rwFlag  Access flag to the data
     *  \return host-allocated shared pointer to the data.
     */
    inline SharedPtr<T> toHost(const data_management::ReadWriteMode& rwFlag) const
    {
        return internal::BufferConverter<T>::toHostSharedPtr(*_impl, rwFlag);
    }

#ifdef DAAL_SYCL_INTERFACE
    /**
     *  Converts data to the SYCL* buffer.
     *  \return one-dimentional SYCL* buffer.
     */
    inline cl::sycl::buffer<T, 1> toSycl() const
    {
        // TODO: Handle the case if _impl is empty
        return internal::BufferConverter<T>::toSycl(*_impl);
    }
#endif

    /**
     *   Returns the total number of elements in the buffer.
     */
    inline size_t size() const
    {
        // TODO: Handle the case if _impl is empty
        return _impl->size();
    }

    /**
     *   Creates a new Buffer object with memory allocated the
     *   same way as in the parent Buffer (from host pointer or SYCL* buffer).
     *   \param[in] size Number of elements to allocate
     *   \return Buffer with allocated memory
     */
    inline Buffer reallocate(size_t size)
    {
        return Buffer(_impl->allocateLike(size));
    }

    /**
     *   Drops underlying reference to the data from the buffer and makes it empty
     */
    inline void reset()
    {
        // TODO: Handle the case if _impl is empty
        _impl.reset();
    }

    /**
     *   Creates Buffer object that points to the same memory as a parent but with offset
     *   \param[in] offset Offset in elements from start of the parent buffer
     *   \param[in] size   Number of elements in the sub-buffer
     */
    inline Buffer<T> getSubBuffer(size_t offset, size_t size) const
    {
        return Buffer<T>(_impl->getSubBuffer(offset, size));
    }

private:
    explicit Buffer(internal::BufferImplTemplateIface<T> *impl) : _impl(impl) { }
    explicit Buffer(const SharedPtr< internal::BufferImplTemplateIface<T> >& impl) : _impl(impl) { }

    SharedPtr<internal::BufferImplTemplateIface<T> > _impl;
};

/** @} */
} // namespace interface1

using interface1::Buffer;

} // namespace services
} // namespace daal

#endif
