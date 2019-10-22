/* file: buffer_impl.h */
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

#ifndef __DAAL_SERVICES_INTERNAL_BUFFER_IMPL_H__
#define __DAAL_SERVICES_INTERNAL_BUFFER_IMPL_H__

#include "services/daal_shared_ptr.h"
#include "data_management/data/numeric_types.h"
#include "services/internal/error_handling_helpers.h"

namespace daal
{
namespace services
{
namespace internal
{

template <typename T> class HostBuffer;
template <typename T> class UsmBufferIface;
template <typename T> class SyclBufferIface;

template <typename T>
class BufferVisitor : public Base {
 public:
  virtual void operator()(const HostBuffer<T> &bufferImpl) = 0;
  virtual void operator()(const UsmBufferIface<T> &bufferImpl) = 0;
  virtual void operator()(const SyclBufferIface<T> &bufferImpl) = 0;
};

template <typename T>
class BufferIface
{
public:
    virtual ~BufferIface() { }
    virtual size_t size() const = 0;
    virtual void apply(BufferVisitor<T> &visitor) const = 0;
    virtual BufferIface<T> *getSubBuffer(size_t offset, size_t size) const = 0;
};

template <typename T>
class ConvertableToHostIface
{
public:
    virtual ~ConvertableToHostIface() { }
    virtual SharedPtr<T> getHostRead(Status *status = NULL) const = 0;
    virtual SharedPtr<T> getHostWrite(Status *status = NULL) const = 0;
    virtual SharedPtr<T> getHostReadWrite(Status *status = NULL) const = 0;
};

template <typename T>
class UsmBufferIface : public BufferIface<T>,
                       public ConvertableToHostIface<T> { };

template <typename T>
class SyclBufferIface : public BufferIface<T>,
                        public ConvertableToHostIface<T> { };

template <typename T>
class HostBuffer : public Base,
                   public BufferIface<T>
{
public:
    explicit HostBuffer(const SharedPtr<T> &data, size_t size)
        : _data(data), _size(size) { }

    explicit HostBuffer(T* data, size_t size)
        : _data(data, services::EmptyDeleter()), _size(size) { }

    size_t size() const DAAL_C11_OVERRIDE
    { return _size; }

    void apply(BufferVisitor<T> &visitor) const DAAL_C11_OVERRIDE
    { visitor(*this); }

    HostBuffer<T> *getSubBuffer(size_t offset, size_t size) const DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT( offset + size <= _size );
        return new HostBuffer<T>(SharedPtr<T>(_data, _data.get() + offset), size);
    }

    const SharedPtr<T> &get() const
    { return _data; }

private:
    SharedPtr<T> _data;
    size_t _size;
};


template<typename T>
class ConvertToHostSharedPtr : public BufferVisitor<T>
{
public:
    explicit ConvertToHostSharedPtr(const data_management::ReadWriteMode& rwFlag)
        : _rwFlag(rwFlag) { }

    void operator()(const HostBuffer<T> &bufferImpl) DAAL_C11_OVERRIDE
    { _hostSharedPtr = bufferImpl.get(); }

    void operator()(const UsmBufferIface<T> &bufferImpl) DAAL_C11_OVERRIDE
    { _hostSharedPtr = convertToHost(bufferImpl); }

    void operator()(const SyclBufferIface<T> &bufferImpl) DAAL_C11_OVERRIDE
    { _hostSharedPtr = convertToHost(bufferImpl); }

    const SharedPtr<T> &getHostPtr() const
    { return _hostSharedPtr; }

    Status getStatus() const
    { return _status; }

private:
    template <typename BufferImpl>
    SharedPtr<T> convertToHost(const BufferImpl &buffer)
    {
        using namespace daal::data_management;
        switch (_rwFlag)
        {
            case readOnly:  return buffer.getHostRead(&_status);
            case writeOnly: return buffer.getHostWrite(&_status);
            case readWrite: return buffer.getHostReadWrite(&_status);
        }
        DAAL_ASSERT(!"Not implemented read-write mode");
        return SharedPtr<T>();
    }

    Status _status;
    SharedPtr<T> _hostSharedPtr;
    data_management::ReadWriteMode _rwFlag;
};

template <typename T>
class HostBufferConverter
{
public:
    SharedPtr<T> toHost(const internal::BufferIface<T> &bufferImpl,
                        const data_management::ReadWriteMode& rwMode,
                        Status *status = NULL)
    {
        ConvertToHostSharedPtr<T> action(rwMode);
        bufferImpl.apply(action);
        tryAssignStatusAndThrow(status, action.getStatus());
        return action.getHostPtr();
    }
};

} // namespace internal
} // namespace services
} // namespace daal

#endif
