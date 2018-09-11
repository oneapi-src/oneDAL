/* file: service_arrays.h */
/*******************************************************************************
* Copyright 2015-2018 Intel Corporation.
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

#ifndef __SERVICE_ARRAYS_H__
#define __SERVICE_ARRAYS_H__

#include "service_allocators.h"

namespace daal
{
namespace services
{
namespace internal
{

template<typename T, typename Allocator, typename ConstructionPolicy, CpuType cpu>
class DynamicArray
{
public:
    DAAL_NEW_DELETE();

    DynamicArray() : _data(nullptr), _size(0) { }

    explicit DynamicArray(size_t size)
    {
        allocate(size);
    }

    DynamicArray(DynamicArray &&other)
    {
        moveImpl(other);
    }

    ~DynamicArray()
    {
        destroy();
    }

    DynamicArray &operator = (DynamicArray &&other)
    {
        moveImpl(other);
        return *this;
    }

    inline T &operator [] (size_t index)
    {
        return _data[index];
    }

    inline const T &operator [] (size_t index) const
    {
        return _data[index];
    }

    inline       T* get()       { return _data; }
    inline const T* get() const { return _data; }

    inline size_t size() const { return _size; }

    inline T *reset(size_t size = 0)
    {
        destroy();
        allocate(size);
        return _data;
    }

    DynamicArray(const DynamicArray &) = delete;
    DynamicArray &operator = (const DynamicArray &) = delete;

private:
    void allocate(size_t size)
    {
        _data = (size) ? Allocator::allocate(size) : nullptr;
        _size = 0;

        if (_data)
        {
            ConstructionPolicy::construct(_data, _data + size);
            _size = size;
        }
    }

    void destroy()
    {
        if (_data)
        {
            ConstructionPolicy::destroy(_data, _data + _size);
            Allocator::deallocate(_data);
        }

        _data = nullptr;
        _size = 0;
    }

    void moveImpl(DynamicArray &&other)
    {
        _data = other._data;
        _size = other._size;

        other._data = nullptr;
        other._size = 0;
    }

private:
    T *_data;
    size_t _size;
};

template<typename T, CpuType cpu, typename ConstructionPolicy = DefaultConstructionPolicy<T, cpu>>
using TArray = DynamicArray<T, DAALMalloc<T, cpu>, ConstructionPolicy, cpu>;

template<typename T, CpuType cpu, typename ConstructionPolicy = DefaultConstructionPolicy<T, cpu>>
using TArrayCalloc = DynamicArray<T, DAALCalloc<T, cpu>, ConstructionPolicy, cpu>;

template<typename T, CpuType cpu, typename ConstructionPolicy = DefaultConstructionPolicy<T, cpu>>
using TArrayScalable = DynamicArray<T, ScalableMalloc<T, cpu>, ConstructionPolicy, cpu>;

template<typename T, CpuType cpu, typename ConstructionPolicy = DefaultConstructionPolicy<T, cpu>>
using TArrayScalableCalloc = DynamicArray<T, ScalableCalloc<T, cpu>, ConstructionPolicy, cpu>;


template<typename T, size_t staticBufferSize, typename Allocator, typename ConstructionPolicy, CpuType cpu>
class StaticallyBufferedDynamicArray
{
public:
    StaticallyBufferedDynamicArray() : _data(nullptr), _size(0) { }

    explicit StaticallyBufferedDynamicArray(size_t size)
    {
        allocate(size);
    }

    ~StaticallyBufferedDynamicArray() { destroy(); }

    StaticallyBufferedDynamicArray(const StaticallyBufferedDynamicArray&) = delete;
    StaticallyBufferedDynamicArray &operator = (const StaticallyBufferedDynamicArray &) = delete;

    inline       T* get()       { return _data; }
    inline const T* get() const { return _data; }

    inline size_t size() const { return _size; }

    inline T *reset(size_t size = 0)
    {
        destroy();
        allocate(size);
        return _data;
    }

    inline T &operator [] (size_t index)
    {
        return _data[index];
    }

    inline const T &operator [] (size_t index) const
    {
        return _data[index];
    }

private:
    void allocate(size_t size)
    {
        _size = 0;

        if (size <= staticBufferSize)
        {
            _data = _buffer;
        }
        else
        {
            _data = (size) ? Allocator::allocate(size) : nullptr;
        }

        if (_data && size)
        {
            ConstructionPolicy::construct(_data, _data + size);
            _size = size;
        }
    }

    void destroy()
    {
        if (_data)
        {
            ConstructionPolicy::destroy(_data, _data + _size);
            if (_data != _buffer)
            {
                Allocator::deallocate(_data);
            }
        }

        _data = nullptr;
        _size = 0;
    }

private:
    T _buffer[staticBufferSize];
    T *_data;
    size_t _size;
};

template<typename T, size_t staticBufferSize, CpuType cpu, typename ConstructionPolicy = DefaultConstructionPolicy<T, cpu>>
using TNArray = StaticallyBufferedDynamicArray<T, staticBufferSize, DAALMalloc<T, cpu>, ConstructionPolicy, cpu>;

} // namespace internal
} // namespace services
} // namespace daal

#endif
