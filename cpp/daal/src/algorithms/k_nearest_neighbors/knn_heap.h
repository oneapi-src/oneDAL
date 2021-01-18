/* file: knn_heap.h */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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

#ifndef __KNN_HEAP_H__
#define __KNN_HEAP_H__

namespace daal
{
namespace internal
{
using namespace daal::services::internal;
using namespace daal::services;
using namespace daal::internal;

template <CpuType cpu, typename T>
DAAL_FORCEINLINE T heapLeftChildIndex(T index)
{
    return 2 * index + 1;
}
template <CpuType cpu, typename T>
DAAL_FORCEINLINE T heapRightChildIndex(T index)
{
    return 2 * index + 2;
}
template <CpuType cpu, typename T>
DAAL_FORCEINLINE T heapParentIndex(T index)
{
    return (index - 1) / 2;
}

template <CpuType cpu, typename RandomAccessIterator>
void pushMaxHeap(RandomAccessIterator first, RandomAccessIterator last)
{
    if (first != last)
    {
        --last;
        auto i = last - first;
        if (i > 0)
        {
            const auto newItem = *last; // It can be moved instead.
            auto prev          = i;
            for (i = heapParentIndex<cpu>(i); prev && (*(first + i) < newItem); i = heapParentIndex<cpu>(i))
            {
                *(first + prev) = *(first + i); // It can be moved instead.
                prev            = i;
            }
            *(first + prev) = newItem; // It can be moved instead.
        }
    }
}

template <CpuType cpu, typename RandomAccessIterator, typename Diff>
DAAL_FORCEINLINE void internalAdjustMaxHeap(RandomAccessIterator first, RandomAccessIterator /*last*/, Diff count, Diff i)
{
    for (auto largest = i;; i = largest)
    {
        const auto l = heapLeftChildIndex<cpu>(i);
        if ((l < count) && (*(first + largest) < *(first + l)))
        {
            largest = l;
        }
        const auto r = heapRightChildIndex<cpu>(i);
        if ((r < count) && (*(first + largest) < *(first + r)))
        {
            largest = r;
        }

        if (largest == i)
        {
            break;
        }
        auto temp          = *(first + i);
        *(first + i)       = *(first + largest);
        *(first + largest) = temp; // Moving can be used instead.
    }
}

template <CpuType cpu, typename RandomAccessIterator>
void popMaxHeap(RandomAccessIterator first, RandomAccessIterator last)
{
    if (1 < last - first)
    {
        --last;
        auto temp = *first;
        *first    = *last;
        *last     = temp; // Moving can be used instead.
        internalAdjustMaxHeap<cpu>(first, last, last - first, first - first);
    }
}

template <CpuType cpu, typename RandomAccessIterator>
void makeMaxHeap(RandomAccessIterator first, RandomAccessIterator last)
{
    const auto count = last - first;
    auto i           = count / 2;
    while (0 < i)
    {
        internalAdjustMaxHeap<cpu>(first, last, count, --i);
    }
}

template <typename T, CpuType cpu>
class Heap
{
public:
    Heap() : _elements(nullptr), _count(0) {}

    ~Heap()
    {
        service_scalable_free<T, cpu>(_elements);
        _elements = nullptr;
    }

    bool init(size_t size)
    {
        _count = 0;
        if (!_elements)
        {
            _elements = static_cast<T *>(service_scalable_malloc<T, cpu>(size));
        }

        return _elements;
    }

    void clear()
    {
        if (_elements)
        {
            service_scalable_free<T, cpu>(_elements);
            _elements = nullptr;
        }
    }

    void reset() { _count = 0; }

    void push(const T & e, size_t k)
    {
        _elements[_count++] = e;
        if (_count == k)
        {
            makeMaxHeap<cpu>(_elements, _elements + _count);
        }
    }

    void replaceMax(const T & e)
    {
        popMaxHeap<cpu>(_elements, _elements + _count);
        _elements[_count - 1] = e;
        pushMaxHeap<cpu>(_elements, _elements + _count);
    }

    void replaceMaxIfNeeded(const T & e, size_t k)
    {
        if (_count < k)
        {
            _elements[_count++] = e;
            pushMaxHeap<cpu>(_elements, _elements + _count);
        }
        else
        {
            if (e.distance < getMax()->distance)
            {
                replaceMax(e);
            }
        }
    }

    size_t size() const { return _count; }

    T * getMax() { return _elements; }

    const T & operator[](size_t index) const { return *(_elements + index); }

private:
    T * _elements;
    size_t _count;
};

template <typename algorithmFpType, CpuType cpu>
struct GlobalNeighbors
{
    algorithmFpType distance;
    size_t index;

    inline bool operator<(const GlobalNeighbors & rhs) const { return (distance < rhs.distance); }
};

} // namespace internal
} // namespace daal

#endif
