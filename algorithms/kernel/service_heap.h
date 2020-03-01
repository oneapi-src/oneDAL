/* file: service_heap.h */
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

/*
//++
//  Implementation of heap algorithms.
//--
*/

#ifndef __SERVICE_HEAP_H__
#define __SERVICE_HEAP_H__

#include "daal_defines.h"
#include "service_utils.h"
#include "service_type_traits.h"

namespace daal
{
namespace algorithms
{
namespace internal
{
using namespace services::internal;

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

template <CpuType cpu, typename RandomAccessIterator, typename Diff, typename Compare>
DAAL_FORCEINLINE void internalAdjustMaxHeap(RandomAccessIterator first, RandomAccessIterator /*last*/, Diff count, Diff i, Compare compare)
{
    for (auto largest = i;; i = largest)
    {
        const auto l = heapLeftChildIndex<cpu>(i);
        if ((l < count) && compare(*(first + largest), *(first + l)))
        {
            largest = l;
        }
        const auto r = heapRightChildIndex<cpu>(i);
        if ((r < count) && compare(*(first + largest), *(first + r)))
        {
            largest = r;
        }

        if (largest == i)
        {
            break;
        }
        iterSwap<cpu>(first + i, first + largest);
    }
}

template <CpuType cpu, typename RandomAccessIterator, typename Diff>
DAAL_FORCEINLINE void internalAdjustMaxHeap(RandomAccessIterator first, RandomAccessIterator last, Diff count, Diff i)
{
    using Type            = typename RemovePointer<RandomAccessIterator>::type;
    const auto comparator = [](const Type & a, const Type & b) -> bool { return a < b; };
    internalAdjustMaxHeap(first, last, count, i, comparator);
}

template <CpuType cpu, typename RandomAccessIterator, typename Compare>
void popMaxHeap(RandomAccessIterator first, RandomAccessIterator last, Compare compare)
{
    if (1 < last - first)
    {
        --last;
        iterSwap<cpu>(first, last);
        internalAdjustMaxHeap<cpu>(first, last, last - first, first - first, compare);
    }
}

template <CpuType cpu, typename RandomAccessIterator>
void popMaxHeap(RandomAccessIterator first, RandomAccessIterator last)
{
    using Type            = typename RemovePointer<cpu, RandomAccessIterator>::type;
    const auto comparator = [](const Type & a, const Type & b) -> bool { return a < b; };
    popMaxHeap<cpu>(first, last, comparator);
}

template <CpuType cpu, typename RandomAccessIterator, typename Compare>
void makeMaxHeap(RandomAccessIterator first, RandomAccessIterator last, Compare compare)
{
    const auto count = last - first;
    auto i           = count / 2;
    while (0 < i)
    {
        internalAdjustMaxHeap<cpu>(first, last, count, --i, compare);
    }
}

template <CpuType cpu, typename RandomAccessIterator>
void makeMaxHeap(RandomAccessIterator first, RandomAccessIterator last)
{
    using Type            = typename RemovePointer<cpu, RandomAccessIterator>::type;
    const auto comparator = [](const Type & a, const Type & b) -> bool { return a < b; };
    makeMaxHeap<cpu>(first, last, comparator);
}

template <CpuType cpu, typename RandomAccessIterator, typename Compare>
DAAL_FORCEINLINE void sortMaxHeap(RandomAccessIterator first, RandomAccessIterator last, Compare compare)
{
    while (1 < last - first)
    {
        popMaxHeap<cpu>(first, --last, compare);
    }
}

template <CpuType cpu, typename RandomAccessIterator>
DAAL_FORCEINLINE void sortMaxHeap(RandomAccessIterator first, RandomAccessIterator last)
{
    using Type            = typename RemovePointer<cpu, RandomAccessIterator>::type;
    const auto comparator = [](const Type & a, const Type & b) -> bool { return a < b; };
    sortMaxHeap<cpu>(first, last, comparator);
}

/*
    Cut from kdtree_knn_classification_predict_dense_default_batch.i to provide more consistent access
*/

template <CpuType cpu, typename RandomAccessIterator, typename Addr>
DAAL_FORCEINLINE void maxHeapUpdate(RandomAccessIterator first, Addr & prev, const Addr i)
{
    *(first + prev) = *(first + i);
    prev            = i;
}

template <CpuType cpu, typename RandomAccessIterator, typename Compare>
void pushMaxHeap(RandomAccessIterator first, RandomAccessIterator last, Compare compare)
{
    auto i = last - first - 1;
    if (0 < i)
    {
        const auto newItem = *(last - 1);
        auto prev          = i;
        for (i = heapParentIndex<cpu>(i); i && compare(*(first + i), newItem); i = heapParentIndex<cpu>(i))
        {
            maxHeapUpdate<cpu, RandomAccessIterator, decltype(i)>(first, prev, i);
        }
        // Last iteration of cycle for case i == 0
        if (!i && (*(first + i) < newItem))
        {
            maxHeapUpdate<cpu, RandomAccessIterator, decltype(i)>(first, prev, i);
        }
        *(first + prev) = newItem;
    }
}

template <CpuType cpu, typename RandomAccessIterator>
void pushMaxHeap(RandomAccessIterator first, RandomAccessIterator last)
{
    using Type            = typename RemovePointer<cpu, RandomAccessIterator>::type;
    const auto comparator = [](const Type & a, const Type & b) -> bool { return a < b; };
    pushMaxHeap<cpu>(first, last, comparator);
}

} // namespace internal
} // namespace algorithms
} // namespace daal

#endif
