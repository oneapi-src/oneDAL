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

#include "services/daal_defines.h"
#include "service/kernel/service_utils.h"
#include "service/kernel/service_type_traits.h"

namespace daal
{
namespace algorithms
{
namespace internal
{
using namespace services::internal;

template <CpuType cpu>
struct Less
{
    template <typename Type>
    DAAL_FORCEINLINE bool operator()(const Type & left, const Type & right) const
    {
        return left < right;
    }
};

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

template <CpuType cpu, typename RandomAccessIterator, typename Diff, typename Compare = Less<cpu> >
DAAL_FORCEINLINE void internalAdjustMaxHeap(RandomAccessIterator first, RandomAccessIterator /*last*/, Diff count, Diff i,
                                            Compare compare = Less<cpu>())
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

template <CpuType cpu, typename RandomAccessIterator, typename Compare = Less<cpu> >
void popMaxHeap(RandomAccessIterator first, RandomAccessIterator last, Compare compare = Less<cpu>())
{
    if (1 < last - first)
    {
        --last;
        iterSwap<cpu>(first, last);
        internalAdjustMaxHeap<cpu, RandomAccessIterator>(first, last, last - first, first - first, compare);
    }
}

template <CpuType cpu, typename RandomAccessIterator, typename Compare = Less<cpu> >
void makeMaxHeap(RandomAccessIterator first, RandomAccessIterator last, Compare compare = Less<cpu>())
{
    const auto count = last - first;
    auto i           = count / 2;
    while (0 < i)
    {
        internalAdjustMaxHeap<cpu, RandomAccessIterator>(first, last, count, --i, compare);
    }
}

template <CpuType cpu, typename RandomAccessIterator, typename Compare = Less<cpu> >
DAAL_FORCEINLINE void sortMaxHeap(RandomAccessIterator first, RandomAccessIterator last, Compare compare = Less<cpu>())
{
    while (1 < last - first)
    {
        popMaxHeap<cpu, RandomAccessIterator>(first, --last, compare);
    }
}

template <CpuType cpu, typename RandomAccessIterator, typename Addr>
DAAL_FORCEINLINE void maxHeapUpdate(RandomAccessIterator first, Addr & prev, const Addr i)
{
    *(first + prev) = *(first + i);
    prev            = i;
}

template <CpuType cpu, typename RandomAccessIterator, typename Compare = Less<cpu> >
void pushMaxHeap(RandomAccessIterator first, RandomAccessIterator last, Compare compare = Less<cpu>())
{
    auto i = last - first - 1;
    if (0 < i)
    {
        const auto newItem = *(last - 1);
        auto prev          = i;
        for (i = heapParentIndex<cpu>(i); i && compare(*(first + i), newItem); i = heapParentIndex<cpu>(i))
        {
            maxHeapUpdate<cpu>(first, prev, i);
        }
        // Last iteration of cycle for case i == 0
        if (!i && (*(first + i) < newItem))
        {
            maxHeapUpdate<cpu>(first, prev, i);
        }
        *(first + prev) = newItem;
    }
}

} // namespace internal
} // namespace algorithms
} // namespace daal

#endif
