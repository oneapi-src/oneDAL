/* file: service_heap.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

/*
//++
//  Implementation of heap algorithms.
//--
*/

#ifndef __SERVICE_HEAP_H__
#define __SERVICE_HEAP_H__

#include "daal_defines.h"
#include "service_utils.h"

namespace daal
{
namespace algorithms
{
namespace internal
{

using namespace services::internal;

template <CpuType cpu, typename T> DAAL_FORCEINLINE T heapLeftChildIndex(T index) { return 2 * index + 1; }
template <CpuType cpu, typename T> DAAL_FORCEINLINE T heapRightChildIndex(T index) { return 2 * index + 2; }
template <CpuType cpu, typename T> DAAL_FORCEINLINE T heapParentIndex(T index) { return (index - 1) / 2; }

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

        if (largest == i) { break; }
        iterSwap<cpu>(first + i, first + largest);
    }
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

template <CpuType cpu, typename RandomAccessIterator, typename Compare>
void makeMaxHeap(RandomAccessIterator first, RandomAccessIterator last, Compare compare)
{
    const auto count = last - first;
    auto i = count / 2;
    while (0 < i) { internalAdjustMaxHeap<cpu>(first, last, count, --i, compare); }
}

template <CpuType cpu, typename RandomAccessIterator, typename Compare>
DAAL_FORCEINLINE void sortMaxHeap(RandomAccessIterator first, RandomAccessIterator last, Compare compare)
{
    while (1 < last - first)
    {
        popMaxHeap<cpu>(first, --last, compare);
    }
}

} // namespace internal
} // namespace algorithms
} // namespace daal

#endif
