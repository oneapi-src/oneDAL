/* file: service_utils.h */
/*******************************************************************************
* Copyright 2015-2017 Intel Corporation
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
//  Declaration of service utilities
//--
*/
#ifndef __SERVICE_UTILS_H__
#define __SERVICE_UTILS_H__

#include "env_detect.h"

namespace daal
{
namespace services
{
namespace internal
{

template<CpuType cpu, typename T>
inline void swap(T & x, T & y)
{
    T tmp = x;
    x = y;
    y = tmp;
}

template <CpuType cpu, typename ForwardIterator, typename T>
ForwardIterator lowerBound(ForwardIterator first, ForwardIterator last, const T & value)
{
    ForwardIterator it;
    auto count = last - first; // distance.
    while (count > 0)
    {
        it = first;
        const auto step = count / 2;
        it += step; // advance.
        if (*it < value)
        {
            first = ++it;
            count -= step + 1;
        }
        else
        {
            count = step;
        }
    }
    return first;
}

template <CpuType cpu, typename ForwardIterator, typename T, typename Compare>
ForwardIterator lowerBound(ForwardIterator first, ForwardIterator last, const T & value, Compare compare)
{
    ForwardIterator it;
    auto count = last - first; // distance.
    while (count > 0)
    {
        it = first;
        const auto step = count / 2;
        it += step; // advance.
        if (compare(*it, value))
        {
            first = ++it;
            count -= step + 1;
        }
        else
        {
            count = step;
        }
    }
    return first;
}

template <CpuType cpu, typename ForwardIterator, typename T>
ForwardIterator upperBound(ForwardIterator first, ForwardIterator last, const T & value)
{
    ForwardIterator it;
    auto count = last - first; // distance.
    while (count > 0)
    {
        it = first;
        const auto step = count / 2;
        it += step; // advance.
        if (!(value < *it))
        {
            first = ++it;
            count -= step + 1;
        }
        else
        {
            count = step;
        }
    }
    return first;
}

template <CpuType cpu, typename ForwardIterator, typename T, typename Compare>
ForwardIterator upperBound(ForwardIterator first, ForwardIterator last, const T & value, Compare compare)
{
    ForwardIterator it;
    auto count = last - first; // distance.
    while (count > 0)
    {
        it = first;
        const auto step = count / 2;
        it += step; // advance.
        if (!compare(value, *it))
        {
            first = ++it;
            count -= step + 1;
        }
        else
        {
            count = step;
        }
    }
    return first;
}

template <CpuType cpu, typename T>
inline const T & min(const T & a, const T & b) { return !(b < a) ? a : b; }

template <CpuType cpu, typename T>
inline const T & max(const T & a, const T & b) { return (a < b) ? b : a; }

template <CpuType cpu, typename BidirectionalIterator, typename Compare>
BidirectionalIterator partition(BidirectionalIterator first, BidirectionalIterator last, Compare compare)
{
    while (first != last)
    {
        while (compare(*first))
        {
            if (++first == last) return first;
        }
        do
        {
            if (--last == first) return first;
        } while (!compare(*last));
        swap<cpu>(*first, *last);
        ++first;
    }
    return first;
}

template <CpuType cpu, typename ForwardIterator>
ForwardIterator maxElement(ForwardIterator first, ForwardIterator last)
{
    if (first == last) { return last; }
    auto largest = first;
    while (++first != last)
    {
        if (*largest < *first) { largest = first; }
    }
    return largest;
}

template <CpuType cpu, typename ForwardIterator, typename Compare>
ForwardIterator maxElement(ForwardIterator first, ForwardIterator last, Compare compare)
{
    if (first == last) { return last; }
    auto largest = first;
    while (++first != last)
    {
        if (compare(*largest, *first)) { largest = first; }
    }
    return largest;
}

} // namespace internal
} // namespace services
} // namespace daal

#endif
