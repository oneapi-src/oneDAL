/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#pragma once

#include <algorithm>

namespace oneapi::dal::backend::primitives {

namespace detail {

struct compare {
    template <typename Type>
    bool operator()(const Type& lhs, const Type& rhs) const {
        return lhs < rhs;
    }
};

template <typename Index>
constexpr inline Index left_child(Index idx) {
    return 2 * idx + 1;
}

template <typename Index>
constexpr inline Index right_child(Index idx) {
    return 2 * idx + 2;
}

template <typename Index>
constexpr inline Index parent(Index idx) {
    return (idx - 1) / 2;
}

template <typename RandomIterator, typename Difference, typename Compare>
inline void adjust(RandomIterator first, Difference count, Difference i, Compare compare) {
    for (auto largest = i;; i = largest) {
        const auto left = left_child(i);
        const bool use_left = (left < count) && compare(*(first + largest), *(first + left));
        largest = use_left ? left : largest;

        const auto right = right_child(i);
        const bool use_right = (right < count) && compare(*(first + largest), *(first + right));
        largest = use_right ? right : largest;

        if (largest == i)
            break;

        std::iter_swap<RandomIterator>(first + i, first + largest);
    }
}

template <typename RandomIterator, typename Compare>
inline void pop_heap_impl(RandomIterator first, RandomIterator last, Compare compare) {
    if ((last - first) > 1) {
        --last;
        std::iter_swap(first, last);
        adjust(first, last - first, first - first, compare);
    }
}

template <typename RandomIterator, typename Compare>
inline void make_heap_impl(RandomIterator first, RandomIterator last, Compare compare) {
    const auto count = last - first;
    for (auto i = count / 2; i > 0; --i) {
        adjust(first, count, i - 1, compare);
    }
}

template <typename RandomIterator, typename Compare>
inline void sort_heap_impl(RandomIterator first, RandomIterator last, Compare compare) {
    while (last > first) {
        pop_heap_impl(first, last--, compare);
    }
}

template <typename T, typename RandomIterator, typename Difference, typename Compare>
inline void push_heap_impl(T value, RandomIterator first, Difference idx, Compare compare) {
    for (auto par = parent(idx); (idx > 0) && compare(*(first + par), value); par = parent(idx)) {
        *(first + idx) = std::move(*(first + par));
        idx = par;
    }
    *(first + idx) = std::move(value);
}

template <typename RandomIterator, typename Compare>
inline void push_heap_impl(RandomIterator first, RandomIterator last, Compare compare) {
    push_heap_impl(std::move(*(last - 1)), first, last - first - 1, compare);
}

} // namespace detail

template <typename RandomIterator, typename Compare = detail::compare>
inline void pop_heap(RandomIterator first, RandomIterator last, Compare compare = Compare{}) {
#ifdef __SYCL_DEVICE_ONLY__
    detail::pop_heap_impl<RandomIterator, Compare>(first, last, compare);
#else
    std::pop_heap<RandomIterator, Compare>(first, last, compare);
#endif
}

template <typename RandomIterator, typename Compare = detail::compare>
inline void make_heap(RandomIterator first, RandomIterator last, Compare compare = Compare{}) {
#ifdef __SYCL_DEVICE_ONLY__
    detail::make_heap_impl<RandomIterator, Compare>(first, last, compare);
#else
    std::make_heap<RandomIterator, Compare>(first, last, compare);
#endif
}

template <typename RandomIterator, typename Compare = detail::compare>
inline void sort_heap(RandomIterator first, RandomIterator last, Compare compare = Compare{}) {
#ifdef __SYCL_DEVICE_ONLY__
    detail::sort_heap_impl<RandomIterator, Compare>(first, last, compare);
#else
    std::sort_heap<RandomIterator, Compare>(first, last, compare);
#endif
}

template <typename RandomIterator, typename Compare = detail::compare>
inline void push_heap(RandomIterator first, RandomIterator last, Compare compare = Compare{}) {
#ifdef __SYCL_DEVICE_ONLY__
    detail::push_heap_impl<RandomIterator, Compare>(first, last, compare);
#else
    std::push_heap<RandomIterator, Compare>(first, last, compare);
#endif
}

template <typename T, typename RandomIterator, typename Compare = detail::compare>
inline void replace_first(T value,
                          RandomIterator first,
                          RandomIterator last,
                          Compare compare = Compare{}) {
    pop_heap(first, last, compare);
    *(last - 1) = std::move(value);
    push_heap(first, last, compare);
}

} // namespace oneapi::dal::backend::primitives
