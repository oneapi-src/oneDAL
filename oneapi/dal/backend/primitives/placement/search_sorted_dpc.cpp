/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include <algorithm>
#include <functional>

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

#include "oneapi/dal/backend/primitives/placement/search_sorted.hpp"

namespace oneapi::dal::backend::primitives {

template <search_alignment alignment, typename Type>
struct comparator {};

template <typename Type>
struct comparator<search_alignment::left, Type> {
    constexpr static auto value = std::less<Type>{};
};

template <typename Type>
struct comparator<search_alignment::right, Type> {
    constexpr static auto value = std::less_equal<Type>{};
};

template <search_alignment alignment, typename Type>
constexpr auto comparator_v = comparator<alignment, Type>::value;

template <bool clip, typename Index>
inline Index clip_place(std::int64_t count, Index result) {
    if constexpr (clip) {
        return std::max(static_cast<Index>(0), std::min<Index>(result, count - 1));
    }
    else {
        return result;
    }
}

template <search_alignment alignment, typename Type, typename Index, bool clip>
sycl::event search_sorted_1d(sycl::queue& queue,
                             const ndview<Type, 1>& data,
                             const ndview<Type, 1>& points,
                             ndview<Index, 1>& results,
                             const event_vector& deps) {
    constexpr auto is_left = alignment == search_alignment::left;
    constexpr auto is_right = alignment == search_alignment::right;
    static_assert(is_left || is_right);

    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(points.has_data());
    ONEDAL_ASSERT(results.has_mutable_data());

    const auto data_count = data.get_count();
    const auto points_count = points.get_count();
    const auto range = make_range_1d(points_count);
    ONEDAL_ASSERT(points_count == results.get_count());

    const auto* const data_ptr = data.get_data();
    const auto* const points_ptr = points.get_data();
    auto* const results_ptr = results.get_mutable_data();

    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);

        h.parallel_for(range, [=](sycl::id<1> idx) {
            constexpr auto cmp = comparator_v<alignment, Type>;

            const auto target = points_ptr[idx];
            Index left_idx = 0, right_idx = static_cast<Index>(data_count);

            while (left_idx < right_idx) {
                const auto mid_idx = left_idx + (right_idx - left_idx) / 2;
                const auto mid_val = data_ptr[mid_idx];

                if (cmp(mid_val, target))
                    left_idx = mid_idx + 1;
                else
                    right_idx = mid_idx;
            }

            results_ptr[idx] = clip_place<clip>(data_count, left_idx);
        });
    });
}

#define INSTANTIATE(A, T, I, C)                                            \
    template sycl::event search_sorted_1d<A, T, I, C>(sycl::queue&,        \
                                                      const ndview<T, 1>&, \
                                                      const ndview<T, 1>&, \
                                                      ndview<I, 1>&,       \
                                                      const event_vector&);

#define INSTANTIATE_CLIP(A, T, I) \
    INSTANTIATE(A, T, I, true)    \
    INSTANTIATE(A, T, I, false)

#define INSTANTIATE_ALIGNMENT(T, I)                \
    INSTANTIATE_CLIP(search_alignment::left, T, I) \
    INSTANTIATE_CLIP(search_alignment::right, T, I)

#define INSTANTIATE_TYPE(I)         \
    INSTANTIATE_ALIGNMENT(float, I) \
    INSTANTIATE_ALIGNMENT(double, I)

INSTANTIATE_TYPE(std::int32_t)
INSTANTIATE_TYPE(std::int64_t)

} // namespace oneapi::dal::backend::primitives
