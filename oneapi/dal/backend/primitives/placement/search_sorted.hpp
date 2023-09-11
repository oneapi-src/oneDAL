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

#pragma once

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

enum class search_alignment : std::int64_t { left = 0b0l, right = 0b1l };

/// @brief Finds indices of bins to place points into
///
/// @tparam Type  Type of values to handle
/// @tparam Index Type of indices to return
/// @tparam clip  Should maximum index be capped
///               by the number of samples
///
/// @param[in]  queue       Queue to run kernels on
/// @param[in]  alignment   Lower (left) or upper (right)
///                         bound selector
/// @param[in]  data        Bin boundaries
/// @param[in]  points      Points to place into bins
/// @param[out] results     Output array
/// @param[in]  deps        Vector of dependencies
/// @return                 Indices of bins
template <typename Type, typename Index, bool clip = false>
inline sycl::event search_sorted_1d(sycl::queue& queue,
                                    search_alignment alignment,
                                    const ndview<Type, 1>& data,
                                    const ndview<Type, 1>& points,
                                    ndview<Index, 1>& results,
                                    const event_vector& deps = {}) {
    constexpr auto left = search_alignment::left;
    constexpr auto right = search_alignment::right;

    if (alignment == left) {
        return search_sorted_1d<search_alignment::left, Type, Index, clip>(queue,
                                                                           data,
                                                                           points,
                                                                           results,
                                                                           deps);
    }

    if (alignment == right) {
        return search_sorted_1d<search_alignment::right, Type, Index, clip>(queue,
                                                                            data,
                                                                            points,
                                                                            results,
                                                                            deps);
    }

    ONEDAL_ASSERT(false);
    return wait_or_pass(deps);
}

/// @brief Finds indices of bins to place points into
///
/// @tparam Type  Type of values to handle
/// @tparam Index Type of indices to return
///
/// @param[in]  queue       Queue to run kernels on
/// @param[in]  clip_result Should maximum index be capped
///                         by the number of samples
/// @param[in]  alignment   Lower (left) or upper (right)
///                         bound selector
/// @param[in]  data        Bin boundaries
/// @param[in]  points      Points to place into bins
/// @param[out] results     Output array
/// @param[in]  deps        Vector of dependencies
/// @return                 Indices of bins
template <typename Type, typename Index>
inline sycl::event search_sorted_1d(sycl::queue& queue,
                                    bool clip_result,
                                    search_alignment alignment,
                                    const ndview<Type, 1>& data,
                                    const ndview<Type, 1>& points,
                                    ndview<Index, 1>& results,
                                    const event_vector& deps = {}) {
    if (clip_result) {
        return search_sorted_1d<Type, Index, true>(queue, alignment, data, points, results, deps);
    }
    else {
        return search_sorted_1d<Type, Index, false>(queue, alignment, data, points, results, deps);
    }
}

/// @brief Finds indices of bins to place points into
///
/// @cite https://github.com/numpy/numpy/blob/maintenance/1.24.x/numpy/core/src/npysort/binsearch.cpp
///
/// @tparam Type      Type of values to handle
/// @tparam Index     Type of indices to return
/// @tparam clip      Should maximum index be capped
///                   by the number of samples
/// @tparam alignment Lower (left) or upper (right)
///                   bound selector
///
/// @param[in]  queue       Queue to run kernels on
/// @param[in]  data        Bin boundaries
/// @param[in]  points      Points to place into bins
/// @param[out] results     Output array
/// @param[in]  deps        Vector of dependencies
/// @return                 Indices of bins
template <search_alignment alignment, typename Type, typename Index, bool clip = false>
sycl::event search_sorted_1d(sycl::queue& queue,
                             const ndview<Type, 1>& data,
                             const ndview<Type, 1>& points,
                             ndview<Index, 1>& results,
                             const event_vector& deps = {});

#endif

} // namespace oneapi::dal::backend::primitives
