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

/// @brief Computes cumulative sum aka inclusive scan
///        on array with the time complexity of
///        $O(N \log_\text{wg} N / \text{wg})$
///        and no additional allocations
///
/// @tparam Type Type of values to handle
///
/// @param[in]      queue       SYCL queue to run kernel on
/// @param[in, out] data        Both input and output of primitive
/// @param[in]      base_stride Default size of a block. Should be
///                             smaller or equal to the largest wg
/// @param[in]      deps        Dependencies for this kernel
/// @return                     SYCL event to track execution
template <typename Type>
sycl::event cumulative_sum_1d(sycl::queue& queue,
                              ndview<Type, 1>& data,
                              std::int64_t base_stride,
                              const event_vector& deps = {});

/// @brief Computes cumulative sum aka inclusive scan
///        on array with the time complexity of
///        $O(N \log_\text{wg} N / \text{wg})$
///        and no additional allocations
///
/// @tparam Type Type of values to handle
///
/// @param[in]      queue       SYCL queue to run kernel on
/// @param[in, out] data        Both input and output of primitive
/// @param[in]      deps        Dependencies for this kernel
/// @return                     SYCL event to track execution
template <typename Type>
sycl::event cumulative_sum_1d(sycl::queue& queue,
                              ndview<Type, 1>& data,
                              const event_vector& deps = {});

namespace detail {

/// @brief Computes partial cumulative sums inplace
///        on the `data` array assuming elements have
///        `curr_stride` stribe between them and size
///        of each block is equal to `base_stride`.
/// @note  Usually, `base_stride== wg`.
/// @note  `base_stride <= curr_stride` always
template <typename Type>
sycl::event block_cumsum(sycl::queue& queue,
                         ndview<Type, 1>& data,
                         std::int64_t base_stride,
                         std::int64_t curr_stride,
                         const event_vector& deps = {});

/// @brief Distributes partially computed cumulative
///        sum values to other elements.
/// @note  Usually, `base_stride== wg`.
/// @note  `base_stride <= curr_stride` always
template <typename Type>
sycl::event distribute_sum(sycl::queue& queue,
                           ndview<Type, 1>& data,
                           std::int64_t base_stride,
                           std::int64_t curr_stride,
                           const event_vector& deps = {});

} // namespace detail

#endif

} // namespace oneapi::dal::backend::primitives
