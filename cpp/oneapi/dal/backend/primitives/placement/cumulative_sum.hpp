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

template <typename Type>
sycl::event cumulative_sum_1d(sycl::queue& queue,
                              ndview<Type, 1>& data,
                              std::int64_t base_stride,
                              const event_vector& deps = {});

template <typename Type>
sycl::event cumulative_sum_1d(sycl::queue& queue,
                              ndview<Type, 1>& data,
                              const event_vector& deps = {});

namespace detail {

template <typename Type>
sycl::event block_cumsum(sycl::queue& queue,
                         ndview<Type, 1>& data,
                         std::int64_t base_stride,
                         std::int64_t curr_stride,
                         const event_vector& deps = {});

template <typename Type>
sycl::event distribute_sum(sycl::queue& queue,
                           ndview<Type, 1>& data,
                           std::int64_t base_stride,
                           std::int64_t curr_stride,
                           const event_vector& deps = {});

} // namespace detail

#endif

} // namespace oneapi::dal::backend::primitives
