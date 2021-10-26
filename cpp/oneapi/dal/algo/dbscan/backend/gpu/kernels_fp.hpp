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

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::dbscan::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

template <typename Float>
struct kernels_fp {
    static std::int64_t get_block_size_in_rows(sycl::queue& queue, std::int64_t column_count);
    static std::int64_t get_part_count_for_partial_centroids(sycl::queue& queue,
                                                             std::int64_t column_count,
                                                             std::int64_t cluster_count);
    template <bool use_weights>
    static sycl::event get_cores_impl(sycl::queue& queue,
                                      const pr::ndview<Float, 2>& data,
                                      const pr::ndview<Float, 2>& weights,
                                      pr::ndview<std::int32_t, 1>& cores,
                                      Float epsilon,
                                      std::int64_t min_observations,
                                      std::int64_t block_start,
                                      std::int64_t block_end,
                                      const bk::event_vector& deps);
    static sycl::event get_cores(sycl::queue& queue,
                                 const pr::ndview<Float, 2>& data,
                                 const pr::ndview<Float, 2>& weights,
                                 pr::ndview<std::int32_t, 1>& cores,
                                 Float epsilon,
                                 std::int64_t min_observations,
                                 std::int64_t block_start = -1,
                                 std::int64_t block_end = -1,
                                 const bk::event_vector& deps = {});
    static std::int32_t start_next_cluster(sycl::queue& queue,
                                           const pr::ndview<std::int32_t, 1>& cores,
                                           pr::ndview<std::int32_t, 1>& responses,
                                           const bk::event_vector& deps = {});

    static sycl::event update_queue(sycl::queue& queue,
                                    const pr::ndview<Float, 2>& data,
                                    const pr::ndview<std::int32_t, 1>& cores,
                                    pr::ndview<std::int32_t, 1>& algo_queue,
                                    std::int32_t queue_begin,
                                    std::int32_t queue_end,
                                    pr::ndview<std::int32_t, 1>& responses,
                                    pr::ndview<std::int32_t, 1>& queue_front,
                                    Float epsilon,
                                    std::int32_t cluster_id,
                                    std::int64_t block_start = -1,
                                    std::int64_t block_end = -1,
                                    const bk::event_vector& deps = {});
    static void set_queue_front_and_value(sycl::queue& queue,
                                          pr::ndarray<std::int32_t, 1>& arr_queue,
                                          pr::ndarray<std::int32_t, 1>& queue_front,
                                          std::int32_t value,
                                          std::int32_t cluster_index);
    static void set_queue_front(sycl::queue& queue,
                                pr::ndarray<std::int32_t, 1>& queue_front,
                                std::int64_t value);
    static std::int32_t get_queue_front(sycl::queue& queue,
                                        const pr::ndarray<std::int32_t, 1>& queue_front);
};

void set_queue_ptr(sycl::queue& queue,
                   pr::ndview<std::int32_t, 1>& algo_queue,
                   pr::ndview<std::int32_t, 1>& queue_front,
                   std::int32_t start_index,
                   const bk::event_vector& deps = {});
void set_arr_value(sycl::queue& queue,
                   pr::ndview<std::int32_t, 1>& arr,
                   std::int32_t offset,
                   std::int32_t value,
                   const bk::event_vector& deps = {});
std::int64_t count_cores(sycl::queue& queue, const pr::ndview<std::int32_t, 1>& cores);
#endif

} // namespace oneapi::dal::dbscan::backend
