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
//#include "oneapi/dal/backend/primitives/selection/kselect_by_rows.hpp"
#include "oneapi/dal/backend/primitives/distance.hpp"
//#include "oneapi/dal/backend/primitives/sort/sort.hpp"
//#include "oneapi/dal/algo/kmeans/common.hpp"

namespace oneapi::dal::kmeans::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace prm = dal::backend::primitives;

template <typename Float>
sycl::event find_candidates_impl(sycl::queue& queue,
                                 prm::ndview<Float, 2>& closest_distances,
                                 std::int64_t num_candidates,
                                 prm::ndview<std::int32_t, 1>& candidate_indices,
                                 prm::ndview<Float, 1>& candidate_distances,
                                 const bk::event_vector& deps = {});

template <typename Float, typename Metric>
sycl::event assign_clusters_impl(sycl::queue& queue,
                                 const prm::ndview<Float, 2>& data,
                                 const prm::ndview<Float, 2>& centroids,
                                 std::int64_t block_rows,
                                 prm::ndview<std::int32_t, 2>& labels,
                                 prm::ndview<Float, 2>& distances,
                                 prm::ndview<Float, 2>& closest_distances,
                                 const bk::event_vector& deps = {});

template <typename Float>
sycl::event reduce_centroids_impl(sycl::queue& queue,
                                  const prm::ndview<Float, 2>& data,
                                  const prm::ndview<std::int32_t, 2>& labels,
                                  const prm::ndview<Float, 2>& centroids,
                                  const prm::ndview<Float, 2>& partial_centroids,
                                  const prm::ndview<std::int32_t, 1>& counters,
                                  const std::uint32_t num_parts,
                                  const bk::event_vector& deps = {});

sycl::event count_clusters_impl(sycl::queue& queue,
                                const prm::ndview<std::int32_t, 2>& labels,
                                std::int64_t num_centroids,
                                prm::ndview<std::int32_t, 1>& counters,
                                prm::ndarray<std::int32_t, 1> num_empty_clusters,
                                const bk::event_vector& deps = {});

template <typename Float>
sycl::event compute_objective_function_impl(sycl::queue& queue,
                                            const prm::ndview<Float, 2>& closest_distances,
                                            prm::ndarray<Float, 1> objective_function,
                                            const bk::event_vector& deps = {});

template <typename Float>
bk::event_vector fill_empty_clusters_impl(sycl::queue& queue,
                                          const prm::ndview<Float, 2>& data,
                                          const prm::ndarray<std::int32_t, 1>& counters,
                                          const prm::ndarray<std::int32_t, 1>& candidate_indices,
                                          const prm::ndarray<Float, 1>& candidate_distances,
                                          prm::ndview<Float, 2>& centroids,
                                          prm::ndarray<std::int32_t, 2>& labels,
                                          Float& objective_function,
                                          const bk::event_vector& deps = {});
#endif

} // namespace oneapi::dal::kmeans::backend
