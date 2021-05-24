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

#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_fp.hpp"

namespace oneapi::dal::kmeans::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;
#define INSTANTIATE_WITH_METRIC(F, M)                                                  \
    template sycl::event assign_clusters<F, M<F>>(sycl::queue & queue,                 \
                                                  const pr::ndview<F, 2>& data,        \
                                                  const pr::ndview<F, 2>& centroids,   \
                                                  std::int64_t block_rows,             \
                                                  pr::ndview<std::int32_t, 2>& labels, \
                                                  pr::ndview<F, 2>& distances,         \
                                                  pr::ndview<F, 2>& closest_distances, \
                                                  const bk::event_vector& deps);

#define INSTANTIATE(F) INSTANTIATE_WITH_METRIC(F, pr::squared_l2_metric) \
    template bk::event_vector fill_empty_clusters(             \
        sycl::queue& queue,                                    \
        const pr::ndview<F, 2>& data,                          \
        const pr::ndarray<std::int32_t, 1>& counters,          \
        const pr::ndarray<std::int32_t, 1>& candidate_indices, \
        const pr::ndarray<F, 1>& candidate_distances,          \
        pr::ndview<F, 2>& centroids,                           \
        pr::ndarray<std::int32_t, 2>& labels,                  \
        F& objective_function,                                 \
        const bk::event_vector& deps);                         \
    template sycl::event find_candidates<F>(sycl::queue & queue,                            \
                                            const pr::ndview<F, 2>& closest_distances,      \
                                            std::int64_t candidate_count,                   \
                                            pr::ndview<std::int32_t, 1>& candidate_indices, \
                                            pr::ndview<F, 1>& candidate_distances,          \
                                            const bk::event_vector& deps);                  \
    template sycl::event merge_reduce_centroids<F>(sycl::queue & queue,                         \
                                                   const pr::ndview<std::int32_t, 1>& counters, \
                                                   const pr::ndview<F, 2>& partial_centroids,   \
                                                   std::int64_t part_count,                     \
                                                   pr::ndview<F, 2>& centroids,                 \
                                                   const bk::event_vector& deps);               \
    template std::int64_t get_block_size_in_rows<F>(sycl::queue & queue,                        \
                                                    std::int64_t column_count);                 \
    template std::int64_t get_part_count_for_partial_centroids<F>(sycl::queue & queue,          \
                                                                  std::int64_t column_count,    \
                                                                  std::int64_t cluster_count);  \
    template sycl::event partial_reduce_centroids<F>(sycl::queue & queue,                       \
                                                     const pr::ndview<F, 2>& data,              \
                                                     const pr::ndview<std::int32_t, 2>& labels, \
                                                     std::int64_t cluster_count,                \
                                                     std::int64_t part_count,                   \
                                                     pr::ndview<F, 2>& partial_centroids,       \
                                                     const bk::event_vector& deps);             \
    template sycl::event compute_objective_function<F>(sycl::queue & queue,                       \
                                                       const pr::ndview<F, 2>& closest_distances, \
                                                       pr::ndview<F, 1>& objective_function,      \
                                                       const bk::event_vector& deps);

#endif

} // namespace oneapi::dal::kmeans::backend
