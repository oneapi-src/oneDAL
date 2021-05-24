/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_integral.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_fp.hpp"

namespace oneapi::dal::kmeans::backend {

namespace pr = dal::backend::primitives;
namespace bk = dal::backend;

template <typename Float>
auto update_clusters(sycl::queue& queue,
                     std::int64_t cluster_count,
                     std::int64_t max_iteration_count,
                     double accuracy_threshold,
                     std::int64_t block_rows,
                     std::int64_t part_count,
                     const pr::ndarray<Float, 2> data,
                     const pr::ndarray<Float, 2> initial_centroids,
                     pr::ndarray<Float, 2> centroids,
                     pr::ndarray<Float, 2> partial_centroids,
                     pr::ndarray<Float, 2> distance_block,
                     pr::ndarray<Float, 2> closest_distances,
                     pr::ndarray<Float, 1> objective_function,
                     pr::ndarray<std::int32_t, 2> labels,
                     pr::ndarray<std::int32_t, 1> counters,
                     pr::ndarray<std::int32_t, 1> candidate_indices,
                     pr::ndarray<Float, 1> candidate_distances,
                     pr::ndarray<std::int32_t, 1> empty_cluster_count,
                     const bk::event_vector& deps = {}) {
    auto assign_event = assign_clusters<Float, pr::squared_l2_metric<Float>>(queue,
                                                                             data,
                                                                             initial_centroids,
                                                                             block_rows,
                                                                             labels,
                                                                             distance_block,
                                                                             closest_distances,
                                                                             deps);
    auto count_event = count_clusters(queue, labels, cluster_count, counters, { assign_event });
    auto objective_function_event = compute_objective_function<Float>(queue,
                                                                      closest_distances,
                                                                      objective_function,
                                                                      { assign_event });
    auto reset_event = partial_centroids.fill(queue, 0.0);
    reset_event.wait_and_throw();
    auto centroids_event = partial_reduce_centroids<Float>(queue,
                                                           data,
                                                           labels,
                                                           cluster_count,
                                                           part_count,
                                                           partial_centroids,
                                                           { count_event });
    centroids_event = merge_reduce_centroids<Float>(queue,
                                                    counters,
                                                    partial_centroids,
                                                    part_count,
                                                    centroids,
                                                    { count_event, centroids_event });
    count_empty_clusters(queue, cluster_count, counters, empty_cluster_count, { count_event });

    std::int64_t candidate_count = empty_cluster_count.to_host(queue).get_data()[0];
    sycl::event find_candidates_event;
    if (candidate_count > 0) {
        find_candidates_event = find_candidates<Float>(queue,
                                                       closest_distances,
                                                       candidate_count,
                                                       candidate_indices,
                                                       candidate_distances);
    }
    Float objective_function_value = objective_function.to_host(queue).get_data()[0];
    bk::event_vector candidate_events;
    if (candidate_count > 0) {
        auto [updated_objective_function_value, copy_events] =
            fill_empty_clusters(queue,
                                data,
                                counters,
                                candidate_indices,
                                candidate_distances,
                                centroids,
                                labels,
                                objective_function_value,
                                { find_candidates_event });
        sycl::event::wait(copy_events);
        objective_function_value = updated_objective_function_value;
    }
    return std::make_tuple(objective_function_value, centroids_event);
}

} // namespace oneapi::dal::kmeans::backend
