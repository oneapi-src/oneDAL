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

#include "oneapi/dal/backend/communicator.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::kmeans::backend {

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;
namespace spmd = oneapi::dal::preview::spmd;

template <typename Float>
class centroid_candidates {
public:
    explicit centroid_candidates(const pr::ndarray<std::int32_t, 1>& indices,
                                 const pr::ndarray<Float, 1>& distances,
                                 const pr::ndarray<std::int32_t, 1>& empty_cluster_indices)
            : candidate_count_(indices.get_dimension(0)),
              indices_(indices),
              distances_(distances),
              empty_cluster_indices_(empty_cluster_indices) {
        ONEDAL_ASSERT(candidate_count_ > 0);
        ONEDAL_ASSERT(empty_cluster_indices.get_dimension(0) == candidate_count_);
        ONEDAL_ASSERT(distances.get_dimension(0) == candidate_count_);
    }

    std::int64_t get_candidate_count() const {
        return candidate_count_;
    }

    const pr::ndarray<std::int32_t, 1>& get_indices() const {
        return indices_;
    }

    const pr::ndarray<Float, 1>& get_distances() const {
        return distances_;
    }

    const pr::ndarray<std::int32_t, 1>& get_empty_cluster_indices() const {
        return empty_cluster_indices_;
    }

private:
    std::int64_t candidate_count_;
    pr::ndarray<std::int32_t, 1> indices_;
    pr::ndarray<Float, 1> distances_;
    pr::ndarray<std::int32_t, 1> empty_cluster_indices_;
};

template <typename Float>
auto find_candidates(sycl::queue& queue,
                     std::int64_t candidate_count,
                     const pr::ndarray<Float, 2>& closest_distances,
                     const pr::ndarray<std::int32_t, 1>& counters,
                     const bk::event_vector& deps = {})
    -> std::tuple<centroid_candidates<Float>, sycl::event>;

template <typename Float>
auto fill_empty_clusters(sycl::queue& queue,
                         bk::communicator<spmd::device_memory_access::usm>& comm,
                         const pr::ndview<Float, 2>& data,
                         const centroid_candidates<Float>& candidates,
                         pr::ndview<Float, 2>& centroids,
                         const bk::event_vector& deps = {}) -> sycl::event;

template <typename Float>
inline Float correct_objective_function(sycl::queue& queue,
                                        const centroid_candidates<Float>& candidates,
                                        const bk::event_vector& deps = {}) {
    sycl::event::wait_and_throw(deps);

    const auto& candidate_distances = candidates.get_distances();
    const std::int64_t candidate_count = candidate_distances.get_dimension(0);
    const auto host_candidate_distances = candidate_distances.to_host(queue);
    const Float* host_candidate_distances_ptr = host_candidate_distances.get_data();

    Float objective_function_correction = 0;
    for (std::int64_t i = 0; i < candidate_count; i++) {
        objective_function_correction -= host_candidate_distances_ptr[i];
    }

    return objective_function_correction;
}

/// Fills centroids that correspond to the empty clusters
///
/// @param[in] queue              The DPC++ queue
/// @param[in] candidate_count    The number of empty clusters need to bu filled
/// @param[in] data               The [n x p] array of all feature vectors
/// @param[in] closest_distances  The distance between each observation and closest centroid,
///                               value at i-th position is $\min_j d(x_i, c_j)$, where $x_i$ is
///                               observation and $c_j$ is centroid
/// @param[in] counters           The number of observations assigned to each cluster,
///                               value at i-th position indicates that i-th clusters
///                               consists of `counters[i]` observations
/// @param[out] centroids         The centroids of [k x p], where $k$ is the number of centroids,
///                               $p$ is the number of features.
/// @param[in] deps               The vectors of events need to be completed before start computations
///
/// @return The correction coefficient needs to be added to the value of the objective function
template <typename Float>
inline auto handle_empty_clusters(sycl::queue& queue,
                                  bk::communicator<spmd::device_memory_access::usm>& comm,
                                  std::int64_t candidate_count,
                                  const pr::ndview<Float, 2>& data,
                                  const pr::ndarray<Float, 2>& closest_distances,
                                  const pr::ndarray<std::int32_t, 1>& counters,
                                  pr::ndview<Float, 2>& centroids,
                                  const bk::event_vector& deps = {})
    -> std::tuple<Float, sycl::event> {
    ONEDAL_ASSERT(candidate_count > 0);
    ONEDAL_ASSERT(data.get_dimension(0) >= candidate_count);
    ONEDAL_ASSERT(closest_distances.get_dimension(0) == data.get_dimension(0));
    ONEDAL_ASSERT(counters.get_dimension(0) >= candidate_count);
    ONEDAL_ASSERT(centroids.get_dimension(0) == counters.get_dimension(0));
    ONEDAL_ASSERT(centroids.get_dimension(1) == data.get_dimension(1));

    auto [candidates, find_candidates_event] =
        find_candidates(queue, candidate_count, closest_distances, counters, deps);

    auto fill_event =
        fill_empty_clusters(queue, comm, data, candidates, centroids, { find_candidates_event });

    const Float correction =
        correct_objective_function(queue, candidates, { find_candidates_event });

    return { correction, fill_event };
}

} // namespace oneapi::dal::kmeans::backend
