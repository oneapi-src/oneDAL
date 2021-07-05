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
#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_integral.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_fp.hpp"

namespace oneapi::dal::kmeans::backend {

namespace pr = dal::backend::primitives;
namespace bk = dal::backend;

template <typename Float>
class cluster_updater {
public:
    cluster_updater(const sycl::queue& q, const dal::backend::spmd_communicator& comm)
            : queue_(q),
              comm_(comm) {}

    auto& set_cluster_count(std::int64_t cluster_count) {
        cluster_count_ = cluster_count;
        return *this;
    }

    auto& set_accuracy_threshold(double accuracy_threshold) {
        accuracy_threshold_ = accuracy_threshold;
        return *this;
    }

    auto& set_part_count(std::int64_t part_count) {
        part_count_ = part_count;
        return *this;
    }

    auto& set_data(const pr::ndarray<Float, 2> data) {
        data_ = data;
        row_count_ = data.get_dimension(0);
        column_count_ = data.get_dimension(1);
        return *this;
    }

    auto& set_initial_centroids(const pr::ndarray<Float, 2> initial_centroids) {
        initial_centroids_ = initial_centroids;
        return *this;
    }

    void allocate_buffers() {
        partial_centroids_ =
            pr::ndarray<Float, 2>::empty(queue_,
                                         { part_count_ * cluster_count_, column_count_ },
                                         sycl::usm::alloc::device);
        counters_ =
            pr::ndarray<std::int32_t, 1>::empty(queue_, cluster_count_, sycl::usm::alloc::device);
        candidate_indices_ =
            pr::ndarray<std::int32_t, 1>::empty(queue_, cluster_count_, sycl::usm::alloc::device);
        candidate_distances_ =
            pr::ndarray<Float, 1>::empty(queue_, cluster_count_, sycl::usm::alloc::device);
        empty_cluster_count_ =
            pr::ndarray<std::int32_t, 1>::empty(queue_, 1, sycl::usm::alloc::device);
        empty_cluster_indices_ =
            pr::ndarray<std::int32_t, 1>::empty(queue_, cluster_count_, sycl::usm::alloc::device);
    }

    auto update(pr::ndarray<Float, 2>& centroids,
                pr::ndarray<Float, 2>& distance_block,
                pr::ndarray<Float, 2>& closest_distances,
                pr::ndarray<Float, 1>& objective_function,
                pr::ndarray<std::int32_t, 2>& labels,
                const bk::event_vector& deps = {}) {
        ONEDAL_ASSERT(data_.get_dimension(0) == row_count_);
        ONEDAL_ASSERT(data_.get_dimension(1) == column_count_);
        ONEDAL_ASSERT(closest_distances.get_dimension(0) == row_count_);
        ONEDAL_ASSERT(partial_centroids_.get_dimension(0) == part_count_ * cluster_count_);
        ONEDAL_ASSERT(partial_centroids_.get_dimension(1) == column_count_);
        ONEDAL_ASSERT(counters_.get_dimension(0) == cluster_count_);
        ONEDAL_ASSERT(candidate_indices_.get_dimension(0) == cluster_count_);
        ONEDAL_ASSERT(candidate_distances_.get_dimension(0) == cluster_count_);
        ONEDAL_ASSERT(empty_cluster_count_.get_dimension(0) == 1);
        ONEDAL_ASSERT(distance_block.get_dimension(1) == cluster_count_);
        ONEDAL_ASSERT(labels.get_dimension(0) == row_count_);
        ONEDAL_ASSERT(labels.get_dimension(1) == 1);
        ONEDAL_ASSERT(objective_function.get_dimension(0) == 1);
        ONEDAL_ASSERT(centroids.get_dimension(0) == cluster_count_);
        ONEDAL_ASSERT(centroids.get_dimension(1) == column_count_);

        const auto block_size_in_rows = distance_block.get_dimension(0);
        auto assign_event =
            kernels_fp<Float>::template assign_clusters<pr::squared_l2_metric<Float>>(
                queue_,
                data_,
                initial_centroids_,
                block_size_in_rows,
                labels,
                distance_block,
                closest_distances,
                deps);

        auto count_event =
            count_clusters(queue_, labels, cluster_count_, counters_, { assign_event });
        auto count_reduce_request = comm_.allreduce(counters_.flatten(queue_), { count_event });

        auto objective_function_event =
            kernels_fp<Float>::compute_objective_function(queue_,
                                                          closest_distances,
                                                          objective_function,
                                                          { assign_event });
        Float objective_function_value = objective_function.to_host(queue_).get_data()[0];
        auto objective_function_request = comm_.allreduce(objective_function_value);

        auto reset_event = partial_centroids_.fill(queue_, 0.0);
        auto centroids_event =
            kernels_fp<Float>::partial_reduce_centroids(queue_,
                                                        data_,
                                                        labels,
                                                        cluster_count_,
                                                        part_count_,
                                                        partial_centroids_,
                                                        { reset_event, count_event });

        // Counters are needed in the `merge_reduce_centroids` function,
        // we wait until cross-rank reduction is finished
        count_reduce_request.wait();
        centroids_event =
            kernels_fp<Float>::merge_reduce_centroids(queue_,
                                                      counters_,
                                                      partial_centroids_,
                                                      part_count_,
                                                      centroids,
                                                      { count_event, centroids_event });

        auto centroids_reduce_request =
            comm_.allreduce(centroids.flatten(queue_), { centroids_event });

        const std::int64_t empty_cluster_count =
            count_empty_clusters(queue_, cluster_count_, counters_, { count_event });

        // Centroids and objective function are needed in the `handle_empty_clusters`,
        // we wait until cross-rank reduction is finished
        centroids_reduce_request.wait();
        objective_function_request.wait();

        if (empty_cluster_count > 0) {
            objective_function_value +=
                handle_empty_clusters(empty_cluster_count, centroids, closest_distances);
        }

        return std::make_tuple(objective_function_value, centroids_event);
    }

private:
    Float handle_empty_clusters(std::int64_t candidate_count,
                                pr::ndarray<Float, 2>& centroids,
                                pr::ndarray<Float, 2>& closest_distances,
                                const bk::event_vector& deps = {}) {
        auto find_candidates_event = kernels_fp<Float>::find_candidates(queue_,
                                                                        candidate_count,
                                                                        closest_distances,
                                                                        candidate_indices_,
                                                                        candidate_distances_,
                                                                        deps);

        auto fill_indices_event = fill_empty_cluster_indices(candidate_count,
                                                             counters_,
                                                             empty_cluster_indices_,
                                                             { find_candidates_event });

        if (comm_.is_distributed()) {
            auto candidates = try_allocate_candidates(candidate_count);

            auto gather_event = kernels_fp<Float>::gather_candidates(queue_,
                                                                     candidate_count,
                                                                     data_,
                                                                     candidate_indices_,
                                                                     candidates,
                                                                     { fill_indices_event });
            reduce_candidates(candidate_count, candidate_distances_, candidates, { gather_event });

            auto scatter_event = kernels_fp<Float>::scatter_candidates(queue_,
                                                                       empty_cluster_indices_,
                                                                       candidates,
                                                                       centroids);
            scatter_event.wait_and_throw();
        }
        else {
            auto fill_event = kernels_fp<Float>::fill_empty_clusters(queue_,
                                                                     data_,
                                                                     candidate_indices_,
                                                                     empty_cluster_indices_,
                                                                     centroids,
                                                                     { fill_indices_event });
            fill_event.wait_and_throw();
        }

        return correct_objective_function(candidate_distances_);
    }

    void reduce_candidates(std::int64_t candidate_count,
                           pr::ndarray<Float, 1>& distances,
                           pr::ndarray<Float, 2>& candidates,
                           const bk::event_vector& deps = {}) {
        ONEDAL_ASSERT(candidate_count > 0);
        ONEDAL_ASSERT(column_count_ > 0);
        ONEDAL_ASSERT(candidates.get_dimension(0) == candidate_count);
        ONEDAL_ASSERT(candidates.get_dimension(1) == column_count_);
        ONEDAL_ASSERT(distances.get_dimension(0) == candidate_count);

        const std::int64_t all_candidate_count =
            dal::detail::check_mul_overflow(comm_.get_rank_count(), candidate_count);

        // Allgather candidates
        const auto host_candidates = candidates.to_host(queue_);
        auto host_all_candidates = pr::ndarray<Float, 3>::empty({ comm_.get_rank_count(), //
                                                                  candidate_count,
                                                                  column_count_ });
        auto candidates_request = comm_.allgather(host_candidates.flatten(), //
                                                  host_all_candidates.flatten());

        // Allgather distances
        const auto host_distances = distances.to_host(queue_);
        auto host_all_distances = pr::ndarray<Float, 2>::empty({ comm_.get_rank_count(), //
                                                                 candidate_count });
        auto distances_request = comm_.allgather(host_distances.flatten(), //
                                                 host_all_distances.flatten());

        auto host_all_indices = bk::make_unique_host<std::int32_t>(all_candidate_count);
        {
            std::int32_t* host_all_indices_ptr = host_all_indices.get();
            for (std::int64_t i = 0; i < all_candidate_count; i++) {
                host_all_indices_ptr[i] = i;
            }
        }

        candidates_request.wait();
        distances_request.wait();

        {
            ONEDAL_ASSERT(candidate_count <= all_candidate_count);
            std::int32_t* host_all_indices_ptr = host_all_indices.get();
            const Float* host_all_distances_ptr = host_all_distances.get_data();

            std::partial_sort(host_all_indices_ptr,
                              host_all_indices_ptr + candidate_count,
                              host_all_indices_ptr + all_candidate_count,
                              [=](std::int32_t i, std::int32_t j) {
                                  return host_all_distances_ptr[i] > host_all_distances_ptr[j];
                              });

            if (candidate_count >= 2) {
                ONEDAL_ASSERT(host_all_distances_ptr[host_all_indices_ptr[0]] >
                              host_all_distances_ptr[host_all_indices_ptr[1]]);
            }
        }

        {
            const Float* host_all_candidates_ptr = host_all_candidates.get_data();
            const Float* host_all_distances_ptr = host_all_distances.get_data();
            const std::int32_t* host_all_indices_ptr = host_all_indices.get();
            Float* host_distances_ptr = host_distances.get_mutable_data();
            Float* host_candidates_ptr = host_candidates.get_mutable_data();

            for (std::int64_t i = 0; i < candidate_count; i++) {
                const std::int64_t src_i = host_all_indices_ptr[i];
                host_distances_ptr[i] = host_all_distances_ptr[src_i];
                bk::copy(host_candidates_ptr + i * column_count_,
                         host_all_candidates_ptr + src_i * column_count_,
                         column_count_);
            }
        }

        {
            const Float* host_distances_ptr = host_distances.get_data();
            const Float* host_candidates_ptr = host_candidates.get_data();
            auto distances_assign_event =
                distances.assign(queue_, host_distances_ptr, candidate_count);
            auto candidates_assign_event =
                candidates.assign(queue_, host_candidates_ptr, candidate_count * column_count_);
            sycl::event::wait({ distances_assign_event, candidates_assign_event });
        }
    }

    sycl::event fill_empty_cluster_indices(std::int64_t candidate_count,
                                           const pr::ndarray<std::int32_t, 1>& counters,
                                           pr::ndarray<std::int32_t, 1>& empty_cluster_indices,
                                           const bk::event_vector& deps) {
        ONEDAL_ASSERT(cluster_count_ > 0);
        ONEDAL_ASSERT(candidate_count > 0);
        ONEDAL_ASSERT(counters.get_dimension(0) == cluster_count_);
        ONEDAL_ASSERT(empty_cluster_indices.get_dimension(0) == candidate_count);

        const auto host_counters = counters.to_host(queue_);
        const auto host_empty_cluster_indices = bk::make_unique_host<std::int32_t>(candidate_count);

        const std::int32_t* host_counters_ptr = host_counters.get_data();
        std::int32_t* host_empty_cluster_indices_ptr = host_empty_cluster_indices.get();

        std::int64_t counter = 0;
        for (std::int64_t i = 0; i < cluster_count_; i++) {
            if (host_counters_ptr[i] > 0) {
                continue;
            }

            host_empty_cluster_indices_ptr[counter] = i;
            counter++;
        }

        // We have to wait as `host_counters` will be deleted once we leave scope of the function
        empty_cluster_indices.assign(queue_, host_empty_cluster_indices_ptr, candidate_count)
            .wait_and_throw();

        return sycl::event{};
    }

    Float correct_objective_function(const pr::ndarray<Float, 1>& candidate_distances,
                                     const bk::event_vector& deps = {}) {
        ONEDAL_ASSERT(candidate_distances.get_dimension(0) > 0);

        const std::int64_t candidate_count = candidate_distances.get_dimension(0);
        const auto host_candidate_distances = candidate_distances.to_host(queue_);
        const Float* host_candidate_distances_ptr = host_candidate_distances.get_data();

        Float objective_function_correction = 0;
        for (std::int64_t i = 0; i < candidate_count; i++) {
            objective_function_correction -= host_candidate_distances_ptr[i];
        }

        return objective_function_correction;
    }

    pr::ndarray<Float, 2>& try_allocate_candidates(std::int64_t candidate_count) {
        ONEDAL_ASSERT(candidate_count > 0);
        ONEDAL_ASSERT(cluster_count_ > 0);
        ONEDAL_ASSERT(column_count_ > 0);

        if (candidates_.get_dimension(0) < candidate_count) {
            candidates_ = pr::ndarray<Float, 2>::empty(queue_,
                                                       { candidate_count, column_count_ },
                                                       sycl::usm::alloc::device);
        }
        return candidates_;
    }

    sycl::queue queue_;
    dal::backend::spmd_communicator comm_;
    std::int64_t row_count_ = 0;
    std::int64_t column_count_ = 0;
    std::int64_t cluster_count_ = 0;
    std::int64_t part_count_ = 0;
    double accuracy_threshold_ = 0;
    pr::ndarray<Float, 2> data_;
    pr::ndarray<Float, 2> initial_centroids_;
    pr::ndarray<Float, 2> partial_centroids_;
    pr::ndarray<std::int32_t, 1> counters_;
    pr::ndarray<std::int32_t, 1> candidate_indices_;
    pr::ndarray<Float, 1> candidate_distances_;
    pr::ndarray<std::int32_t, 1> empty_cluster_count_;
    pr::ndarray<std::int32_t, 1> empty_cluster_indices_;
    pr::ndarray<Float, 2> candidates_;
};
} // namespace oneapi::dal::kmeans::backend
