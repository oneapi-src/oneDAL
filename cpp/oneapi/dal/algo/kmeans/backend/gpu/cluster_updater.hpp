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
    cluster_updater() {}
    auto& set_cluster_count(std::int64_t cluster_count) {
        cluster_count_ = cluster_count;
        return *this;
    }
    auto& set_accuracy_threshold(double accuracy_threshold) {
        accuracy_threshold_ = accuracy_threshold;
        return *this;
    }
    auto set_part_count(std::int64_t part_count) {
        part_count_ = part_count;
        return *this;
    }
    auto set_queue(sycl::queue& queue) {
        queue_ = queue;
        return *this;
    }
    auto set_data(const pr::ndarray<Float, 2> data) {
        data_ = data;
        row_count_ = data.get_dimension(0);
        column_count_ = data.get_dimension(1);
        return *this;
    }
    auto set_initial_centroids(const pr::ndarray<Float, 2> initial_centroids) {
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
    }
    auto update(pr::ndarray<Float, 2> centroids,
                pr::ndarray<Float, 2> distance_block,
                pr::ndarray<Float, 2> closest_distances,
                pr::ndarray<Float, 1> objective_function,
                pr::ndarray<std::int32_t, 2> labels,
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
        assign_event.wait_and_throw();
        auto count_event =
            count_clusters(queue_, labels, cluster_count_, counters_, { assign_event });
        auto objective_function_event =
            kernels_fp<Float>::compute_objective_function(queue_,
                                                          closest_distances,
                                                          objective_function,
                                                          { assign_event });
        auto reset_event = partial_centroids_.fill(queue_, 0.0);
        reset_event.wait_and_throw();
        auto centroids_event = kernels_fp<Float>::partial_reduce_centroids(queue_,
                                                                           data_,
                                                                           labels,
                                                                           cluster_count_,
                                                                           part_count_,
                                                                           partial_centroids_,
                                                                           { count_event });
        centroids_event =
            kernels_fp<Float>::merge_reduce_centroids(queue_,
                                                      counters_,
                                                      partial_centroids_,
                                                      part_count_,
                                                      centroids,
                                                      { count_event, centroids_event });
        count_empty_clusters(queue_,
                             cluster_count_,
                             counters_,
                             empty_cluster_count_,
                             { count_event });

        std::int64_t candidate_count = empty_cluster_count_.to_host(queue_).get_data()[0];
        sycl::event find_candidates_event;
        if (candidate_count > 0) {
            find_candidates_event = kernels_fp<Float>::find_candidates(queue_,
                                                                       closest_distances,
                                                                       candidate_count,
                                                                       candidate_indices_,
                                                                       candidate_distances_);
        }
        Float objective_function_value = objective_function.to_host(queue_).get_data()[0];
        bk::event_vector candidate_events;
        if (candidate_count > 0) {
            auto [updated_objective_function_value, copy_events] =
                kernels_fp<Float>::fill_empty_clusters(queue_,
                                                       data_,
                                                       counters_,
                                                       candidate_indices_,
                                                       candidate_distances_,
                                                       centroids,
                                                       labels,
                                                       objective_function_value,
                                                       { find_candidates_event });
            sycl::event::wait(copy_events);
            objective_function_value = updated_objective_function_value;
        }
        return std::make_tuple(objective_function_value, centroids_event);
    }

private:
    sycl::queue queue_;
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
};
} // namespace oneapi::dal::kmeans::backend
