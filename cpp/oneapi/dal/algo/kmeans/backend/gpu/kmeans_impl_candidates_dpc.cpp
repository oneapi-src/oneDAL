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

#include "oneapi/dal/backend/primitives/sort/sort.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/kmeans_impl.hpp"

namespace oneapi::dal::kmeans::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

template <typename Float>
sycl::event find_candidates(sycl::queue& queue,
                            const pr::ndview<Float, 2>& closest_distances,
                            std::int64_t candidate_count,
                            pr::ndview<std::int32_t, 1>& candidate_indices,
                            pr::ndview<Float, 1>& candidate_distances,
                            const bk::event_vector& deps) {
    ONEDAL_ASSERT(closest_distances.get_dimension(0) > candidate_count);
    ONEDAL_ASSERT(closest_distances.get_dimension(1) == 1);
    ONEDAL_ASSERT(candidate_indices.get_dimension(0) == candidate_indices.get_dimension(0));
    ONEDAL_ASSERT(candidate_indices.get_dimension(0) >= candidate_count);
    const auto elem_count = closest_distances.get_dimension(0);
    auto indices =
        pr::ndarray<std::int32_t, 1>::empty(queue, { elem_count }, sycl::usm::alloc::device);
    auto values = pr::ndview<Float, 1>::wrap(closest_distances.get_mutable_data(), { elem_count });
    std::int32_t* indices_ptr = indices.get_mutable_data();
    auto fill_event = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(elem_count), [=](sycl::id<1> idx) {
            indices_ptr[idx] = idx;
        });
    });
    auto sort_event = pr::radix_sort_indices_inplace<Float, std::int32_t>{ queue }(values,
                                                                                   indices,
                                                                                   { fill_event });
    sort_event.wait_and_throw();
    auto candidate_indices_ptr = candidate_indices.get_mutable_data();
    auto candidate_distances_ptr = candidate_distances.get_mutable_data();
    auto values_ptr = values.get_mutable_data();
    auto copy_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ sort_event });
        cgh.parallel_for(sycl::range<1>(candidate_count), [=](sycl::id<1> idx) {
            candidate_distances_ptr[idx] = values_ptr[idx];
            candidate_indices_ptr[idx] = indices_ptr[idx];
        });
    });
    copy_event.wait_and_throw();
    return copy_event;
}

template <typename Float>
bk::event_vector fill_empty_clusters(sycl::queue& queue,
                                     const pr::ndview<Float, 2>& data,
                                     const pr::ndarray<std::int32_t, 1>& counters,
                                     const pr::ndarray<std::int32_t, 1>& candidate_indices,
                                     const pr::ndarray<Float, 1>& candidate_distances,
                                     pr::ndview<Float, 2>& centroids,
                                     pr::ndarray<std::int32_t, 2>& labels,
                                     Float& objective_function,
                                     const bk::event_vector& deps) {
    ONEDAL_ASSERT(data.get_dimension(1) == centroids.get_dimension(1));
    ONEDAL_ASSERT(data.get_dimension(0) >= centroids.get_dimension(0));
    ONEDAL_ASSERT(counters.get_dimension(1) == centroids.get_dimension(0));
    ONEDAL_ASSERT(candidate_indices.get_dimension(0) <= centroids.get_dimension(0));
    ONEDAL_ASSERT(candidate_distances.get_dimension(0) <= centroids.get_dimension(0));
    ONEDAL_ASSERT(labels.get_dimension(0) >= data.get_dimension(0));
    ONEDAL_ASSERT(labels.get_dimension(1) == 1);

    bk::event_vector events;
    const auto column_count = data.get_dimension(1);
    const auto candidate_count = candidate_indices.get_dimension(0);
    sycl::event::wait(deps);
    auto host_counters = counters.to_host(queue);
    auto counters_ptr = host_counters.get_data();

    auto host_labels = labels.to_host(queue);
    auto labels_ptr = host_labels.get_mutable_data();

    auto centroids_ptr = centroids.get_mutable_data();
    auto data_ptr = data.get_data();

    auto host_candidate_distances = candidate_distances.to_host(queue);
    auto candidate_distances_ptr = host_candidate_distances.get_data();

    auto host_candidate_indices = candidate_indices.to_host(queue);
    auto candidate_indices_ptr = host_candidate_indices.get_data();
    std::int64_t cpos = 0;

    for (std::int64_t ic = 0; ic < candidate_count; ic++) {
        if (counters_ptr[ic] > 0)
            continue;
        auto index = candidate_indices_ptr[cpos];
        auto value = candidate_distances_ptr[cpos];
        labels_ptr[index] = ic;
        objective_function -= value;
        auto copy_event = queue.submit([&](sycl::handler& cgh) {
            cgh.memcpy(centroids_ptr + ic * column_count * sizeof(Float),
                       data_ptr + index * column_count * sizeof(Float),
                       sizeof(Float) * column_count);
        });
        events.push_back(copy_event);
        cpos++;
    }
    return events;
}

#define INSTANTIATE(F)                                                                      \
    template sycl::event find_candidates<F>(sycl::queue & queue,                            \
                                            const pr::ndview<F, 2>& closest_distances,      \
                                            std::int64_t candidate_count,                   \
                                            pr::ndview<std::int32_t, 1>& candidate_indices, \
                                            pr::ndview<F, 1>& candidate_distances,          \
                                            const bk::event_vector& deps);                  \
                                                                                            \
    template bk::event_vector fill_empty_clusters(                                          \
        sycl::queue& queue,                                                                 \
        const pr::ndview<F, 2>& data,                                                       \
        const pr::ndarray<std::int32_t, 1>& counters,                                       \
        const pr::ndarray<std::int32_t, 1>& candidate_indices,                              \
        const pr::ndarray<F, 1>& candidate_distances,                                       \
        pr::ndview<F, 2>& centroids,                                                        \
        pr::ndarray<std::int32_t, 2>& labels,                                               \
        F& objective_function,                                                              \
        const bk::event_vector& deps);

INSTANTIATE(float)
INSTANTIATE(double)

#endif

} // namespace oneapi::dal::kmeans::backend
