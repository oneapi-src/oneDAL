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

template <typename T>
struct find_candidates_kernel {};

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
        cgh.depends_on(sort_event);
        cgh.parallel_for<find_candidates_kernel<Float>>(
            sycl::range<1>(candidate_count),
            [=](sycl::id<1> idx) {
                candidate_distances_ptr[idx] = values_ptr[idx];
                candidate_indices_ptr[idx] = indices_ptr[idx];
            });
    });
    copy_event.wait_and_throw();
    return copy_event;
}
#define INSTANTIATE(F)                                                                      \
    template sycl::event find_candidates<F>(sycl::queue & queue,                            \
                                            const pr::ndview<F, 2>& closest_distances,      \
                                            std::int64_t candidate_count,                   \
                                            pr::ndview<std::int32_t, 1>& candidate_indices, \
                                            pr::ndview<F, 1>& candidate_distances,          \
                                            const bk::event_vector& deps);

INSTANTIATE(float)
INSTANTIATE(double)

#endif

} // namespace oneapi::dal::kmeans::backend
