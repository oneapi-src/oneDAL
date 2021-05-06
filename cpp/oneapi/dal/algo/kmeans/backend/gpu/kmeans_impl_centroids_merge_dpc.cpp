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

#include "oneapi/dal/algo/kmeans/backend/gpu/kmeans_impl.hpp"

namespace oneapi::dal::kmeans::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

static std::int64_t get_gpu_sg_size(sycl::queue& queue) {
    // TODO optimization/dispatching
    return 16;
}

template <typename T>
struct centroid_reduction {};

template <typename T>
struct centroid_merge {};

template <typename Float>
sycl::event merge_reduce_centroids(sycl::queue& queue,
                                   const pr::ndview<std::int32_t, 1>& counters,
                                   const pr::ndview<Float, 2>& partial_centroids,
                                   std::int64_t part_count,
                                   pr::ndview<Float, 2>& centroids,
                                   const bk::event_vector& deps) {
    ONEDAL_ASSERT(partial_centroids.get_dimension(0) == centroids.get_dimension(0) * part_count);
    ONEDAL_ASSERT(partial_centroids.get_dimension(1) == centroids.get_dimension(1));
    ONEDAL_ASSERT(counters.get_dimension(0) == centroids.get_dimension(0));
    const Float* partial_centroids_ptr = partial_centroids.get_data();
    Float* centroids_ptr = centroids.get_mutable_data();
    const std::int32_t* counters_ptr = counters.get_data();
    const auto column_count = centroids.get_dimension(1);
    const auto cluster_count = centroids.get_dimension(0);
    const auto sg_size_to_set = get_gpu_sg_size(queue);

    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for<centroid_merge<Float>>(
            bk::make_multiple_nd_range_2d({ sg_size_to_set, column_count * cluster_count },
                                          { sg_size_to_set, 1 }),
            [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const std::int64_t sg_id = sg.get_group_id()[0];
                const std::int64_t wg_id = item.get_global_id(1);
                const std::int64_t sg_count = sg.get_group_range()[0];
                const std::int64_t sg_global_id = wg_id * sg_count + sg_id;
                if (sg_global_id >= column_count * cluster_count)
                    return;
                const std::int64_t sg_cluster_id = sg_global_id / column_count;
                const std::int64_t local_id = sg.get_local_id()[0];
                const std::int64_t local_range = sg.get_local_range()[0];
                Float sum = 0.0;
                for (std::int64_t i = local_id; i < part_count; i += local_range) {
                    sum += partial_centroids_ptr[i * cluster_count * column_count + sg_global_id];
                }
                sum = reduce(sg, sum, sycl::ONEAPI::plus<Float>());

                if (local_id == 0) {
                    auto count = counters_ptr[sg_cluster_id];
                    if (count > 0) {
                        centroids_ptr[sg_global_id] = sum / count;
                    }
                }
            });
    });
}

#define INSTANTIATE(F)                                                                          \
    template sycl::event merge_reduce_centroids<F>(sycl::queue & queue,                         \
                                                   const pr::ndview<std::int32_t, 1>& counters, \
                                                   const pr::ndview<F, 2>& partial_centroids,   \
                                                   std::int64_t part_count,                     \
                                                   pr::ndview<F, 2>& centroids,                 \
                                                   const bk::event_vector& deps);

INSTANTIATE(float)
INSTANTIATE(double)

#endif

} // namespace oneapi::dal::kmeans::backend
