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

template <typename Float>
std::int64_t get_block_size_in_rows(sycl::queue& queue, std::int64_t column_count) {
    // TODO optimization
    std::int64_t block_size_in_bytes = bk::device_global_mem_cache_size(queue);
    return block_size_in_bytes / column_count / sizeof(Float);
}

template <typename Float>
std::int64_t get_part_count_for_partial_centroids(sycl::queue& queue,
                                                  std::int64_t column_count,
                                                  std::int64_t cluster_count) {
    // TODO optimization
    std::int64_t block_size_in_bytes =
        std::min(bk::device_max_mem_alloc_size(queue), bk::device_global_mem_size(queue) / 4);
    std::int64_t part_count = 128;
    dal::detail::check_mul_overflow(cluster_count, column_count);
    dal::detail::check_mul_overflow(cluster_count * column_count, part_count);
    while (cluster_count * column_count * part_count > block_size_in_bytes) {
        part_count /= 2;
    }
    ONEDAL_ASSERT(part_count > 0);
    return part_count;
}

static std::int64_t get_gpu_sg_size(sycl::queue& queue) {
    // TODO optimization/dispatching
    return 16;
}


template<typename T>
struct centroid_reduction {};


template <typename Float>
sycl::event partial_reduce_centroids(sycl::queue& queue,
                             const pr::ndview<Float, 2>& data,
                             const pr::ndview<std::int32_t, 2>& labels,
                             std::int64_t cluster_count,
                             std::int64_t part_count,
                             pr::ndview<Float, 2>& partial_centroids,
                             const bk::event_vector& deps) {
    ONEDAL_ASSERT(data.get_shape()[1] == centroids.get_shape()[1]);
    ONEDAL_ASSERT(data.get_shape()[1] == partial_centroids.get_shape()[1]);
    ONEDAL_ASSERT(labels.get_shape()[0] >= data.get_shape()[0]);
    ONEDAL_ASSERT(labels.get_shape()[1] == 1);
    ONEDAL_ASSERT(partial_centroids.get_shape()[0] >= cluster_count * part_count);
    const Float* data_ptr = data.get_data();
    const std::int32_t* label_ptr = labels.get_data();
    Float* partial_centroids_ptr = partial_centroids.get_mutable_data();
    const auto row_count = data.get_shape()[0];
    const auto column_count = data.get_shape()[1];
    const auto sg_size_to_set = get_gpu_sg_size(queue);
    return queue
        .submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.parallel_for<centroid_reduction<Float>>(
                bk::make_multiple_nd_range_2d({ sg_size_to_set, part_count },
                                              { sg_size_to_set, 1 }),
                [=](sycl::nd_item<2> item) {
                    auto sg = item.get_sub_group();
                    const std::int64_t sg_id = sg.get_group_id()[0];
                    const std::int64_t wg_id = item.get_global_id(1);
                    const std::int64_t sg_count = sg.get_group_range()[0];
                    const std::int64_t sg_global_id = wg_id * sg_count + sg_id;
                    if (sg_global_id >= part_count)
                        return;
                    const std::int64_t local_id = sg.get_local_id()[0];
                    const std::int64_t local_range = sg.get_local_range()[0];
                    for (std::int64_t i = sg_global_id; i < row_count; i += part_count) {
                        std::int32_t cl = -1;
                        if (local_id == 0) {
                            cl = label_ptr[i];
                        }
                        cl = reduce(sg, cl, sycl::ONEAPI::maximum<std::int32_t>());
                        for (std::int64_t j = local_id; j < column_count; j += local_range) {
                            partial_centroids_ptr[sg_global_id * cluster_count * column_count +
                                                  cl * column_count + j] +=
                                data_ptr[i * column_count + j];
                        }
                    }
                });
        });
}


#define INSTANTIATE(F)                                                                            \
    template std::int64_t get_block_size_in_rows<F>(sycl::queue & queue,                          \
                                                    std::int64_t column_count);                   \
    template std::int64_t get_part_count_for_partial_centroids<F>(sycl::queue & queue,            \
                                                                  std::int64_t column_count,      \
                                                                  std::int64_t cluster_count);    \
    template sycl::event partial_reduce_centroids<F>(sycl::queue& queue,                                        \
                             const pr::ndview<F, 2>& data,  \
                             const pr::ndview<std::int32_t, 2>& labels, \
                             std::int64_t cluster_count, \
                             std::int64_t part_count, \
                             pr::ndview<F, 2>& partial_centroids, \
                             const bk::event_vector& deps);

INSTANTIATE(float)
INSTANTIATE(double)

#endif

} // namespace oneapi::dal::kmeans::backend
