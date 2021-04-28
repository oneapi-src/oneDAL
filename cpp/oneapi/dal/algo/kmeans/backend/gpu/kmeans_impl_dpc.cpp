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

#include "oneapi/dal/backend/primitives/selection/kselect_by_rows_single_col.hpp"
#include "oneapi/dal/backend/primitives/sort/sort.hpp"
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

static std::int64_t get_gpu_wg_count(sycl::queue& queue) {
    // TODO optimization/dispatching
    return 128;
}

template <typename Float>
sycl::event find_candidates(sycl::queue& queue,
                            const pr::ndview<Float, 2>& closest_distances,
                            std::int64_t candidate_count,
                            pr::ndview<std::int32_t, 1>& candidate_indices,
                            pr::ndview<Float, 1>& candidate_distances,
                            const bk::event_vector& deps) {
    ONEDAL_ASSERT(closest_distances.get_shape()[0] > candidate_count);
    ONEDAL_ASSERT(closest_distances.get_shape()[1] == 1);
    ONEDAL_ASSERT(candidate_indices.get_shape()[0] == candidate_indices.get_shape()[0]);
    ONEDAL_ASSERT(candidate_indices.get_shape()[0] >= candidate_count);
    auto elem_count = closest_distances.get_shape()[0];
    auto indices =
        pr::ndarray<std::int32_t, 1>::empty(queue, { elem_count }, sycl::usm::alloc::device);
    auto values = pr::ndview<Float, 1>::wrap(closest_distances.get_mutable_data(), { elem_count });
    std::int32_t* indices_ptr = indices.get_mutable_data();
    auto fill_event = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(elem_count), [=](sycl::item<1> item) {
            std::int32_t ind = item.get_id()[0];
            indices_ptr[ind] = ind;
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
        cgh.parallel_for(sycl::range<1>(candidate_count), [=](sycl::item<1> item) {
            std::int32_t ind = item.get_id()[0];
            candidate_distances_ptr[ind] = values_ptr[ind];
            candidate_indices_ptr[ind] = indices_ptr[ind];
        });
    });
    copy_event.wait_and_throw();
    return copy_event;
}

template <typename Float, typename Metric>
sycl::event assign_clusters(sycl::queue& queue,
                            const pr::ndview<Float, 2>& data,
                            const pr::ndview<Float, 2>& centroids,
                            std::int64_t block_rows,
                            pr::ndview<std::int32_t, 2>& labels,
                            pr::ndview<Float, 2>& distances,
                            pr::ndview<Float, 2>& closest_distances,
                            const bk::event_vector& deps) {
    ONEDAL_ASSERT(data.get_shape()[1] == centroids.get_shape()[1]);
    ONEDAL_ASSERT(data.get_shape()[0] >= centroids.get_shape()[0]);
    ONEDAL_ASSERT(labels.get_shape()[0] >= data.get_shape()[0]);
    ONEDAL_ASSERT(labels.get_shape()[1] == 1);
    ONEDAL_ASSERT(closest_distances.get_shape()[0] >= data.get_shape()[0]);
    ONEDAL_ASSERT(closest_distances.get_shape()[1] == 1);
    ONEDAL_ASSERT(distances.get_shape()[0] >= block_rows);
    ONEDAL_ASSERT(distances.get_shape()[1] >= centroids.get_shape()[0]);
    sycl::event selection_event;
    auto row_count = data.get_shape()[0];
    auto column_count = data.get_shape()[1];
    auto centroid_count = centroids.get_shape()[0];
    pr::distance<Float, Metric> block_distances(queue);
    auto block_count = row_count / block_rows + std::int64_t(row_count % block_rows > 0);
    for (std::int64_t iblock = 0; iblock < block_count; iblock++) {
        auto row_offset = block_rows * iblock;
        auto cur_rows = std::min(block_rows, row_count - row_offset);
        auto distance_block =
            pr::ndview<Float, 2>::wrap(distances.get_mutable_data(), { cur_rows, centroid_count });
        auto data_block = pr::ndview<Float, 2>::wrap(data.get_data() + row_offset * column_count,
                                                     { cur_rows, column_count });
        auto distance_event =
            block_distances(data_block, centroids, distance_block, { selection_event });
        auto label_block =
            pr::ndview<int32_t, 2>::wrap(labels.get_mutable_data() + row_offset, { cur_rows, 1 });
        auto closest_distance_block =
            pr::ndview<Float, 2>::wrap(closest_distances.get_mutable_data() + row_offset,
                                       { cur_rows, 1 });
        selection_event = pr::kselect_by_rows_single_col<Float>{}.operator()(queue,
                                                                             distance_block,
                                                                             1,
                                                                             closest_distance_block,
                                                                             label_block,
                                                                             { distance_event });
    }
    return selection_event;
}

template <typename Float>
sycl::event reduce_centroids(sycl::queue& queue,
                             const pr::ndview<Float, 2>& data,
                             const pr::ndview<std::int32_t, 2>& labels,
                             const pr::ndview<std::int32_t, 1>& counters,
                             std::int64_t part_count,
                             pr::ndview<Float, 2>& centroids,
                             pr::ndview<Float, 2>& partial_centroids,
                             const bk::event_vector& deps) {
    ONEDAL_ASSERT(data.get_shape()[1] == centroids.get_shape()[1]);
    ONEDAL_ASSERT(data.get_shape()[1] == partial_centroids.get_shape()[1]);
    ONEDAL_ASSERT(data.get_shape()[0] >= centroids.get_shape()[0]);
    ONEDAL_ASSERT(labels.get_shape()[0] >= data.get_shape()[0]);
    ONEDAL_ASSERT(labels.get_shape()[1] == 1);
    ONEDAL_ASSERT(partial_centroids.get_shape()[0] >= centroids.get_shape()[0] * part_count);
    const Float* data_ptr = data.get_data();
    const std::int32_t* label_ptr = labels.get_data();
    Float* partial_centroids_ptr = partial_centroids.get_mutable_data();
    Float* centroids_ptr = centroids.get_mutable_data();
    const std::int32_t* counters_ptr = counters.get_data();
    const auto row_count = data.get_shape()[0];
    const auto column_count = data.get_shape()[1];
    const auto centroid_count = centroids.get_shape()[0];
    const auto sg_size_to_set = get_gpu_sg_size(queue);
    queue
        .submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.parallel_for(
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
                            partial_centroids_ptr[sg_global_id * centroid_count * column_count +
                                                  cl * column_count + j] +=
                                data_ptr[i * column_count + j];
                        }
                    }
                });
        })
        .wait_and_throw();

    auto merge_centroids_event = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            bk::make_multiple_nd_range_2d({ sg_size_to_set, column_count * centroid_count },
                                          { sg_size_to_set, 1 }),
            [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const std::int64_t sg_id = sg.get_group_id()[0];
                const std::int64_t wg_id = item.get_global_id(1);
                const std::int64_t sg_count = sg.get_group_range()[0];
                const std::int64_t sg_global_id = wg_id * sg_count + sg_id;
                if (sg_global_id >= column_count * centroid_count)
                    return;
                const std::int64_t sg_cluster_id = sg_global_id / column_count;
                const std::int64_t local_id = sg.get_local_id()[0];
                const std::int64_t local_range = sg.get_local_range()[0];
                Float sum = 0.0;
                for (std::int64_t i = local_id; i < part_count; i += local_range) {
                    sum += partial_centroids_ptr[i * centroid_count * column_count + sg_global_id];
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
    return merge_centroids_event;
}

sycl::event count_clusters(sycl::queue& queue,
                           const pr::ndview<std::int32_t, 2>& labels,
                           std::int64_t centroid_count,
                           pr::ndview<std::int32_t, 1>& counters,
                           pr::ndarray<std::int32_t, 1>& empty_cluster_count,
                           const bk::event_vector& deps) {
    ONEDAL_ASSERT(counters.get_shape()[0] >= centroid_count);
    ONEDAL_ASSERT(empty_cluster_count.get_shape()[0] == 1);
    ONEDAL_ASSERT(labels.get_shape()[1] == 1);
    const std::int32_t* label_ptr = labels.get_data();
    std::int32_t* counter_ptr = counters.get_mutable_data();
    const auto sg_size_to_set = get_gpu_sg_size(queue);
    const auto wg_count_to_set = get_gpu_wg_count(queue);
    auto cluster_count_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        const auto row_count = labels.get_shape()[0];
        cgh.parallel_for(
            bk::make_multiple_nd_range_2d({ sg_size_to_set, wg_count_to_set },
                                          { sg_size_to_set, 1 }),
            [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const std::int64_t sg_id = sg.get_group_id()[0];
                const std::int64_t wg_id = item.get_global_id(1);
                const std::int64_t wg_count = item.get_global_range(1);
                const std::int64_t sg_count = sg.get_group_range()[0];
                const std::int64_t sg_global_id = wg_id * sg_count + sg_id;
                const std::int64_t total_sg_count = wg_count * sg_count;

                const std::int64_t local_id = sg.get_local_id()[0];
                const std::int64_t local_range = sg.get_local_range()[0];

                const std::int64_t block_size =
                    row_count / total_sg_count + std::int64_t(row_count % total_sg_count > 0);
                const std::int64_t offset = block_size * sg_global_id;
                const std::int64_t end =
                    (offset + block_size) > row_count ? row_count : (offset + block_size);
                for (std::int64_t i = offset + local_id; i < end; i += local_range) {
                    const std::int32_t cl = label_ptr[i];
                    sycl::ONEAPI::atomic_ref<std::int32_t,
                                             cl::sycl::ONEAPI::memory_order::relaxed,
                                             cl::sycl::ONEAPI::memory_scope::device,
                                             cl::sycl::access::address_space::global_device_space>
                        counter_atomic(counter_ptr[cl]);
                    counter_atomic.fetch_add(1);
                }
            });
    });
    std::int32_t* value_ptr = empty_cluster_count.get_mutable_data();
    auto empty_cluster_count_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ cluster_count_event });
        cgh.parallel_for(
            bk::make_multiple_nd_range_2d({ sg_size_to_set, 1 }, { sg_size_to_set, 1 }),
            [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const std::int64_t sg_id = sg.get_group_id()[0];
                if (sg_id > 0)
                    return;
                const std::int64_t local_id = sg.get_local_id()[0];
                const std::int64_t local_range = sg.get_local_range()[0];
                std::int32_t sum = 0;
                for (std::int64_t i = local_id; i < centroid_count; i += local_range) {
                    sum += counter_ptr[i] == 0;
                }
                sum = reduce(sg, sum, sycl::ONEAPI::plus<std::int32_t>());
                if (local_id == 0) {
                    value_ptr[0] = sum;
                }
            });
    });
    return empty_cluster_count_event;
}

template <typename Float>
sycl::event compute_objective_function(sycl::queue& queue,
                                       const pr::ndview<Float, 2>& closest_distances,
                                       pr::ndview<Float, 1>& objective_function,
                                       const bk::event_vector& deps) {
    ONEDAL_ASSERT(closest_distances.get_shape()[1] == 1);
    ONEDAL_ASSERT(objective_function.get_shape()[0] == 1);
    const Float* distance_ptr = closest_distances.get_data();
    Float* value_ptr = objective_function.get_mutable_data();
    const auto row_count = closest_distances.get_shape()[0];
    const auto sg_size_to_set = get_gpu_sg_size(queue);
    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(
            bk::make_multiple_nd_range_2d({ sg_size_to_set, 1 }, { sg_size_to_set, 1 }),
            [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const std::int64_t sg_id = sg.get_group_id()[0];
                if (sg_id > 0)
                    return;
                const std::int64_t local_id = sg.get_local_id()[0];
                const std::int64_t local_range = sg.get_local_range()[0];
                Float sum = 0;
                for (std::int64_t i = local_id; i < row_count; i += local_range) {
                    sum += distance_ptr[i];
                }
                sum = reduce(sg, sum, sycl::ONEAPI::plus<Float>());
                if (local_id == 0) {
                    value_ptr[0] = sum;
                }
            });
    });
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
    ONEDAL_ASSERT(data.get_shape()[1] == centroids.get_shape()[1]);
    ONEDAL_ASSERT(data.get_shape()[0] >= centroids.get_shape()[0]);
    ONEDAL_ASSERT(counters.get_shape()[1] == centroids.get_shape()[0]);
    ONEDAL_ASSERT(candidate_indices.get_shape()[0] <= centroids.get_shape()[0]);
    ONEDAL_ASSERT(candidate_distances.get_shape()[0] <= centroids.get_shape()[0]);
    ONEDAL_ASSERT(labels.get_shape()[0] >= data.get_shape()[0]);
    ONEDAL_ASSERT(labels.get_shape()[1] == 1);

    bk::event_vector events;
    auto column_count = data.get_shape()[1];
    auto candidate_count = candidate_indices.get_shape()[0];
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

#define INSTANTIATE_WITH_METRIC(F, M)                                                  \
    template sycl::event assign_clusters<F, M<F>>(sycl::queue & queue,                 \
                                                  const pr::ndview<F, 2>& data,        \
                                                  const pr::ndview<F, 2>& centroids,   \
                                                  std::int64_t block_rows,             \
                                                  pr::ndview<std::int32_t, 2>& labels, \
                                                  pr::ndview<F, 2>& distances,         \
                                                  pr::ndview<F, 2>& closest_distances, \
                                                  const bk::event_vector& deps);

#define INSTANTIATE(F)                                                                            \
    template std::int64_t get_block_size_in_rows<F>(sycl::queue & queue,                          \
                                                    std::int64_t column_count);                   \
    template std::int64_t get_part_count_for_partial_centroids<F>(sycl::queue & queue,            \
                                                                  std::int64_t column_count,      \
                                                                  std::int64_t cluster_count);    \
    template sycl::event find_candidates<F>(sycl::queue & queue,                                  \
                                            const pr::ndview<F, 2>& closest_distances,            \
                                            std::int64_t candidate_count,                         \
                                            pr::ndview<std::int32_t, 1>& candidate_indices,       \
                                            pr::ndview<F, 1>& candidate_distances,                \
                                            const bk::event_vector& deps);                        \
                                                                                                  \
    template sycl::event reduce_centroids<F>(sycl::queue & queue,                                 \
                                             const pr::ndview<F, 2>& data,                        \
                                             const pr::ndview<std::int32_t, 2>& labels,           \
                                             const pr::ndview<std::int32_t, 1>& counters,         \
                                             std::int64_t part_count,                             \
                                             pr::ndview<F, 2>& centroids,                         \
                                             pr::ndview<F, 2>& partial_centroids,                 \
                                             const bk::event_vector& deps);                       \
                                                                                                  \
    template sycl::event compute_objective_function<F>(sycl::queue & queue,                       \
                                                       const pr::ndview<F, 2>& closest_distances, \
                                                       pr::ndview<F, 1>& objective_function,      \
                                                       const bk::event_vector& deps);             \
                                                                                                  \
    template bk::event_vector fill_empty_clusters(                                                \
        sycl::queue& queue,                                                                       \
        const pr::ndview<F, 2>& data,                                                             \
        const pr::ndarray<std::int32_t, 1>& counters,                                             \
        const pr::ndarray<std::int32_t, 1>& candidate_indices,                                    \
        const pr::ndarray<F, 1>& candidate_distances,                                             \
        pr::ndview<F, 2>& centroids,                                                              \
        pr::ndarray<std::int32_t, 2>& labels,                                                     \
        F& objective_function,                                                                    \
        const bk::event_vector& deps);                                                            \
    INSTANTIATE_WITH_METRIC(F, pr::squared_l2_metric)

INSTANTIATE(float)
INSTANTIATE(double)

#endif

} // namespace oneapi::dal::kmeans::backend
