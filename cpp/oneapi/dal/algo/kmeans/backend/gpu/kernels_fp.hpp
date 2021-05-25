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

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/distance.hpp"
#include "oneapi/dal/backend/primitives/sort/sort.hpp"

namespace oneapi::dal::kmeans::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

inline std::int64_t get_recommended_sg_size(const sycl::queue& queue) {
    // TODO optimization/dispatching
    return 16;
}

inline std::int64_t get_recommended_wg_count(sycl::queue& queue) {
    // TODO optimization/dispatching
    return 128;
}

inline std::int64_t get_scaled_wg_size_per_row(const sycl::queue& queue,
                                               std::int64_t column_count,
                                               std::int64_t preffered_wg_size) {
    const std::int64_t sg_max_size = bk::device_max_sg_size(queue);
    const std::int64_t row_adjusted_sg_num =
        column_count / sg_max_size + std::int64_t(column_count % sg_max_size > 0);
    std::int64_t expected_sg_num = std::min(preffered_wg_size / sg_max_size, row_adjusted_sg_num);
    if (expected_sg_num < 1)
        expected_sg_num = 1;
    return dal::detail::check_mul_overflow(expected_sg_num, sg_max_size);
}

inline std::int64_t get_gpu_sg_size(sycl::queue& queue) {
    // TODO optimization/dispatching
    return 16;
}

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

template <typename T>
struct select_min_distance {};

template <typename T>
struct set_indices {};

template <typename T>
struct find_candidates_kernel {};

template <typename T>
struct fill_empty_cluster_kernel {};

template <typename T>
struct centroid_reduction {};

template <typename T>
struct reset_partial_centroids {};

template <typename T>
struct centroid_merge {};

template <typename T>
struct compute_obj_function {};

template <typename Float>
std::int64_t get_block_size_in_rows(sycl::queue& queue, std::int64_t column_count) {
    // TODO optimization
    std::int64_t block_size_in_bytes = bk::device_global_mem_cache_size(queue);
    std::int64_t block_size_in_rows = block_size_in_bytes / column_count / sizeof(Float);
    ONEDAL_ASSERT(block_size_in_rows > 0);
    return block_size_in_rows;
}

template <typename Float>
std::int64_t get_part_count_for_partial_centroids(sycl::queue& queue,
                                                  std::int64_t column_count,
                                                  std::int64_t cluster_count) {
    // TODO optimization
    constexpr std::int64_t mem_block_count = 4; // To ensure all blocks fit in memory
    const std::int64_t block_size_in_bytes =
        std::min(bk::device_max_mem_alloc_size(queue),
                 bk::device_global_mem_size(queue) / mem_block_count);
    std::int64_t part_count = 128; // Number of partial centroids. Reasonable initial guess.
    dal::detail::check_mul_overflow(cluster_count, column_count);
    dal::detail::check_mul_overflow(cluster_count * column_count, part_count);
    std::int64_t fp_size = dal::detail::get_data_type_size(dal::detail::make_data_type<Float>());
    dal::detail::check_mul_overflow(cluster_count * column_count * part_count, fp_size);
    const std::int64_t part_size = cluster_count * column_count * fp_size;
    while (part_count * part_size > block_size_in_bytes / 2) {
        part_count /= 2;
    }
    ONEDAL_ASSERT(part_count > 0);
    return part_count;
}

template <typename Float>
sycl::event select(sycl::queue& queue,
                   const pr::ndview<Float, 2>& data,
                   pr::ndview<Float, 2>& selection,
                   pr::ndview<std::int32_t, 2>& indices,
                   const bk::event_vector& deps = {}) {
    ONEDAL_ASSERT(indices.get_dimension(0) == data.get_dimension(0));
    ONEDAL_ASSERT(indices.get_dimension(1) == 1);
    ONEDAL_ASSERT(selection.get_dimension(0) == data.get_dimension(0));
    ONEDAL_ASSERT(selection.get_dimension(1) == 1);

    const std::int64_t col_count = data.get_dimension(1);
    const std::int64_t row_count = data.get_dimension(0);
    const std::int64_t stride = data.get_dimension(1);

    const std::int64_t preffered_wg_size = 128;
    const std::int64_t wg_size = get_scaled_wg_size_per_row(queue, col_count, preffered_wg_size);

    const Float* data_ptr = data.get_data();
    Float* selection_ptr = selection.get_mutable_data();
    std::int32_t* indices_ptr = indices.get_mutable_data();
    const auto fp_max = dal::detail::limits<Float>::max();

    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for<select_min_distance<Float>>(
            bk::make_multiple_nd_range_2d({ wg_size, row_count }, { wg_size, 1 }),
            [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const std::uint32_t sg_id = sg.get_group_id()[0];
                const std::uint32_t wg_id = item.get_global_id(1);
                const std::uint32_t sg_num = sg.get_group_range()[0];
                const std::uint32_t sg_global_id = wg_id * sg_num + sg_id;
                if (sg_global_id >= row_count)
                    return;
                const std::uint32_t in_offset = sg_global_id * stride;
                const std::uint32_t out_offset = sg_global_id;

                const std::uint32_t local_id = sg.get_local_id()[0];
                const std::uint32_t local_range = sg.get_local_range()[0];

                std::int32_t index = -1;
                Float value = fp_max;
                for (std::uint32_t i = local_id; i < col_count; i += local_range) {
                    const Float cur_val = data_ptr[in_offset + i];
                    if (cur_val < value) {
                        index = i;
                        value = cur_val;
                    }
                }

                sg.barrier();

                const Float final_value = reduce(sg, value, sycl::ONEAPI::minimum<Float>());
                const bool present = (final_value == value);
                const std::int32_t pos =
                    exclusive_scan(sg, present ? 1 : 0, sycl::ONEAPI::plus<std::int32_t>());
                const bool owner = present && pos == 0;
                const std::int32_t final_index =
                    -reduce(sg, owner ? -index : 1, sycl::ONEAPI::minimum<std::int32_t>());

                if (local_id == 0) {
                    indices_ptr[out_offset] = final_index;
                    selection_ptr[out_offset] = final_value;
                }
            });
    });
    return event;
}

template <typename Float, typename Metric>
sycl::event assign_clusters(sycl::queue& queue,
                            const pr::ndview<Float, 2>& data,
                            const pr::ndview<Float, 2>& centroids,
                            std::int64_t block_rows,
                            pr::ndview<std::int32_t, 2>& labels,
                            pr::ndview<Float, 2>& distances,
                            pr::ndview<Float, 2>& closest_distances,
                            const bk::event_vector& deps = {}) {
    ONEDAL_ASSERT(data.get_dimension(1) == centroids.get_dimension(1));
    ONEDAL_ASSERT(data.get_dimension(0) >= centroids.get_dimension(0));
    ONEDAL_ASSERT(labels.get_dimension(0) >= data.get_dimension(0));
    ONEDAL_ASSERT(labels.get_dimension(1) == 1);
    ONEDAL_ASSERT(closest_distances.get_dimension(0) >= data.get_dimension(0));
    ONEDAL_ASSERT(closest_distances.get_dimension(1) == 1);
    ONEDAL_ASSERT(distances.get_dimension(0) >= block_rows);
    ONEDAL_ASSERT(distances.get_dimension(1) >= centroids.get_dimension(0));
    sycl::event selection_event;
    const auto row_count = data.get_dimension(0);
    const auto column_count = data.get_dimension(1);
    const auto centroid_count = centroids.get_dimension(0);
    pr::distance<Float, Metric> block_distances(queue);
    auto block_count = row_count / block_rows + std::int64_t(row_count % block_rows > 0);
    for (std::int64_t iblock = 0; iblock < block_count; iblock++) {
        const auto row_offset = block_rows * iblock;
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
        selection_event =
            select(queue, distance_block, closest_distance_block, label_block, { distance_event });
    }
    return selection_event;
}

template <typename Float>
std::tuple<Float, bk::event_vector> fill_empty_clusters(
    sycl::queue& queue,
    const pr::ndview<Float, 2>& data,
    const pr::ndarray<std::int32_t, 1>& counters,
    const pr::ndarray<std::int32_t, 1>& candidate_indices,
    const pr::ndarray<Float, 1>& candidate_distances,
    pr::ndview<Float, 2>& centroids,
    pr::ndarray<std::int32_t, 2>& labels,
    Float objective_function,
    const bk::event_vector& deps = {}) {
    ONEDAL_ASSERT(data.get_dimension(1) == centroids.get_dimension(1));
    ONEDAL_ASSERT(data.get_dimension(0) >= centroids.get_dimension(0));
    ONEDAL_ASSERT(counters.get_dimension(0) == centroids.get_dimension(0));
    ONEDAL_ASSERT(candidate_indices.get_dimension(0) <= centroids.get_dimension(0));
    ONEDAL_ASSERT(candidate_distances.get_dimension(0) <= centroids.get_dimension(0));
    ONEDAL_ASSERT(labels.get_dimension(0) >= data.get_dimension(0));
    ONEDAL_ASSERT(labels.get_dimension(1) == 1);

    const auto cluster_count = centroids.get_dimension(0);

    bk::event_vector events;
    const auto column_count = data.get_dimension(1);
    [[maybe_unused]] const auto candidate_count = candidate_indices.get_dimension(0);
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

    for (std::int64_t ic = 0; ic < cluster_count; ic++) {
        if (counters_ptr[ic] > 0)
            continue;
        ONEDAL_ASSERT(cpos < candidate_count);
        auto index = candidate_indices_ptr[cpos];
        auto value = candidate_distances_ptr[cpos];
        labels_ptr[index] = ic;
        objective_function -= value;
        auto copy_event = queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<fill_empty_cluster_kernel<Float>>(
                sycl::range<1>(column_count),
                [=](sycl::id<1> idx) {
                    centroids_ptr[idx + ic * column_count] = data_ptr[idx + index * column_count];
                });
        });
        events.push_back(copy_event);
        cpos++;
    }
    return std::make_tuple(objective_function, events);
}

template <typename Float>
sycl::event find_candidates(sycl::queue& queue,
                            pr::ndview<Float, 2>& closest_distances,
                            std::int64_t candidate_count,
                            pr::ndview<std::int32_t, 1>& candidate_indices,
                            pr::ndview<Float, 1>& candidate_distances,
                            const bk::event_vector& deps = {}) {
    ONEDAL_ASSERT(candidate_count > 0);
    ONEDAL_ASSERT(closest_distances.get_dimension(0) > candidate_count);
    ONEDAL_ASSERT(closest_distances.get_dimension(1) == 1);
    ONEDAL_ASSERT(candidate_indices.get_dimension(0) == candidate_indices.get_dimension(0));
    ONEDAL_ASSERT(candidate_indices.get_dimension(0) >= candidate_count);
    const auto elem_count = closest_distances.get_dimension(0);
    auto indices =
        pr::ndarray<std::int32_t, 1>::empty(queue, { elem_count }, sycl::usm::alloc::device);
    auto values = pr::ndview<Float, 1>::wrap(closest_distances.get_mutable_data(), { elem_count });
    auto values_ptr = values.get_mutable_data();
    std::int32_t* indices_ptr = indices.get_mutable_data();
    auto fill_event = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<set_indices<Float>>(sycl::range<1>(elem_count), [=](sycl::id<1> idx) {
            indices_ptr[idx] = idx;
            values_ptr[idx] *= -1.0;
        });
    });
    pr::radix_sort_indices_inplace<Float, std::int32_t>{ queue }(values, indices, { fill_event })
        .wait_and_throw();
    auto candidate_indices_ptr = candidate_indices.get_mutable_data();
    auto candidate_distances_ptr = candidate_distances.get_mutable_data();
    auto copy_event = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<find_candidates_kernel<Float>>(
            sycl::range<1>(candidate_count),
            [=](sycl::id<1> idx) {
                candidate_distances_ptr[idx] = -1.0 * values_ptr[idx];
                candidate_indices_ptr[idx] = indices_ptr[idx];
            });
    });
    copy_event.wait_and_throw();
    return copy_event;
}

template <typename Float>
sycl::event merge_reduce_centroids(sycl::queue& queue,
                                   const pr::ndview<std::int32_t, 1>& counters,
                                   const pr::ndview<Float, 2>& partial_centroids,
                                   std::int64_t part_count,
                                   pr::ndview<Float, 2>& centroids,
                                   const bk::event_vector& deps = {}) {
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

template <typename Float>
sycl::event partial_reduce_centroids(sycl::queue& queue,
                                     const pr::ndview<Float, 2>& data,
                                     const pr::ndview<std::int32_t, 2>& labels,
                                     std::int64_t cluster_count,
                                     std::int64_t part_count,
                                     pr::ndview<Float, 2>& partial_centroids,
                                     const bk::event_vector& deps = {}) {
    ONEDAL_ASSERT(data.get_dimension(1) == partial_centroids.get_dimension(1));
    ONEDAL_ASSERT(partial_centroids.get_dimension(0) == cluster_count * part_count);
    ONEDAL_ASSERT(labels.get_dimension(0) == data.get_dimension(0));
    ONEDAL_ASSERT(labels.get_dimension(1) == 1);
    const Float* data_ptr = data.get_data();
    const std::int32_t* label_ptr = labels.get_data();
    Float* partial_centroids_ptr = partial_centroids.get_mutable_data();
    const auto row_count = data.get_dimension(0);
    const auto column_count = data.get_dimension(1);
    const auto sg_size_to_set = get_recommended_sg_size(queue);
    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for<centroid_reduction<Float>>(
            bk::make_multiple_nd_range_2d({ sg_size_to_set, part_count }, { sg_size_to_set, 1 }),
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

template <typename Float>
sycl::event compute_objective_function(sycl::queue& queue,
                                       const pr::ndview<Float, 2>& closest_distances,
                                       pr::ndview<Float, 1>& objective_function,
                                       const bk::event_vector& deps = {}) {
    ONEDAL_ASSERT(closest_distances.get_dimension(1) == 1);
    ONEDAL_ASSERT(objective_function.get_dimension(0) == 1);
    const Float* distance_ptr = closest_distances.get_data();
    Float* value_ptr = objective_function.get_mutable_data();
    const auto row_count = closest_distances.get_dimension(0);
    const auto sg_size_to_set = get_recommended_sg_size(queue);
    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for<compute_obj_function<Float>>(
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
#endif

} // namespace oneapi::dal::kmeans::backend
