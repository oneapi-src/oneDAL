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
#include "oneapi/dal/backend/primitives/selection/kselect_by_rows.hpp"
#include "oneapi/dal/backend/primitives/distance.hpp"
#include "oneapi/dal/backend/primitives/sort/sort.hpp"
#include "oneapi/dal/algo/kmeans/common.hpp"

namespace oneapi::dal::kmeans::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace prm = dal::backend::primitives;

template <typename Float>
auto find_candidates_impl(sycl::queue& queue, prm::ndview<Float, 1>& closest_distances,
                        std::int64_t num_candidates,
                        prm::ndview<std::int32_t, 1>& candidate_indices,
                        prm::ndview<Float, 1>& candidate_distances,
                        const bk::event_vector& deps) {
    auto elem_count = closest_distances.get_shape()[0];
    auto indices = prm::ndarray<std::int32_t, 1>::empty(queue, { elem_count }, sycl::usm::alloc::device);
    std::int32_t* indices_ptr = indices.get_mutable_data();
    auto fill_event = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(elem_count), [=](sycl::item<1> item) {
            std::int32_t ind = item.get_id()[0];
            indices_ptr[ind] = ind;
        });
    });
    auto sort_event = prm::radix_sort_indices_inplace<Float, std::int32_t>{ queue }(closest_distances, indices, {fill_event});
    sort_event.wait_and_throw();
    auto candidate_indices_ptr = candidate_indices.get_mutable_data();
    auto candidate_distances_ptr = candidate_distances.get_mutable_data();
    auto closest_distances_ptr = closest_distances.get_mutable_data();
    auto copy_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on({sort_event});
        cgh.parallel_for(sycl::range<1>(num_candidates), [=](sycl::item<1> item) {
            std::int32_t ind = item.get_id()[0];
            candidate_distances_ptr[ind] = closest_distances_ptr[ind];
            candidate_indices_ptr[ind] = indices_ptr[ind];
        });
    });
    copy_event.wait_and_throw();
    return copy_event;
}


template <typename Float, typename Metric>
auto assign_clusters_impl(sycl::queue& queue, const prm::ndview<Float, 2>& data,
                        const prm::ndview<Float, 2>& centroids,
                        std::int64_t block_rows,
                        prm::ndview<std::int32_t, 2>& labels,
                        prm::ndview<Float, 2>& distances,
                        prm::ndview<Float, 2>& closest_distances,
                        const bk::event_vector& deps) { 
    sycl::event selection_event;
    auto row_count = data.get_shape()[0];
    auto column_count = data.get_shape()[1];
    auto num_centroids = centroids.get_shape()[0];
    prm::distance<Float, Metric> block_distances(queue);
    auto num_blocks = row_count / block_rows + std::int64_t(row_count % block_rows > 0);
    for (std::int64_t iblock = 0; iblock < num_blocks; iblock++) {
        auto cur_rows = block_rows;
        auto row_offset = block_rows * iblock;
        if (cur_rows + row_offset > row_count) {
            cur_rows = row_count - row_offset;
        }
        auto distance_block =
                    prm::ndarray<Float, 2>::wrap(distances.get_mutable_data(), { cur_rows, num_centroids});
        auto data_block =
            prm::ndarray<Float, 2>::wrap(data.get_data() + row_offset * column_count,
                                            { cur_rows, column_count});
        auto distance_event = block_distances(data_block, centroids, distance_block, { selection_event });
/*        std::cout << "Waiting for distances" << std::endl;
        distance_event.wait_and_throw();
        for(int i = 0; i < cur_rows; i ++) {
            std::cout << "D: " << i;
            for(int j = 0; j < num_centroids; j++)
                std::cout << " " << distance_block.get_data()[i * num_centroids + j];
            std::cout << std::endl;
        }*/
        auto label_block =
            prm::ndarray<int32_t, 2>::wrap(labels.get_mutable_data() + row_offset,
                                            { cur_rows, 1 });
        auto closest_distance_block =
            prm::ndarray<Float, 2>::wrap(closest_distances.get_mutable_data() + row_offset,
                                            { cur_rows, 1 });
        prm::kselect_by_rows<Float> selector(queue, distances.get_shape(), 1);
        selection_event = selector(queue,
                                    distance_block,
                                    1,
                                    closest_distance_block,
                                    label_block,
                                    { distance_event });
/*        selection_event.wait_and_throw();
        for(int i = 0; i < cur_rows; i ++) {
            std::cout << "S: " << label_block.get_data()[i] << " " << closest_distance_block.get_data()[i] << std::endl;
        }*/
    }
    return selection_event;
}

template <typename Float>
sycl::event reduce_centroids_impl(sycl::queue& queue, const prm::ndview<Float, 2>& data,
                            const prm::ndview<std::int32_t, 2>& labels,
                            const prm::ndview<Float, 2>& centroids,
                            const prm::ndview<Float, 2>& partial_centroids,
                            const prm::ndview<std::int32_t, 1>& counters,
                            const std::uint32_t num_parts,
                            const bk::event_vector& deps = {}) {
    const Float* data_ptr = data.get_data();
    const std::int32_t* label_ptr = labels.get_data();
    Float* partial_centroids_ptr = partial_centroids.get_mutable_data();
    Float* centroids_ptr = centroids.get_mutable_data();
    const std::int32_t* counters_ptr = counters.get_data();
    const auto row_count = data.get_shape()[0];
    const auto column_count = data.get_shape()[1];
    const auto num_centroids = centroids.get_shape()[0];
    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(
            bk::make_multiple_nd_range_2d({ 16, num_parts }, { 16, 1 }),
            [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const std::uint32_t sg_id = sg.get_group_id()[0];
                const std::uint32_t wg_id = item.get_global_id(1);
                const std::uint32_t sg_num = sg.get_group_range()[0];
                const std::uint32_t sg_global_id = wg_id * sg_num + sg_id;
                if (sg_global_id >= num_parts)
                    return;
                const std::uint32_t local_id = sg.get_local_id()[0];
                const std::uint32_t local_range = sg.get_local_range()[0];
                for (std::uint32_t i = sg_global_id; i < row_count; i += num_parts) {
                    std::int32_t cl = -1;
                    if (local_id == 0) {
                        cl = label_ptr[i];
                    }
                    cl = reduce(sg, cl, sycl::ONEAPI::maximum<std::int32_t>());
                    for (std::uint32_t j = local_id; j < column_count; j += local_range) {
                        partial_centroids_ptr[sg_global_id * num_centroids * column_count +
                                                cl * column_count + j] +=
                            data_ptr[i * column_count + j];
                    }
                }
            });
    });
    event.wait_and_throw();

    auto final_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ event });
        cgh.parallel_for(
            bk::make_multiple_nd_range_2d({ 16, column_count * num_centroids }, { 16, 1 }),
            [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const std::uint32_t sg_id = sg.get_group_id()[0];
                const std::uint32_t wg_id = item.get_global_id(1);
                const std::uint32_t sg_num = sg.get_group_range()[0];
                const std::uint32_t sg_global_id = wg_id * sg_num + sg_id;
                if (sg_global_id >= column_count * num_centroids)
                    return;
                const std::uint32_t sg_cluster_id = sg_global_id / column_count;
                const std::uint32_t local_id = sg.get_local_id()[0];
                const std::uint32_t local_range = sg.get_local_range()[0];
                Float sum = 0.0;
                for (std::uint32_t i = local_id; i < num_parts; i += local_range) {
                    sum +=
                        partial_centroids_ptr[i * num_centroids * column_count + sg_global_id];
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
    return final_event;
}

sycl::event count_clusters_impl(sycl::queue& queue, const prm::ndview<std::int32_t, 2>& labels,
                                std::int64_t num_centroids,
                                prm::ndview<std::int32_t, 1>& counters,
                                prm::ndarray<std::int32_t, 1> num_empty_clusters,
                                const bk::event_vector& deps = {}) {
    const std::int32_t* label_ptr = labels.get_data();
    std::int32_t* counter_ptr = counters.get_mutable_data();
    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        const auto row_count = labels.get_shape()[0];
        cgh.parallel_for(
            bk::make_multiple_nd_range_2d({ 16, 128 }, { 16, 1 }),
            [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const std::uint32_t sg_id = sg.get_group_id()[0];
                const std::uint32_t wg_id = item.get_global_id(1);
                const std::uint32_t wg_num = item.get_global_range(1);
                const std::uint32_t sg_num = sg.get_group_range()[0];
                const std::uint32_t sg_global_id = wg_id * sg_num + sg_id;
                const std::uint32_t total_sg_num = wg_num * sg_num;

                const std::uint32_t local_id = sg.get_local_id()[0];
                const std::uint32_t local_range = sg.get_local_range()[0];

                const std::uint32_t block_size =
                    row_count / total_sg_num + std::uint32_t(row_count % total_sg_num > 0);
                const std::uint32_t offset = block_size * sg_global_id;
                const std::uint32_t end =
                    (offset + block_size) > row_count ? row_count : (offset + block_size);
                for (std::uint32_t i = offset + local_id; i < end; i += local_range) {
                    const std::int32_t cl = label_ptr[i];
                    sycl::ONEAPI::atomic_ref<
                        std::int32_t,
                        cl::sycl::ONEAPI::memory_order::relaxed,
                        cl::sycl::ONEAPI::memory_scope::device,
                        cl::sycl::access::address_space::global_device_space>
                        counter_atomic(counter_ptr[cl]);
                    counter_atomic.fetch_add(1);
                }
            });
    });
    std::int32_t* value_ptr = num_empty_clusters.get_mutable_data();
    auto final_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ event });
        cgh.parallel_for(bk::make_multiple_nd_range_2d({ 16, 1 }, { 16, 1 }),
                            [=](sycl::nd_item<2> item) {
                                auto sg = item.get_sub_group();
                                const std::uint32_t sg_id = sg.get_group_id()[0];
                                if (sg_id > 0)
                                    return;
                                const std::uint32_t local_id = sg.get_local_id()[0];
                                const std::uint32_t local_range = sg.get_local_range()[0];
                                std::uint32_t sum = 0;
                                for (std::uint32_t i = local_id; i < num_centroids;
                                    i += local_range) {
                                    sum += counter_ptr[i] == 0;
                                }
                                sum = reduce(sg, sum, sycl::ONEAPI::plus<std::int32_t>());
                                if (local_id == 0) {
                                    value_ptr[0] = sum;
                                }
                            });
    });
    return final_event;
}

template <typename Float>
sycl::event compute_objective_function_impl(sycl::queue& queue,
                                        const prm::ndview<Float, 1>& closest_distances,
                                        prm::ndarray<Float, 1> objective_function,
                                        const bk::event_vector& deps = {}) {
    const Float* distance_ptr = closest_distances.get_data();
    Float* value_ptr = objective_function.get_mutable_data();
    const auto row_count = closest_distances.get_shape()[0];
    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(bk::make_multiple_nd_range_2d({ 16, 1 }, { 16, 1 }),
                            [=](sycl::nd_item<2> item) {
                                auto sg = item.get_sub_group();
                                const std::uint32_t sg_id = sg.get_group_id()[0];
                                if (sg_id > 0)
                                    return;
                                const std::uint32_t local_id = sg.get_local_id()[0];
                                const std::uint32_t local_range = sg.get_local_range()[0];
                                Float sum = 0;
                                for (std::uint32_t i = local_id; i < row_count; i += local_range) {
                                    sum += distance_ptr[i];
                                }
                                sum = reduce(sg, sum, sycl::ONEAPI::plus<Float>());
                                if (local_id == 0) {
                                    value_ptr[0] = sum;
                                }
                            });
    });
    return event;
}

#endif

} // namespace oneapi::dal::kmeans::backend
