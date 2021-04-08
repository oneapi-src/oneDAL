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

namespace oneapi::dal::kmeans::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace prm = dal::backend::primitives;

using descriptor_t = detail::descriptor_base<task::clustering>;

template <typename Float>
struct distance_struct {
    static sycl::event compute(const prm::ndview<Float, 2>& data,
                               const prm::ndview<Float, 2>& centroids,
                               prm::ndview<Float, 2>& distances,
                               const bk::event_vector& deps = {}) {
        return sycl::event();
    }
};

// TODO automation of range set
template <typename Float>
class kmeans_impl {
public:
    kmeans_impl(sycl::queue& queue,
                std::int64_t row_count,
                std::int64_t column_count,
                const descriptor_t& desc)
            : queue_(queue),
              row_count_(row_count),
              column_count_(column_count),
              num_centroids_(desc.get_cluster_count()) {
        // TODO allocate blocks for distances, partial indices
        // TODO calc number of blocks, etc.
    }
    auto update_clusters(const prm::ndview<Float, 2>& data,
                         const prm::ndview<Float, 2>& centroids,
                         prm::ndview<std::int32_t, 1>& labels,
                         const bk::event_vector& deps = {}) {
        // TODO counters.fill(0);
        sycl::event selection_event;
        sycl::event count_event;
        for (std::uint32_t iblock = 0; iblock < num_blocks_; iblock++) {
            auto block_rows = block_rows_;
            auto row_offset = block_rows_ * iblock;
            if (block_rows + row_offset > row_count_) {
                block_rows = row_count_ - row_offset;
            }
            auto data_block =
                prm::ndarray<Float, 2>::wrap(data.get_data() + row_offset * column_count_,
                                             { block_rows, column_count_ });
            auto distance_block =
                prm::ndarray<Float, 2>::wrap(distances_.get_data(), { block_rows, num_centroids_ });
            auto distance_event = distance_struct<Float>::compute(data_block,
                                                                  centroids,
                                                                  distances_,
                                                                  { selection_event });
            auto label_block =
                prm::ndarray<int32_t, 2>::wrap(labels.get_mutable_data() + row_offset,
                                               { block_rows, 1 });
            auto closest_distance_block =
                prm::ndarray<Float, 2>::wrap(closest_distances_.get_mutable_data() + row_offset,
                                             { block_rows, 1 });
            prm::kselect_by_rows<Float> selector(queue_, distances_.get_shape(), 1);
            selection_event = selector(queue_,
                                       distance_block,
                                       1,
                                       closest_distance_block,
                                       label_block,
                                       { distance_event });
            count_event = count_clusters(label_block, { count_event, selection_event });
        }
        auto obj_event = compute_objective_function({ selection_event });
        return std::make_tuple(selection_event, obj_event, count_event);
    }
    auto reduce_centroids(const prm::ndview<Float, 2>& data,
                          const prm::ndview<std::int32_t, 1>& labels,
                          const prm::ndview<Float, 2>& centroids,
                          const bk::event_vector& deps = {}) {
        // TODO partial_centroids_.fill(0);
        const Float* data_ptr = data.get_data();
        const std::int32_t* label_ptr = labels.get_data();
        Float* partial_centroids_ptr = partial_centroids_.get_mutable_data();
        Float* centroids_ptr = centroids.get_mutable_data();
        const std::int32_t* counters_ptr = counters_.get_data();
        auto event = queue_.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            const auto row_count = this->row_count_;
            const auto column_count = this->column_count_;
            const auto num_centroids = this->num_centroids_;
            const auto num_parts = this->num_parts_;
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
                    for (std::uint32_t i = sg_global_id; i < row_count; i += sg_num) {
                        std::int32_t cl = -1;
                        if (local_id == 0) {
                            cl = label_ptr[i];
                        }
                        cl = reduce(sg, cl, sycl::ONEAPI::maximum<std::int32_t>());
                        for (std::uint32_t j = local_id; j < column_count; j += local_range) {
                            partial_centroids_ptr[sg_global_id * num_centroids * column_count +
                                                  cl * column_count + j] =
                                data_ptr[cl * column_count + local_id];
                        }
                    }
                });
        });
        auto final_event = queue_.submit([&](sycl::handler& cgh) {
            cgh.depends_on({ event });
            const auto column_count = this->column_count_;
            const auto num_centroids = this->num_centroids_;
            const auto num_parts = this->num_parts_;
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

                    const std::uint32_t local_id = sg.get_local_id()[0];
                    const std::uint32_t local_range = sg.get_local_range()[0];
                    Float sum = 0.0;
                    for (std::uint32_t i = local_id; i < num_parts; i += local_range) {
                        sum +=
                            partial_centroids_ptr[i * num_centroids * column_count + sg_global_id];
                    }
                    sum = reduce(sg, sum, sycl::ONEAPI::plus<Float>());
                    if (local_id == 0)
                        centroids_ptr[sg_global_id] = sum / counters_ptr[sg_global_id];
                });
        });
        return final_event;
    }

    sycl::event count_clusters(prm::ndview<std::int32_t, 2>& labels,
                               const bk::event_vector& deps = {}) {
        const std::int32_t* label_ptr = labels.get_data();
        std::int32_t* counter_ptr = counters_.get_mutable_data();
        std::uint32_t* value_ptr = &num_empty_clusters_;
        auto event = queue_.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            const auto row_count = this->row_count_;
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
        auto final_event = queue_.submit([&](sycl::handler& cgh) {
            cgh.depends_on({ event });
            const auto num_centroids = this->num_centroids_;
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
    sycl::event compute_objective_function(const bk::event_vector& deps = {}) {
        const Float* distance_ptr = closest_distances_.get_data();
        Float* value_ptr = &objective_function_;
        auto event = queue_.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            const auto row_count = this->row_count_;
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

    std::int64_t get_num_empty_clusters() {
        return num_empty_clusters_;
    }
    Float get_objective_function() {
        return objective_function_;
    }

private:
    prm::ndarray<Float, 2> distances_;
    prm::ndarray<Float, 2> partial_centroids_;
    prm::ndarray<std::int32_t, 1> counters_;
    prm::ndarray<Float, 1> closest_distances_;
    sycl::queue queue_;
    std::int64_t row_count_;
    std::int64_t column_count_;
    std::int64_t num_centroids_;
    std::uint32_t num_empty_clusters_;
    std::uint32_t num_parts_ = 126;
    std::uint32_t num_blocks_;
    std::uint32_t block_rows_;
    Float objective_function_;
};
#endif

} // namespace oneapi::dal::kmeans::backend
