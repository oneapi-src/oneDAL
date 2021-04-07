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

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

struct distance_struct {
    static sycl::event compute(const ndview<Float, 2>& data,
                                const ndview<Float, 2>& centroids,
                                ndview<Float, 2>& distances,
                                const event_vector& deps = {}) { return sycl::event(); }
};

template <typename Float>
class kmeans_core {
public:
    kmeans_core(sycl::queue& queue, std::int64_t row_count, std::int64_t column_count, descriptor& desc) {
        // allocate blocks for distances, partial indices
    }
    auto update_clusters(const ndview<Float, 2>& data,
                                const ndview<Float, 2>& centroids,
                                ndview<std::int32_t, 1>& labels,
                                const event_vector& deps = {}) {
        // counters.fill(0);
        sycl::event selection_event;
        sycl::event count_event;
        for(std::uint32_t iblock = 0; iblock < num_blocks_; iblock++) {
            auto block_rows = block_rows_;
            auto row_offset = block_rows_ * iblock;
            if(block_rows + row_offset > row_count) {
                block_rows = row_count - row_offset;
            }
            auto data_block = ndarray<Float, 2>::wrap(queue_, data.get_data() + row_offset * columns_, {block_rows, columns_});
            auto distance_block = ndarray<Float, 2>::wrap(queue_, distances_.get_data(), {block_rows, num_centroids_});
            auto distance_event = distance_struct::compute(data_block, centroids, distances_, {selection_event});
            auto label_block = ndarray<int32_t, 1>::wrap(queue_, labels.get_mutable_data() + row_offset, block_rows);
            auto closest_distance_block = ndarray<Float, 2>::wrap(queue_, closest_distances_.get_mutable_data() + row_offset, {block_rows, 1});
            kselect_by_rows selector(queue_, distances_.get_shape(), 1);
            selection_event = selector(queue_, distance_block, 1, closest_distance_block, label_block, {distance_event}));
            count_event = count_clusters(label_block, {count_event, selection_event});
        }
        auto obj_event = compute_objective_function(data, labels, closest_distances_, {selection_event});
        return std::make_tuple(selection_event, obj_event, count_event);
    }
    auto reduce_centroids(const ndview<Float, 2>& data,
                          const ndview<std::int32_t, 1>& labels,
                                const ndview<Float, 2>& centroids,
                                const event_vector& deps = {}) {
        // partial_centroids_.fill(0);
        Float * data_ptr = data.get_data();
        std::int32_t* label_ptr = labels.get_data();
        Float * partial_centroids_ptr = partial_centroids.get_mutable_data();
        Float * centroids_ptr = centroids.get_mutable_data();
        auto event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.parallel_for(range<1>{16, num_parts_}, { 16, 1 }), [=](sycl::nd_item<2> item) {
                const std::uint32_t sg_id = sg.get_group_id()[0];
                const std::uint32_t wg_id = item.get_global_id(1);
                const std::uint32_t wg_num = item.get_global_range(1);
                const std::uint32_t sg_num = sg.get_group_range()[0];
                const std::uint32_t sg_global_id = wg_id * sg_num + sg_id;
                if(sg_global_id >= num_parts_) return;

                const std::uint32_t local_id = sg.get_local_id()[0];
                const std::uint32_t local_range = sg.get_local_range()[0];
                for(std::uint32_t i = sg_global_id; i < row_count; i += sg_num) {
                    std::int32_t cl = -1;
                    if(local_id == 0) {
                        std::int32_t cl = label_ptr[i];
                    }    
                    cl = reduce(sg, sum, sycl::ONEAPI::maximum<std::int32_t>());
                    for(std::uint32_t j = local_id; j < col_count; j+ = local_range){
                        partial_centroids_ptr[sg_global_id * num_clusters_ * col_count + cl * col_count + j] = data_ptr[cl * col_count + local_id];
                    }
                }
            });
        });    
        auto final_event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on({event});
            cgh.parallel_for(range<1>{16, col_count_ * num_clusters_}, { 16, 1 }), [=](sycl::nd_item<2> item) {
                const std::uint32_t sg_id = sg.get_group_id()[0];
                const std::uint32_t wg_id = item.get_global_id(1);
                const std::uint32_t wg_num = item.get_global_range(1);
                const std::uint32_t sg_num = sg.get_group_range()[0];
                const std::uint32_t sg_global_id = wg_id * sg_num + sg_id;
                if(sg_global_id >= col_count) return;

                const std::uint32_t local_id = sg.get_local_id()[0];
                const std::uint32_t local_range = sg.get_local_range()[0];
                Float * sum = 0.0;
                for(std::int32_t i = local_id; i < num_parts_; i += local_range) {
                    sum += partial_centroids_ptr[i * num_clusters_ * col_count + sg_global_id];
                }
                sum = reduce(sg, sum, sycl::ONEAPI::plus<Float>());
                if(local_id == 0)
                    centroids_ptr[sg_global_id] = sum;
            });
        });   
        return final_event; 
    }

    sycl::event count_clusters(sycl::queue& queue, ndview<std::int32_t, 1>& labels,
                           const event_vector& deps = {})  {
        std::int32t_t* label_ptr = closest_distances_.get_data();
        std::uint32_t* counter_ptr = counters_.get_mutable_data();
        std::uint32_t* value_ptr = &num_empty_clusters_;
        auto event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.parallel_for(range<1>{16, 128}, { 16, 1 }), [=](sycl::nd_item<2> item) {
                const std::uint32_t sg_id = sg.get_group_id()[0];
                const std::uint32_t wg_id = item.get_global_id(1);
                const std::uint32_t wg_num = item.get_global_range(1);
                const std::uint32_t sg_num = sg.get_group_range()[0];
                const std::uint32_t sg_global_id = wg_id * sg_num + sg_id;
                const std::uint32_t total_sg_num = wg_num * sg_num;
                const std::uint32_t block_size = row_count / total_sg_num + std::uint32_t(row_count % total_sg_num > 0);
                const std::uint32_t offset = block_size * sg_global_id;
                const std::uint32_t end = (offset + block_size) > row_count ? row_count : (offset + block_size);
                for(std::uint32_t i = offset + local_id; i < end; i += local_range) {
                    const std::int32_t cl = label_ptr[i];
                    sycl::atomic_ref counter_atomic(counter_ptr[cl]);
                    counter_atomic.fetch_add(1);
                }
            });
        });
        auto final_event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on({event});
            cgh.parallel_for(range<1>{16, 1}, { 16, 1 }), [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const std::uint32_t sg_id = sg.get_group_id()[0];
                if(sg_id > 0) return;
                const std::uint32_t local_id = sg.get_local_id()[0];
                const std::uint32_t local_range = sg.get_local_range()[0];
                std::uint32_t sum = 0;
                for(std::uint32_t i = local_id; i < num_clusters_; i += local_range) {
                    sum += counter_ptr[i] == 0;
                }
                sum = reduce(sg, sum, sycl::ONEAPI::plus<std::int32_t>());
                if(local_id == 0) {
                    value_ptr[0] = sum;
                }
            });
        });
        return final_event;
    }
    sycl::event compute_obective_function(sycl::queue& queue,
                           const event_vector& deps = {})  {
        Float * distance_ptr = closest_distances_.get_data();
        Float * value_ptr = &objective_function_;
        auto event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.parallel_for(make_multiple_nd_range_2d({16, 1}, { 16, 1 }), [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const std::uint32_t sg_id = sg.get_group_id()[0];
                if(sg_id > 0) return;
                const std::uint32_t local_id = sg.get_local_id()[0];
                const std::uint32_t local_range = sg.get_local_range()[0];
                Float sum = 0;
                for(std::uint32_t i = local_id; i < row_count_; i += local_range) {
                    sum += distance_ptr[i];
                }
                sum = reduce(sg, sum, sycl::ONEAPI::plus<Float>());
                if(local_id == 0) {
                    value_ptr[0] = sum;
                }
            });
        });
        return event;
    }
    sycl::event compute_obective_function(sycl::queue& queue,
                           const event_vector& deps = {})  {
        Float * distance_ptr = closest_distances_.get_data();
        Float * value_ptr = &objective_function_;
        auto event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.parallel_for(make_multiple_nd_range_2d({16, 1}, { 16, 1 }), [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const std::uint32_t sg_id = sg.get_group_id()[0];
                if(sg_id > 0) return;
                const std::uint32_t local_id = sg.get_local_id()[0];
                const std::uint32_t local_range = sg.get_local_range()[0];
                Float sum = 0;
                for(std::uint32_t i = local_id; i < row_count_; i += local_range) {
                    sum += distance_ptr[i];
                }
                sum = reduce(sg, sum, sycl::ONEAPI::plus<Float>());
                if(local_id == 0) {
                    value_ptr[0] = sum;
                }
            });
        });
        return event;
    }    
    std::int64_t get_num_empty_clusters() { return num_empty_clusters_; }
    Float get_objective_function() { return objective_function_; }
private:
    std::ndarray<Float, 2> distances_;
    std::ndarray<Float, 2> partial_centroids_;
    std::int32_t num_empty_clusters_;
    Float objective_function_;
};
#endif

} // namespace oneapi::dal::backend::primitives
