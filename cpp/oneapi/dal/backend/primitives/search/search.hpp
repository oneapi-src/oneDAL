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

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/distance.hpp"
#include "oneapi/dal/backend/primitives/selection.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float>
std::int64_t propose_train_block(const sycl::queue& q, std::int64_t width);

template <typename Float>
std::int64_t propose_query_block(const sycl::queue& q, std::int64_t width);

template<typename Float>
class search_temp_objects;

template <typename Float, typename Distance>
class search_engine {
    using temp_t = search_temp_objects>Float>;

public:
    search_engine(sycl::queue& queue, const ndview<Float, 2>& train_data)
            : search_engine(queue,
                            train_data,
                            propose_train_block(queue, train_data.get_dimension(1))) {}

    search_engine(sycl::queue& queue, const ndview<Float, 2>& train_data, std::int64_t train_block)
            : search_engine(queue, train_data, train_block, Distance(queue)) {}

    search_engine(sycl::queue& queue,
                  const ndview<Float, 2>& train_data,
                  std::int64_t train_block,
                  const Distance& distance_instance)
            : queue_(queue),
              train_data_(train_data),
              train_blocking_(train_data.get_dimension(0), train_block),
              distance_instance_(distance_instance) {}

    template <typename Callback>
    sycl::event operator()(const ndview<Float, 2>& query_data,
                           Callback& callback,
                           std::int64_t k_neighbors = 1,
                           const event_vector& deps = {}) const {
        const auto query_block = propose_query_block(queue_, query_data.get_dimension(1));
        return this->operator()(query_data.callback, k_neighbors, query_block, deps);
    }

    template <typename Callback>
    sycl::event operator()(const ndview<Float, 2>& query_data,
                           Callback& callback,
                           std::int64_t query_block,
                           std::int64_t k_neighbors = 1,
                           const event_vector& deps = {}) const {
        const uniform_blocking query_blocking(query_data.get_dimension(0), query_block);
        auto tmp_objs = create_temporary_objects(query_blocking);
        sycl::event
        for(std::int64_t qb_id = 0; qb_id < query_blocking.get_block_count(); ++qb_id) {
            const auto query_slice = get_row_slice(query_data,
                                                   query_blocking.get_block_start_index(qb_id),
                                                   query_blocking.get_block_end_index(qb_id));
            for(std::int64_t tb_id = 0; tb_id < train_blocking_.get_block_count(); ++tb_id) {
                const auto train_slice = get_row_slice(train_data_,
                                                       train_blocking_.get_block_start_index(tb_id),
                                                       train_blocking_.get_block_end_index(tb_id));
                auto dst_event = compute_distances(tmp_objs, train_slice, query_slice, )
                auto sel_event = select_neighbors(tmp_objs, k_neighbors, {dst_event});
                auto cb_event =
            }
        }
        return dispose_temporary_objects(tmp_objs, cb_event);
    }

private:
    temp_t create_temporary_objects(const uniform_blocking& query_blocking);
    sycl::event dispose_temporary_objects(const )


    sycl::queue& queue_;
    const ndview<Float, 2>& train_data_;
    const Distance distance_instance_;
    const uniform_blocking train_blocking_;
};

#endif

} // namespace oneapi::dal::backend::primitives
