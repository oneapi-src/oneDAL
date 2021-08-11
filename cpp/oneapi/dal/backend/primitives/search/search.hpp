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

template<typename Float, typename Distance>
class search_temp_objects;

template <typename Float, typename Distance>
class search_engine {
    using temp_t = search_temp_objects<Float, Distance>;
    constexpr static inline bool is_l2_squared =
        std::is_same_v<Distance, squared_l2_distance<Float>>;

    constexpr static inline std::int64_t selection_sub_blocks = 16;

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
              distance_instance_(distance_instance),
              train_blocking_(train_data.get_dimension(0), train_block),
              selection_blocking_(train_blocking_.get_block_count(), selection_sub_blocks) {}

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
        const ndshape<2> typical_blocking{get_train_blocking().get_block(), query_blocking.get_block()};
        k_select_by_rows selection(get_queue(), typical_blocking, k_neighbors);
        auto tmp_objs = create_temporary_objects(query_blocking, k_neighbors);
        sycl::event last_event;
        for(std::int64_t qb_id = 0; qb_id < query_blocking.get_block_count(); ++qb_id) {
            const auto query_slice = query_data.get_row_slice(
                                                   query_blocking.get_block_start_index(qb_id),
                                                   query_blocking.get_block_end_index(qb_id));
            auto search_event = do_search(query_slice, k_neighbors, tm_objs, selection, deps + last_event);
            last_event = callback(qb_id, get_indices(tmp_objs), get_distances(tmp_objs), { search_event });
        }
        return dispose_temporary_objects(std::move(tmp_objs), { last_event  );
    }

protected:
    sycl::queue& get_queue();
    const Distance& get_distance() const;
    const uniform_blocking& get_train_blocking();
    const uniform_blocking& get_selection_blocking();
    ndview<Float, 2> get_train_block(std::int64_t idx);
    temp_t create_temporary_objects(const uniform_blocking& query_blocking);
    sycl::event dispose_temporary_objects(temp_t&& tmp_objs, const event_vector& deps);
    sycl::event distance(const ndview<Float, 2> query,
                         const ndview<Float, 2> train,
                         const event_vector& deps) const;
    sycl::event do_search(const ndview<Float, 2>& query,
                          std::int64_t k_neighbors,
                          temp_t& temp_objs,
                          k_select_by_rows& select,
                          const event_vector& deps);
    static ndview<Float, 2> get_distances(temp_t& tmp_objs);
    static ndview<std::int64_t, 2> get_indices(temp_t& tmp_objs);

private:
    sycl::queue& queue_;
    const Distance distance_instance_;
    const ndview<Float, 2>& train_data_;
    const uniform_blocking train_blocking_;
    const uniform_blocking selection_blocking_;
};

#endif

} // namespace oneapi::dal::backend::primitives
