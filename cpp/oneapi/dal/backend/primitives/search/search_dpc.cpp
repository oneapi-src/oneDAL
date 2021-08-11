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

#include "oneapi/dal/backend/primitives/search.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
std::int64_t propose_train_block(const sycl::queue& q, std::int64_t width) {
    constexpr std::int64_t result = 4 * 4096;
    return result;
}

template <typename Float>
std::int64_t propose_query_block(const sycl::queue& q, std::int64_t width) {
    constexpr std::int64_t result = 4 * 2048 / sizeof(Float);
    return result;
}

template<typename Float, typename Distance>
struct search_temp_objects {
    search_temp_objects(sycl::queue& q,
                        std::int64_t k,
                        std::int64_t query_block,
                        std::int64_t train_block
                        std::int64_t select_block)
        : k_{k},
          distances_{ndarray<Float, 2>::empty(q, {query_block, train_block})},
          out_indices_(ndarray<std::int64_t, 2>::empty(q, {query_block, select_block})),
          out_distances_(ndarray<Float, 2>::empty(q, {query_block, k * select_block}))
          part_indices_(ndarray<std::int32_t, 2>::empty(q, {query_block, k * select_block})),
          part_distances_(ndarray<Float, 2>::empty(q, {query_block, k * select_block})) {};

    std::int64_t get_k() const {
        return k_;
    }

    ndview<Float, 2>& get_distances() {
        return distances_;
    }

    ndview<Float, 2>& get_out_indices() {
        return out_indices_;
    }

    ndview<Float, 2>& get_out_distances() {
        return out_distances_;
    }

    ndview<Float, 2>& get_part_indices() {
        return part_indices_;
    }

    ndview<Float, 2>& get_part_distances() {
        return part_distances_;
    }

    ndview<Float, 2> get_part_indices_block(std::int64_t idx) {
        const auto from = idx * get_k();
        const auto to = (idx + 1) * get_k();
        return get_part_indices().get_col_slice(from, to);
    }

    ndview<Float, 2> get_part_indices_block(std::int64_t idx) {
        const auto from = idx * get_k();
        const auto to = (idx + 1) * get_k();
        return get_part_indices().get_col_slice(from, to);
    }

    ndview<Float, 2> get_part_distances_block(std::int64_t idx) {
        const auto from = idx * get_k();
        const auto to = (idx + 1) * get_k();
        return get_part_distances().get_col_slice(from, to);
    }

private:
    const std::int64_t k_;
    ndarray<Float, 2> distances_;
    ndarray<std::int64_t, 2> out_indices_;
    ndarray<Float, 2> out_distances_;
    ndarray<Float, 2> out_distances_;
    ndarray<std::int32_t, 2> part_indices_;
    ndarray<Float, 2> part_distances_;
};

template <typename Float, typename Distance>
sycl::queue& search_engine<Float, Distance>::get_queue() {
    return this->queue_;
}

template <typename Float, typename Distance>
const Distance& search_engine<Float, Distance>::get_distance() const {
    return this->distance_instance_;
}

template <typename Float, typename Distance>
sycl::event search_engine<Float, Distance>::distance(const ndview<Float, 2> query,
                                                     const ndview<Float, 2> train,
                                                     const event_vector& deps) const {
    return get_distance()(query, train, deps);
}

template <typename Float, typename Distance>
const uniform_blocking& search_engine<Float, Distance>::get_train_blocking() const {
    return this->train_blocking_;
}

template <typename Float, typename Distance>
const uniform_blocking& search_engine<Float, Distance>::get_selection_blocking() const {
    return this->selection_blocking_;
}

template <typename Float, typename Distance>
auto search_engine<Float, Distance>::create_temporary_objects(const uniform_blocking& query_blocking, std::int64_t k_neighbors) const -> temp_t {
    return search_temp_objects(get_queue(),
                               k_neighbors,
                               query_blocking.get_block(),
                               get_train_blocking().get_block(),
                               select_sub_blocks);
}

template <typename Float, typename Distance>
sycl::event search_engine<Float, Distance>::dispose_temporary_objects(temp_t&& tmp_objs, const event_vector& deps) const {
    sycl::event::wait_and_throw(deps);
    return sycl::event();
}

template <typename Float, typename Distance>
static ndview<Float, 2> search_engine<Float, Distance>::get_out_indices(temp_t& tmp_objs) {
    return temp_objs.get_out_indices();
}

template <typename Float, typename Distance>
static ndview<Float, 2> search_engine<Float, Distance>::get_out_distances(temp_t& tmp_objs) {
    return temp_objs.get_out_distances();
}

template <typename Float, typename Distance>
sycl::event search_engine<Float, Distance>::do_search(const ndview<Float, 2>& query,
                                                      std::int64_t k_neighbors,
                                                      temp_t& temp_objs,
                                                      k_select_by_rows
                                                      event_vector& deps) {
    sycl::event last_event = reset(temp_objs, deps);
    //Iterations over larger blocks
    for(std::int64_t sb_id = 0; sb_id < get_train_blocking().get_block_count(); ++sb_id) {
        const std::int64_t start_tb = get_train_blocking().get_block_start_index(sb_id);
        const std::int64_t end_tb = get_train_blocking().get_block_end_index(sb_id);
        //Iterations over smaller blocks
        for(std::int64_t tb_id = start_tb; tb_id < end_tb; ++tb_id) {
            const auto train = get_train_block(tb_id);
            auto dist_event = distance(query,
                                       train,
                                       temp_objs.get_distances(),
                                       { last_event });
            const auto rel_idx = tb_id - start_tb;
            auto part_inds = temp_objs.get_out_indices_block(rel_idx);
            auto part_dsts = temp_objs.get_out_distances_block(rel_idx);
            last_event = select(temp_objs.get_distances(),
                                k_neighbors,
                                part_dsts,
                                part_inds,
                                { dist_event });
        }
        auto merge_event = merge(temp_objs, { last_event });
        const std::int64_t  = get_train_blocking().get_start_index(start_tb);
        last_event = treat_indices(temp_objs, { merge_event });
        last_event = back_copy(temp_objs, { last_event })
    }
    return last_event;
}

template<typename Float>
merge_kernel




#define INSTANTIATE(F)                                                          \
template std::int64_t propose_train_block<F>(const sycl::queue&, std::int64_t); \
template std::int64_t propose_query_block<F>(const sycl::queue&, std::int64_t); \
template class search<F, distance<F, lp_metric<F>>>;                            \
template class search<F, distance<F, squared_l2_metric<F>>};

INSTANTIATE(float);
INSTANTIATE(double);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
