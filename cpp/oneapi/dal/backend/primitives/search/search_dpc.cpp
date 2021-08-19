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

#include "oneapi/dal/detail/common.hpp"

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/search/search.hpp"
#include "oneapi/dal/backend/primitives/distance/distance.hpp"
#include "oneapi/dal/backend/primitives/selection/select_indexed.hpp"
#include "oneapi/dal/backend/primitives/selection/kselect_by_rows.hpp"

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

template <typename Float, typename Distance>
class search_temp_objects_base {
public:
    search_temp_objects_base(sycl::queue& q,
                             std::int64_t k,
                             std::int64_t query_block,
                             std::int64_t train_block,
                             std::int64_t select_block)
            : k_{ k },
              qblock_{ query_block },
              sblock_{ select_block },
              distances_{ ndarray<Float, 2>::empty(q,
                                                   { query_block, train_block },
                                                   sycl::usm::alloc::device) },
              out_indices_(
                  ndarray<std::int32_t, 2>::empty(q, { query_block, k }, sycl::usm::alloc::device)),
              out_distances_(
                  ndarray<Float, 2>::empty(q, { query_block, k }, sycl::usm::alloc::device)),
              part_indices_(ndarray<std::int32_t, 2>::empty(q,
                                                            { query_block, k * (select_block + 1) },
                                                            sycl::usm::alloc::device)),
              part_distances_(ndarray<Float, 2>::empty(q,
                                                       { query_block, k * (select_block + 1) },
                                                       sycl::usm::alloc::device)) {
        dal::detail::check_mul_overflow(query_block, k);
        dal::detail::check_mul_overflow(query_block, train_block);
        dal::detail::check_mul_overflow(query_block, k * (select_block + 1));
    }

    std::int64_t get_k() const {
        return k_;
    }

    std::int64_t get_query_block() const {
        return qblock_;
    }

    std::int64_t get_select_block() const {
        return sblock_;
    }

    ndview<Float, 2>& get_distances() {
        return distances_;
    }

    ndview<std::int32_t, 2>& get_out_indices() {
        return out_indices_;
    }

    ndview<Float, 2>& get_out_distances() {
        return out_distances_;
    }

    ndview<std::int32_t, 2>& get_part_indices() {
        return part_indices_;
    }

    ndview<Float, 2>& get_part_distances() {
        return part_distances_;
    }

    ndview<std::int32_t, 2> get_part_indices_block(std::int64_t idx) {
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
    const std::int64_t k_, qblock_, sblock_;
    ndarray<Float, 2> distances_;
    ndarray<std::int32_t, 2> out_indices_;
    ndarray<Float, 2> out_distances_;
    ndarray<std::int32_t, 2> part_indices_;
    ndarray<Float, 2> part_distances_;
};

template <typename Float, typename Distance>
class search_temp_objects : public search_temp_objects_base<Float, Distance> {
    using base_t = search_temp_objects_base<Float, Distance>;

public:
    search_temp_objects(sycl::queue& q,
                        std::int64_t k,
                        std::int64_t query_block,
                        std::int64_t train_block,
                        std::int64_t select_block)
            : base_t(q, k, query_block, train_block, select_block) {}
};

template <typename Float, typename Distance>
search_engine<Float, Distance>::search_engine(sycl::queue& queue,
                                              const ndview<Float, 2>& train_data)
        : search_engine(queue,
                        train_data,
                        propose_train_block<Float>(queue, train_data.get_dimension(1))) {}

template <typename Float, typename Distance>
search_engine<Float, Distance>::search_engine(sycl::queue& queue,
                                              const ndview<Float, 2>& train_data,
                                              std::int64_t train_block)
        : search_engine(queue, train_data, train_block, Distance(queue)) {}

template <typename Float, typename Distance>
search_engine<Float, Distance>::search_engine(sycl::queue& queue,
                                              const ndview<Float, 2>& train_data,
                                              std::int64_t train_block,
                                              const Distance& distance_instance)
        : queue_(queue),
          distance_instance_(distance_instance),
          train_data_(train_data),
          train_blocking_(train_data.get_dimension(0), train_block),
          selection_blocking_(train_blocking_.get_block_count(), selection_sub_blocks) {}

template <typename Float, typename Distance>
sycl::queue& search_engine<Float, Distance>::get_queue() const {
    return this->queue_;
}

template <typename Float, typename Distance>
const Distance& search_engine<Float, Distance>::get_distance_impl() const {
    return this->distance_instance_;
}

template <typename Float, typename Distance>
sycl::event search_engine<Float, Distance>::distance(const ndview<Float, 2>& query,
                                                     const ndview<Float, 2>& train,
                                                     ndview<Float, 2>& dists,
                                                     const event_vector& deps) const {
    ONEDAL_ASSERT(query.has_data());
    ONEDAL_ASSERT(train.has_data());
    ONEDAL_ASSERT(dists.has_mutable_data());
    ONEDAL_ASSERT(query.get_dimension(1) == train.get_dimension(1));
    ONEDAL_ASSERT(train.get_dimension(0) == dists.get_dimension(1));
    ONEDAL_ASSERT(query.get_dimension(0) == dists.get_dimension(0));
    return get_distance_impl()(query, train, dists, deps);
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
ndview<Float, 2> search_engine<Float, Distance>::get_train_block(std::int64_t i) const {
    const auto from = get_train_blocking().get_block_start_index(i);
    const auto to = get_train_blocking().get_block_end_index(i);
    return train_data_.get_row_slice(from, to);
}

template <typename Float, typename Distance>
auto search_engine<Float, Distance>::create_temporary_objects(
    const uniform_blocking& query_blocking,
    std::int64_t k_neighbors) const -> temp_ptr_t {
    return new temp_t(get_queue(),
                      k_neighbors,
                      query_blocking.get_block(),
                      get_train_blocking().get_block(),
                      selection_sub_blocks);
}

template <typename Float, typename Distance>
sycl::event search_engine<Float, Distance>::dispose_temporary_objects(
    temp_ptr_t temp,
    const event_vector& deps) const {
    sycl::event::wait_and_throw(deps);
    delete temp;
    return sycl::event();
}

template <typename Float, typename Distance>
auto search_engine<Float, Distance>::create_selection_objects(std::int64_t query_block,
                                                              std::int64_t k_neighbors) const
    -> selc_t {
    const auto train_block = get_train_blocking().get_block();
    dal::detail::check_mul_overflow(k_neighbors, (selection_sub_blocks + 1));
    const auto width =
        std::max<std::int64_t>(k_neighbors * (selection_sub_blocks + 1), train_block);
    const ndshape<2> typical_blocking(query_block, width);
    return selc_t(get_queue(), typical_blocking, k_neighbors);
}

template <typename Float, typename Distance>
ndview<std::int32_t, 2> search_engine<Float, Distance>::get_indices(temp_ptr_t tmp_objs) {
    return tmp_objs->get_out_indices();
}

template <typename Float, typename Distance>
ndview<Float, 2> search_engine<Float, Distance>::get_distances(temp_ptr_t tmp_objs) {
    return tmp_objs->get_out_distances();
}

template <typename Float, typename Distance>
sycl::event search_engine<Float, Distance>::reset(temp_ptr_t tmp_objs,
                                                  const event_vector& deps) const {
    constexpr Float default_dst_value = detail::limits<Float>::max();
    constexpr std::int32_t default_idx_value = -1;
    auto out_dsts = fill(get_queue(), tmp_objs->get_out_distances(), default_dst_value, deps);
    auto out_idcs = fill(get_queue(), tmp_objs->get_out_indices(), default_idx_value, deps);
    auto part_dsts = fill(get_queue(), tmp_objs->get_part_distances(), default_dst_value, deps);
    auto part_idcs = fill(get_queue(), tmp_objs->get_part_indices(), default_idx_value, deps);
    const auto fill_events = out_dsts + out_idcs + part_dsts + part_idcs;
    return fill(get_queue(), tmp_objs->get_distances(), default_dst_value, fill_events);
}

template <typename Float, typename Distance>
sycl::event search_engine<Float, Distance>::treat_indices(ndview<std::int32_t, 2>& indices,
                                                          std::int64_t start_index,
                                                          const event_vector& deps) const {
    ONEDAL_ASSERT(indices.has_mutable_data());
    auto* const ids_ptr = indices.get_mutable_data();
    const auto ids_str = indices.get_leading_stride();
    const ndshape<2> ids_shape = indices.get_shape();
    const auto tr_range = make_range_2d(ids_shape[0], ids_shape[1]);
    return get_queue().submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(tr_range, [=](sycl::id<2> idx) {
            *(ids_ptr + ids_str * idx[0] + idx[1]) += start_index;
        });
    });
}

template <typename Float, typename Distance>
sycl::event search_engine<Float, Distance>::select_indexed(const ndview<std::int32_t, 2>& src,
                                                           ndview<std::int32_t, 2>& dst,
                                                           const event_vector& deps) const {
    namespace pr = oneapi::dal::backend::primitives;
    ONEDAL_ASSERT(src.has_data());
    ONEDAL_ASSERT(dst.has_mutable_data());
    return pr::select_indexed(get_queue(), dst, src, dst, deps);
}

template <typename Float, typename Distance>
sycl::event search_engine<Float, Distance>::do_search(const ndview<Float, 2>& query,
                                                      std::int64_t k_neighbors,
                                                      temp_ptr_t temp_objs,
                                                      selc_t& select,
                                                      const event_vector& deps) const {
    ONEDAL_ASSERT(temp_objs->get_k() == k_neighbors);
    ONEDAL_ASSERT(temp_objs->get_select_block() == selection_sub_blocks);
    ONEDAL_ASSERT(temp_objs->get_query_block() >= query.get_dimension(0));
    sycl::event last_event = reset(temp_objs, deps);
    const auto query_block_size = query.get_dimension(0);
    //Iterations over larger blocks
    for (std::int64_t sb_id = 0; sb_id < get_selection_blocking().get_block_count(); ++sb_id) {
        const std::int64_t start_tb = get_selection_blocking().get_block_start_index(sb_id);
        const std::int64_t end_tb = get_selection_blocking().get_block_end_index(sb_id);
        //Iterations over smaller blocks
        for (std::int64_t tb_id = start_tb; tb_id < end_tb; ++tb_id) {
            const auto train = get_train_block(tb_id);
            const auto train_block_size = get_train_blocking().get_block_length(tb_id);
            ONEDAL_ASSERT(train.get_dimension(0) == train_block_size);
            auto dists = temp_objs->get_distances()
                             .get_col_slice(0, train_block_size)
                             .get_row_slice(0, query_block_size);
            auto dist_event = distance(query, train, dists, { last_event });

            const auto rel_idx = tb_id - start_tb;
            auto part_inds =
                temp_objs->get_part_indices_block(rel_idx + 1).get_row_slice(0, query_block_size);
            auto part_dsts =
                temp_objs->get_part_distances_block(rel_idx + 1).get_row_slice(0, query_block_size);
            auto selt_event =
                select(get_queue(), dists, k_neighbors, part_dsts, part_inds, { dist_event });

            const auto st_idx = get_train_blocking().get_block_start_index(tb_id);
            last_event = treat_indices(part_inds, st_idx, { selt_event });
        }
        dal::detail::check_mul_overflow(k_neighbors, (1 + end_tb - start_tb));
        const std::int64_t cols = k_neighbors * (1 + end_tb - start_tb);
        auto dists = temp_objs->get_part_distances().get_col_slice(0, cols);
        auto selt_event = select(get_queue(),
                                 dists,
                                 k_neighbors,
                                 temp_objs->get_out_distances(),
                                 temp_objs->get_out_indices(),
                                 { last_event });
        auto inds_event = select_indexed(temp_objs->get_part_indices(),
                                         temp_objs->get_out_indices(),
                                         { selt_event });

        auto part_indcs = temp_objs->get_part_indices_block(0);
        last_event = copy(get_queue(), part_indcs, temp_objs->get_out_indices(), { inds_event });
    }
    return last_event;
}

#define INSTANTIATE(F)                                                              \
    template std::int64_t propose_train_block<F>(const sycl::queue&, std::int64_t); \
    template std::int64_t propose_query_block<F>(const sycl::queue&, std::int64_t); \
    template class search_temp_objects<F, distance<F, lp_metric<F>>>;               \
    template class search_temp_objects<F, distance<F, squared_l2_metric<F>>>;       \
    template class search_engine<F, distance<F, lp_metric<F>>>;                     \
    template class search_engine<F, distance<F, squared_l2_metric<F>>>;

INSTANTIATE(float);
INSTANTIATE(double);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
