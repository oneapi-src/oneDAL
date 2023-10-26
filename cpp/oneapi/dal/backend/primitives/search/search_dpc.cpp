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
#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/blas/gemm.hpp"
#include "oneapi/dal/backend/primitives/search/search.hpp"
#include "oneapi/dal/backend/primitives/distance/distance.hpp"
#include "oneapi/dal/backend/primitives/selection/select_indexed.hpp"
#include "oneapi/dal/backend/primitives/selection/kselect_by_rows.hpp"
#include "oneapi/dal/backend/primitives/distance/cosine_distance_misc.hpp"
#include "oneapi/dal/backend/primitives/distance/squared_l2_distance_misc.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
std::int64_t propose_train_block(const sycl::queue& q, std::int64_t width) {
    constexpr std::int64_t result = 4096 * 8 / sizeof(Float);
    return result;
}

template <typename Float>
std::int64_t propose_query_block(const sycl::queue& q, std::int64_t width) {
    constexpr std::int64_t result = 8192 * 8 / sizeof(Float);
    return result;
}

template <typename Index>
sycl::event treat_indices(sycl::queue& q,
                          ndview<Index, 2>& indices,
                          std::int64_t start_index,
                          const event_vector& deps) {
    ONEDAL_PROFILER_TASK(search.treat_indices, q);
    ONEDAL_ASSERT(indices.has_mutable_data());
    auto* const ids_ptr = indices.get_mutable_data();
    const auto ids_str = indices.get_leading_stride();
    const ndshape<2> ids_shape = indices.get_shape();
    const auto tr_range = make_range_2d(ids_shape[0], ids_shape[1]);
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(tr_range, [=](sycl::id<2> idx) {
            *(ids_ptr + ids_str * idx[0] + idx[1]) += start_index;
        });
    });
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
              tblock_{ train_block },
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

protected:
    const std::int64_t k_, qblock_, tblock_, sblock_;
    ndarray<Float, 2> distances_;
    ndarray<std::int32_t, 2> out_indices_;
    ndarray<Float, 2> out_distances_;
    ndarray<std::int32_t, 2> part_indices_;
    ndarray<Float, 2> part_distances_;
};

template <typename Float, typename Distance>
search_temp_objects_deleter<Float, Distance>::search_temp_objects_deleter(event_ptr_t event)
        : last_event_(event) {}

template <typename Float, typename Distance>
auto search_temp_objects_deleter<Float, Distance>::get_last_event() -> event_ptr_t& {
    return this->last_event_;
}

template <typename Float, typename Distance>
void search_temp_objects_deleter<Float, Distance>::operator()(temp_t* obj) const {
    this->last_event_->wait_and_throw();
    delete obj;
}

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

template <typename Float>
class search_temp_objects<Float, squared_l2_distance<Float>>
        : public search_temp_objects_base<Float, squared_l2_distance<Float>> {
    using distance_t = squared_l2_distance<Float>;
    using base_t = search_temp_objects_base<Float, distance_t>;

public:
    search_temp_objects(sycl::queue& q,
                        std::int64_t k,
                        std::int64_t query_block,
                        std::int64_t train_block,
                        std::int64_t select_block)
            : base_t(q, k, query_block, train_block, select_block),
              query_norms_(ndarray<Float, 1>::empty(q, { query_block }, sycl::usm::alloc::device)) {
    }

    template <ndorder torder>
    auto& init_train_norms(sycl::queue& queue,
                           const ndview<Float, 2, torder>& train,
                           const event_vector& deps = {}) {
        const std::int32_t samples_count = train.get_dimension(0);
        train_blocking_ = uniform_blocking(samples_count, this->tblock_);
        train_events_ = event_vector(train_blocking_.get_block_count());
        train_norms_ = ndarray<Float, 1>::empty(queue, { samples_count }, sycl::usm::alloc::device);
        for (std::int64_t tb = 0; tb < train_blocking_.get_block_count(); ++tb) {
            const auto from = train_blocking_.get_block_start_index(tb);
            const auto to = train_blocking_.get_block_end_index(tb);
            auto train_block = train.get_row_slice(from, to);
            auto norms_block = get_train_norms_block(tb);
            train_events_[tb] = compute_squared_l2_norms(queue, train_block, norms_block, deps);
        }
        return *this;
    }

    ndview<Float, 1> get_train_norms_block(std::int64_t tb) const {
        const auto from = train_blocking_.get_block_start_index(tb);
        const auto to = train_blocking_.get_block_end_index(tb);
        return train_norms_.get_slice(from, to);
    }

    sycl::event get_train_norms_event(std::int64_t tb) const {
        return train_events_.at(tb);
    }

    ndview<Float, 1>& get_query_norms() {
        return query_norms_;
    }

protected:
    event_vector train_events_;
    ndarray<Float, 1> train_norms_;
    ndarray<Float, 1> query_norms_;
    uniform_blocking train_blocking_;
};

template <typename Float>
class search_temp_objects<Float, cosine_distance<Float>>
        : public search_temp_objects_base<Float, cosine_distance<Float>> {
    using distance_t = cosine_distance<Float>;
    using base_t = search_temp_objects_base<Float, distance_t>;

public:
    search_temp_objects(sycl::queue& q,
                        std::int64_t k,
                        std::int64_t query_block,
                        std::int64_t train_block,
                        std::int64_t select_block)
            : base_t(q, k, query_block, train_block, select_block),
              query_inv_norms_(
                  ndarray<Float, 1>::empty(q, { query_block }, sycl::usm::alloc::device)) {}

    template <ndorder torder>
    auto& init_train_inv_norms(sycl::queue& queue,
                               const ndview<Float, 2, torder>& train,
                               const event_vector& deps = {}) {
        ONEDAL_ASSERT(train.has_data());
        const std::int64_t samples_count = train.get_dimension(0);
        train_blocking_ = uniform_blocking(samples_count, this->tblock_);
        train_events_ = event_vector(train_blocking_.get_block_count());
        train_inv_norms_ =
            ndarray<Float, 1>::empty(queue, { samples_count }, sycl::usm::alloc::device);
        for (std::int64_t tb = 0; tb < train_blocking_.get_block_count(); ++tb) {
            const auto from = train_blocking_.get_block_start_index(tb);
            const auto to = train_blocking_.get_block_end_index(tb);
            auto train_block = train.get_row_slice(from, to);
            auto inv_norms_block = get_train_inv_norms_block(tb);
            train_events_[tb] =
                compute_inversed_l2_norms(queue, train_block, inv_norms_block, deps);
        }
        return *this;
    }

    ndview<Float, 1> get_train_inv_norms_block(std::int64_t tb) const {
        const auto from = train_blocking_.get_block_start_index(tb);
        const auto to = train_blocking_.get_block_end_index(tb);
        return train_inv_norms_.get_slice(from, to);
    }

    sycl::event get_train_inv_norms_event(std::int64_t tb) const {
        return train_events_.at(tb);
    }

    ndview<Float, 1>& get_query_inv_norms() {
        return query_inv_norms_;
    }

protected:
    event_vector train_events_;
    ndarray<Float, 1> train_inv_norms_;
    ndarray<Float, 1> query_inv_norms_;
    uniform_blocking train_blocking_;
};

template <typename Float, typename Distance>
std::shared_ptr<search_temp_objects<Float, Distance>> create_search_objects(
    sycl::queue& q,
    std::int64_t k,
    std::int64_t query_block,
    std::int64_t train_block,
    std::int64_t select_block,
    std::shared_ptr<sycl::event> dep) {
    const auto deleter = search_temp_objects_deleter<Float, Distance>(dep);
    auto* const object =
        new search_temp_objects<Float, Distance>(q, k, query_block, train_block, select_block);
    return std::shared_ptr<search_temp_objects<Float, Distance>>(object, deleter);
}

template <typename Float, typename Distance, typename Impl, ndorder torder>
search_engine_base<Float, Distance, Impl, torder>::search_engine_base(
    sycl::queue& queue,
    const ndview<Float, 2, torder>& train_data)
        : search_engine_base(queue,
                             train_data,
                             propose_train_block<Float>(queue, train_data.get_dimension(1))) {}

template <typename Float, typename Distance, typename Impl, ndorder torder>
search_engine_base<Float, Distance, Impl, torder>::search_engine_base(
    sycl::queue& queue,
    const ndview<Float, 2, torder>& train_data,
    std::int64_t train_block)
        : search_engine_base(queue, train_data, train_block, Distance(queue)) {}

template <typename Float, typename Distance, typename Impl, ndorder torder>
search_engine_base<Float, Distance, Impl, torder>::search_engine_base(
    sycl::queue& queue,
    const ndview<Float, 2, torder>& train_data,
    std::int64_t train_block,
    const Distance& distance_instance)
        : queue_(queue),
          distance_instance_(distance_instance),
          train_data_(train_data),
          train_blocking_(train_data.get_dimension(0), train_block),
          selection_blocking_(train_blocking_.get_block_count(), selection_sub_blocks) {}

template <typename Float, typename Distance, typename Impl, ndorder torder>
sycl::queue& search_engine_base<Float, Distance, Impl, torder>::get_queue() const {
    return this->queue_;
}

template <typename Float, typename Distance, typename Impl, ndorder torder>
const Distance& search_engine_base<Float, Distance, Impl, torder>::get_distance_impl() const {
    return this->distance_instance_;
}

template <typename Float, typename Distance, typename Impl, ndorder torder>
template <ndorder qorder>
sycl::event search_engine_base<Float, Distance, Impl, torder>::distance(
    const ndview<Float, 2, qorder>& query,
    const ndview<Float, 2, torder>& train,
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

template <typename Float, typename Distance, typename Impl, ndorder torder>
const uniform_blocking& search_engine_base<Float, Distance, Impl, torder>::get_train_blocking()
    const {
    return this->train_blocking_;
}

template <typename Float, typename Distance, typename Impl, ndorder torder>
const uniform_blocking& search_engine_base<Float, Distance, Impl, torder>::get_selection_blocking()
    const {
    return this->selection_blocking_;
}

template <typename Float, typename Distance, typename Impl, ndorder torder>
ndview<Float, 2, torder> search_engine_base<Float, Distance, Impl, torder>::get_train_block(
    std::int64_t i) const {
    const auto from = get_train_blocking().get_block_start_index(i);
    const auto to = get_train_blocking().get_block_end_index(i);
    return train_data_.get_row_slice(from, to);
}

template <typename Float, typename Distance, typename Impl, ndorder torder>
auto search_engine_base<Float, Distance, Impl, torder>::create_temporary_objects(
    std::int64_t query_block,
    std::int64_t k_neighbors,
    event_ptr_t last_event) const -> temp_ptr_t {
    return create_search_objects<Float, Distance>(this->get_queue(),
                                                  k_neighbors,
                                                  query_block,
                                                  this->get_train_blocking().get_block(),
                                                  selection_sub_blocks,
                                                  last_event);
}

template <typename Float, typename Distance, typename Impl, ndorder torder>
auto search_engine_base<Float, Distance, Impl, torder>::create_selection_objects(
    std::int64_t query_block,
    std::int64_t k_neighbors) const -> selc_ptr_t {
    const auto train_block = get_train_blocking().get_block();
    dal::detail::check_mul_overflow(k_neighbors, (selection_sub_blocks + 1));
    const auto width =
        std::max<std::int64_t>(k_neighbors * (selection_sub_blocks + 1), train_block);
    const ndshape<2> typical_blocking(query_block, width);
    return std::make_shared<selc_t>(get_queue(), typical_blocking, k_neighbors);
}

template <typename Float, typename Distance, typename Impl, ndorder torder>
ndview<std::int32_t, 2> search_engine_base<Float, Distance, Impl, torder>::get_indices(
    temp_ptr_t tmp_objs) {
    return tmp_objs->get_out_indices();
}

template <typename Float, typename Distance, typename Impl, ndorder torder>
ndview<Float, 2> search_engine_base<Float, Distance, Impl, torder>::get_distances(
    temp_ptr_t tmp_objs) {
    return tmp_objs->get_out_distances();
}

template <typename Float, typename Distance, typename Impl, ndorder torder>
sycl::event search_engine_base<Float, Distance, Impl, torder>::reset(
    temp_ptr_t tmp_objs,
    const event_vector& deps) const {
    constexpr std::int32_t default_idx_value = -1;
    constexpr Float default_dst_value = detail::limits<Float>::max();
    auto out_dsts = fill(get_queue(), tmp_objs->get_out_distances(), default_dst_value, deps);
    auto out_idcs = fill(get_queue(), tmp_objs->get_out_indices(), default_idx_value, deps);
    auto part_dsts = fill(get_queue(), tmp_objs->get_part_distances(), default_dst_value, deps);
    auto part_idcs = fill(get_queue(), tmp_objs->get_part_indices(), default_idx_value, deps);
    const auto fill_events = out_dsts + out_idcs + part_dsts + part_idcs;
    return fill(get_queue(), tmp_objs->get_distances(), default_dst_value, fill_events);
}

template <typename Float, typename Distance, typename Impl, ndorder torder>
sycl::event search_engine_base<Float, Distance, Impl, torder>::select_indexed(
    const ndview<std::int32_t, 2>& src,
    ndview<std::int32_t, 2>& dst,
    const event_vector& deps) const {
    namespace pr = oneapi::dal::backend::primitives;
    ONEDAL_ASSERT(src.has_data());
    ONEDAL_ASSERT(dst.has_mutable_data());
    return pr::select_indexed(get_queue(), dst, src, dst, deps);
}

template <typename Float, typename Distance, typename Impl, ndorder torder>
template <ndorder qorder>
sycl::event search_engine_base<Float, Distance, Impl, torder>::do_search(
    const ndview<Float, 2, qorder>& query,
    std::int64_t k_neighbors,
    temp_ptr_t temp_objs,
    selc_ptr_t selt_objs,
    const event_vector& deps) const {
    ONEDAL_PROFILER_TASK(search.base, this->get_queue());

    ONEDAL_ASSERT(temp_objs->get_k() == k_neighbors);
    ONEDAL_ASSERT(temp_objs->get_select_block() == selection_sub_blocks);
    ONEDAL_ASSERT(temp_objs->get_query_block() >= query.get_dimension(0));
    sycl::event last_event = reset(temp_objs, deps);
    const auto query_block_size = query.get_dimension(0);
    //Iterations over larger blocks
    for (std::int64_t sb_id = 0; sb_id < get_selection_blocking().get_block_count(); ++sb_id) {
        ONEDAL_PROFILER_TASK(search.base.selection_blocking, this->get_queue());
        const std::int64_t start_tb = get_selection_blocking().get_block_start_index(sb_id);
        const std::int64_t end_tb = get_selection_blocking().get_block_end_index(sb_id);
        //Iterations over smaller blocks
        for (std::int64_t tb_id = start_tb; tb_id < end_tb; ++tb_id) {
            ONEDAL_PROFILER_TASK(search.base.inner_blocking, this->get_queue());
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
                (*selt_objs)(get_queue(), dists, k_neighbors, part_dsts, part_inds, { dist_event });

            const auto st_idx = get_train_blocking().get_block_start_index(tb_id);
            last_event = treat_indices(this->get_queue(), part_inds, st_idx, { selt_event });
        }
        dal::detail::check_mul_overflow(k_neighbors, (1 + end_tb - start_tb));
        const std::int64_t cols = k_neighbors * (1 + end_tb - start_tb);
        auto dists = temp_objs->get_part_distances().get_col_slice(0, cols);
        auto selt_event = (*selt_objs)(this->get_queue(),
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

template <typename Float, typename Distance, ndorder torder>
search_engine<Float, Distance, torder>::search_engine(sycl::queue& queue,
                                                      const ndview<Float, 2, torder>& train_data,
                                                      std::int64_t train_block)
        : search_engine(queue, train_data, train_block, Distance(queue)) {}

template <typename Float, typename Distance, ndorder torder>
search_engine<Float, Distance, torder>::search_engine(sycl::queue& queue,
                                                      const ndview<Float, 2, torder>& train_data,
                                                      std::int64_t train_block,
                                                      const Distance& distance_instance)
        : base_t(queue, train_data, train_block, distance_instance) {}

template <typename Float, ndorder torder>
search_engine<Float, squared_l2_distance<Float>, torder>::search_engine(
    sycl::queue& queue,
    const ndview<Float, 2, torder>& train_data,
    std::int64_t train_block)
        : search_engine(queue, train_data, train_block, squared_l2_distance<Float>(queue)) {}

template <typename Float, ndorder torder>
search_engine<Float, squared_l2_distance<Float>, torder>::search_engine(
    sycl::queue& queue,
    const ndview<Float, 2, torder>& train_data,
    std::int64_t train_block,
    const squared_l2_distance<Float>& distance_instance)
        : base_t(queue, train_data, train_block, distance_instance) {}

template <typename Float, ndorder torder>
template <ndorder qorder>
sycl::event search_engine<Float, squared_l2_distance<Float>, torder>::distance(
    const ndview<Float, 2, qorder>& query,
    const ndview<Float, 2, torder>& train,
    ndview<Float, 2>& dists,
    const ndview<Float, 1>& query_norms,
    const ndview<Float, 1>& train_norms,
    const event_vector& deps) const {
    ONEDAL_ASSERT(query.has_data());
    ONEDAL_ASSERT(train.has_data());
    ONEDAL_ASSERT(dists.has_mutable_data());
    ONEDAL_ASSERT(query.get_dimension(1) == train.get_dimension(1));
    ONEDAL_ASSERT(train.get_dimension(0) == dists.get_dimension(1));
    ONEDAL_ASSERT(query.get_dimension(0) == dists.get_dimension(0));
    return this->get_distance_impl()(query, train, dists, query_norms, train_norms, deps);
}

template <typename Float, ndorder torder>
template <ndorder qorder>
sycl::event search_engine<Float, squared_l2_distance<Float>, torder>::do_search(
    const ndview<Float, 2, qorder>& query,
    std::int64_t k_neighbors,
    temp_ptr_t temp_objs,
    selc_ptr_t selt_objs,
    const event_vector& deps) const {
    ONEDAL_PROFILER_TASK(search.squared_l2, this->get_queue());

    ONEDAL_ASSERT(temp_objs->get_k() == k_neighbors);
    ONEDAL_ASSERT(temp_objs->get_select_block() == base_t::selection_sub_blocks);
    ONEDAL_ASSERT(temp_objs->get_query_block() >= query.get_dimension(0));
    temp_objs->init_train_norms(this->get_queue(), this->train_data_, deps);
    sycl::event last_event = this->reset(temp_objs, deps);
    sycl::event tevent, selt_event, inds_event, ip_event;
    const auto query_block_size = query.get_dimension(0);
    auto qnorms = temp_objs->get_query_norms().get_slice(0, query_block_size);
    auto qevent = compute_squared_l2_norms(this->get_queue(), query, qnorms, deps);
    //Iterations over larger blocks
    for (std::int64_t sb_id = 0; sb_id < this->get_selection_blocking().get_block_count();
         ++sb_id) {
        ONEDAL_PROFILER_TASK(search.squared_l2.selection_blocking, this->get_queue());
        const std::int64_t start_tb = this->get_selection_blocking().get_block_start_index(sb_id);
        const std::int64_t end_tb = this->get_selection_blocking().get_block_end_index(sb_id);
        //Iterations over smaller blocks
        for (std::int64_t tb_id = start_tb; tb_id < end_tb; ++tb_id) {
            ONEDAL_PROFILER_TASK(search.squared_l2.inner_blocking, this->get_queue());
            const auto train = this->get_train_block(tb_id);
            const auto train_block_size = this->get_train_blocking().get_block_length(tb_id);
            auto tnorms = temp_objs->get_train_norms_block(tb_id);
            ONEDAL_ASSERT(train.get_dimension(0) == train_block_size);
            auto ip = temp_objs->get_distances()
                          .get_col_slice(0, train_block_size)
                          .get_row_slice(0, query_block_size);
            {
                ONEDAL_PROFILER_TASK(tblock.distance, this->get_queue());
                tevent = temp_objs->get_train_norms_event(tb_id);
                ip_event = gemm(this->get_queue(),
                                query,
                                train.t(),
                                ip,
                                Float(-2),
                                Float(0),
                                { last_event });
            }
            const auto rel_idx = tb_id - start_tb;
            auto part_inds =
                temp_objs->get_part_indices_block(rel_idx + 1).get_row_slice(0, query_block_size);
            auto part_dsts =
                temp_objs->get_part_distances_block(rel_idx + 1).get_row_slice(0, query_block_size);
            {
                ONEDAL_PROFILER_TASK(tblock.selection, this->get_queue());
                // TODO: does complexity of this differ from others?
                selt_event = selt_objs->select_sq_l2(this->get_queue(),
                                                     qnorms,
                                                     tnorms,
                                                     ip,
                                                     k_neighbors,
                                                     part_dsts,
                                                     part_inds,
                                                     { ip_event, qevent, tevent });
            }
            {
                ONEDAL_PROFILER_TASK(tblock.treat, this->get_queue());
                const auto st_idx = this->get_train_blocking().get_block_start_index(tb_id);
                last_event = treat_indices(this->get_queue(), part_inds, st_idx, { selt_event });
            }
        }
        {
            ONEDAL_PROFILER_TASK(sblock.selection, this->get_queue());
            dal::detail::check_mul_overflow(k_neighbors, (1 + end_tb - start_tb));
            const std::int64_t cols = k_neighbors * (1 + end_tb - start_tb);
            auto dists = temp_objs->get_part_distances().get_col_slice(0, cols);
            selt_event = (*selt_objs)(this->get_queue(),
                                      dists,
                                      k_neighbors,
                                      temp_objs->get_out_distances(),
                                      temp_objs->get_out_indices(),
                                      { last_event });
        }
        {
            ONEDAL_PROFILER_TASK(sblock.select_indexed, this->get_queue());
            inds_event = this->select_indexed(temp_objs->get_part_indices(),
                                              temp_objs->get_out_indices(),
                                              { selt_event });
        }
        {
            ONEDAL_PROFILER_TASK(sblock.copy, this->get_queue());
            auto part_indcs = temp_objs->get_part_indices_block(0);
            last_event =
                copy(this->get_queue(), part_indcs, temp_objs->get_out_indices(), { inds_event });
        }
    }
    return last_event;
}

template <typename Float, ndorder torder>
search_engine<Float, cosine_distance<Float>, torder>::search_engine(
    sycl::queue& queue,
    const ndview<Float, 2, torder>& train_data,
    std::int64_t train_block)
        : search_engine(queue, train_data, train_block, cosine_distance<Float>(queue)) {}

template <typename Float, ndorder torder>
search_engine<Float, cosine_distance<Float>, torder>::search_engine(
    sycl::queue& queue,
    const ndview<Float, 2, torder>& train_data,
    std::int64_t train_block,
    const cosine_distance<Float>& distance_instance)
        : base_t(queue, train_data, train_block, distance_instance) {}

template <typename Float, ndorder torder>
template <ndorder qorder>
sycl::event search_engine<Float, cosine_distance<Float>, torder>::distance(
    const ndview<Float, 2, qorder>& query,
    const ndview<Float, 2, torder>& train,
    ndview<Float, 2>& dists,
    const ndview<Float, 1>& query_inv_norms,
    const ndview<Float, 1>& train_inv_norms,
    const event_vector& deps) const {
    ONEDAL_ASSERT(query.has_data());
    ONEDAL_ASSERT(train.has_data());
    ONEDAL_ASSERT(dists.has_mutable_data());
    ONEDAL_ASSERT(query.get_dimension(1) == train.get_dimension(1));
    ONEDAL_ASSERT(train.get_dimension(0) == dists.get_dimension(1));
    ONEDAL_ASSERT(query.get_dimension(0) == dists.get_dimension(0));
    return this->get_distance_impl()(query, train, dists, query_inv_norms, train_inv_norms, deps);
}

template <typename Float, ndorder torder>
template <ndorder qorder>
sycl::event search_engine<Float, cosine_distance<Float>, torder>::do_search(
    const ndview<Float, 2, qorder>& query,
    std::int64_t k_neighbors,
    temp_ptr_t temp_objs,
    selc_ptr_t selt_objs,
    const event_vector& deps) const {
    ONEDAL_PROFILER_TASK(search.cosine, this->get_queue());

    ONEDAL_ASSERT(temp_objs->get_k() == k_neighbors);
    ONEDAL_ASSERT(temp_objs->get_select_block() == base_t::selection_sub_blocks);
    ONEDAL_ASSERT(temp_objs->get_query_block() >= query.get_dimension(0));
    temp_objs->init_train_inv_norms(this->get_queue(), this->train_data_, deps);
    sycl::event last_event = this->reset(temp_objs, deps);
    const auto query_block_size = query.get_dimension(0);
    auto qinorms = temp_objs->get_query_inv_norms().get_slice(0, query_block_size);
    auto qievent = compute_inversed_l2_norms(this->get_queue(), query, qinorms, deps);
    //Iterations over larger blocks
    for (std::int64_t sb_id = 0; sb_id < this->get_selection_blocking().get_block_count();
         ++sb_id) {
        ONEDAL_PROFILER_TASK(search.cosine.selection_blocking, this->get_queue());
        const std::int64_t start_tb = this->get_selection_blocking().get_block_start_index(sb_id);
        const std::int64_t end_tb = this->get_selection_blocking().get_block_end_index(sb_id);
        //Iterations over smaller blocks
        for (std::int64_t tb_id = start_tb; tb_id < end_tb; ++tb_id) {
            ONEDAL_PROFILER_TASK(search.cosine.inner_blocking, this->get_queue());
            const auto train = this->get_train_block(tb_id);
            const auto train_block_size = this->get_train_blocking().get_block_length(tb_id);
            auto tinorms = temp_objs->get_train_inv_norms_block(tb_id);
            auto tievent = temp_objs->get_train_inv_norms_event(tb_id);
            ONEDAL_ASSERT(train.get_dimension(0) == train_block_size);
            auto dists = temp_objs->get_distances()
                             .get_col_slice(0, train_block_size)
                             .get_row_slice(0, query_block_size);

            auto gemm_event = compute_cosine_inner_product(this->get_queue(),
                                                           query,
                                                           train,
                                                           dists,
                                                           { last_event });
            auto dist_event = finalize_cosine(this->get_queue(),
                                              qinorms,
                                              tinorms,
                                              dists,
                                              { gemm_event, qievent, tievent });

            const auto rel_idx = tb_id - start_tb;
            auto part_inds =
                temp_objs->get_part_indices_block(rel_idx + 1).get_row_slice(0, query_block_size);
            auto part_dsts =
                temp_objs->get_part_distances_block(rel_idx + 1).get_row_slice(0, query_block_size);
            auto selt_event = (*selt_objs)(this->get_queue(),
                                           dists,
                                           k_neighbors,
                                           part_dsts,
                                           part_inds,
                                           { dist_event });

            const auto st_idx = this->get_train_blocking().get_block_start_index(tb_id);
            last_event = treat_indices(this->get_queue(), part_inds, st_idx, { selt_event });
        }
        dal::detail::check_mul_overflow(k_neighbors, (1 + end_tb - start_tb));
        const std::int64_t cols = k_neighbors * (1 + end_tb - start_tb);
        auto dists = temp_objs->get_part_distances().get_col_slice(0, cols);
        auto selt_event = (*selt_objs)(this->get_queue(),
                                       dists,
                                       k_neighbors,
                                       temp_objs->get_out_distances(),
                                       temp_objs->get_out_indices(),
                                       { last_event });
        auto inds_event = this->select_indexed(temp_objs->get_part_indices(),
                                               temp_objs->get_out_indices(),
                                               { selt_event });

        auto part_indcs = temp_objs->get_part_indices_block(0);
        last_event =
            copy(this->get_queue(), part_indcs, temp_objs->get_out_indices(), { inds_event });
    }
    return last_event;
}

#define INSTANTIATE(F, A, B)                                                                       \
    template sycl::event search_engine<F, squared_l2_distance<F>, A>::do_search(                   \
        const ndview<F, 2, B>&,                                                                    \
        std::int64_t,                                                                              \
        std::shared_ptr<search_temp_objects<F, squared_l2_distance<F>>>,                           \
        std::shared_ptr<kselect_by_rows<F>>,                                                       \
        const event_vector&) const;                                                                \
    template sycl::event search_engine<F, cosine_distance<F>, A>::do_search(                       \
        const ndview<F, 2, B>&,                                                                    \
        std::int64_t,                                                                              \
        std::shared_ptr<search_temp_objects<F, cosine_distance<F>>>,                               \
        std::shared_ptr<kselect_by_rows<F>>,                                                       \
        const event_vector&) const;                                                                \
    template sycl::event search_engine<F, lp_distance<F>, A>::do_search(                           \
        const ndview<F, 2, B>&,                                                                    \
        std::int64_t,                                                                              \
        std::shared_ptr<search_temp_objects<F, lp_distance<F>>>,                                   \
        std::shared_ptr<kselect_by_rows<F>>,                                                       \
        const event_vector&) const;                                                                \
    template sycl::event search_engine<F, chebyshev_distance<F>, A>::do_search(                    \
        const ndview<F, 2, B>&,                                                                    \
        std::int64_t,                                                                              \
        std::shared_ptr<search_temp_objects<F, chebyshev_distance<F>>>,                            \
        std::shared_ptr<kselect_by_rows<F>>,                                                       \
        const event_vector&) const;                                                                \
    template sycl::event search_engine<F, squared_l2_distance<F>, A>::distance(                    \
        const ndview<F, 2, B>&,                                                                    \
        const ndview<F, 2, A>&,                                                                    \
        ndview<F, 2>&,                                                                             \
        const ndview<F, 1>&,                                                                       \
        const ndview<F, 1>&,                                                                       \
        const event_vector&) const;                                                                \
    template sycl::event search_engine<F, cosine_distance<F>, A>::distance(const ndview<F, 2, B>&, \
                                                                           const ndview<F, 2, A>&, \
                                                                           ndview<F, 2>&,          \
                                                                           const ndview<F, 1>&,    \
                                                                           const ndview<F, 1>&,    \
                                                                           const event_vector&)    \
        const;                                                                                     \
    template sycl::event search_engine<F, lp_distance<F>, A>::distance(const ndview<F, 2, B>&,     \
                                                                       const ndview<F, 2, A>&,     \
                                                                       ndview<F, 2>&,              \
                                                                       const event_vector&) const; \
    template sycl::event search_engine<F, chebyshev_distance<F>, A>::distance(                     \
        const ndview<F, 2, B>&,                                                                    \
        const ndview<F, 2, A>&,                                                                    \
        ndview<F, 2>&,                                                                             \
        const event_vector&) const;

#define INSTANTIATE_B(F, A)                                                                   \
    INSTANTIATE(F, A, ndorder::c)                                                             \
    INSTANTIATE(F, A, ndorder::f)                                                             \
    template class search_engine_base<F,                                                      \
                                      distance<F, lp_metric<F>>,                              \
                                      search_engine<F, distance<F, lp_metric<F>>, A>,         \
                                      A>;                                                     \
    template class search_engine_base<F,                                                      \
                                      distance<F, squared_l2_metric<F>>,                      \
                                      search_engine<F, distance<F, squared_l2_metric<F>>, A>, \
                                      A>;                                                     \
    template class search_engine_base<F,                                                      \
                                      distance<F, cosine_metric<F>>,                          \
                                      search_engine<F, distance<F, cosine_metric<F>>, A>,     \
                                      A>;                                                     \
    template class search_engine_base<F,                                                      \
                                      distance<F, chebyshev_metric<F>>,                       \
                                      search_engine<F, distance<F, chebyshev_metric<F>>, A>,  \
                                      A>;                                                     \
    template class search_engine<F, distance<F, lp_metric<F>>, A>;                            \
    template class search_engine<F, distance<F, cosine_metric<F>>, A>;                        \
    template class search_engine<F, distance<F, chebyshev_metric<F>>, A>;                     \
    template class search_engine<F, distance<F, squared_l2_metric<F>>, A>;

#define INSTANTIATE_F(F)                                                             \
    INSTANTIATE_B(F, ndorder::c)                                                     \
    INSTANTIATE_B(F, ndorder::f)                                                     \
    template std::int64_t propose_train_block<F>(const sycl::queue&, std::int64_t);  \
    template std::int64_t propose_query_block<F>(const sycl::queue&, std::int64_t);  \
    template class search_temp_objects<F, distance<F, lp_metric<F>>>;                \
    template class search_temp_objects<F, distance<F, cosine_metric<F>>>;            \
    template class search_temp_objects<F, distance<F, chebyshev_metric<F>>>;         \
    template class search_temp_objects<F, distance<F, squared_l2_metric<F>>>;        \
    template class search_temp_objects_deleter<F, distance<F, lp_metric<F>>>;        \
    template class search_temp_objects_deleter<F, distance<F, cosine_metric<F>>>;    \
    template class search_temp_objects_deleter<F, distance<F, chebyshev_metric<F>>>; \
    template class search_temp_objects_deleter<F, distance<F, squared_l2_metric<F>>>;

INSTANTIATE_F(float)
INSTANTIATE_F(double)

template sycl::event treat_indices(sycl::queue&,
                                   ndview<std::int64_t, 2>&,
                                   std::int64_t,
                                   const event_vector&);
template sycl::event treat_indices(sycl::queue&,
                                   ndview<std::int32_t, 2>&,
                                   std::int64_t,
                                   const event_vector&);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
