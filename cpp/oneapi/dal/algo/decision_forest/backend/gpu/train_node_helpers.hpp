/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/dal/algo/decision_forest/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::decision_forest::backend {

enum class destination_type { host, device };

template <typename Index = std::int32_t>
class node {
    static_assert(std::is_integral_v<Index>);

public:
    static constexpr Index get_prop_count() {
        return node_prop_count_;
    }

    static constexpr Index get_medium_node_max_row_count() {
        return medium_node_max_row_count_;
    }
    static constexpr Index get_small_node_max_row_count() {
        return small_node_max_row_count_;
    }
    static constexpr Index get_elementary_node_max_row_count() {
        return elementary_node_max_row_count_;
    }

    static constexpr Index ind_ofs() {
        return ind_ofs_;
    }
    static constexpr Index ind_lrc() {
        return ind_lrc_;
    }
    static constexpr Index ind_fid() {
        return ind_fid_;
    }
    static constexpr Index ind_bin() {
        return ind_bin_;
    }
    static constexpr Index ind_lch_grc() {
        return ind_lch_grc_;
    }
    static constexpr Index ind_win() {
        return ind_win_;
    }
    static constexpr Index ind_grc() {
        return ind_grc_;
    }
    static constexpr Index ind_lch_lrc() {
        return ind_lch_lrc_;
    }

private:
    static constexpr inline Index medium_node_max_row_count_ = 8192;
    static constexpr inline Index small_node_max_row_count_ = 256;
    static constexpr inline Index elementary_node_max_row_count_ = 32;

    // left part rows count, response
    // node_prop_count_ is going to be removed here after migration to node_list_manager
    // node props mapping
    constexpr static Index ind_ofs_ = 0; // property index for local row offset
    constexpr static Index ind_lrc_ = 1; // property index for local row count
    constexpr static Index ind_fid_ = 2; // property index for local row count
    constexpr static Index ind_bin_ = 3; // property index for local row count
    constexpr static Index ind_lch_grc_ = 4; // property index for left child global row count
    constexpr static Index ind_win_ = 5; // property index for winner class
    constexpr static Index ind_grc_ = 6; // property index for global row count
    constexpr static Index ind_lch_lrc_ = 7; // property index for left child local row count

    static constexpr inline Index node_prop_count_ = 8;
};

#ifdef ONEDAL_DATA_PARALLEL
using alloc = sycl::usm::alloc;

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

template <typename Index = std::int32_t>
class node_list {
    static_assert(std::is_integral_v<Index>);

    using node_t = node<Index>;

public:
    node_list() = delete;
    node_list(const sycl::queue& queue) : queue_(queue), count_(0) {}
    node_list(const sycl::queue& queue, Index count) : queue_(queue), count_(count) {
        std::int64_t elem_count =
            de::check_mul_overflow<std::int64_t>(count_ * node_t::get_prop_count());
        list_ = pr::ndarray<Index, 1>::empty(queue_, { elem_count }, alloc::device);
    }

    node_list(const sycl::queue& queue, const pr::ndarray<Index, 1>& list, Index count)
            : queue_(queue),
              list_(list),
              count_(count) {
        ONEDAL_ASSERT(state_is_valid());
    }

    Index get_count() const {
        return count_;
    }

    pr::ndarray<Index, 1>& get_list() {
        return list_;
    }

    const pr::ndarray<Index, 1>& get_list() const {
        return list_;
    }

    bool state_is_valid() const {
        ONEDAL_ASSERT(list_.get_count() >= get_count() * node_t::get_prop_count());
        return true;
    }

private:
    sycl::queue queue_;

    pr::ndarray<Index, 1> list_;
    Index count_;
};

template <typename Index = std::int32_t>
class node_group {
    static_assert(std::is_integral_v<Index>);

public:
    static constexpr Index get_prop_count() {
        return node_group_prop_count_;
    }

private:
    static constexpr inline Index node_group_prop_count_ =
        2; // group_index offset and max_row_count in group
};

template <typename Index = std::int32_t>
class node_group_list;

template <typename Index = std::int32_t>
class node_group_view : public node_group<Index> {
    static_assert(std::is_integral_v<Index>);

    using node_t = node<Index>;
    using node_group_t = node_group<Index>;
    friend node_group_list<Index>;

public:
    Index get_group_index() const {
        return idx_;
    }
    Index get_node_indices_offset() const {
        return group_node_indices_offset_;
    }

    const pr::ndarray<Index, 1>& get_node_indices_list() const {
        return node_indices_list_;
    }

    Index get_max_row_count() const {
        return max_row_count_;
    }

    Index get_max_row_block_count() const {
        const Index row_block_size = node_t::get_small_node_max_row_count();
        return max_row_count_ / row_block_size + bool(max_row_count_ % row_block_size);
    }

    Index get_node_count() const {
        return node_count_;
    }

    const Index* get_node_indices_list_ptr() const {
        return node_indices_list_ptr_ + group_node_indices_offset_;
    }

    bool state_is_valid() const {
        return true; // all validity check are performed in constructor
    }

private:
    node_group_view(const pr::ndarray<Index, 1>& node_group_list,
                    const pr::ndarray<Index, 1>& node_indices_list,
                    Index idx,
                    Index total_group_count)
            : node_indices_list_(node_indices_list),
              idx_(idx) {
        ONEDAL_ASSERT(node_group_list.has_data());
        ONEDAL_ASSERT(node_indices_list.has_data());
        ONEDAL_ASSERT(idx_ >= 0);
        ONEDAL_ASSERT(idx_ <= total_group_count);

        const Index* group_list_ptr = node_group_list.get_data();
        node_count_ = group_list_ptr[(idx_ + 1) * node_group_t::get_prop_count() + 0] -
                      group_list_ptr[idx_ * node_group_t::get_prop_count() + 0];
        ONEDAL_ASSERT(node_count_ >= 0);
        ONEDAL_ASSERT(node_count_ <= de::limits<Index>::max());

        group_node_indices_offset_ = group_list_ptr[idx_ * node_group_t::get_prop_count() + 0];

        ONEDAL_ASSERT(node_indices_list.get_count() >= (group_node_indices_offset_ + node_count_));
        node_indices_list_ptr_ = node_indices_list.get_data();

        max_row_count_ = group_list_ptr[idx_ * node_group_t::get_prop_count() + 1];
        ONEDAL_ASSERT(max_row_count_ >= 0);
        ONEDAL_ASSERT(max_row_count_ <= de::limits<Index>::max());
    }

private:
    const pr::ndarray<Index, 1> node_indices_list_;
    const Index* node_indices_list_ptr_;

    Index group_node_indices_offset_;
    Index max_row_count_;
    Index node_count_;
    Index idx_;
};

template <typename Index>
class node_group_list {
    static_assert(std::is_integral_v<Index>);

    using node_t = node<Index>;
    using node_list_t = node_list<Index>;
    using node_group_t = node_group<Index>;
    using node_group_view_t = node_group_view<Index>;
    using node_group_list_t = node_group_list<Index>;

public:
    node_group_list() = delete;
    node_group_list(sycl::queue queue) : queue_(queue), node_list_(queue_) {
        constexpr Index elem_count = group_count_ + 1;
        // +1 is required because 0 elem stores a group offset in node indices list

        node_group_list_ =
            pr::ndarray<Index, 1>::empty(queue_,
                                         { elem_count * node_group_t::get_prop_count() },
                                         alloc::device);
        Index bound_list[elem_count] = { de::limits<Index>::max(),
                                         node_t::get_medium_node_max_row_count(),
                                         node_t::get_small_node_max_row_count(),
                                         node_t::get_elementary_node_max_row_count(),
                                         0 };
        group_bound_list_ =
            pr::ndarray<Index, 1>::wrap(bound_list, { elem_count }).to_device(queue_);
    }

    sycl::event filter(const node_list_t& node_list, const bk::event_vector& deps) {
        ONEDAL_ASSERT(node_list.get_count() > 0);
        indices_count_ = node_list.get_count();
        node_list_ = node_list;
        if (node_indices_list_.get_count() < indices_count_) {
            node_indices_list_ =
                pr::ndarray<Index, 1>::empty(queue_, { indices_count_ }, alloc::device);
        }
        filter_event_ = filter_internal(node_list, deps);
        is_group_list_on_host_ = false;
        return filter_event_;
    }

    pr::ndarray<Index, 1>& get_list() {
        return node_group_list_;
    }

    pr::ndarray<Index, 1>& get_node_indices_list() {
        return node_indices_list_;
    }

    const pr::ndarray<Index, 1>& get_bound_list() const {
        return group_bound_list_;
    }

    static constexpr Index get_count() {
        return group_count_;
    }

    node_group_view_t get_group_view(Index group_idx, const bk::event_vector& deps = {}) {
        filter_event_.wait_and_throw();
        if (!is_group_list_on_host_) {
            node_group_list_host_ = node_group_list_.to_host(queue_, deps);
            is_group_list_on_host_ = true;
        }

        return std::move(
            node_group_view_t(node_group_list_host_, node_indices_list_, group_idx, get_count()));
    }

    bool state_is_valid() const {
        ONEDAL_ASSERT(node_group_list_.get_count() ==
                      (get_count() + 1) * node_group_t::get_prop_count());
        ONEDAL_ASSERT(group_bound_list_.get_count() == get_count() + 1);
        ONEDAL_ASSERT(node_indices_list_.get_count() >= indices_count_);
        return true;
    }

private:
    sycl::event filter_internal(const node_list_t& node_list, const bk::event_vector& deps);

private:
    static constexpr inline Index group_count_ = 4;
    static constexpr inline Index min_local_size_ = 256;
    //static constexpr inline Index big_node_low_bound_blocks_num_ = 32;
    // nodes with bigger row count than following one will require more than one hist

    sycl::queue queue_;
    sycl::event filter_event_;

    pr::ndarray<Index, 1> node_group_list_;
    pr::ndarray<Index, 1> node_group_list_host_;

    pr::ndarray<Index, 1> group_bound_list_;

    pr::ndarray<Index, 1> node_indices_list_;
    Index indices_count_;

    bool is_group_list_on_host_ = false;

    node_list_t node_list_;
};
#endif //#ifdef ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::decision_forest::backend
