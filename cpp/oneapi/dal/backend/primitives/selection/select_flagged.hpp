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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

/// Functor for direct access of mask values
template <typename Flag, typename Integer>
class mask_direct {
public:
    mask_direct(const Flag* mask_ptr) : mask_ptr_(mask_ptr) {}
    Flag operator[](Integer i) const {
        return mask_ptr_[i];
    }

private:
    const Flag* mask_ptr_;
};

/// Functor for indirect access (via index array) of mask values
template <typename Flag, typename Integer>
class mask_indirect {
public:
    mask_indirect(const Flag* mask_ptr, const Integer* ind_ptr)
            : mask_ptr_(mask_ptr),
              ind_ptr_(ind_ptr) {}
    Flag operator[](Integer i) const {
        return mask_ptr_[ind_ptr_[i]];
    }

private:
    const Flag* mask_ptr_;
    const Integer* ind_ptr_;
};

template <typename Flag, typename Data, typename Integer>
struct data2mask_map {
    using mask_t = mask_indirect<Flag, Data>;
};

template <typename Flag, typename Integer>
struct data2mask_map<Flag, float, Integer> {
    using mask_t = mask_direct<Flag, Integer>;
};

template <typename Flag, typename Integer>
struct data2mask_map<Flag, double, Integer> {
    using mask_t = mask_direct<Flag, Integer>;
};

/// @tparam Data type used for storing input data values
/// @tparam Flag type used for storing input mask values
template <typename Data, typename Flag>
class select_flagged_base {
protected:
    using integer_t = std::uint32_t;
    using mask_t = typename data2mask_map<Flag, Data, integer_t>::mask_t;

    /// Performs initialization of auxiliary variables and required auxiliary buffers
    /// @param[in]  queue The queue
    select_flagged_base(const sycl::queue& queue);
    select_flagged_base(const select_flagged_base&) = delete;
    virtual ~select_flagged_base();
    select_flagged_base& operator=(const select_flagged_base&) = delete;

    void init(sycl::queue& queue, std::int64_t elem_count);

    sycl::event select_flagged_base_impl(const mask_t& mask_accessor,
                                         const ndview<Data, 1>& in,
                                         ndview<Data, 1>& out,
                                         std::int64_t& selected_elem_count,
                                         const event_vector& deps);

private:
    sycl::event scan(sycl::queue& queue,
                     const mask_t& mask_accessor,
                     ndarray<integer_t, 1>& part_sum,
                     integer_t elem_count,
                     integer_t local_size,
                     integer_t local_sum_count,
                     sycl::event& deps);

    sycl::event sum_scan(sycl::queue& queue,
                         const ndarray<integer_t, 1>& part_sum,
                         ndarray<integer_t, 1>& part_prefix_sum,
                         integer_t local_size,
                         integer_t local_sum_count,
                         ndarray<integer_t, 1>& total_sum,
                         sycl::event& deps);

    sycl::event reorder(sycl::queue& queue,
                        const mask_t& mask_accessor,
                        const ndview<Data, 1>& in,
                        ndview<Data, 1>& out,
                        ndarray<integer_t, 1>& part_prefix_sum,
                        integer_t elem_count,
                        integer_t local_size,
                        integer_t local_sum_count,
                        sycl::event& deps);

    sycl::queue queue_;
    sycl::event select_flagged_base_event_;

    ndarray<integer_t, 1> part_sum_;
    ndarray<integer_t, 1> part_prefix_sum_;
    ndarray<integer_t, 1> total_sum_;

    integer_t elem_count_;
    integer_t local_size_;
    integer_t local_sum_count_;
    integer_t sum_buff_size_;

    static constexpr inline integer_t max_local_sum_count_ = 256;
    static constexpr inline integer_t preferable_sbg_size_ = 16;
};

template <typename Data, typename Flag>
class select_flagged : public select_flagged_base<Data, Flag> {
    static_assert(std::is_floating_point<Data>::value);

    using base_t = select_flagged_base<Data, Flag>;

public:
    using mask_t = typename base_t::mask_t;

    /// Performs initialization of auxiliary variables and required auxiliary buffers
    /// @param[in]  queue The queue
    select_flagged(const sycl::queue& queue) : select_flagged_base<Data, Flag>(queue) {}
    select_flagged(const select_flagged&) = delete;
    select_flagged& operator=(const select_flagged&) = delete;

    /// Performs flagged copy of values from input vector into output one base on input mask values
    /// Copies of the selected values are compacted into output in their original relative ordering.
    /// Value i is being copied out if bool(mask[i]) eq true.
    /// NOTE: auxiliary buffers and variables are reset in case if number of elements in input
    ///       differs from the number of input elements processed on previous call
    ///
    /// @param[in]  mask The [n] input vector of flaggs used for selecting values
    /// @param[in]  in   The [n] input vector of values
    /// @param[out] out  The [n] output vector of selected values
    /// @param[out] selected_elem_count  The number of selected values
    sycl::event operator()(const ndview<Flag, 1>& mask,
                           const ndview<Data, 1>& in,
                           ndview<Data, 1>& out,
                           std::int64_t& selected_elem_count,
                           const event_vector& deps = {}) {
        ONEDAL_ASSERT(in.get_count() == mask.get_count());

        mask_t mask_accessor{ mask.get_data() };
        return base_t::select_flagged_base_impl(mask_accessor, in, out, selected_elem_count, deps);
    }
};

template <typename Data, typename Flag>
class select_flagged_index : public select_flagged_base<Data, Flag> {
    static_assert(std::numeric_limits<Data>::is_integer);

    using base_t = select_flagged_base<Data, Flag>;

public:
    using mask_t = typename base_t::mask_t;

    /// Performs initialization of auxiliary variables and required auxiliary buffers
    /// @param[in]  queue The queue
    select_flagged_index(const sycl::queue& queue) : select_flagged_base<Data, Flag>(queue) {}
    select_flagged_index(const select_flagged_index&) = delete;
    select_flagged_index& operator=(const select_flagged_index&) = delete;

    /// Performs flagged copy of indices from input vector into output one base on input mask values
    /// Copies of the selected indices are compacted into output in their original relative ordering.
    /// Index i is being copied out if bool(mask[in[i]]) eq true.
    /// NOTE: auxiliary buffers and variables are reset in case if number of elements in input
    ///       differs from the number of input elements processed on previous call
    ///
    /// @param[in]  mask The [n] input vector of flaggs used for selecting indices
    /// @param[in]  in   The [n] input vector of indices
    /// @param[out] out  The [n] output vector of selected indices
    /// @param[out] selected_elem_count  The number of selected indices
    sycl::event operator()(const ndview<Flag, 1>& mask,
                           const ndview<Data, 1>& in,
                           ndview<Data, 1>& out,
                           std::int64_t& selected_elem_count,
                           const event_vector& deps = {}) {
        ONEDAL_ASSERT(in.get_count() == mask.get_count());

        mask_t mask_accessor{ mask.get_data(), in.get_data() };
        return base_t::select_flagged_base_impl(mask_accessor, in, out, selected_elem_count, deps);
    }
};

#endif

} // namespace oneapi::dal::backend::primitives
