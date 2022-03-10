/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_helpers.hpp"

#ifdef ONEDAL_DATA_PARALLEL

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_best_split_sp_opt_impl.hpp"

namespace oneapi::dal::decision_forest::backend {

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

using alloc = sycl::usm::alloc;
using address = sycl::access::address_space;

using sycl::ext::oneapi::plus;
using sycl::ext::oneapi::minimum;
using sycl::ext::oneapi::maximum;

template <typename T>
using enable_if_float_t = std::enable_if_t<detail::is_valid_float_v<T>>;

template <typename F>
struct float_accuracy;

template <>
struct float_accuracy<float> {
    static constexpr float val = float(1e-5);
};

template <>
struct float_accuracy<double> {
    static constexpr double val = double(1e-10);
};

bool device_has_enough_local_mem(const sycl::queue& queue, std::int64_t required_byte_size) {
    auto device = queue.get_device();
    auto has_local_mem =
        device.is_host() ||
        (device.get_info<sycl::info::device::local_mem_type>() != sycl::info::local_mem_type::none);
    std::int64_t local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
    return has_local_mem && (local_mem_size / 2 >= required_byte_size);
}

template <typename T, typename Index = size_t>
inline T* fill_zero(T* dst, Index elem_count) {
    for (Index i = 0; i < elem_count; ++i) {
        dst[i] = T(0);
    }
    return dst;
}

template <typename T, typename Index = size_t>
inline T* fill_with_group(sycl::nd_item<2>& item, T* dst, Index elem_count, T val) {
    const Index local_id = item.get_local_id()[0];
    const Index local_size = item.get_local_range()[0];
    for (Index i = local_id; i < elem_count; i += local_size) {
        dst[i] = val;
    }

    item.barrier(sycl::access::fence_space::local_space);

    return dst;
}

template <typename Float, typename Index>
inline void add_val_to_hist(
    typename task_types<Float, Index, task::classification>::hist_type_t* hist_ptr,
    Float val) {
    Index class_id = static_cast<Index>(val);
    hist_ptr[class_id] += 1;
}

template <typename Float, typename Index>
inline void add_val_to_hist(
    typename task_types<Float, Index, task::regression>::hist_type_t* hist_ptr,
    Float val) {
    hist_ptr[0] += Float(1);
    Float inv_n = Float(1) / hist_ptr[0];
    Float delta = val - hist_ptr[1]; // y[i] - mean
    hist_ptr[1] += delta * inv_n; // updated mean
    hist_ptr[2] += delta * (val - hist_ptr[1]); // updated sum2Cent
}

// merge src and dst stat
template <typename Float, typename Index, typename T = enable_if_float_t<Float>>
inline void merge_stat(Float* dst, const Float* src, Index) {
    if (Float(0) == src[0])
        return;

    Float sum_n1n2 = dst[0] + src[0];
    Float mul_n1n2 = dst[0] * src[0];
    Float delta_scl = mul_n1n2 / sum_n1n2;
    Float mean_scl = (Float)1 / sum_n1n2;
    Float delta = src[1] - dst[1];

    dst[2] = dst[2] + src[2] + delta * delta * delta_scl;
    dst[1] = (dst[1] * dst[0] + src[1] * src[0]) * mean_scl;
    dst[0] = sum_n1n2;
}

template <typename Index>
inline void merge_stat(Index* dst, const Index* src, Index count) {
    for (Index i = 0; i < count; ++i) {
        dst[i] += src[i];
    }
}

template <typename Index>
inline void merge_stat(Index* dst, Index* accum, const Index* src, Index count) {
    for (Index i = 0; i < count; ++i) {
        *accum += src[i];
        dst[i] += src[i];
    }
}

template <typename Float, typename Index, typename T = enable_if_float_t<Float>>
inline void sub_stat(Float* dst, const Float* src, const Float* mrg, Index elem_count) {
    dst[0] = mrg[0] - src[0];

    dst[1] = (dst[0] >= Float(1)) ? (mrg[1] + (src[0] * (mrg[1] - src[1])) / dst[0]) : Float(0);

    Float sum_n1n2 = mrg[0];
    Float mul_n1n2 = src[0] * dst[0];
    Float delta_scl = mul_n1n2 / sum_n1n2;
    Float delta = src[1] - dst[1];

    dst[2] = (dst[0] >= Float(1)) ? ((mrg[2] - src[2] - delta * delta * delta_scl)) : Float(0);
}

template <typename Float>
inline bool float_eq(Float a, Float b) {
    return sycl::fabs(a - b) <= float_accuracy<Float>::val;
}

template <typename Float>
inline bool float_gt(Float a, Float b) {
    return (a - b) > float_accuracy<Float>::val;
}

template <typename Float, typename Index, typename Task>
struct split_info {
    using task_t = Task;
    using impl_const_t = impl_const<Index, task_t>;
    using hist_type_t = typename task_types<Float, Index, Task>::hist_type_t;
    using byte_t = std::uint8_t;
    static constexpr Index cache_buf_int_size = 3;
    static constexpr Index cache_buf_float_size = 2;

    static constexpr Index get_cache_byte_size() {
        return cache_buf_int_size * sizeof(Index) + cache_buf_float_size * sizeof(Float);
    }
    static constexpr Index get_cache_with_hist_byte_size(Index hist_prop_count) {
        return cache_buf_int_size * sizeof(Index) + cache_buf_float_size * sizeof(Float) +
               hist_prop_count * sizeof(hist_type_t);
    }

    inline void init(hist_type_t* in_left_hist, Index in_hist_prop_count) {
        left_hist = in_left_hist;
        hist_prop_count = in_hist_prop_count;
    }

    inline void init_clear(sycl::nd_item<2>& item,
                           hist_type_t* in_left_hist,
                           Index in_hist_prop_count) {
        init(in_left_hist, in_hist_prop_count);
        clear_scalar();
        clear_hist();
    }

    inline void clear_scalar() {
        ftr_id = impl_const_t::leaf_mark_;
        ftr_bin = impl_const_t::leaf_mark_;
        left_count = 0;
        right_count = 0;
        left_imp = Float(0);
        right_imp = Float(0);
        imp_dec = -de::limits<Float>::max();
    }

    inline void clear_hist(sycl::nd_item<2>& item) {
        fill_with_group(item, left_hist, hist_prop_count, hist_type_t(0));
    }

    inline void clear_hist() {
        for (Index i = 0; i < hist_prop_count; i++) {
            left_hist[i] = hist_type_t(0);
        }
    }

    inline byte_t* store(byte_t* buf_ptr, Index idx, Index total_block_count) {
        Index* index_buf_ptr = get_buf_ptr<Index>(&buf_ptr, total_block_count * cache_buf_int_size);
        Float* float_buf_ptr =
            get_buf_ptr<Float>(&buf_ptr, total_block_count * cache_buf_float_size);

        Index* ftr_id_ptr = index_buf_ptr + idx * cache_buf_int_size + 0;
        Index* ftr_bin_ptr = index_buf_ptr + idx * cache_buf_int_size + 1;
        Index* left_count_ptr = index_buf_ptr + idx * cache_buf_int_size + 2;

        Float* left_imp_ptr = float_buf_ptr + idx * cache_buf_float_size + 0;
        Float* imp_dec_ptr = float_buf_ptr + idx * cache_buf_float_size + 1;

        ftr_id_ptr[0] = ftr_id;
        ftr_bin_ptr[0] = ftr_bin;
        left_count_ptr[0] = left_count;
        left_imp_ptr[0] = left_imp;
        imp_dec_ptr[0] = imp_dec;

        return buf_ptr;
    }

    inline byte_t* load(byte_t* buf_ptr, Index idx, Index total_block_count) {
        Index* index_buf_ptr = get_buf_ptr<Index>(&buf_ptr, total_block_count * cache_buf_int_size);
        Float* float_buf_ptr =
            get_buf_ptr<Float>(&buf_ptr, total_block_count * cache_buf_float_size);

        Index* ftr_id_ptr = index_buf_ptr + idx * cache_buf_int_size + 0;
        Index* ftr_bin_ptr = index_buf_ptr + idx * cache_buf_int_size + 1;
        Index* left_count_ptr = index_buf_ptr + idx * cache_buf_int_size + 2;

        Float* left_imp_ptr = float_buf_ptr + idx * cache_buf_float_size + 0;
        Float* imp_dec_ptr = float_buf_ptr + idx * cache_buf_float_size + 1;

        ftr_id = ftr_id_ptr[0];
        ftr_bin = ftr_bin_ptr[0];
        left_count = left_count_ptr[0];
        left_imp = left_imp_ptr[0];
        imp_dec = imp_dec_ptr[0];
        //left_hist = hist_buf_ptr + idx * hist_prop_count;

        return buf_ptr;
    }

    inline void store_with_hist(byte_t* buf_ptr, Index idx, Index total_block_count) {
        buf_ptr = store(buf_ptr, idx, total_block_count);
        hist_type_t* hist_buf_ptr =
            get_buf_ptr<hist_type_t>(&buf_ptr, total_block_count * hist_prop_count);
        hist_type_t* hist_ptr = hist_buf_ptr + idx * hist_prop_count;

        for (Index i = 0; i < hist_prop_count; ++i) {
            hist_ptr[i] = left_hist[i];
        }
    }

    inline void load_with_hist(byte_t* buf_ptr, Index idx, Index total_block_count) {
        buf_ptr = load(buf_ptr, idx, total_block_count);
        hist_type_t* hist_buf_ptr =
            get_buf_ptr<hist_type_t>(&buf_ptr, total_block_count * hist_prop_count);
        hist_type_t* hist_ptr = hist_buf_ptr + idx * hist_prop_count;

        for (Index i = 0; i < hist_prop_count; ++i) {
            left_hist[i] = hist_ptr[i];
        }
    }

    inline void set_left_hist_ptr(hist_type_t* hist_ptr) {
        left_hist = hist_ptr;
    }

    Index ftr_id;
    Index ftr_bin;
    Index left_count;
    Index right_count;
    Float left_imp;
    Float right_imp;
    Float imp_dec;

    hist_type_t* left_hist;
    Index hist_prop_count;
};

template <typename Float, typename Index, typename Task>
struct split_smp {
    using task_t = Task;
    using impl_const_t = impl_const<Index, task_t>;
    using hist_type_t = typename task_types<Float, Index, Task>::hist_type_t;
    using split_info_t = split_info<Float, Index, Task>;

    // classififcation version
    inline void merge_bin_hist(Index& left_count_,
                               hist_type_t* left_class_hist_,
                               const hist_type_t* bin_hist_ptr,
                               Index class_count) {
        merge_stat(&left_class_hist_[0], &left_count_, bin_hist_ptr, class_count);
    }

    // regression version
    inline void merge_bin_hist(hist_type_t* left_hist_,
                               const hist_type_t* bin_hist_ptr,
                               Index elem_count) {
        merge_stat(&left_hist_[0], bin_hist_ptr, elem_count);
    }

    // classififcation version
    inline void calc_imp_dec(split_info_t& si,
                             const Index* node_ptr,
                             const Float* imp_list_ptr_,
                             const hist_type_t* node_class_hist_list_ptr_,
                             Index class_count,
                             Index node_id) {
        Index node_row_count = node_ptr[impl_const_t::ind_grc];
        const Float* node_imp_ptr = imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;
        Float node_imp = node_imp_ptr[0];
        const Index* node_class_hist_ptr = node_class_hist_list_ptr_ + node_id * class_count;

        si.right_count = node_row_count - si.left_count;

        const Float divL = (0 < si.left_count)
                               ? Float(1) / (Float(si.left_count) * Float(si.left_count))
                               : Float(0);
        const Float divR = (0 < si.right_count)
                               ? Float(1) / (Float(si.right_count) * Float(si.right_count))
                               : Float(0);

        si.left_imp = Float(1);
        si.right_imp = Float(1);

        for (Index class_id = 0; class_id < class_count; ++class_id) {
            si.left_imp -= Float(si.left_hist[class_id]) * Float(si.left_hist[class_id]) * divL;
            si.right_imp -= Float(node_class_hist_ptr[class_id] - si.left_hist[class_id]) *
                            Float(node_class_hist_ptr[class_id] - si.left_hist[class_id]) * divR;
        }

        si.left_imp = sycl::max(si.left_imp, Float(0));
        si.right_imp = sycl::max(si.right_imp, Float(0));

        si.imp_dec =
            node_imp - (Float(si.left_count) * si.left_imp + Float(si.right_count) * si.right_imp) /
                           Float(node_row_count);
    }

    // regression version
    inline void calc_imp_dec(split_info_t& si,
                             const Index* node_ptr,
                             const Float* imp_list_ptr,
                             Index node_id) {
        constexpr Index buff_size = impl_const_t::node_imp_prop_count_ + 1;
        Index node_row_count = node_ptr[impl_const_t::ind_grc];
        si.left_count = static_cast<Index>(si.left_hist[0]);

        const Float* node_imp_ptr = imp_list_ptr + node_id * impl_const_t::node_imp_prop_count_;

        hist_type_t node_hist[buff_size] = { static_cast<Float>(node_row_count),
                                             node_imp_ptr[0],
                                             node_imp_ptr[1] };
        hist_type_t right_hist[buff_size] = { hist_type_t(0) };

        // getting hist for right part
        sub_stat<Float, Index, task_t>(&right_hist[0], &si.left_hist[0], &node_hist[0], buff_size);

        si.right_count = node_row_count - si.left_count;
        si.imp_dec = node_imp_ptr[1] - (si.left_hist[2] + right_hist[2]);
    }

    inline bool test_split_is_best(const split_info_t& bs,
                                   const split_info_t& ts,
                                   Index node_id,
                                   Float node_imp,
                                   Float impurity_threshold,
                                   Index min_observations_in_leaf_node) {
        return (
            Float(0) < ts.imp_dec && !float_eq(node_imp, Float(0)) &&
            node_imp >= impurity_threshold &&
            (bs.ftr_bin == impl_const_t::leaf_mark_ || float_gt(ts.imp_dec, bs.imp_dec) ||
             (float_eq(ts.imp_dec, bs.imp_dec) &&
              (ts.ftr_id < bs.ftr_id || (bs.ftr_id == ts.ftr_id && ts.ftr_bin < bs.ftr_bin)))) &&
            ts.left_count >= min_observations_in_leaf_node &&
            ts.right_count >= min_observations_in_leaf_node);
    }

    // universal
    inline void choose_best_split(split_info_t& bs,
                                  split_info_t& ts,
                                  const Float* node_imp_list_ptr,
                                  Index hist_elem_count,
                                  Index node_id,
                                  Float impurity_threshold,
                                  Index min_observations_in_leaf_node) {
        // TODO move check for imp 0 to node split func
        Float node_imp = Float(0);
        if constexpr (std::is_same_v<task_t, task::classification>) {
            const Float* node_imp_ptr =
                node_imp_list_ptr + node_id * impl_const_t::node_imp_prop_count_;
            node_imp = node_imp_ptr[0];
        }
        else {
            const Float* node_imp_ptr =
                node_imp_list_ptr + node_id * impl_const_t::node_imp_prop_count_;
            node_imp = node_imp_ptr[1] / Float(ts.left_count + ts.right_count);
        }

        if (test_split_is_best(bs,
                               ts,
                               node_id,
                               node_imp,
                               impurity_threshold,
                               min_observations_in_leaf_node)) {
            bs.ftr_id = ts.ftr_id;
            bs.ftr_bin = ts.ftr_bin;
            bs.imp_dec = ts.imp_dec;

            bs.left_count = ts.left_count;
            bs.left_imp = ts.left_imp;

            for (Index i = 0; i < hist_elem_count; ++i) {
                bs.left_hist[i] = ts.left_hist[i];
            }
        }
    }

    // classififcation version
    inline void update_left_child_imp(Float* left_imp_list_ptr,
                                      hist_type_t* left_class_hist_list_ptr,
                                      Float bs_left_imp_,
                                      const hist_type_t* bs_left_class_hist_,
                                      Index node_id,
                                      Index class_count) {
        Float* left_node_imp_ptr = left_imp_list_ptr + node_id * impl_const_t::node_imp_prop_count_;

        left_node_imp_ptr[0] = bs_left_imp_;
        hist_type_t* left_node_class_hist_ptr = left_class_hist_list_ptr + node_id * class_count;

        for (Index class_id = 0; class_id < class_count; ++class_id) {
            left_node_class_hist_ptr[class_id] = bs_left_class_hist_[class_id];
        }
    }

    // regression version
    inline void update_left_child_imp(Float* left_imp_list_ptr,
                                      const hist_type_t* bs_left_hist,
                                      Index node_id) {
        Float* left_node_imp_ptr = left_imp_list_ptr + node_id * impl_const_t::node_imp_prop_count_;

        left_node_imp_ptr[0] = bs_left_hist[1];
        left_node_imp_ptr[1] = bs_left_hist[2];
    }

    inline void update_node_bs_info(const split_info_t& bs,
                                    Index* node_ptr,
                                    Float* node_imp_decr_ptr,
                                    Index node_id,
                                    Index index_max,
                                    bool update_imp_dec_required) {
        node_ptr[impl_const_t::ind_fid] =
            (bs.ftr_id == index_max) ? impl_const_t::leaf_mark_ : bs.ftr_id;
        node_ptr[impl_const_t::ind_bin] =
            (bs.ftr_bin == index_max) ? impl_const_t::leaf_mark_ : bs.ftr_bin;
        node_ptr[impl_const_t::ind_lch_grc] = bs.left_count;

        if (update_imp_dec_required) {
            if constexpr (std::is_same_v<task_t, task::classification>) {
                node_imp_decr_ptr[node_id] = bs.imp_dec;
            }
            else {
                node_imp_decr_ptr[node_id] = bs.imp_dec / node_ptr[impl_const_t::ind_grc];
            }
        }
    }

    inline bool my_split_is_best_for_sbg(sycl::nd_item<2>& item,
                                         split_info_t& bs,
                                         Index* node_ptr,
                                         Index node_id,
                                         Index index_max) {
        auto sbg = item.get_sub_group();
        const Index sub_group_local_id = sbg.get_local_id();

        const Float bestImpDec = sycl::reduce_over_group(sbg, bs.imp_dec, maximum<Float>());

        const Index impDecIsBest = float_eq(bestImpDec, bs.imp_dec);

        const Index bestFeatureId =
            sycl::reduce_over_group(sbg, impDecIsBest ? bs.ftr_id : index_max, minimum<Index>());
        const Index bestFeatureValue = sycl::reduce_over_group(
            sbg,
            (bestFeatureId == bs.ftr_id && impDecIsBest) ? bs.ftr_bin : index_max,
            minimum<Index>());

        const bool noneSplitFoundBySubGroup =
            ((impl_const_t::leaf_mark_ == bestFeatureId) && (0 == sub_group_local_id));
        const bool mySplitIsBest = (impl_const_t::leaf_mark_ != bestFeatureId &&
                                    bs.ftr_id == bestFeatureId && bs.ftr_bin == bestFeatureValue);
        return (noneSplitFoundBySubGroup || mySplitIsBest);
    }
};

template <typename Float, typename Bin, typename Index, typename Task>
std::int64_t
train_best_split_sp_opt_impl<Float, Bin, Index, Task>::define_local_size_for_small_single_pass(
    const sycl::queue& queue,
    std::int64_t selected_ftr_count) {
    //auto vec_size = bk::device_native_vector_size<Float>(queue);
    auto vec_size = bk::device_max_sg_size(queue);
    auto max_local_size = std::min(bk::device_max_wg_size(queue), vec_size * vec_size);
    auto max_sbg_count = max_local_size / vec_size;
    auto local_size =
        std::max(std::int64_t(std::min(bk::up_pow2(selected_ftr_count), max_sbg_count) * vec_size),
                 std::int64_t(128));

    return local_size;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event
train_best_split_sp_opt_impl<Float, Bin, Index, Task>::compute_best_split_single_pass_large(
    sycl::queue& queue,
    const context_t& ctx,
    const pr::ndarray<Bin, 2>& data,
    const pr::ndview<Float, 1>& response,
    const pr::ndarray<Index, 1>& tree_order,
    const pr::ndarray<Index, 1>& selected_ftr_list,
    const pr::ndarray<Index, 1>& bin_offset_list,
    const imp_data_t& imp_data_list,
    const pr::ndarray<Index, 1>& node_ind_list,
    Index node_ind_ofs,
    pr::ndarray<Index, 1>& node_list,
    imp_data_t& left_child_imp_data_list,
    pr::ndarray<Float, 1>& node_imp_dec_list,
    bool update_imp_dec_required,
    Index node_count,
    const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(compute_best_split_single_pass, queue);

    using split_smp_t = split_smp<Float, Index, Task>;

    Index hist_prop_count = 0;
    if constexpr (std::is_same_v<std::decay_t<Task>, task::classification>) {
        hist_prop_count = ctx.class_count_;
    }
    else {
        hist_prop_count = impl_const<Index, task::regression>::hist_prop_count_;
    }

    ONEDAL_ASSERT(data.get_count() == ctx.row_count_ * ctx.column_count_);
    ONEDAL_ASSERT(response.get_count() == ctx.row_count_);
    ONEDAL_ASSERT(tree_order.get_count() == ctx.tree_in_block_ * ctx.selected_row_total_count_);
    ONEDAL_ASSERT(selected_ftr_list.get_count() >= node_count * ctx.selected_ftr_count_);
    ONEDAL_ASSERT(bin_offset_list.get_count() == ctx.column_count_ + 1);
    ONEDAL_ASSERT(imp_data_list.imp_list_.get_count() >=
                  node_count * impl_const_t::node_imp_prop_count_);
    if constexpr (std::is_same_v<Task, task::classification>) {
        ONEDAL_ASSERT(imp_data_list.class_hist_list_.get_count() >= node_count * ctx.class_count_);
    }
    ONEDAL_ASSERT(node_ind_list.get_count() >= (node_ind_ofs + node_count));
    ONEDAL_ASSERT(node_list.get_count() >=
                  (node_ind_ofs + node_count) * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(left_child_imp_data_list.imp_list_.get_count() >=
                  node_count * impl_const_t::node_imp_prop_count_);
    if constexpr (std::is_same_v<Task, task::classification>) {
        ONEDAL_ASSERT(left_child_imp_data_list.class_hist_list_.get_count() >=
                      node_count * ctx.class_count_);
    }

    if (update_imp_dec_required) {
        ONEDAL_ASSERT(node_imp_dec_list.get_count() == node_count);
    }

    [[maybe_unused]] const Bin* data_ptr = data.get_data();
    [[maybe_unused]] const Float* response_ptr = response.get_data();
    [[maybe_unused]] const Index* tree_order_ptr = tree_order.get_data();

    [[maybe_unused]] const Index* selected_ftr_list_ptr = selected_ftr_list.get_data();

    imp_data_list_ptr<Float, Index, Task> imp_list_ptr(imp_data_list);

    [[maybe_unused]] const Index* node_indices_ptr = node_ind_list.get_data();
    [[maybe_unused]] Index* node_list_ptr = node_list.get_mutable_data();
    [[maybe_unused]] Float* node_imp_decr_list_ptr =
        update_imp_dec_required ? node_imp_dec_list.get_mutable_data() : nullptr;

    imp_data_list_ptr_mutable<Float, Index, Task> left_imp_list_ptr(left_child_imp_data_list);

    [[maybe_unused]] const Index column_count = ctx.column_count_;

    [[maybe_unused]] const Index selected_ftr_count = ctx.selected_ftr_count_;

    [[maybe_unused]] const Index index_max = ctx.index_max_;

    Index max_sbg_size = bk::device_max_sg_size(queue);
    Index max_wg_size = 256;
    auto max_sbg_count =
        (max_wg_size / max_sbg_size) * 2; // *2 in case if compiler will reduce sbg_size

    Index local_size = max_wg_size;

    std::size_t local_hist_buf_size = 0;

    std::size_t local_bs_buf_int_size = 0;
    std::size_t local_bs_buf_float_size = 0;

    std::size_t global_aux_ftr_buf_int_size = 1 + max_sbg_size;
    //std::size_t global_split_slot_buf_int_size = 3;
    //std::size_t global_split_slot_buf_float_size = 2;
    //std::size_t global_split_slot_hist_buf_size = hist_prop_count;

    local_bs_buf_int_size = 3; // bs_ftr_id, bs_ftr_bin and bs_left_count
    local_bs_buf_int_size += 2; // min_bin value holder, val holder for atomic reduce add
    local_bs_buf_float_size = max_sbg_count * 2;
    local_hist_buf_size = hist_prop_count * 2; // x2 because bs_hist and ts_hist

    sycl::event last_event;

    [[maybe_unused]] const Index class_count = ctx.class_count_;
    [[maybe_unused]] const Float imp_threshold = ctx.impurity_threshold_;
    [[maybe_unused]] const Index min_obs_leaf = ctx.min_observations_in_leaf_node_;

    [[maybe_unused]] const Float min_imp_dec = -de::limits<Float>::max();

    [[maybe_unused]] const Float* node_imp_list_ptr = imp_list_ptr.imp_list_ptr_;
    [[maybe_unused]] Float* left_child_imp_list_ptr = left_imp_list_ptr.imp_list_ptr_;

    // following vars are not used for regression, but should present to compile kernel
    [[maybe_unused]] const Index* class_hist_list_ptr =
        imp_list_ptr.get_class_hist_list_ptr_or_null();
    [[maybe_unused]] Index* left_child_class_hist_list_ptr =
        left_imp_list_ptr.get_class_hist_list_ptr_or_null();

    [[maybe_unused]] constexpr Index buff_size = impl_const_t::private_hist_buff_size;

    Index wg_in_block_count = 8192;
    //Index max_wg_count_for_node = 4;
    Index ftr_worker_per_node_count = 1;
    Index node_in_block_count = wg_in_block_count / ftr_worker_per_node_count;

    //std::size_t hist_buf_size = hist_prop_count * 2; // x2 because of one hist for best split and another for test split
    //std::size_t local_buf_byte_size = target_sbg_count * (hist_buf_size * sizeof(hist_type_t) + split_info<Float, Index, Task>::get_cache_byte_size());
    //std::int64_t global_buf_byte_size = wg_in_block_count * (hist_prop_count * sizeof(hist_type_t) + split_info<Float, Index, Task>::get_cache_with_hist_byte_size(hist_prop_count));
    std::int64_t global_buf_byte_size =
        wg_in_block_count *
        split_info<Float, Index, Task>::get_cache_with_hist_byte_size(hist_prop_count);

    auto global_buf_byte =
        pr::ndarray<byte_t, 1>::empty(queue, { global_buf_byte_size }, alloc::device);
    byte_t* global_buf_byte_ptr = global_buf_byte.get_mutable_data();
    auto global_aux_ftr_buf_int = pr::ndarray<Index, 1>::empty(
        queue,
        { std::int64_t(node_in_block_count * global_aux_ftr_buf_int_size) },
        alloc::device);

    [[maybe_unused]] Index* global_aux_ftr_buf_int_ptr = global_aux_ftr_buf_int.get_mutable_data();
    //if(!device_has_enough_local_mem(queue, local_buf_byte_size)) {

    //    throw "Device doesn't have enough local memory!";
    //}

    for (Index processed_node_cnt = 0; processed_node_cnt < node_count;
         processed_node_cnt += node_in_block_count, node_ind_ofs += node_in_block_count) {
        auto fill_aux_ftr_event = global_aux_ftr_buf_int.fill(queue, 0);
        bk::event_vector fill_deps{ fill_aux_ftr_event };

        const sycl::nd_range<2> nd_range =
            bk::make_multiple_nd_range_2d({ local_size, wg_in_block_count }, { local_size, 1 });

        last_event = queue.submit([&](cl::sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.depends_on(fill_deps);

            local_accessor_rw_t<std::uint8_t> local_byte_buf(
                local_bs_buf_int_size * sizeof(Index) + local_bs_buf_float_size * sizeof(Float) +
                    local_hist_buf_size * sizeof(hist_type_t),
                cgh);

            cgh.parallel_for(nd_range, [=](cl::sycl::nd_item<2> item) {
                auto sbg = item.get_sub_group();
                const Index group_id = item.get_group().get_id(1);
                [[maybe_unused]] const Index ftr_group_id = group_id % ftr_worker_per_node_count;
                [[maybe_unused]] const Index node_idx =
                    item.get_global_id()[1] / ftr_worker_per_node_count;
                if ((processed_node_cnt + node_idx) > (node_count - 1) ||
                    ftr_group_id > (selected_ftr_count - 1)) {
                    return;
                }

                [[maybe_unused]] const Index node_id = node_indices_ptr[node_ind_ofs + node_idx];
                [[maybe_unused]] Index* node_ptr =
                    node_list_ptr + node_id * impl_const_t::node_prop_count_;

                [[maybe_unused]] const Index local_id = item.get_local_id()[0];
                [[maybe_unused]] const Index local_size = item.get_local_range()[0];

                [[maybe_unused]] const Index sub_group_id = sbg.get_group_id();
                [[maybe_unused]] const Index sub_group_local_id = sbg.get_local_id();
                [[maybe_unused]] const Index sub_group_size = sbg.get_local_range()[0];
                //const Index sub_group_count = item.get_local_range()[0] / sub_group_size;

                [[maybe_unused]] const Index row_ofs = node_ptr[impl_const_t::ind_ofs];
                [[maybe_unused]] const Index row_count = node_ptr[impl_const_t::ind_lrc];

                split_smp_t sp_hlp;
                split_info<Float, Index, Task> bs;

                // slm pointers declaration
                std::uint8_t* local_byte_buf_ptr = local_byte_buf.get_pointer().get();
                //Index* local_bs_buf_int_ptr = local_bs_buf_int.get_pointer().get();
                Index* local_bs_buf_int_ptr =
                    get_buf_ptr<Index>(&local_byte_buf_ptr, local_bs_buf_int_size);
                //hist_type_t* local_hist_buf_ptr = local_hist_buf.get_pointer().get();
                hist_type_t* local_hist_buf_ptr =
                    get_buf_ptr<hist_type_t>(&local_byte_buf_ptr, local_hist_buf_size);
                //Float* local_bs_buf_float_ptr = local_bs_buf_float.get_pointer().get();
                Float* local_bs_buf_float_ptr =
                    get_buf_ptr<Float>(&local_byte_buf_ptr, local_bs_buf_float_size);

                [[maybe_unused]] Index* min_bin_buf_ptr = local_bs_buf_int_ptr + 3;
                [[maybe_unused]] Index* count_buf_ptr = local_bs_buf_int_ptr + 4;

                bs.init_clear(item, local_hist_buf_ptr + 0 * hist_prop_count, hist_prop_count);

                Index processed_ftr_count = 0;
                for (Index ftr_idx = ftr_group_id; ftr_idx < selected_ftr_count;
                     ftr_idx += ftr_worker_per_node_count) {
                    processed_ftr_count++;

                    split_info<Float, Index, Task> ts;
                    ts.init(local_hist_buf_ptr + 1 * hist_prop_count, hist_prop_count);
                    ts.ftr_id = selected_ftr_list_ptr[node_id * selected_ftr_count + ftr_idx];
                    Index i = local_id;
                    Index id = (i < row_count) ? tree_order_ptr[row_ofs + i] : index_max;
                    Index bin =
                        (i < row_count) ? data_ptr[id * column_count + ts.ftr_id] : index_max;
                    Float response = (i < row_count) ? response_ptr[id] : Float(0);
                    Index response_int = (i < row_count) ? static_cast<Index>(response) : Index(-1);

                    //[[maybe_unused]] Index ts.ftr_bin = Index(-1);
                    ts.ftr_bin = Index(-1);

                    ts.ftr_bin = reduce_min_over_group(item,
                                                       min_bin_buf_ptr,
                                                       bin > ts.ftr_bin ? bin : index_max);
                    while (ts.ftr_bin < index_max) {
                        ts.left_count = 0;
                        ts.right_count = 0;
                        ts.left_imp = Float(0);
                        ts.right_imp = Float(0);
                        ts.imp_dec = Float(0);
                        ts.clear_hist(item);

                        const Index count = (bin <= ts.ftr_bin) ? 1 : 0;

                        const Index left_count = reduce_add_over_group(item, count_buf_ptr, count);
                        //if(local_id == 0) { ts.left_count = left_count; }
                        ts.left_count = left_count;

                        [[maybe_unused]] Float sum = Float(0);
                        [[maybe_unused]] Float sum2cent = Float(0);
                        if constexpr (std::is_same_v<Task, task::classification>) {
                            const Index val = (bin <= ts.ftr_bin) ? response_int : Index(-1);
                            Index all_class_count = 0;

                            for (Index class_id = 0; class_id < class_count - 1; class_id++) {
                                Index total_class_count =
                                    reduce_add_over_group(item,
                                                          count_buf_ptr,
                                                          Index(class_id == val));
                                all_class_count += total_class_count;
                                ts.left_hist[class_id] = total_class_count;
                            }

                            ts.left_hist[class_count - 1] = ts.left_count - all_class_count;
                        }
                        else {
                            const Float val = (bin <= ts.ftr_bin) ? response : Float(0);
                            Float sum =
                                reduce_add_over_group(item, local_bs_buf_float_ptr, val, row_count);
                            //Float sum = sp_hlp.reduce_add_over_group(item, local_bs_buf_float_ptr, val);

                            Float mean = sum / left_count;
                            ts.right_count = row_count - left_count;

                            Float val_s2c =
                                (bin <= ts.ftr_bin) ? (val - mean) * (val - mean) : Float(0);

                            sum2cent = reduce_add_over_group(item,
                                                             local_bs_buf_float_ptr,
                                                             val_s2c,
                                                             row_count);
                            //sum2cent = sp_hlp.reduce_add_over_group(item, local_bs_buf_float_ptr, val_s2c);

                            ts.left_hist[0] = Float(left_count);
                            ts.left_hist[1] = mean;
                            ts.left_hist[2] = sum2cent;
                        }

                        if (local_id == 0) {
                            if constexpr (std::is_same_v<Task, task::classification>) {
                                sp_hlp.calc_imp_dec(ts,
                                                    node_ptr,
                                                    node_imp_list_ptr,
                                                    class_hist_list_ptr,
                                                    class_count,
                                                    node_id);
                                sp_hlp.choose_best_split(bs,
                                                         ts,
                                                         node_imp_list_ptr,
                                                         class_count,
                                                         node_id,
                                                         imp_threshold,
                                                         min_obs_leaf);
                            }
                            else {
                                sp_hlp.calc_imp_dec(ts, node_ptr, node_imp_list_ptr, node_id);
                                sp_hlp.choose_best_split(bs,
                                                         ts,
                                                         node_imp_list_ptr,
                                                         impl_const_t::hist_prop_count_,
                                                         node_id,
                                                         imp_threshold,
                                                         min_obs_leaf);
                            }
                        }

                        ts.ftr_bin = reduce_min_over_group(item,
                                                           min_bin_buf_ptr,
                                                           bin > ts.ftr_bin ? bin : index_max);
                    }
                }

                if (sub_group_id > 0) {
                    return;
                }

                [[maybe_unused]] Index total_processed_ftr_count = 0;

                if (local_id == 0) {
                    bs.store_with_hist(
                        global_buf_byte_ptr +
                            node_idx * ftr_worker_per_node_count *
                                split_info<Float, Index, Task>::get_cache_with_hist_byte_size(
                                    hist_prop_count),
                        ftr_group_id,
                        ftr_worker_per_node_count);
                    total_processed_ftr_count = atomic_global_sum(
                        global_aux_ftr_buf_int_ptr + node_idx * global_aux_ftr_buf_int_size,
                        processed_ftr_count);
                }

                // read slm marker
                total_processed_ftr_count =
                    sycl::group_broadcast(sbg, total_processed_ftr_count, 0);

                if (sub_group_id == 0 && total_processed_ftr_count == selected_ftr_count) {
                    //bs.loadi_with_hist(global_split_buf_int_ptr, global_split_buf_float_ptr, global_split_hist_buf_ptr, node_idx, max_sbg_size, sub_group_local_id);

                    bs.load_with_hist(
                        global_buf_byte_ptr +
                            node_idx * ftr_worker_per_node_count *
                                split_info<Float, Index, Task>::get_cache_with_hist_byte_size(
                                    hist_prop_count),
                        sub_group_local_id % ftr_worker_per_node_count,
                        ftr_worker_per_node_count);

                    if (sp_hlp.my_split_is_best_for_sbg(item, bs, node_ptr, node_id, index_max)) {
                        sp_hlp.update_node_bs_info(bs,
                                                   node_ptr,
                                                   node_imp_decr_list_ptr,
                                                   node_id,
                                                   index_max,
                                                   update_imp_dec_required);
                        if constexpr (std::is_same_v<Task, task::classification>) {
                            sp_hlp.update_left_child_imp(left_child_imp_list_ptr,
                                                         left_child_class_hist_list_ptr,
                                                         bs.left_imp,
                                                         bs.left_hist,
                                                         node_id,
                                                         class_count);
                        }
                        else {
                            sp_hlp.update_left_child_imp(left_child_imp_list_ptr,
                                                         bs.left_hist,
                                                         node_id);
                        }
                    }
                }
            });
        });

        last_event.wait_and_throw();
    }

    return last_event;
}

//// new single pass
template <typename Float, typename Bin, typename Index, typename Task>
sycl::event
train_best_split_sp_opt_impl<Float, Bin, Index, Task>::compute_best_split_single_pass_small(
    sycl::queue& queue,
    const context_t& ctx,
    const pr::ndarray<Bin, 2>& data,
    const pr::ndview<Float, 1>& response,
    const pr::ndarray<Index, 1>& tree_order,
    const pr::ndarray<Index, 1>& selected_ftr_list,
    const pr::ndarray<Index, 1>& bin_offset_list,
    const imp_data_t& imp_data_list,
    const node_group_view_t& node_group,
    node_list_t& level_node_list,
    imp_data_t& left_child_imp_data_list,
    pr::ndarray<Float, 1>& node_imp_dec_list,
    bool update_imp_dec_required,
    const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(compute_best_split_single_pass, queue);

    using split_smp_t = split_smp<Float, Index, Task>;

    Index hist_prop_count = 0;
    if constexpr (std::is_same_v<std::decay_t<Task>, task::classification>) {
        hist_prop_count = ctx.class_count_;
    }
    else {
        hist_prop_count = impl_const<Index, task::regression>::hist_prop_count_;
    }

    Index node_count = node_group.get_node_count();
    Index node_ind_ofs = node_group.get_node_indices_offset();
    ONEDAL_ASSERT(data.get_count() == ctx.row_count_ * ctx.column_count_);
    ONEDAL_ASSERT(response.get_count() == ctx.row_count_);
    ONEDAL_ASSERT(tree_order.get_count() == ctx.tree_in_block_ * ctx.selected_row_total_count_);
    ONEDAL_ASSERT(selected_ftr_list.get_count() >= node_count * ctx.selected_ftr_count_);
    ONEDAL_ASSERT(bin_offset_list.get_count() == ctx.column_count_ + 1);
    ONEDAL_ASSERT(imp_data_list.imp_list_.get_count() >=
                  node_count * impl_const_t::node_imp_prop_count_);
    if constexpr (std::is_same_v<Task, task::classification>) {
        ONEDAL_ASSERT(imp_data_list.class_hist_list_.get_count() >= node_count * ctx.class_count_);
    }
    ONEDAL_ASSERT(node_ind_list.get_count() >= (node_ind_ofs + node_count));
    ONEDAL_ASSERT(node_list.get_count() >=
                  (node_ind_ofs + node_count) * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(left_child_imp_data_list.imp_list_.get_count() >=
                  node_count * impl_const_t::node_imp_prop_count_);
    if constexpr (std::is_same_v<Task, task::classification>) {
        ONEDAL_ASSERT(left_child_imp_data_list.class_hist_list_.get_count() >=
                      node_count * ctx.class_count_);
    }

    if (update_imp_dec_required) {
        ONEDAL_ASSERT(node_imp_dec_list.get_count() == node_count);
    }

    [[maybe_unused]] const Bin* data_ptr = data.get_data();
    [[maybe_unused]] const Float* response_ptr = response.get_data();
    [[maybe_unused]] const Index* tree_order_ptr = tree_order.get_data();

    [[maybe_unused]] const Index* selected_ftr_list_ptr = selected_ftr_list.get_data();

    imp_data_list_ptr<Float, Index, Task> imp_list_ptr(imp_data_list);

    [[maybe_unused]] const Index* node_indices_ptr = node_group.get_node_indices_list_ptr();
    [[maybe_unused]] Index* node_list_ptr = level_node_list.get_list().get_mutable_data();
    //[[maybe_unused]] const Index* node_indices_ptr = node_ind_list.get_data();
    //[[maybe_unused]] Index* node_list_ptr = node_list.get_mutable_data();
    [[maybe_unused]] Float* node_imp_decr_list_ptr =
        update_imp_dec_required ? node_imp_dec_list.get_mutable_data() : nullptr;

    imp_data_list_ptr_mutable<Float, Index, Task> left_imp_list_ptr(left_child_imp_data_list);

    [[maybe_unused]] const Index column_count = ctx.column_count_;

    [[maybe_unused]] const Index selected_ftr_count = ctx.selected_ftr_count_;

    [[maybe_unused]] const Index index_max = ctx.index_max_;

    Index max_sbg_size = bk::device_max_sg_size(queue);
    Index target_sbg_size = max_sbg_size > 16 ? 32 : 16;
    //Index max_wg_size = bk::device_max_wg_size(queue);
    Index max_wg_size = 512;

    Index local_size = max_wg_size;
    //auto local_size = max_sbg_size;

    Index target_sbg_count = (local_size / target_sbg_size);

    std::size_t hist_buf_size =
        hist_prop_count * 2; // x2 because of one hist for best split and another for test split
    std::size_t local_buf_byte_size =
        target_sbg_count * (hist_buf_size * sizeof(hist_type_t) +
                            split_info<Float, Index, Task>::get_cache_byte_size());

    sycl::event last_event;

    [[maybe_unused]] const Index class_count = ctx.class_count_;
    [[maybe_unused]] const Float imp_threshold = ctx.impurity_threshold_;
    [[maybe_unused]] const Index min_obs_leaf = ctx.min_observations_in_leaf_node_;

    [[maybe_unused]] const Float min_imp_dec = -de::limits<Float>::max();

    [[maybe_unused]] const Float* node_imp_list_ptr = imp_list_ptr.imp_list_ptr_;
    [[maybe_unused]] Float* left_child_imp_list_ptr = left_imp_list_ptr.imp_list_ptr_;

    // following vars are not used for regression, but should present to compile kernel
    [[maybe_unused]] const Index* class_hist_list_ptr =
        imp_list_ptr.get_class_hist_list_ptr_or_null();
    [[maybe_unused]] Index* left_child_class_hist_list_ptr =
        left_imp_list_ptr.get_class_hist_list_ptr_or_null();

    //[[maybe_unused]] constexpr Index buff_size = impl_const_t::private_hist_buff_size;

    Index wg_in_block_count = 8192;
    Index node_in_block_count = wg_in_block_count;

    constexpr Index target_row_count = 32;

    //define_local_size_for_small_single_pass(queue, selected_ftr_count);

    if (!device_has_enough_local_mem(queue, local_buf_byte_size)) {
        throw "Device doesn't have enough local memory!";
    }

    for (Index processed_node_cnt = 0; processed_node_cnt < node_count;
         processed_node_cnt += node_in_block_count, node_ind_ofs += node_in_block_count) {
        const sycl::nd_range<2> nd_range =
            bk::make_multiple_nd_range_2d({ local_size, wg_in_block_count }, { local_size, 1 });

        last_event = queue.submit([&](cl::sycl::handler& cgh) {
            cgh.depends_on(deps);
            local_accessor_rw_t<std::uint8_t> local_byte_buf(local_buf_byte_size, cgh);

            cgh.parallel_for(
                nd_range,
                [=](cl::sycl::nd_item<2> item) [[intel::reqd_sub_group_size(target_row_count)]] {
                    auto sbg = item.get_sub_group();

                    [[maybe_unused]] const Index node_idx = item.get_global_id()[1];
                    [[maybe_unused]] const Index node_id = node_indices_ptr[node_idx];
                    [[maybe_unused]] Index* node_ptr =
                        node_list_ptr + node_id * impl_const_t::node_prop_count_;

                    [[maybe_unused]] const Index local_id = item.get_local_id()[0];
                    [[maybe_unused]] const Index local_size = item.get_local_range()[0];

                    [[maybe_unused]] const Index sub_group_id = sbg.get_group_id();
                    [[maybe_unused]] const Index sub_group_local_id = sbg.get_local_id();
                    [[maybe_unused]] const Index sub_group_size = sbg.get_local_range()[0];
                    //const Index target_sbg_count = item.get_local_range()[0] / sub_group_size;

                    [[maybe_unused]] const Index row_ofs = node_ptr[impl_const_t::ind_ofs];
                    [[maybe_unused]] const Index row_count = node_ptr[impl_const_t::ind_lrc];

                    //if((processed_node_cnt + node_idx) > (node_count - 1) || row_count > target_row_count) { return; }
                    if ((processed_node_cnt + node_idx) > (node_count - 1)) {
                        return;
                    }

                    [[maybe_unused]] split_smp_t sp_hlp;
                    split_info<Float, Index, Task> bs;

                    // slm pointers declaration
                    std::uint8_t* local_byte_buf_ptr = local_byte_buf.get_pointer().get();

                    hist_type_t* local_bs_hist_buf_ptr =
                        get_buf_ptr<hist_type_t>(&local_byte_buf_ptr,
                                                 target_sbg_count * hist_prop_count);
                    hist_type_t* local_ts_hist_buf_ptr =
                        get_buf_ptr<hist_type_t>(&local_byte_buf_ptr,
                                                 target_sbg_count * hist_prop_count);

                    bs.init_clear(item,
                                  local_bs_hist_buf_ptr + sub_group_id * hist_prop_count,
                                  hist_prop_count);

                    for (Index ftr_idx = sub_group_id; ftr_idx < selected_ftr_count;
                         ftr_idx += target_sbg_count) {
                        split_info<Float, Index, Task> ts;
                        ts.init(local_ts_hist_buf_ptr + sub_group_id * hist_prop_count,
                                hist_prop_count);
                        ts.ftr_id = selected_ftr_list_ptr[node_id * selected_ftr_count + ftr_idx];

                        Index i = sub_group_local_id;
                        Index id = (i < row_count) ? tree_order_ptr[row_ofs + i] : index_max;
                        Index bin =
                            (i < row_count) ? data_ptr[id * column_count + ts.ftr_id] : index_max;
                        Float response = (i < row_count) ? response_ptr[id] : Float(0);
                        Index response_int =
                            (i < row_count) ? static_cast<Index>(response) : Index(-1);

                        ts.ftr_bin = Index(-1);

                        while ((ts.ftr_bin =
                                    sycl::reduce_over_group(sbg,
                                                            bin > ts.ftr_bin ? bin : index_max,
                                                            minimum<Index>())) < index_max) {
                            const Index count = (bin <= ts.ftr_bin) ? 1 : 0;

                            ts.left_count = sycl::reduce_over_group(sbg, count, plus<Index>());

                            if constexpr (std::is_same_v<Task, task::classification>) {
                                const Index val = (bin <= ts.ftr_bin) ? response_int : Index(-1);
                                Index all_class_count = 0;

                                for (Index class_id = 0; class_id < class_count - 1; class_id++) {
                                    Index total_class_count =
                                        sycl::reduce_over_group(sbg,
                                                                Index(class_id == val),
                                                                plus<Index>());
                                    all_class_count += total_class_count;
                                    ts.left_hist[class_id] = total_class_count;
                                }

                                ts.left_hist[class_count - 1] = ts.left_count - all_class_count;
                            }
                            else {
                                const Float val = (bin <= ts.ftr_bin) ? response : Float(0);
                                const Float sum = sycl::reduce_over_group(sbg, val, plus<Float>());

                                const Float mean = sum / ts.left_count;
                                ts.right_count = row_count - ts.left_count;

                                Float val_s2c =
                                    (bin <= ts.ftr_bin) ? (val - mean) * (val - mean) : Float(0);

                                Float sum2cent =
                                    sycl::reduce_over_group(sbg, val_s2c, plus<Float>());

                                ts.left_hist[0] = ts.left_count;
                                ts.left_hist[1] = mean;
                                ts.left_hist[2] = sum2cent;
                            }

                            if (sub_group_local_id == 0) {
                                if constexpr (std::is_same_v<Task, task::classification>) {
                                    sp_hlp.calc_imp_dec(ts,
                                                        node_ptr,
                                                        node_imp_list_ptr,
                                                        class_hist_list_ptr,
                                                        class_count,
                                                        node_id);
                                    sp_hlp.choose_best_split(bs,
                                                             ts,
                                                             node_imp_list_ptr,
                                                             class_count,
                                                             node_id,
                                                             imp_threshold,
                                                             min_obs_leaf);
                                }
                                else {
                                    sp_hlp.calc_imp_dec(ts, node_ptr, node_imp_list_ptr, node_id);
                                    sp_hlp.choose_best_split(bs,
                                                             ts,
                                                             node_imp_list_ptr,
                                                             impl_const_t::hist_prop_count_,
                                                             node_id,
                                                             imp_threshold,
                                                             min_obs_leaf);
                                }
                            }
                        }
                    }

                    byte_t* local_buf_ptr = get_buf_ptr<byte_t>(
                        &local_byte_buf_ptr,
                        target_sbg_count * split_info<Float, Index, Task>::get_cache_byte_size());

                    if (sub_group_local_id == 0) {
                        bs.store(local_buf_ptr, sub_group_id, target_sbg_count);
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (sub_group_id == 0) {
                        bs.load(local_buf_ptr,
                                sub_group_local_id % target_sbg_count,
                                target_sbg_count);
                        bs.set_left_hist_ptr(local_bs_hist_buf_ptr +
                                             (sub_group_local_id % target_sbg_count) *
                                                 bs.hist_prop_count);

                        if (sp_hlp
                                .my_split_is_best_for_sbg(item, bs, node_ptr, node_id, index_max)) {
                            sp_hlp.update_node_bs_info(bs,
                                                       node_ptr,
                                                       node_imp_decr_list_ptr,
                                                       node_id,
                                                       index_max,
                                                       update_imp_dec_required);
                            if constexpr (std::is_same_v<Task, task::classification>) {
                                sp_hlp.update_left_child_imp(left_child_imp_list_ptr,
                                                             left_child_class_hist_list_ptr,
                                                             bs.left_imp,
                                                             bs.left_hist,
                                                             node_id,
                                                             class_count);
                            }
                            else {
                                sp_hlp.update_left_child_imp(left_child_imp_list_ptr,
                                                             bs.left_hist,
                                                             node_id);
                            }
                        }
                    }
                });
        });

        last_event.wait_and_throw();
    }

    return last_event;
}

#define INSTANTIATE(F, B, I, T) template class train_best_split_sp_opt_impl<F, B, I, T>;

INSTANTIATE(float, std::uint32_t, std::int32_t, task::classification);
INSTANTIATE(float, std::uint32_t, std::int32_t, task::regression);

INSTANTIATE(double, std::uint32_t, std::int32_t, task::classification);
INSTANTIATE(double, std::uint32_t, std::int32_t, task::regression);

} // namespace oneapi::dal::decision_forest::backend

#endif
