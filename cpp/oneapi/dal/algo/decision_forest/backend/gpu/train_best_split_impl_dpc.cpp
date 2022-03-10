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

#include "oneapi/dal/algo/decision_forest/backend/gpu/dbg_helpers.hpp"

#ifdef ONEDAL_DATA_PARALLEL

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_best_split_impl.hpp"

namespace oneapi::dal::decision_forest::backend {

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

using alloc = sycl::usm::alloc;
using address = sycl::access::address_space;

using sycl::ext::oneapi::plus;
using sycl::ext::oneapi::minimum;
using sycl::ext::oneapi::maximum;

using bin_map_t = std::uint64_t;
constexpr inline std::int32_t bin_block_count =
    4; // number of elements in bin_map_t array which is used for trackin already processed bins
constexpr inline std::int32_t bin_in_block_count = sizeof(bin_map_t) * 8;

template <typename Data>
using local_accessor_rw_t =
    sycl::accessor<Data, 1, sycl::access::mode::read_write, sycl::access::target::local>;

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

template <typename Index>
inline void mark_bin_processed(bin_map_t* bin_map, Index bin_idx) {
    bin_map_t mask = 1ul << (bin_idx % bin_in_block_count);
    Index block_idx = bin_idx / bin_in_block_count;
    if (block_idx < bin_block_count) {
        bin_map[block_idx] &= mask;
    }
}

template <typename Index>
inline bool is_bin_processed(const bin_map_t* bin_map, Index bin_idx) {
    bin_map_t mask = 1ul << (bin_idx % bin_in_block_count);
    Index block_idx = bin_idx / bin_in_block_count;
    return block_idx < bin_block_count ? bin_map[block_idx] & mask : false;
}

template <typename Float, typename Index, typename Task>
struct split_info {
    using task_t = Task;
    using impl_const_t = impl_const<Index, task_t>;
    using hist_type_t = typename task_types<Float, Index, Task>::hist_type_t;

    inline void init(hist_type_t* in_left_hist, Index in_hist_prop_count) {
        left_hist = in_left_hist;
        hist_prop_count = in_hist_prop_count;
    }

    inline void init_clear(sycl::nd_item<2>& item,
                           hist_type_t* in_left_hist,
                           Index in_hist_prop_count) {
        init(in_left_hist, in_hist_prop_count);
        clear_scalar();
        clear_hist(item);
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

    inline void store(Index* global_split_buf_int_ptr,
                      Float* global_split_buf_float_ptr,
                      hist_type_t* global_split_hist_buf_ptr,
                      Index node_idx,
                      Index max_sbg_size,
                      Index idx) {
        const Index global_split_slot_buf_int_size = 3;
        const Index global_split_slot_buf_float_size = 2;
        [[maybe_unused]] Index* glb_ftr_id_ptr =
            global_split_buf_int_ptr + node_idx * max_sbg_size * global_split_slot_buf_int_size +
            max_sbg_size * 0;
        [[maybe_unused]] Index* glb_ftr_bin_ptr =
            global_split_buf_int_ptr + node_idx * max_sbg_size * global_split_slot_buf_int_size +
            max_sbg_size * 1;
        [[maybe_unused]] Index* glb_left_count_ptr =
            global_split_buf_int_ptr + node_idx * max_sbg_size * global_split_slot_buf_int_size +
            max_sbg_size * 2;

        [[maybe_unused]] Float* glb_left_imp_ptr =
            global_split_buf_float_ptr +
            node_idx * max_sbg_size * global_split_slot_buf_float_size + max_sbg_size * 0;
        [[maybe_unused]] Float* glb_imp_dec_ptr =
            global_split_buf_float_ptr +
            node_idx * max_sbg_size * global_split_slot_buf_float_size + max_sbg_size * 1;

        [[maybe_unused]] hist_type_t* glb_hist_ptr = global_split_hist_buf_ptr +
                                                     node_idx * max_sbg_size * hist_prop_count +
                                                     idx * hist_prop_count;

        glb_ftr_id_ptr[idx] = ftr_id;
        glb_ftr_bin_ptr[idx] = ftr_bin;
        glb_left_count_ptr[idx] = left_count;
        glb_left_imp_ptr[idx] = left_imp;
        glb_imp_dec_ptr[idx] = imp_dec;
        for (Index class_id = 0; class_id < hist_prop_count; ++class_id) {
            glb_hist_ptr[class_id] = left_hist[class_id];
        }
    }

    inline void load(Index* global_split_buf_int_ptr,
                     Float* global_split_buf_float_ptr,
                     hist_type_t* global_split_hist_buf_ptr,
                     Index node_idx,
                     Index max_sbg_size,
                     Index idx) {
        const Index global_split_slot_buf_int_size = 3;
        const Index global_split_slot_buf_float_size = 2;
        [[maybe_unused]] Index* glb_ftr_id_ptr =
            global_split_buf_int_ptr + node_idx * max_sbg_size * global_split_slot_buf_int_size +
            max_sbg_size * 0;
        [[maybe_unused]] Index* glb_ftr_bin_ptr =
            global_split_buf_int_ptr + node_idx * max_sbg_size * global_split_slot_buf_int_size +
            max_sbg_size * 1;
        [[maybe_unused]] Index* glb_left_count_ptr =
            global_split_buf_int_ptr + node_idx * max_sbg_size * global_split_slot_buf_int_size +
            max_sbg_size * 2;

        [[maybe_unused]] Float* glb_left_imp_ptr =
            global_split_buf_float_ptr +
            node_idx * max_sbg_size * global_split_slot_buf_float_size + max_sbg_size * 0;
        [[maybe_unused]] Float* glb_imp_dec_ptr =
            global_split_buf_float_ptr +
            node_idx * max_sbg_size * global_split_slot_buf_float_size + max_sbg_size * 1;

        //[[maybe_unused]] hist_type_t* glb_hist_ptr = global_split_hist_buf_ptr + node_idx * max_sbg_size * hist_prop_count;
        [[maybe_unused]] hist_type_t* glb_hist_ptr = global_split_hist_buf_ptr +
                                                     node_idx * max_sbg_size * hist_prop_count +
                                                     idx * hist_prop_count;

        ftr_id = glb_ftr_id_ptr[idx];
        ftr_bin = glb_ftr_bin_ptr[idx];
        left_count = glb_left_count_ptr[idx];
        left_imp = glb_left_imp_ptr[idx];
        imp_dec = glb_imp_dec_ptr[idx];
        left_hist = glb_hist_ptr;
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
    /*
    inline void init_split_info(sycl::nd_item<2>& item, split_info_t& si, hist_type_t* in_left_hist, Index in_hist_prop_count) {
        si.left_hist = in_left_hist;
        si.hist_prop_count = in_hist_prop_count;
        si.ftr_id = impl_const_t::leaf_mark_;
        si.ftr_bin = impl_const_t::leaf_mark_;
        si.left_count = 0;
        si.right_count = 0;
        si.left_imp = Float(0);
        si.right_imp = Float(0);
        si.imp_dec = -de::limits<Float>::max();
        fill_with_group(item, si.left_hist, si.hist_prop_count, hist_type_t(0));
        //init(in_left_hist, in_hist_prop_count);
        //clear_scalar();
        //clear_hist(item);
    }
*/
    // classififcation version
    inline void add_val(Index& left_count_,
                        hist_type_t* left_class_hist_,
                        Index ftr_bin_,
                        Index obs_bin,
                        Float obs_response) {
        Index class_id = static_cast<Index>(obs_response);

        left_count_ += Index(obs_bin <= ftr_bin_);
        left_class_hist_[class_id] += Index(obs_bin <= ftr_bin_);
    }

    inline void merge_bin_hist(Index& left_count_,
                               hist_type_t* left_class_hist_,
                               const hist_type_t* bin_hist_ptr,
                               Index class_count) {
        merge_stat(&left_class_hist_[0], &left_count_, bin_hist_ptr, class_count);
    }

    // regression version
    inline void add_val(hist_type_t* left_hist_,
                        Index ftr_bin_,
                        Index obs_bin,
                        Float obs_response) {
        if (obs_bin <= ftr_bin_) {
            add_val_to_hist<Float, Index>(&left_hist_[0], obs_response);
        }
    }

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
    // classififcation version
    inline void calc_imp_dec(Index& right_count_,
                             Float& left_imp_,
                             Float& right_imp_,
                             Float& imp_dec_,
                             Index left_count_,
                             const Index* node_ptr,
                             const Float* imp_list_ptr_,
                             const hist_type_t* node_class_hist_list_ptr_,
                             const hist_type_t* left_class_hist_,
                             Index class_count,
                             Index node_id) {
        Index node_row_count = node_ptr[impl_const_t::ind_grc];
        const Float* node_imp_ptr = imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;
        Float node_imp = node_imp_ptr[0];
        const Index* node_class_hist_ptr = node_class_hist_list_ptr_ + node_id * class_count;

        right_count_ = node_row_count - left_count_;

        const Float divL =
            (0 < left_count_) ? Float(1) / (Float(left_count_) * Float(left_count_)) : Float(0);
        const Float divR =
            (0 < right_count_) ? Float(1) / (Float(right_count_) * Float(right_count_)) : Float(0);

        left_imp_ = Float(1);
        right_imp_ = Float(1);

        for (Index class_id = 0; class_id < class_count; ++class_id) {
            left_imp_ -=
                Float(left_class_hist_[class_id]) * Float(left_class_hist_[class_id]) * divL;
            right_imp_ -= Float(node_class_hist_ptr[class_id] - left_class_hist_[class_id]) *
                          Float(node_class_hist_ptr[class_id] - left_class_hist_[class_id]) * divR;
        }

        left_imp_ = sycl::max(left_imp_, Float(0));
        right_imp_ = sycl::max(right_imp_, Float(0));

        imp_dec_ = node_imp - (Float(left_count_) * left_imp_ + Float(right_count_) * right_imp_) /
                                  Float(node_row_count);
    }

    // regression version
    inline void calc_imp_dec(Index& left_count_,
                             Index& right_count_,
                             Float& imp_dec_,
                             const Index* node_ptr,
                             const Float* imp_list_ptr,
                             const hist_type_t* left_hist_,
                             Index node_id) {
        constexpr Index buff_size = impl_const_t::node_imp_prop_count_ + 1;
        Index node_row_count = node_ptr[impl_const_t::ind_grc];
        left_count_ = static_cast<Index>(left_hist_[0]);

        const Float* node_imp_ptr = imp_list_ptr + node_id * impl_const_t::node_imp_prop_count_;

        hist_type_t node_hist[buff_size] = { static_cast<Float>(node_row_count),
                                             node_imp_ptr[0],
                                             node_imp_ptr[1] };
        hist_type_t right_hist[buff_size] = { hist_type_t(0) };

        // getting hist for right part
        sub_stat<Float, Index, task_t>(&right_hist[0], &left_hist_[0], &node_hist[0], buff_size);

        right_count_ = node_row_count - left_count_;
        imp_dec_ = node_imp_ptr[1] - (left_hist_[2] + right_hist[2]);
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

    inline bool test_split_is_best(Index bs_ftr_id_,
                                   Index bs_ftr_bin_,
                                   Index bs_left_count_,
                                   Float bs_imp_dec_,
                                   Index ts_ftr_id_,
                                   Index ts_ftr_bin_,
                                   Index ts_left_count_,
                                   Index ts_right_count_,
                                   Float ts_imp_dec_,
                                   Index node_id,
                                   Float node_imp,
                                   Float impurity_threshold_,
                                   Index min_observations_in_leaf_node) {
        return (Float(0) < ts_imp_dec_ && !float_eq(node_imp, Float(0)) &&
                node_imp >= impurity_threshold_ &&
                (bs_ftr_bin_ == impl_const_t::leaf_mark_ || float_gt(ts_imp_dec_, bs_imp_dec_) ||
                 (float_eq(ts_imp_dec_, bs_imp_dec_) &&
                  (ts_ftr_id_ < bs_ftr_id_ ||
                   (bs_ftr_id_ == ts_ftr_id_ && ts_ftr_bin_ < bs_ftr_bin_)))) &&
                ts_left_count_ >= min_observations_in_leaf_node &&
                ts_right_count_ >= min_observations_in_leaf_node);
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
    inline void choose_best_split(Index& bs_ftr_id_,
                                  Index& bs_ftr_bin_,
                                  Index& bs_left_count_,
                                  Float& bs_left_imp_,
                                  Float& bs_imp_dec_,
                                  hist_type_t* bs_left_class_hist_,
                                  Index ts_ftr_id_,
                                  Index ts_ftr_bin_,
                                  Index ts_left_count_,
                                  Index ts_right_count_,
                                  Float ts_left_imp_,
                                  Float ts_imp_dec_,
                                  const hist_type_t* ts_left_class_hist_,
                                  const Float* node_imp_list_ptr,
                                  Index class_count_,
                                  Index node_id,
                                  Float impurity_threshold_,
                                  Index min_observations_in_leaf_node) {
        // TODO move check for imp 0 to node split func
        const Float* node_imp_ptr =
            node_imp_list_ptr + node_id * impl_const_t::node_imp_prop_count_;
        Float node_imp = node_imp_ptr[0];

        if (test_split_is_best(bs_ftr_id_,
                               bs_ftr_bin_,
                               bs_left_count_,
                               bs_imp_dec_,
                               ts_ftr_id_,
                               ts_ftr_bin_,
                               ts_left_count_,
                               ts_right_count_,
                               ts_imp_dec_,
                               node_id,
                               node_imp,
                               impurity_threshold_,
                               min_observations_in_leaf_node)) {
            bs_ftr_id_ = ts_ftr_id_;
            bs_ftr_bin_ = ts_ftr_bin_;
            bs_imp_dec_ = ts_imp_dec_;

            bs_left_count_ = ts_left_count_;
            bs_left_imp_ = ts_left_imp_;
            for (Index class_id = 0; class_id < class_count_; ++class_id) {
                bs_left_class_hist_[class_id] = ts_left_class_hist_[class_id];
            }
        }
    }

    // regression version
    inline void choose_best_split(Index& bs_ftr_id_,
                                  Index& bs_ftr_bin_,
                                  Index& bs_left_count_,
                                  Float& bs_imp_dec_,
                                  hist_type_t* bs_left_hist_,
                                  Index ts_ftr_id_,
                                  Index ts_ftr_bin_,
                                  Index ts_left_count_,
                                  Index ts_right_count_,
                                  Float ts_imp_dec_,
                                  const hist_type_t* ts_left_hist_,
                                  const Float* node_imp_list_ptr,
                                  Index hist_elem_count,
                                  Index node_id,
                                  Float impurity_threshold_,
                                  Index min_observations_in_leaf_node) {
        const Float* node_imp_ptr =
            node_imp_list_ptr + node_id * impl_const_t::node_imp_prop_count_;
        Float node_imp = node_imp_ptr[1];
        node_imp = node_imp / Float(ts_left_count_ + ts_right_count_);

        if (test_split_is_best(bs_ftr_id_,
                               bs_ftr_bin_,
                               bs_left_count_,
                               bs_imp_dec_,
                               ts_ftr_id_,
                               ts_ftr_bin_,
                               ts_left_count_,
                               ts_right_count_,
                               ts_imp_dec_,
                               node_id,
                               node_imp,
                               impurity_threshold_,
                               min_observations_in_leaf_node)) {
            bs_ftr_id_ = ts_ftr_id_;
            bs_ftr_bin_ = ts_ftr_bin_;
            bs_imp_dec_ = ts_imp_dec_;

            bs_left_count_ = ts_left_count_;

            for (Index i = 0; i < hist_elem_count; ++i) {
                bs_left_hist_[i] = ts_left_hist_[i];
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
    inline void update_node_bs_info(Index& bs_ftr_id_,
                                    Index& bs_ftr_bin_,
                                    Index& bs_left_count_,
                                    Float& bs_imp_dec_,
                                    Index* node_ptr,
                                    Float* node_imp_decr_ptr,
                                    Index node_id,
                                    Index index_max,
                                    bool update_imp_dec_required) {
        node_ptr[impl_const_t::ind_fid] =
            (bs_ftr_id_ == index_max) ? impl_const_t::leaf_mark_ : bs_ftr_id_;
        node_ptr[impl_const_t::ind_bin] =
            (bs_ftr_bin_ == index_max) ? impl_const_t::leaf_mark_ : bs_ftr_bin_;
        node_ptr[impl_const_t::ind_lch_grc] = bs_left_count_;

        if (update_imp_dec_required) {
            if constexpr (std::is_same_v<task_t, task::classification>) {
                node_imp_decr_ptr[node_id] = bs_imp_dec_;
            }
            else {
                node_imp_decr_ptr[node_id] = bs_imp_dec_ / node_ptr[impl_const_t::ind_grc];
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

    inline bool my_split_is_best_for_sbg(sycl::nd_item<2>& item,
                                         Index& bs_ftr_id_,
                                         Index& bs_ftr_bin_,
                                         Index& bs_left_count_,
                                         Float& bs_imp_dec_,
                                         Index* node_ptr,
                                         Float* node_imp_decr_ptr,
                                         Index node_id,
                                         Index index_max,
                                         bool update_imp_dec_required) {
        auto sbg = item.get_sub_group();
        const Index sub_group_local_id = sbg.get_local_id();

        const Float bestImpDec = sycl::reduce_over_group(sbg, bs_imp_dec_, maximum<Float>());

        const Index impDecIsBest = float_eq(bestImpDec, bs_imp_dec_);

        const Index bestFeatureId =
            sycl::reduce_over_group(sbg, impDecIsBest ? bs_ftr_id_ : index_max, minimum<Index>());
        const Index bestFeatureValue = sycl::reduce_over_group(
            sbg,
            (bestFeatureId == bs_ftr_id_ && impDecIsBest) ? bs_ftr_bin_ : index_max,
            minimum<Index>());

        const bool noneSplitFoundBySubGroup =
            ((impl_const_t::leaf_mark_ == bestFeatureId) && (0 == sub_group_local_id));
        const bool mySplitIsBest = (impl_const_t::leaf_mark_ != bestFeatureId &&
                                    bs_ftr_id_ == bestFeatureId && bs_ftr_bin_ == bestFeatureValue);
        return (noneSplitFoundBySubGroup || mySplitIsBest);
    }

    // classififcation version
    inline void choose_best_split_for_sbg(sycl::nd_item<2>& item,
                                          split_info_t& bs,
                                          Index* node_ptr,
                                          Float* node_imp_decr_ptr,
                                          Float* left_imp_list_ptr,
                                          hist_type_t* left_class_hist_list_ptr,
                                          Index node_id,
                                          Index index_max,
                                          Index class_count,
                                          bool update_imp_dec_required) {
        if (my_split_is_best_for_sbg(item, bs, node_ptr, node_id, index_max)) {
            update_node_bs_info(bs,
                                node_ptr,
                                node_imp_decr_ptr,
                                node_id,
                                index_max,
                                update_imp_dec_required);
            update_left_child_imp(left_imp_list_ptr,
                                  left_class_hist_list_ptr,
                                  bs.left_imp,
                                  bs.left_hist,
                                  node_id,
                                  class_count);
        }
    }
    inline void choose_best_split_for_sbg(sycl::nd_item<2>& item,
                                          Index& bs_ftr_id_,
                                          Index& bs_ftr_bin_,
                                          Index& bs_left_count_,
                                          Float& bs_left_imp_,
                                          Float& bs_imp_dec_,
                                          hist_type_t* bs_left_class_hist_,
                                          Index* node_ptr,
                                          Float* node_imp_decr_ptr,
                                          Float* left_imp_list_ptr,
                                          hist_type_t* left_class_hist_list_ptr,
                                          Index node_id,
                                          Index index_max,
                                          Index class_count,
                                          bool update_imp_dec_required) {
        if (item.get_sub_group().get_group_id() > 0) {
            return;
        }

        if (my_split_is_best_for_sbg(item,
                                     bs_ftr_id_,
                                     bs_ftr_bin_,
                                     bs_left_count_,
                                     bs_imp_dec_,
                                     node_ptr,
                                     node_imp_decr_ptr,
                                     node_id,
                                     index_max,
                                     update_imp_dec_required)) {
            update_node_bs_info(bs_ftr_id_,
                                bs_ftr_bin_,
                                bs_left_count_,
                                bs_imp_dec_,
                                node_ptr,
                                node_imp_decr_ptr,
                                node_id,
                                index_max,
                                update_imp_dec_required);
            update_left_child_imp(left_imp_list_ptr,
                                  left_class_hist_list_ptr,
                                  bs_left_imp_,
                                  bs_left_class_hist_,
                                  node_id,
                                  class_count);
        }
    }

    // regression version
    inline void choose_best_split_for_sbg(sycl::nd_item<2>& item,
                                          Index& bs_ftr_id_,
                                          Index& bs_ftr_bin_,
                                          Index& bs_left_count_,
                                          Float& bs_imp_dec_,
                                          hist_type_t* bs_left_hist_,
                                          Index* node_ptr,
                                          Float* node_imp_decr_ptr,
                                          Float* left_imp_list_ptr,
                                          Index node_id,
                                          Index index_max,
                                          bool update_imp_dec_required) {
        if (item.get_sub_group().get_group_id() > 0) {
            return;
        }

        if (my_split_is_best_for_sbg(item,
                                     bs_ftr_id_,
                                     bs_ftr_bin_,
                                     bs_left_count_,
                                     bs_imp_dec_,
                                     node_ptr,
                                     node_imp_decr_ptr,
                                     node_id,
                                     index_max,
                                     update_imp_dec_required)) {
            update_node_bs_info(bs_ftr_id_,
                                bs_ftr_bin_,
                                bs_left_count_,
                                bs_imp_dec_,
                                node_ptr,
                                node_imp_decr_ptr,
                                node_id,
                                index_max,
                                update_imp_dec_required);
            update_left_child_imp(left_imp_list_ptr, bs_left_hist_, node_id);
        }
    }

    /*
    inline Index reduce_min_over_group(sycl::nd_item<2>& item,
                  Index* slm_buf,
                  Index val) {
        auto sbg = item.get_sub_group();

        //try remove this condition for perf check
        if (sbg.get_group_id() == 0 && sbg.get_local_id() == 0) {
            slm_buf[0] = val;
        }
        item.barrier(sycl::access::fence_space::local_space);

        Index sbg_min = sycl::reduce_over_group(sbg, val, minimum<Index>());
        if(sbg.get_local_id() == 0) {
            atomic_local_min(&slm_buf[0], sbg_min);
        }

        item.barrier(sycl::access::fence_space::local_space);

        return slm_buf[0];
    }

    template <typename T>
    inline T reduce_add_over_group(sycl::nd_item<2>& item,
                  T* slm_buf,
                  T val) {
        auto sbg = item.get_sub_group();

        //try remove this condition for perf check
        if (sbg.get_group_id() == 0 && sbg.get_local_id() == 0) {
            slm_buf[0] = Index(0);
        }
        item.barrier(sycl::access::fence_space::local_space);

        T sbg_sum = sycl::reduce_over_group(sbg, val, plus<T>());
        if(sbg.get_local_id() == 0) {
            atomic_local_add(&slm_buf[0], sbg_sum);
        }

        item.barrier(sycl::access::fence_space::local_space);

        return slm_buf[0];
    }

    inline Float reduce_add_over_group(sycl::nd_item<2>& item,
                  Float* slm_buf,
                  Float  val,
                  Index  elem_count) {
        auto sbg = item.get_sub_group();
        const Index sub_group_size = sbg.get_local_range()[0];
        const Index local_id = item.get_local_id()[0];

        Float count_val = val;
        slm_buf[sbg.get_group_id()] = sycl::reduce_over_group(sbg, count_val, plus<Float>());

        Index i = elem_count;
        i = i / sub_group_size + bool(i % sub_group_size);

        for(; i > 1; i = i / sub_group_size + bool(i % sub_group_size)) {
            item.barrier(sycl::access::fence_space::local_space);

            count_val = (local_id < i) ? slm_buf[local_id] : Float(0);
            slm_buf[sbg.get_group_id()] = sycl::reduce_over_group(sbg, count_val, plus<Float>());
        }

        item.barrier(sycl::access::fence_space::local_space);

        return slm_buf[0];
    }
*/
    inline Index reserve_global_slot(Index* global_slot_flag_list, Index desired_idx, Index count) {
        for (Index i = desired_idx; i < count; i = ((i == count - 1) ? 0 : ++i)) {
            if (atomic_global_cmpxchg(&global_slot_flag_list[i], 0, 1) == 0) {
                return i;
            }
        }
        return desired_idx;
    }

    inline void release_global_slot(Index* global_slot_flag_list, Index idx) {
        global_slot_flag_list[idx] = 0;
    }
};

template <typename Float, typename Bin, typename Index, typename Task, bool use_private_mem>
sycl::event
train_best_split_impl<Float, Bin, Index, Task, use_private_mem>::compute_best_split_by_histogram(
    sycl::queue& queue,
    const context_t& ctx,
    const pr::ndarray<hist_type_t, 1>& node_hist_list,
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
    ONEDAL_PROFILER_TASK(compute_best_split_by_histogram, queue);

    using split_smp_t = split_smp<Float, Index, Task>;

    Index hist_prop_count = 0;
    if constexpr (std::is_same_v<std::decay_t<Task>, task::classification>) {
        hist_prop_count = ctx.class_count_;
    }
    else {
        hist_prop_count = impl_const<Index, task::regression>::hist_prop_count_;
    }

    ONEDAL_ASSERT(node_hist_list.get_count() == hist_prop_count * ctx.max_bin_count_among_ftrs_ *
                                                    ctx.selected_ftr_count_ * node_count);
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

    const hist_type_t* node_hist_list_ptr = node_hist_list.get_data();
    const Index* selected_ftr_list_ptr = selected_ftr_list.get_data();

    const Index* bin_offset_list_ptr = bin_offset_list.get_data();
    imp_data_list_ptr<Float, Index, Task> imp_list_ptr(imp_data_list);

    const Index* node_indices_ptr = node_ind_list.get_data();
    Index* node_list_ptr = node_list.get_mutable_data();
    Float* node_imp_decr_list_ptr =
        update_imp_dec_required ? node_imp_dec_list.get_mutable_data() : nullptr;

    imp_data_list_ptr_mutable<Float, Index, Task> left_imp_list_ptr(left_child_imp_data_list);

    const Index max_bin_count_among_ftrs = ctx.max_bin_count_among_ftrs_;

    const Index selected_ftr_count = ctx.selected_ftr_count_;

    constexpr Index buff_size = impl_const_t::private_hist_buff_size;

    const Index class_count = ctx.class_count_;
    const Float imp_threshold = ctx.impurity_threshold_;
    const Index min_obs_leaf = ctx.min_observations_in_leaf_node_;
    const Index index_max = ctx.index_max_;

    const Float min_imp_dec = de::limits<Float>::min();

    const Float* node_imp_list_ptr = imp_list_ptr.imp_list_ptr_;
    Float* left_child_imp_list_ptr = left_imp_list_ptr.imp_list_ptr_;

    // following vars are not used for regression, but should present to compile kernel
    const Index* class_hist_list_ptr = imp_list_ptr.get_class_hist_list_ptr_or_null();
    Index* left_child_class_hist_list_ptr = left_imp_list_ptr.get_class_hist_list_ptr_or_null();

    auto local_size = bk::device_max_sg_size(queue);

    std::size_t local_hist_buf_size = 0;
    if constexpr (use_private_mem) {
        local_hist_buf_size = 1; // just some non zero value
    }
    else {
        local_hist_buf_size = (2 * local_size) * hist_prop_count; // x2 - for each item 2 hists
    }

    const sycl::nd_range<2> nd_range =
        bk::make_multiple_nd_range_2d({ local_size, node_count }, { local_size, 1 });

    sycl::event last_event;

    last_event = queue.submit([&](cl::sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<hist_type_t> local_hist_buf(local_hist_buf_size, cgh);
        cgh.parallel_for(nd_range, [=](cl::sycl::nd_item<2> item) {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id() > 0) {
                return;
            }
            const Index node_idx = item.get_global_id()[1];
            const Index node_id = node_indices_ptr[node_ind_ofs + node_idx];
            Index* node_ptr = node_list_ptr + node_id * impl_const_t::node_prop_count_;

            const Index sub_group_local_id = sbg.get_local_id();
            const Index sub_group_size = sbg.get_local_range()[0];

            split_smp_t sp_hlp;

            Index bs_ftr_bin = impl_const_t::leaf_mark_;
            Index bs_ftr_id = impl_const_t::leaf_mark_;
            Index bs_left_count = 0;

            hist_type_t* local_hist_buf_ptr = local_hist_buf.get_pointer().get();

            hist_type_t* bs_left_hist = nullptr;

            hist_type_t prv_bs_left_hist[buff_size] = { hist_type_t(0) };
            if constexpr (use_private_mem) {
                bs_left_hist = &prv_bs_left_hist[0];
            }
            else {
                bs_left_hist =
                    fill_zero(local_hist_buf_ptr + (sub_group_local_id * 2 + 0) * hist_prop_count,
                              hist_prop_count);
            }

            Float bs_left_imp = Float(0);
            Float bs_imp_dec = min_imp_dec;

            for (Index ftr_idx = sub_group_local_id; ftr_idx < selected_ftr_count;
                 ftr_idx += sub_group_size) {
                const Index ts_ftr_id =
                    selected_ftr_list_ptr[node_id * selected_ftr_count + ftr_idx];
                const Index ts_ftr_bin_count =
                    bin_offset_list_ptr[ts_ftr_id + 1] - bin_offset_list_ptr[ts_ftr_id];

                const hist_type_t* node_hist_ptr =
                    node_hist_list_ptr +
                    node_idx * selected_ftr_count * max_bin_count_among_ftrs * hist_prop_count;
                const hist_type_t* ftr_hist_ptr =
                    node_hist_ptr + ftr_idx * max_bin_count_among_ftrs * hist_prop_count;

                Index ts_left_count = 0;
                Index ts_right_count = 0;

                hist_type_t* ts_left_hist = nullptr;
                hist_type_t prv_ts_left_hist[buff_size] = { hist_type_t(0) };
                if constexpr (use_private_mem) {
                    ts_left_hist = &prv_ts_left_hist[0];
                }
                else {
                    ts_left_hist = fill_zero(
                        local_hist_buf_ptr + (sub_group_local_id * 2 + 1) * hist_prop_count,
                        hist_prop_count);
                }

                Float ts_left_imp = Float(0);
                Float ts_right_imp = Float(0);
                Float ts_imp_dec = Float(0);

                for (Index ts_ftr_bin = 0; ts_ftr_bin < ts_ftr_bin_count; ++ts_ftr_bin) {
                    const Index bin_ofs = ts_ftr_bin * hist_prop_count;

                    if constexpr (std::is_same_v<Task, task::classification>) {
                        sp_hlp.merge_bin_hist(ts_left_count,
                                              ts_left_hist,
                                              ftr_hist_ptr + bin_ofs,
                                              hist_prop_count);
                    }
                    else {
                        sp_hlp.merge_bin_hist(ts_left_hist,
                                              ftr_hist_ptr + bin_ofs,
                                              hist_prop_count);
                    }

                    if constexpr (std::is_same_v<Task, task::classification>) {
                        sp_hlp.calc_imp_dec(ts_right_count,
                                            ts_left_imp,
                                            ts_right_imp,
                                            ts_imp_dec,
                                            ts_left_count,
                                            node_ptr,
                                            node_imp_list_ptr,
                                            class_hist_list_ptr,
                                            ts_left_hist,
                                            class_count,
                                            node_id);

                        sp_hlp.choose_best_split(bs_ftr_id,
                                                 bs_ftr_bin,
                                                 bs_left_count,
                                                 bs_left_imp,
                                                 bs_imp_dec,
                                                 bs_left_hist,
                                                 ts_ftr_id,
                                                 ts_ftr_bin,
                                                 ts_left_count,
                                                 ts_right_count,
                                                 ts_left_imp,
                                                 ts_imp_dec,
                                                 ts_left_hist,
                                                 node_imp_list_ptr,
                                                 class_count,
                                                 node_id,
                                                 imp_threshold,
                                                 min_obs_leaf);
                    }
                    else {
                        sp_hlp.calc_imp_dec(ts_left_count,
                                            ts_right_count,
                                            ts_imp_dec,
                                            node_ptr,
                                            node_imp_list_ptr,
                                            ts_left_hist,
                                            node_id);

                        sp_hlp.choose_best_split(bs_ftr_id,
                                                 bs_ftr_bin,
                                                 bs_left_count,
                                                 bs_imp_dec,
                                                 bs_left_hist,
                                                 ts_ftr_id,
                                                 ts_ftr_bin,
                                                 ts_left_count,
                                                 ts_right_count,
                                                 ts_imp_dec,
                                                 ts_left_hist,
                                                 node_imp_list_ptr,
                                                 buff_size,
                                                 node_id,
                                                 imp_threshold,
                                                 min_obs_leaf);
                    }
                }
            }

            if constexpr (std::is_same_v<Task, task::classification>) {
                sp_hlp.choose_best_split_for_sbg(item,
                                                 bs_ftr_id,
                                                 bs_ftr_bin,
                                                 bs_left_count,
                                                 bs_left_imp,
                                                 bs_imp_dec,
                                                 bs_left_hist,
                                                 node_ptr,
                                                 node_imp_decr_list_ptr,
                                                 left_child_imp_list_ptr,
                                                 left_child_class_hist_list_ptr,
                                                 node_id,
                                                 index_max,
                                                 class_count,
                                                 update_imp_dec_required);
            }
            else {
                sp_hlp.choose_best_split_for_sbg(item,
                                                 bs_ftr_id,
                                                 bs_ftr_bin,
                                                 bs_left_count,
                                                 bs_imp_dec,
                                                 bs_left_hist,
                                                 node_ptr,
                                                 node_imp_decr_list_ptr,
                                                 left_child_imp_list_ptr,
                                                 node_id,
                                                 index_max,
                                                 update_imp_dec_required);
            }
        });
    });

    last_event.wait_and_throw();

    return last_event;
}

#define ENABLE_PRINT

template <typename Float, typename Bin, typename Index, typename Task, bool use_private_mem>
sycl::event
train_best_split_impl<Float, Bin, Index, Task, use_private_mem>::compute_best_split_single_pass(
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

    const Bin* data_ptr = data.get_data();
    const Float* response_ptr = response.get_data();
    const Index* tree_order_ptr = tree_order.get_data();

    const Index* selected_ftr_list_ptr = selected_ftr_list.get_data();

    imp_data_list_ptr<Float, Index, Task> imp_list_ptr(imp_data_list);

    const Index* node_indices_ptr = node_ind_list.get_data();
    Index* node_list_ptr = node_list.get_mutable_data();
    Float* node_imp_decr_list_ptr =
        update_imp_dec_required ? node_imp_dec_list.get_mutable_data() : nullptr;

    imp_data_list_ptr_mutable<Float, Index, Task> left_imp_list_ptr(left_child_imp_data_list);

    const Index column_count = ctx.column_count_;

    const Index selected_ftr_count = ctx.selected_ftr_count_;

    const Index index_max = ctx.index_max_;

    auto local_size = bk::device_max_sg_size(queue);

    std::size_t local_hist_buf_size = 0;
    if constexpr (use_private_mem) {
        local_hist_buf_size = 1; // just some non zero value
    }
    else {
        local_hist_buf_size = (2 * local_size) * hist_prop_count; // x2 - for each item 2 hists
    }

    const sycl::nd_range<2> nd_range =
        bk::make_multiple_nd_range_2d({ local_size, node_count }, { local_size, 1 });

    sycl::event last_event;

    const Index class_count = ctx.class_count_;
    const Float imp_threshold = ctx.impurity_threshold_;
    const Index min_obs_leaf = ctx.min_observations_in_leaf_node_;

    const Float min_imp_dec = de::limits<Float>::min();

    const Float* node_imp_list_ptr = imp_list_ptr.imp_list_ptr_;
    Float* left_child_imp_list_ptr = left_imp_list_ptr.imp_list_ptr_;

    // following vars are not used for regression, but should present to compile kernel
    const Index* class_hist_list_ptr = imp_list_ptr.get_class_hist_list_ptr_or_null();
    Index* left_child_class_hist_list_ptr = left_imp_list_ptr.get_class_hist_list_ptr_or_null();

    constexpr Index buff_size = impl_const_t::private_hist_buff_size;
    constexpr Index min_supported_row_count = 33;

    _P("single pass original");
    //_P("node imp:");
    //print_nd_arr(queue, imp_data_list.imp_list_, node_count, impl_const_t::node_imp_prop_count_);

    last_event = queue.submit([&](cl::sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<hist_type_t> local_hist_buf(local_hist_buf_size, cgh);
#ifdef ENABLE_PRINT
        sycl::stream out(65536, 128, cgh);
#endif
        cgh.parallel_for(nd_range, [=](cl::sycl::nd_item<2> item) {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id() > 0) {
                return;
            }
            const Index node_idx = item.get_global_id()[1];
            const Index node_id = node_indices_ptr[node_ind_ofs + node_idx];
            Index* node_ptr = node_list_ptr + node_id * impl_const_t::node_prop_count_;

            const Index sub_group_local_id = sbg.get_local_id();
            const Index sub_group_size = sbg.get_local_range()[0];

            const Index row_ofs = node_ptr[impl_const_t::ind_ofs];
            const Index row_count = node_ptr[impl_const_t::ind_lrc];

#ifdef ENABLE_PRINT
            if (node_idx == 0 && sub_group_local_id == 0) {
                out << "-------- node_id " << node_id << ", row_count " << row_count << ", row_ofs "
                    << row_ofs << sycl::endl;
            }
#endif
            if (row_count < min_supported_row_count) {
                return;
            }

            split_smp_t sp_hlp;

            Index bs_ftr_bin = impl_const_t::leaf_mark_;
            Index bs_ftr_id = impl_const_t::leaf_mark_;
            Index bs_left_count = 0;

            hist_type_t* local_hist_buf_ptr = local_hist_buf.get_pointer().get();

            hist_type_t* bs_left_hist = nullptr;
            hist_type_t prv_bs_left_hist[buff_size] = { hist_type_t(0) };
            if constexpr (use_private_mem) {
                bs_left_hist = &prv_bs_left_hist[0];
            }
            else {
                bs_left_hist =
                    fill_zero(local_hist_buf_ptr + (sub_group_local_id * 2 + 0) * hist_prop_count,
                              hist_prop_count);
            }

            Float bs_left_imp = Float(0);
            Float bs_imp_dec = min_imp_dec;

            for (Index ftr_idx = sub_group_local_id; ftr_idx < selected_ftr_count;
                 ftr_idx += sub_group_size) {
                const Index ts_ftr_id =
                    selected_ftr_list_ptr[node_id * selected_ftr_count + ftr_idx];

                bin_map_t bin_map[bin_block_count] = { 0 };

                // calculating classes histogram rows count <= bins num
                for (Index i = 0; i < row_count; ++i) {
                    Index curr_row_id = tree_order_ptr[row_ofs + i];
                    Index ts_ftr_bin = data_ptr[curr_row_id * column_count + ts_ftr_id];

                    bool bin_not_processed = !is_bin_processed(bin_map, ts_ftr_bin);
                    if (bin_not_processed) {
                        Index ts_left_count = 0;
                        Index ts_right_count = 0;

                        hist_type_t* ts_left_hist = nullptr;
                        hist_type_t prv_ts_left_hist[buff_size] = { hist_type_t(0) };
                        if constexpr (use_private_mem) {
                            ts_left_hist = &prv_ts_left_hist[0];
                        }
                        else {
                            ts_left_hist = fill_zero(
                                local_hist_buf_ptr + (sub_group_local_id * 2 + 1) * hist_prop_count,
                                hist_prop_count);
                        }

                        Float ts_left_imp = Float(0);
                        Float ts_right_imp = Float(0);
                        Float ts_imp_dec = Float(0);

                        for (Index row_idx = 0; row_idx < row_count; ++row_idx) {
                            Index id = tree_order_ptr[row_ofs + row_idx];
                            Index bin = data_ptr[id * column_count + ts_ftr_id];

                            if constexpr (std::is_same_v<Task, task::classification>) {
                                sp_hlp.add_val(ts_left_count,
                                               ts_left_hist,
                                               ts_ftr_bin,
                                               bin,
                                               response_ptr[id]);
                            }
                            else {
                                sp_hlp.add_val(ts_left_hist, ts_ftr_bin, bin, response_ptr[id]);
                            }
                        }

                        mark_bin_processed(bin_map, ts_ftr_bin);

                        if constexpr (std::is_same_v<Task, task::classification>) {
                            sp_hlp.calc_imp_dec(ts_right_count,
                                                ts_left_imp,
                                                ts_right_imp,
                                                ts_imp_dec,
                                                ts_left_count,
                                                node_ptr,
                                                node_imp_list_ptr,
                                                class_hist_list_ptr,
                                                ts_left_hist,
                                                class_count,
                                                node_id);

                            sp_hlp.choose_best_split(bs_ftr_id,
                                                     bs_ftr_bin,
                                                     bs_left_count,
                                                     bs_left_imp,
                                                     bs_imp_dec,
                                                     bs_left_hist,
                                                     ts_ftr_id,
                                                     ts_ftr_bin,
                                                     ts_left_count,
                                                     ts_right_count,
                                                     ts_left_imp,
                                                     ts_imp_dec,
                                                     ts_left_hist,
                                                     node_imp_list_ptr,
                                                     class_count,
                                                     node_id,
                                                     imp_threshold,
                                                     min_obs_leaf);
                        }
                        else {
                            sp_hlp.calc_imp_dec(ts_left_count,
                                                ts_right_count,
                                                ts_imp_dec,
                                                node_ptr,
                                                node_imp_list_ptr,
                                                ts_left_hist,
                                                node_id);

#ifdef ENABLE_PRINT
                            //if(node_idx == 0) {
                            //    out << "-------- node_id " << node_id << ", ts_ftr_id " << ts_ftr_id << ", ts_ftr_bin " << ts_ftr_bin
                            //    << ", ts_right_count " << ts_right_count << ", ts_left_count " << ts_left_count << ", ts_imp_dec " << ts_imp_dec
                            //    << "----- , ts_left_imp " << ts_left_imp << ", ts_right_imp " << ts_right_imp << sycl::endl;
                            //    out << "----- ts_left_hist[0] " << ts_left_hist[0] << "----- ts_left_hist[1] " << ts_left_hist[1] << ", ts_left_hist[2] " << ts_left_hist[2]
                            //    << sycl::endl;
                            //}
#endif
                            sp_hlp.choose_best_split(bs_ftr_id,
                                                     bs_ftr_bin,
                                                     bs_left_count,
                                                     bs_imp_dec,
                                                     bs_left_hist,
                                                     ts_ftr_id,
                                                     ts_ftr_bin,
                                                     ts_left_count,
                                                     ts_right_count,
                                                     ts_imp_dec,
                                                     ts_left_hist,
                                                     node_imp_list_ptr,
                                                     buff_size,
                                                     node_id,
                                                     imp_threshold,
                                                     min_obs_leaf);
                        }
                    }
                }
            }

            if constexpr (std::is_same_v<Task, task::classification>) {
                sp_hlp.choose_best_split_for_sbg(item,
                                                 bs_ftr_id,
                                                 bs_ftr_bin,
                                                 bs_left_count,
                                                 bs_left_imp,
                                                 bs_imp_dec,
                                                 bs_left_hist,
                                                 node_ptr,
                                                 node_imp_decr_list_ptr,
                                                 left_child_imp_list_ptr,
                                                 left_child_class_hist_list_ptr,
                                                 node_id,
                                                 index_max,
                                                 class_count,
                                                 update_imp_dec_required);
            }
            else {
                sp_hlp.choose_best_split_for_sbg(item,
                                                 bs_ftr_id,
                                                 bs_ftr_bin,
                                                 bs_left_count,
                                                 bs_imp_dec,
                                                 bs_left_hist,
                                                 node_ptr,
                                                 node_imp_decr_list_ptr,
                                                 left_child_imp_list_ptr,
                                                 node_id,
                                                 index_max,
                                                 update_imp_dec_required);
            }
        });
    });

    //last_event.wait_and_throw();

    return last_event;
}
/*
template <typename T, typename Index>
T* get_buf_ptr(std::uint8_t** buf_ptr, Index elem_count) {
    T* res_ptr = reinterpret_cast<T*>(*buf_ptr);
    (*buf_ptr) += elem_count * sizeof(T);
    return res_ptr;
}
*/

//// new single pass
template <typename Float, typename Bin, typename Index, typename Task, bool use_private_mem>
sycl::event
train_best_split_impl<Float, Bin, Index, Task, use_private_mem>::compute_best_split_single_pass_new(
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

    //_P("sp _new");
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

    ///auto ind_list_host = node_ind_list.to_host(queue);
    ///auto ind_list_host_ptr = ind_list_host.get_data();
    ///auto node_list_host = node_list.to_host(queue);
    ///auto node_list_host_ptr = node_list_host.get_data();

    //Index cnt_16 = 0;
    //Index cnt_32 = 0;
    //Index cnt_64 = 0;
    //Index cnt_128 = 0;

    //for(Index i = 0; i < node_count; i++) {
    //    const Index node_id = ind_list_host_ptr[node_ind_ofs + i];
    //    const Index* node_ptr = node_list_host_ptr + node_id * impl_const_t::node_prop_count_;
    //    auto cnt = node_ptr[impl_const_t::ind_lrc];
    //    if(cnt <= 16) cnt_16++;
    //    else if(cnt <= 32) cnt_32++;
    //    else if(cnt <= 64) cnt_64++;
    //    else if(cnt <= 128) cnt_128++;
    //}
    //    _P("nodes rc: 16(%d), 32(%d), 64(%d), 128(%d), other(%d)", cnt_16, cnt_32, cnt_64, cnt_128, node_count - (cnt_16 + cnt_32 + cnt_64 + cnt_128));

    Index max_sbg_size = bk::device_max_sg_size(queue);
    //auto max_sbg_size = 64;
    //auto max_wg_size = bk::device_max_wg_size(queue);
    Index max_wg_size = 256;
    auto max_sbg_count =
        (max_wg_size / max_sbg_size) * 2; // *2 in case if compiler will reduce sbg_size

    Index local_size = max_wg_size;
    //auto local_size = max_sbg_size;

    std::size_t local_hist_buf_size = 0;

    std::size_t local_bs_buf_int_size = 0;
    std::size_t local_bs_buf_float_size = 0;

    //Index max_data_elem_count = 256;
    //std::size_t local_ftr_buf_int_size = max_data_elem_count * selected_ftr_count;
    //std::size_t local_resp_buf_float_size = max_data_elem_count;

    // required local_buffers size
    // min_bin - 1, count_holder - 1
    // bs_hist, ts_hist-  2*(classification - class_count, regression - 3 count, mean, sum2cent)

    // required global buffers size
    // split_hist (classification - class_count, regression - 3 count, mean, sum2cent)
    // (int) ftr_id, ftr_bin, left_count,
    // (int) ftr_slot_lock_flag, ftr_slot_merged_count
    // (Float) left_imp, imp_dec

    // 1 counter for global count of processed ftrs for node, and max_sbg_size - num of slot flags
    std::size_t global_aux_ftr_buf_int_size = 1 + max_sbg_size;
    std::size_t global_split_slot_buf_int_size = 3;
    std::size_t global_split_slot_buf_float_size = 2;
    std::size_t global_split_slot_hist_buf_size = hist_prop_count;

    local_bs_buf_int_size = 3; // bs_ftr_id, bs_ftr_bin and bs_left_count
    local_bs_buf_int_size += 2; // min_bin value holder, val holder for atomic reduce add
    local_bs_buf_float_size = max_sbg_count * 2;
    local_hist_buf_size = hist_prop_count * 2; // x2 because bs_hist and ts_hist

    //auto dev_local_size = bk::device_local_mem_size(queue);
    //_P("selected_ftr_count %d, device local size = %ld", selected_ftr_count, dev_local_size);
    //auto dev_local_size = (std::int64_t)(queue.get_device().template get_info<sycl::info::device::local_mem_size>());

    //_P("lsize %ld, lsize(KB) %ld", dev_local_size, dev_local_size / 1024);
    //_P("required %ld (KB)", ((local_hist_buf_size + local_bs_buf_int_size + local_bs_buf_float_size) * 4) / 1024);

    //check local size fit to requirements

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

    //_P("bs sp new");
    //_P("hist_prop_count %d, min_observations_in_leaf_node %d", hist_prop_count, min_obs_leaf);
    //_P("node imp:");
    //print_nd_arr(queue, imp_data_list.imp_list_, node_count, impl_const_t::node_imp_prop_count_);

    Index wg_in_block_count = 8192;
    Index max_wg_count_for_node = 4;
    Index node_in_block_count = wg_in_block_count / max_wg_count_for_node;

    auto global_aux_ftr_buf_int = pr::ndarray<Index, 1>::empty(
        queue,
        { std::int64_t(node_in_block_count * global_aux_ftr_buf_int_size) },
        alloc::device);

    auto global_split_buf_int = pr::ndarray<Index, 1>::empty(
        queue,
        { std::int64_t(node_in_block_count * max_sbg_size * global_split_slot_buf_int_size) },
        alloc::device);
    auto global_split_buf_float = pr::ndarray<Float, 1>::empty(
        queue,
        { std::int64_t(node_in_block_count * max_sbg_size * global_split_slot_buf_float_size) },
        alloc::device);
    auto global_split_hist_buf = pr::ndarray<hist_type_t, 1>::empty(
        queue,
        { std::int64_t(node_in_block_count * max_sbg_size * global_split_slot_hist_buf_size) },
        alloc::device);

    [[maybe_unused]] Index* global_aux_ftr_buf_int_ptr = global_aux_ftr_buf_int.get_mutable_data();
    [[maybe_unused]] Index* global_split_buf_int_ptr = global_split_buf_int.get_mutable_data();
    [[maybe_unused]] Float* global_split_buf_float_ptr = global_split_buf_float.get_mutable_data();
    [[maybe_unused]] hist_type_t* global_split_hist_buf_ptr =
        global_split_hist_buf.get_mutable_data();

    //auto has_local_mem = device.is_host() ||
    //                   (device.get_info<sycl::info::device::local_mem_type>() !=
    //                    sycl::info::local_mem_type::none);
    //  auto local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
    //  if (!has_local_mem || local_mem_size < (wgroup_size * sizeof(int32_t))) {
    //    throw "Device doesn't have enough local memory!";
    //  }
    //Index min_supported_row_count = 33;
    for (Index processed_node_cnt = 0; processed_node_cnt < node_count;
         processed_node_cnt += node_in_block_count, node_ind_ofs += node_in_block_count) {
        auto fill_aux_ftr_event = global_aux_ftr_buf_int.fill(queue, 0);
        auto fill_int_event = global_split_buf_int.fill(queue, -1);
        auto fill_float_event = global_split_buf_float.fill(queue, min_imp_dec);

        bk::event_vector fill_deps{ fill_aux_ftr_event, fill_int_event, fill_float_event };

        const sycl::nd_range<2> nd_range =
            //bk::make_multiple_nd_range_2d({ local_size, nc_size}, { local_size, 1 });
            //bk::make_multiple_nd_range_2d({ local_size, node_count}, { local_size, 1 });
            bk::make_multiple_nd_range_2d({ local_size, wg_in_block_count }, { local_size, 1 });

        last_event = queue.submit([&](cl::sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.depends_on(fill_deps);
#ifdef ENABLE_PRINT
            sycl::stream out(65536, 128, cgh);
#endif
            local_accessor_rw_t<std::uint8_t> local_byte_buf(
                local_bs_buf_int_size * sizeof(Index) + local_bs_buf_float_size * sizeof(Float) +
                    local_hist_buf_size * sizeof(hist_type_t),
                cgh);

            cgh.parallel_for(nd_range, [=](cl::sycl::nd_item<2> item) {
                auto sbg = item.get_sub_group();
                const Index group_id = item.get_group().get_group_id(1);
                [[maybe_unused]] const Index ftr_group_id = group_id % max_wg_count_for_node;
                [[maybe_unused]] const Index node_idx =
                    item.get_global_id()[1] / max_wg_count_for_node;
                //out << " ------ node_ind_ofs " << node_ind_ofs << " node idx " << node_idx << " node count " << node_count << sycl::endl;
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

                //if(row_count < min_supported_row_count) { return; }

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

                //Index* bs_ftr_id_buf_ptr = local_bs_buf_int_ptr + 0;
                //Index* bs_ftr_bin_buf_ptr = local_bs_buf_int_ptr + 1;
                //Index* bs_left_count_buf_ptr = local_bs_buf_int_ptr + 2;

                [[maybe_unused]] Index* min_bin_buf_ptr = local_bs_buf_int_ptr + 3;
                [[maybe_unused]] Index* count_buf_ptr = local_bs_buf_int_ptr + 4;

                //Float* bs_left_imp_buf_ptr = local_bs_buf_float_ptr + 0;
                //Float* bs_imp_dec_buf_ptr = local_bs_buf_float_ptr + 1;

                //hist_type_t* bs_left_hist = fill_with_group(item, local_hist_buf_ptr + 0 * hist_prop_count, hist_prop_count, hist_type_t(0));
                //hist_type_t* left_hist = fill_with_group(item, local_hist_buf_ptr + 0 * hist_prop_count, hist_prop_count, hist_type_t(0));
                bs.init_clear(item, local_hist_buf_ptr + 0 * hist_prop_count, hist_prop_count);
                //sp_hlp.init_split_info(item, bs, local_hist_buf_ptr + 0 * hist_prop_count, hist_prop_count);

                //#ifdef ENABLE_PRINT
                //if (local_id == 0) {
                //            if(node_idx == 0) {
                //                //out << "byte_p " << local_byte_buf_ptr << ", p_int " << local_bs_buf_int_ptr << "p_hist " << local_hist_buf_ptr << ", p_float " << local_bs_buf_float_exp_ptr
                //                out << " int_size " << local_bs_buf_int_size << " hist_size " << local_hist_buf_size << sycl::endl;
                //                out << " float_size " << local_bs_buf_float_size << sycl::endl;
                //            }
                //}
                //#endif

                Index processed_ftr_count = 0;
                for (Index ftr_idx = ftr_group_id; ftr_idx < selected_ftr_count;
                     ftr_idx += max_wg_count_for_node) {
                    processed_ftr_count++;

                    split_info<Float, Index, Task> ts;
                    ts.init(local_hist_buf_ptr + 1 * hist_prop_count, hist_prop_count);
                    //ts.left_hist = local_hist_buf_ptr + 1 * hist_prop_count;
                    //const Index ts.ftr_id = selected_ftr_list_ptr[node_id * selected_ftr_count + ftr_idx];
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
                                //if(local_id == 0) { ts.left_hist[class_id] = total_class_count; }
                            }

                            ts.left_hist[class_count - 1] = ts.left_count - all_class_count;
                            //if(local_id == 0) { ts.left_hist[class_count - 1] = ts.left_count - all_class_count; }
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
#ifdef ENABLE_PRINT
                                if (node_idx == 0) {
                                    out << "before -------- node_id " << node_id << ", bs_ftr_id "
                                        << bs.ftr_id << ", bs_ftr_bin " << bs.ftr_bin
                                        << ", bs_right_count " << bs.right_count
                                        << ", bs_left_count " << bs.left_count << ", bs_imp_dec "
                                        << bs.imp_dec << "----- , bs_left_imp " << bs.left_imp
                                        << ", bs_right_imp " << bs.right_imp << sycl::endl;
                                    out << "----- bs_left_hist[0] " << bs.left_hist[0]
                                        << "----- bs_left_hist[1] " << bs.left_hist[1]
                                        << ", bs_left_hist[2] " << bs.left_hist[2] << sycl::endl;
                                }
#endif
                                sp_hlp.calc_imp_dec(ts.left_count,
                                                    ts.right_count,
                                                    ts.imp_dec,
                                                    node_ptr,
                                                    node_imp_list_ptr,
                                                    ts.left_hist,
                                                    node_id);

                                //#ifdef ENABLE_PRINT
                                //            if(node_idx == 0) {
                                //                out << "-------- node_id " << node_id << ", ts_ftr_id " << ts.ftr_id << ", ts_ftr_bin " << ts.ftr_bin
                                //                << ", ts_right_count " << ts.right_count << ", ts_left_count " << ts.left_count << ", ts_imp_dec " << ts.imp_dec
                                //                << "----- , ts_left_imp " << ts.left_imp << ", ts_right_imp " << ts.right_imp << sycl::endl;
                                //                out << "----- ts_left_hist[0] " << ts.left_hist[0] << "----- ts_left_hist[1] " << ts.left_hist[1] << ", ts_left_hist[2] " << ts.left_hist[2]
                                //                << sycl::endl;
                                //            }
                                //#endif
                                sp_hlp.choose_best_split(bs,
                                                         ts,
                                                         node_imp_list_ptr,
                                                         impl_const_t::hist_prop_count_,
                                                         node_id,
                                                         imp_threshold,
                                                         min_obs_leaf);
#ifdef ENABLE_PRINT
                                if (node_idx == 0) {
                                    out << "after -------- node_id " << node_id << ", bs_ftr_id "
                                        << bs.ftr_id << ", bs_ftr_bin " << bs.ftr_bin
                                        << ", bs_right_count " << bs.right_count
                                        << ", bs_left_count " << bs.left_count << ", bs_imp_dec "
                                        << bs.imp_dec << "----- , bs_left_imp " << bs.left_imp
                                        << ", bs_right_imp " << bs.right_imp << sycl::endl;
                                    out << "----- bs_left_hist[0] " << bs.left_hist[0]
                                        << "----- bs_left_hist[1] " << bs.left_hist[1]
                                        << ", bs_left_hist[2] " << bs.left_hist[2] << sycl::endl;
                                }
#endif
                            }
                        }
                        //ts.left_hist = sycl::group_broadcast(item.get_group(), ts.left_hist, 0);
                        //bs.left_hist = sycl::group_broadcast(item.get_group(), bs.left_hist, 0);

                        ts.ftr_bin = reduce_min_over_group(item,
                                                           min_bin_buf_ptr,
                                                           bin > ts.ftr_bin ? bin : index_max);
                    }
                }

                if (sub_group_id > 0) {
                    return;
                }

                [[maybe_unused]] Index total_processed_ftr_count = 0;
                [[maybe_unused]] Index slot_ind = 0;

                if (local_id == 0) {
#ifdef ENABLE_PRINT
                    if (row_count == 43) {
                        out << "before store -------- node_id " << node_id << ", bs_ftr_id "
                            << bs.ftr_id << ", bs_ftr_bin " << bs.ftr_bin << ", bs_right_count "
                            << bs.right_count << ", bs_left_count " << bs.left_count
                            << ", bs_imp_dec " << bs.imp_dec << "----- , bs_left_imp "
                            << bs.left_imp << ", bs_right_imp " << bs.right_imp << sycl::endl;
                        out << "----- bs_left_hist[0] " << bs.left_hist[0]
                            << "----- bs_left_hist[1] " << bs.left_hist[1] << sycl::endl;
                    }
#endif
                    bs.store(global_split_buf_int_ptr,
                             global_split_buf_float_ptr,
                             global_split_hist_buf_ptr,
                             node_idx,
                             max_sbg_size,
                             group_id % sub_group_size);
                    total_processed_ftr_count = atomic_global_sum(
                        global_aux_ftr_buf_int_ptr + node_idx * global_aux_ftr_buf_int_size,
                        processed_ftr_count);
                }

                // read slm marker
                total_processed_ftr_count =
                    sycl::group_broadcast(sbg, total_processed_ftr_count, 0);

                if (sub_group_id == 0 && total_processed_ftr_count == selected_ftr_count) {
                    bs.load(global_split_buf_int_ptr,
                            global_split_buf_float_ptr,
                            global_split_hist_buf_ptr,
                            node_idx,
                            max_sbg_size,
                            sub_group_local_id);

                    if constexpr (std::is_same_v<Task, task::classification>) {
                        //sp_hlp.choose_best_split_for_sbg(item,
                        //                                 bs,
                        //                                 node_ptr,
                        //                                 node_imp_decr_list_ptr,
                        //                                 left_child_imp_list_ptr,
                        //                                 left_child_class_hist_list_ptr,
                        //                                 node_id,
                        //                                 index_max,
                        //                                 class_count,
                        //                                 update_imp_dec_required);
                        if (sp_hlp
                                .my_split_is_best_for_sbg(item, bs, node_ptr, node_id, index_max)) {
                            sp_hlp.update_node_bs_info(bs,
                                                       node_ptr,
                                                       node_imp_decr_list_ptr,
                                                       node_id,
                                                       index_max,
                                                       update_imp_dec_required);
                            sp_hlp.update_left_child_imp(left_child_imp_list_ptr,
                                                         left_child_class_hist_list_ptr,
                                                         bs.left_imp,
                                                         bs.left_hist,
                                                         node_id,
                                                         class_count);
                        }
                    }
                    else {
                        //#ifdef ENABLE_PRINT
                        //            if(node_idx == 0) {
                        //                out << "load -------- node_id " << node_id << ", bs_ftr_id " << bs.ftr_id << ", bs_ftr_bin " << bs.ftr_bin
                        //                << ", bs_right_count " << bs.right_count << ", bs_left_count " << bs.left_count << ", bs_imp_dec " << bs.imp_dec
                        //                << "----- , bs_left_imp " << bs.left_imp << ", bs_right_imp " << bs.right_imp << sycl::endl;
                        //                out << "----- bs_left_hist[0] " << bs.left_hist[0] << "----- bs_left_hist[1] " << bs.left_hist[1] << ", bs_left_hist[2] " << bs.left_hist[2]
                        //                << sycl::endl;
                        //            }
                        //#endif
                        //node_ptr[impl_const_t::ind_lch_lrc] = 555;
                        sp_hlp.choose_best_split_for_sbg(item,
                                                         bs.ftr_id,
                                                         bs.ftr_bin,
                                                         bs.left_count,
                                                         bs.imp_dec,
                                                         bs.left_hist,
                                                         node_ptr,
                                                         node_imp_decr_list_ptr,
                                                         left_child_imp_list_ptr,
                                                         node_id,
                                                         index_max,
                                                         update_imp_dec_required);
                    }
                }
            });
        });

        last_event.wait_and_throw();
    }

    return last_event;
}

#define INSTANTIATE(F, B, I, T, M) template class train_best_split_impl<F, B, I, T, M>;

INSTANTIATE(float, std::uint32_t, std::int32_t, task::classification, true);
INSTANTIATE(float, std::uint32_t, std::int32_t, task::classification, false);
INSTANTIATE(float, std::uint32_t, std::int32_t, task::regression, true);
INSTANTIATE(float, std::uint32_t, std::int32_t, task::regression, false);

INSTANTIATE(double, std::uint32_t, std::int32_t, task::classification, true);
INSTANTIATE(double, std::uint32_t, std::int32_t, task::classification, false);
INSTANTIATE(double, std::uint32_t, std::int32_t, task::regression, true);
INSTANTIATE(double, std::uint32_t, std::int32_t, task::regression, false);

} // namespace oneapi::dal::decision_forest::backend

#endif
