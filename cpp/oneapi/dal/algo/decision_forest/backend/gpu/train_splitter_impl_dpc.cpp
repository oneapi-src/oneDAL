/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifdef ONEDAL_DATA_PARALLEL

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_splitter_impl.hpp"

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
using local_accessor_rw_t = sycl::local_accessor<Data, 1>;

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

template <typename T, typename Index = std::size_t>
inline T* fill_zero(T* dst, Index elem_count) {
    for (Index i = 0; i < elem_count; ++i) {
        dst[i] = T(0);
    }
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
struct split_smp {
    using task_t = Task;
    using impl_const_t = impl_const<Index, task_t>;
    using hist_type_t = typename task_types<Float, Index, Task>::hist_type_t;

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
        const Index valNotFound = index_max;

        const Float bestImpDec = sycl::reduce_over_group(sbg, bs_imp_dec_, maximum<Float>());

        const Index impDecIsBest = float_eq(bestImpDec, bs_imp_dec_);

        const Index bestFeatureId =
            sycl::reduce_over_group(sbg, impDecIsBest ? bs_ftr_id_ : valNotFound, minimum<Index>());
        const Index bestFeatureValue = sycl::reduce_over_group(
            sbg,
            (bestFeatureId == bs_ftr_id_ && impDecIsBest) ? bs_ftr_bin_ : valNotFound,
            minimum<Index>());

        const bool noneSplitFoundBySubGroup =
            ((impl_const_t::leaf_mark_ == bestFeatureId) && (0 == sub_group_local_id));
        const bool mySplitIsBest = (impl_const_t::leaf_mark_ != bestFeatureId &&
                                    bs_ftr_id_ == bestFeatureId && bs_ftr_bin_ == bestFeatureValue);
        if (noneSplitFoundBySubGroup || mySplitIsBest) {
            node_ptr[impl_const_t::ind_fid] =
                bs_ftr_id_ == valNotFound ? impl_const_t::leaf_mark_ : bs_ftr_id_;
            node_ptr[impl_const_t::ind_bin] =
                bs_ftr_bin_ == valNotFound ? impl_const_t::leaf_mark_ : bs_ftr_bin_;
            node_ptr[impl_const_t::ind_lch_grc] = bs_left_count_;

            if (update_imp_dec_required) {
                if constexpr (std::is_same_v<task_t, task::classification>) {
                    node_imp_decr_ptr[node_id] = bs_imp_dec_;
                }
                else {
                    node_imp_decr_ptr[node_id] = bs_imp_dec_ / node_ptr[impl_const_t::ind_grc];
                }
            }

            return true;
        }
        return false;
    }

    // classififcation version
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
        auto sbg = item.get_sub_group();
        if (sbg.get_group_id() > 0) {
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
        auto sbg = item.get_sub_group();
        if (sbg.get_group_id() > 0) {
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
            update_left_child_imp(left_imp_list_ptr, bs_left_hist_, node_id);
        }
    }
};

template <typename Float, typename Bin, typename Index, typename Task, bool use_private_mem>
sycl::event
train_splitter_impl<Float, Bin, Index, Task, use_private_mem>::compute_random_split_by_histogram(
    sycl::queue& queue,
    const context_t& ctx,
    const pr::ndarray<hist_type_t, 1>& node_hist_list,
    const pr::ndarray<Index, 1>& selected_ftr_list,
    const pr::ndarray<Float, 1>& random_bins_com,
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
    ONEDAL_PROFILER_TASK(compute_random_split_by_histogram, queue);

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
        ONEDAL_ASSERT(node_imp_dec_list.get_count() >= node_count);
    }

    const hist_type_t* node_hist_list_ptr = node_hist_list.get_data();
    const Index* selected_ftr_list_ptr = selected_ftr_list.get_data();

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

    std::size_t local_hist_buf_size = 2 * hist_prop_count;
    ;

    const sycl::nd_range<1> nd_range = bk::make_multiple_nd_range_1d({ node_count }, { 1 });

    auto ft_rnd_ptr = random_bins_com.get_data();

    sycl::event last_event;

    last_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<hist_type_t> local_hist_buf(local_hist_buf_size, cgh);

        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            const Index node_idx = item.get_global_id()[0];
            const Index node_id = node_indices_ptr[node_ind_ofs + node_idx];
            Index* node_ptr = node_list_ptr + node_id * impl_const_t::node_prop_count_;

            split_smp_t sp_hlp;

            Index bs_ftr_bin = impl_const_t::leaf_mark_;
            Index bs_ftr_id = impl_const_t::leaf_mark_;
            Index bs_left_count = 0;
#if __SYCL_COMPILER_VERSION >= 20230828
            hist_type_t* local_hist_buf_ptr =
                local_hist_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
#else
            hist_type_t* local_hist_buf_ptr = local_hist_buf.get_pointer().get();
#endif
            hist_type_t* bs_left_hist =
                fill_zero(local_hist_buf_ptr + 0 * hist_prop_count, hist_prop_count);

            Float bs_left_imp = Float(0);
            Float bs_imp_dec = min_imp_dec;

            const hist_type_t* node_hist_ptr = node_hist_list_ptr + node_idx * selected_ftr_count *
                                                                        max_bin_count_among_ftrs *
                                                                        hist_prop_count;

            for (Index ftr_idx = 0; ftr_idx < selected_ftr_count; ftr_idx++) {
                const Index ts_ftr_id =
                    selected_ftr_list_ptr[node_id * selected_ftr_count + ftr_idx];

                const hist_type_t* ftr_hist_ptr =
                    node_hist_ptr + ftr_idx * max_bin_count_among_ftrs * hist_prop_count;

                Index ts_left_count = 0;
                Index ts_right_count = 0;

                hist_type_t* ts_left_hist =
                    fill_zero(local_hist_buf_ptr + 1 * hist_prop_count, hist_prop_count);

                Float ts_left_imp = Float(0);
                Float ts_right_imp = Float(0);
                Float ts_imp_dec = Float(0);

                Index min_bin = max_bin_count_among_ftrs;
                Index max_bin = 0;
                for (Index bin_idx = 0; bin_idx < max_bin_count_among_ftrs; bin_idx++) {
                    if constexpr (std::is_same_v<Task, task::classification>) {
                        bool nonzero = false;
                        // Find first non-empty hist for all classes
                        for (Index prop_idx = 0; prop_idx < hist_prop_count; prop_idx++) {
                            if (ftr_hist_ptr[bin_idx * hist_prop_count + prop_idx] > 0) {
                                nonzero = true;
                                break;
                            }
                        }
                        if (nonzero) {
                            min_bin = sycl::min(min_bin, bin_idx);
                            max_bin = sycl::max(max_bin, bin_idx);
                        }
                    }
                    else {
                        // check left count hist[0] > 0
                        if (ftr_hist_ptr[bin_idx * hist_prop_count + 0] > 0) {
                            min_bin = sycl::min(min_bin, bin_idx);
                            max_bin = sycl::max(max_bin, bin_idx);
                        }
                    }
                }

                const Float random_val = ft_rnd_ptr[node_id * selected_ftr_count + ftr_idx];
                const Index random_bin_ofs =
                    static_cast<Index>((max_bin - min_bin + 1) * random_val);
                Index ts_ftr_bin = min_bin + random_bin_ofs;

                if constexpr (std::is_same_v<Task, task::classification>) {
                    for (Index bin_idx = min_bin; bin_idx <= ts_ftr_bin; bin_idx++) {
                        Index ofs = bin_idx * hist_prop_count;
                        sp_hlp.merge_bin_hist(ts_left_count,
                                              ts_left_hist,
                                              ftr_hist_ptr + ofs,
                                              hist_prop_count);
                    }
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
                    for (Index bin_idx = min_bin; bin_idx <= ts_ftr_bin; bin_idx++) {
                        Index ofs = bin_idx * hist_prop_count;
                        sp_hlp.merge_bin_hist(ts_left_hist, ftr_hist_ptr + ofs, hist_prop_count);
                    }

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

            node_ptr[impl_const_t::ind_fid] =
                bs_ftr_id == index_max ? impl_const_t::leaf_mark_ : bs_ftr_id;
            node_ptr[impl_const_t::ind_bin] =
                bs_ftr_bin == index_max ? impl_const_t::leaf_mark_ : bs_ftr_bin;
            node_ptr[impl_const_t::ind_lch_grc] = bs_left_count;

            if (update_imp_dec_required) {
                if constexpr (std::is_same_v<Task, task::classification>) {
                    node_imp_decr_list_ptr[node_id] = bs_imp_dec;
                }
                else {
                    node_imp_decr_list_ptr[node_id] = bs_imp_dec / node_ptr[impl_const_t::ind_grc];
                }
            }

            if constexpr (std::is_same_v<Task, task::classification>) {
                sp_hlp.update_left_child_imp(left_child_imp_list_ptr,
                                             left_child_class_hist_list_ptr,
                                             bs_left_imp,
                                             bs_left_hist,
                                             node_id,
                                             hist_prop_count);
            }
            else {
                sp_hlp.update_left_child_imp(left_child_imp_list_ptr, bs_left_hist, node_id);
            }
        });
    });

    last_event.wait_and_throw();

    return last_event;
}

template <typename Float, typename Bin, typename Index, typename Task, bool use_private_mem>
sycl::event
train_splitter_impl<Float, Bin, Index, Task, use_private_mem>::compute_best_split_by_histogram(
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
        ONEDAL_ASSERT(node_imp_dec_list.get_count() >= node_count);
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

    last_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<hist_type_t> local_hist_buf(local_hist_buf_size, cgh);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
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

#if __SYCL_COMPILER_VERSION >= 20230828
            hist_type_t* local_hist_buf_ptr =
                local_hist_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
#else
            hist_type_t* local_hist_buf_ptr = local_hist_buf.get_pointer().get();
#endif

            hist_type_t* bs_left_hist = nullptr;

            hist_type_t prv_bs_left_hist[buff_size] = { hist_type_t(0) };
            if constexpr (use_private_mem) {
                bs_left_hist = &prv_bs_left_hist[0];
            }
            else {
                bs_left_hist =
                    fill_zero(local_hist_buf_ptr + (sub_group_local_id * 2 + 0) * class_count,
                              class_count);
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
                    ts_left_hist =
                        fill_zero(local_hist_buf_ptr + (sub_group_local_id * 2 + 1) * class_count,
                                  class_count);
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
                        sp_hlp.merge_bin_hist(ts_left_hist,
                                              ftr_hist_ptr + bin_ofs,
                                              hist_prop_count);
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

#define INSTANTIATE(F, B, I, T, M) template class train_splitter_impl<F, B, I, T, M>;

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
