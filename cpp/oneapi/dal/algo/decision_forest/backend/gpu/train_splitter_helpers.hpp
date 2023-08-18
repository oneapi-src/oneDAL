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

#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_helpers.hpp"

#ifdef ONEDAL_DATA_PARALLEL

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
        device.get_info<sycl::info::device::local_mem_type>() != sycl::info::local_mem_type::none;
    std::int64_t local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
    return has_local_mem && (local_mem_size / 2 >= required_byte_size);
}

template <typename T, typename Index = std::int64_t>
inline T* fill_zero(T* dst, Index elem_count) {
    for (Index i = 0; i < elem_count; ++i) {
        dst[i] = T(0);
    }
    return dst;
}

template <typename T, typename Index = std::int64_t>
inline T* fill_with_group(sycl::nd_item<2>& item, T* dst, Index elem_count, T val) {
    const Index local_id = item.get_local_id(0);
    const Index local_size = item.get_local_range(0);
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
    Float mean_scl = Float(1) / sum_n1n2;
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
    static constexpr Index cache_buf_int_size = 3; // ftr_id, ftr_bin, left_count
    static constexpr Index cache_buf_float_size = 2; // left_imp, and imp_dec

    static constexpr Index get_cache_without_hist_byte_size() {
        return cache_buf_int_size * sizeof(Index) + cache_buf_float_size * sizeof(Float);
    }
    static constexpr Index get_cache_byte_size(Index hist_prop_count) {
        return get_cache_without_hist_byte_size() + hist_prop_count * sizeof(hist_type_t);
    }

    inline void init(Index in_hist_prop_count) {
        hist_prop_count = in_hist_prop_count;
    }

    inline void init(hist_type_t* in_left_hist, Index in_hist_prop_count) {
        left_hist = in_left_hist;
        hist_prop_count = in_hist_prop_count;
    }

    inline void init_clear(sycl::nd_item<2>& item,
                           hist_type_t* in_left_hist,
                           Index in_hist_prop_count) {
        init(in_left_hist, in_hist_prop_count);
        clear();
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

    inline void clear() {
        clear_scalar();
        clear_hist();
    }

    inline byte_t* store_without_hist(byte_t* buf_ptr, Index idx, Index total_block_count) {
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

    inline byte_t* load_without_hist(byte_t* buf_ptr, Index idx, Index total_block_count) {
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

        return buf_ptr;
    }

    inline void store(byte_t* buf_ptr, Index idx, Index total_block_count) {
        buf_ptr = store_without_hist(buf_ptr, idx, total_block_count);
        hist_type_t* hist_buf_ptr =
            get_buf_ptr<hist_type_t>(&buf_ptr, total_block_count * hist_prop_count);
        hist_type_t* hist_ptr = hist_buf_ptr + idx * hist_prop_count;

        for (Index i = 0; i < hist_prop_count; ++i) {
            hist_ptr[i] = left_hist[i];
        }
    }

    inline void load(byte_t* buf_ptr, Index idx, Index total_block_count) {
        buf_ptr = load_without_hist(buf_ptr, idx, total_block_count);
        hist_type_t* hist_buf_ptr =
            get_buf_ptr<hist_type_t>(&buf_ptr, total_block_count * hist_prop_count);

        left_hist = hist_buf_ptr + idx * hist_prop_count;
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
                                  const split_info_t& ts,
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

        const Float best_imp_dec = sycl::reduce_over_group(sbg, bs.imp_dec, maximum<Float>());

        const Index imp_dec_is_best = float_eq(best_imp_dec, bs.imp_dec);

        const Index best_ftr_id =
            sycl::reduce_over_group(sbg, imp_dec_is_best ? bs.ftr_id : index_max, minimum<Index>());
        const Index best_ftr_val = sycl::reduce_over_group(
            sbg,
            (best_ftr_id == bs.ftr_id && imp_dec_is_best) ? bs.ftr_bin : index_max,
            minimum<Index>());

        const bool none_split_found_by_sbg =
            ((impl_const_t::leaf_mark_ == best_ftr_id) && (0 == sub_group_local_id));
        const bool my_split_is_best = (impl_const_t::leaf_mark_ != best_ftr_id &&
                                       bs.ftr_id == best_ftr_id && bs.ftr_bin == best_ftr_val);
        return (none_split_found_by_sbg || my_split_is_best);
    }
};

} // namespace oneapi::dal::decision_forest::backend

#endif
