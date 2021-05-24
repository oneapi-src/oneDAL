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

#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include <CL/sycl/ONEAPI/experimental/builtins.hpp>

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_kernel_hist_impl.hpp"

namespace oneapi::dal::decision_forest::backend {

namespace de = dal::detail;
namespace be = dal::backend;
namespace pr = dal::backend::primitives;

using alloc = cl::sycl::usm::alloc;
using address = cl::sycl::access::address_space;

using cl::sycl::ONEAPI::broadcast;
using cl::sycl::ONEAPI::reduce;
using cl::sycl::ONEAPI::plus;
using cl::sycl::ONEAPI::minimum;
using cl::sycl::ONEAPI::maximum;
using cl::sycl::ONEAPI::exclusive_scan;

template <typename T>
using enable_if_float_t = std::enable_if_t<detail::is_valid_float_v<T>>;

template <typename Data>
using local_accessor_rw_t = cl::sycl::
    accessor<Data, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>;

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

template <typename T>
inline T atomic_global_add(T* ptr, T operand) {
    return cl::sycl::atomic_fetch_add<T, address::global_space>(
        { cl::sycl::multi_ptr<T, address::global_space>{ ptr } },
        operand);
}

template <typename Float, typename Bin, typename Index, typename Task>
std::uint64_t train_kernel_hist_impl<Float, Bin, Index, Task>::get_part_hist_required_mem_size(
    Index selected_ftr_count,
    Index max_bin_count_among_ftrs,
    Index class_count) const {
    // mul overflow for nSelectedFeatures * ctx_.max_bin_count_among_ftrs_ and for nHistBins * _nHistProps were checked before kernel call in compute
    const std::uint64_t hist_bin_count = selected_ftr_count * max_bin_count_among_ftrs;
    if constexpr (std::is_same_v<Task, task::classification>) {
        return hist_bin_count * class_count;
    }
    else {
        return hist_bin_count * impl_const_t::hist_prop_count_;
    }
}

template <typename Float, typename Bin, typename Index, typename Task>
void train_kernel_hist_impl<Float, Bin, Index, Task>::validate_input(const descriptor_t& desc,
                                                                     const table& data,
                                                                     const table& labels) const {
    if (data.get_row_count() > de::limits<Index>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_range_of_rows());
    }
    if (data.get_column_count() > de::limits<Index>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_range_of_columns());
    }
    if (desc.get_tree_count() > de::limits<Index>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_number_of_trees());
    }
    if (desc.get_min_observations_in_leaf_node() > de::limits<Index>::max()) {
        throw domain_error(msg::invalid_number_of_min_observations_in_leaf_node());
    }
    if (desc.get_features_per_node() > de::limits<Index>::max()) {
        throw domain_error(msg::invalid_number_of_feature_per_node());
    }
    if (desc.get_max_bins() > de::limits<Index>::max()) {
        throw domain_error(msg::invalid_number_of_max_bins());
    }
    if (desc.get_min_bin_size() > de::limits<Index>::max()) {
        throw domain_error(msg::invalid_value_for_min_bin_size());
    }
    if constexpr (std::is_same_v<Task, task::classification>) {
        if (desc.get_class_count() > de::limits<Index>::max()) {
            throw domain_error(msg::invalid_number_of_classes());
        }
    }
}

template <typename Float, typename Bin, typename Index, typename Task>
void train_kernel_hist_impl<Float, Bin, Index, Task>::init_params(context_t& ctx,
                                                                  const descriptor_t& desc,
                                                                  const table& data,
                                                                  const table& responses) {
    if constexpr (std::is_same_v<Task, task::classification>) {
        ctx.class_count_ = de::integral_cast<Index>(desc.get_class_count());
    }
    ctx.row_count_ = de::integral_cast<Index>(data.get_row_count());
    ctx.column_count_ = de::integral_cast<Index>(data.get_column_count());

    ctx.tree_count_ = de::integral_cast<Index>(desc.get_tree_count());

    ctx.bootstrap_ = desc.get_bootstrap();
    ctx.max_tree_depth_ = desc.get_max_tree_depth();

    if constexpr (std::is_same_v<Task, task::classification>) {
        ctx.selected_ftr_count_ = desc.get_features_per_node() ? desc.get_features_per_node()
                                                               : std::sqrt(ctx.column_count_);
    }
    else {
        ctx.selected_ftr_count_ = desc.get_features_per_node()
                                      ? desc.get_features_per_node()
                                      : ctx.column_count_ / 3 ? ctx.column_count_ / 3 : 1;
    }
    ctx.selected_row_count_ = desc.get_observations_per_tree_fraction() * ctx.row_count_;

    ctx.min_observations_in_leaf_node_ = desc.get_min_observations_in_leaf_node();
    ctx.impurity_threshold_ = desc.get_impurity_threshold();

    ctx.min_observations_in_leaf_node_ = desc.get_min_observations_in_leaf_node();
    ctx.impurity_threshold_ = desc.get_impurity_threshold();

    if (0 >= ctx.selected_row_count_) {
        throw domain_error(msg::invalid_value_for_observations_per_tree_fraction());
    }

    ctx.preferable_local_size_for_part_hist_kernel_ = ctx.preferable_group_size_;
    while (ctx.preferable_local_size_for_part_hist_kernel_ >
           std::max(ctx.selected_ftr_count_, ctx.min_preferable_local_size_for_part_hist_kernel_)) {
        ctx.preferable_local_size_for_part_hist_kernel_ >>= 1;
    }

    auto vimp = desc.get_variable_importance_mode();
    ctx.mda_required_ =
        (variable_importance_mode::mda_raw == vimp || variable_importance_mode::mda_scaled == vimp);
    ctx.mdi_required_ = variable_importance_mode::mdi == vimp;
    ctx.mda_scaled_required_ = (variable_importance_mode::mda_scaled == vimp);

    auto emm = desc.get_error_metric_mode();
    ctx.oob_required_ =
        (check_mask_flag(emm, error_metric_mode::out_of_bag_error) ||
         check_mask_flag(emm, error_metric_mode::out_of_bag_error_per_observation) ||
         ctx.mda_required_);
    ctx.oob_err_required_ = check_mask_flag(emm, error_metric_mode::out_of_bag_error);
    ctx.oob_err_obs_required_ =
        check_mask_flag(emm, error_metric_mode::out_of_bag_error_per_observation);

    // init ftr -> bins map and related params
    indexed_features<Float, Bin, Index> ind_ftrs(queue_,
                                                 desc.get_min_bin_size(),
                                                 desc.get_max_bins());
    ind_ftrs(data).wait_and_throw();

    ctx.total_bin_count_ = ind_ftrs.get_total_bin_count();
    full_data_nd_ = ind_ftrs.get_full_data();
    ftr_bin_offsets_nd_ = ind_ftrs.get_bin_offsets();

    bin_borders_host_.resize(ctx.column_count_);
    for (Index i = 0; i < ctx.column_count_; i++) {
        bin_borders_host_[i] = ind_ftrs.get_bin_borders(i).to_host(queue_);
    }

    data_host_ =
        pr::flatten_table_1d<Float, row_accessor>(queue_, data, alloc::device).to_host(queue_);

    response_nd_ = pr::flatten_table_1d<Float, row_accessor>(queue_, responses, alloc::device);

    response_host_ = response_nd_.to_host(queue_);

    // calculating the maximal number of bins for feature among all features
    ctx.max_bin_count_among_ftrs_ = 0;
    for (Index clmn_idx = 0; clmn_idx < ctx.column_count_; clmn_idx++) {
        auto ftr_bins = ind_ftrs.get_bin_count(clmn_idx);
        ctx.max_bin_count_among_ftrs_ = std::max(ctx.max_bin_count_among_ftrs_, ftr_bins);
    }

    ctx.float_min_ = de::limits<Float>::min();
    ctx.index_max_ = de::limits<Index>::max();

    // define number of trees which can be built in parallel
    const std::uint64_t device_global_mem_size =
        queue_.get_device().get_info<cl::sycl::info::device::global_mem_size>();
    const std::uint64_t device_max_mem_alloc_size =
        queue_.get_device().get_info<cl::sycl::info::device::max_mem_alloc_size>();

    const auto part_hist_size = get_part_hist_required_mem_size(ctx.selected_ftr_count_,
                                                                ctx.max_bin_count_among_ftrs_,
                                                                ctx.class_count_);
    const auto max_mem_alloc_size =
        std::min(device_max_mem_alloc_size, std::uint64_t(ctx.max_mem_alloc_size_for_algo_));

    std::uint64_t used_mem_size =
        sizeof(Float) * ctx.row_count_ * (ctx.column_count_ + 1); // input table size + response
    used_mem_size +=
        ind_ftrs.get_required_mem_size(ctx.row_count_, ctx.column_count_, desc.get_max_bins());
    used_mem_size += ctx.oob_required_ ? sizeof(Float) * ctx.row_count_ * ctx.class_count_ : 0;
    used_mem_size += part_hist_size; // space at least for one part hist

    std::uint64_t available_global_mem_size =
        device_global_mem_size > used_mem_size ? device_global_mem_size - used_mem_size : 0;

    std::uint64_t available_mem_size_for_tree_block =
        std::min(max_mem_alloc_size,
                 static_cast<std::uint64_t>(available_global_mem_size *
                                            ctx.global_mem_fraction_for_tree_block_));

    std::uint64_t required_mem_size_for_one_tree =
        ctx.oob_required_ ? train_service_kernels_.get_oob_rows_required_mem_size(
                                ctx.row_count_,
                                1 /* for 1 tree */,
                                desc.get_observations_per_tree_fraction())
                          : 0;

    required_mem_size_for_one_tree += sizeof(Index) * ctx.selected_row_count_ * 2;

    ctx.tree_in_block_ = de::integral_cast<Index>(available_mem_size_for_tree_block /
                                                  required_mem_size_for_one_tree);

    if (ctx.tree_in_block_ <= 0) {
        // not enough memory even for one tree
        throw domain_error(msg::not_enough_memory_to_build_one_tree());
    }

    ctx.tree_in_block_ = std::min(ctx.tree_count_, ctx.tree_in_block_);

    available_global_mem_size =
        available_global_mem_size > (ctx.tree_in_block_ * required_mem_size_for_one_tree)
            ? available_global_mem_size - (ctx.tree_in_block_ * required_mem_size_for_one_tree)
            : 0;
    // size for one part hist was already reserved, add some more if there is available mem
    ctx.max_part_hist_cumulative_size_ = std::min(
        max_mem_alloc_size,
        static_cast<std::uint64_t>(part_hist_size + available_global_mem_size *
                                                        ctx.global_mem_fraction_for_part_hist_));

    ctx.oob_prop_count_ = impl_const_t::oob_aux_prop_count_;
    if constexpr (std::is_same_v<Task, task::classification>) {
        ctx.oob_prop_count_ = ctx.class_count_;
    }
}

template <typename Float, typename Bin, typename Index, typename Task>
void train_kernel_hist_impl<Float, Bin, Index, Task>::allocate_buffers(const context_t& ctx) {
    de::check_mul_overflow(ctx.selected_row_count_, ctx.tree_in_block_);
    selected_rows_host_ =
        pr::ndarray<Index, 1>::empty({ ctx.selected_row_count_ * ctx.tree_in_block_ });

    // main tree order and auxilliary one are used for partitioning
    tree_order_lev_ = pr::ndarray<Index, 1>::empty(queue_,
                                                   { ctx.selected_row_count_ * ctx.tree_in_block_ },
                                                   alloc::device);
    tree_order_lev_buf_ =
        pr::ndarray<Index, 1>::empty(queue_,
                                     { ctx.selected_row_count_ * ctx.tree_in_block_ },
                                     alloc::device);
    if (ctx.oob_required_) {
        // oob_per_obs_list contains class_count number of counters for all out of bag observations for all trees
        de::check_mul_overflow(ctx.row_count_, ctx.class_count_);
        auto [oob_per_obs_list, event] =
            pr::ndarray<hist_type_t, 1>::zeros(queue_,
                                               { ctx.row_count_ * ctx.oob_prop_count_ },
                                               alloc::device);
        oob_per_obs_list_ = oob_per_obs_list;
        event.wait_and_throw();
    }

    // blocks for MDA scaled error calculation
    if (ctx.mda_scaled_required_) {
        var_imp_variance_host_ = pr::ndarray<Float, 1>::zeros({ ctx.column_count_ });
    }

    if (ctx.mdi_required_ || ctx.mda_required_) {
        res_var_imp_ = pr::ndarray<Float, 1>::empty(queue_, { ctx.column_count_ }, alloc::device);
        res_var_imp_.fill(queue_, 0); // addd deps??
    }
}

template <typename Float, typename Bin, typename Index, typename Task>
std::tuple<pr::ndarray<Index, 1>, cl::sycl::event>
train_kernel_hist_impl<Float, Bin, Index, Task>::gen_features(
    Index node_count,
    const pr::ndarray<Index, 1>& node_vs_tree_map_list,
    dal::array<engine_impl>& engine_arr,
    const context_t& ctx) {
    de::check_mul_overflow((node_count + 1), ctx.selected_ftr_count_);
    // first part is used for features indices, +1 block - part for generator
    auto selected_features_host =
        pr::ndarray<Index, 1>::empty({ (node_count + 1) * ctx.selected_ftr_count_ });
    auto selected_features_com =
        pr::ndarray<Index, 1>::empty(queue_,
                                     { node_count * ctx.selected_ftr_count_ },
                                     alloc::device);

    auto selected_features_host_ptr = selected_features_host.get_mutable_data();
    //std::int32_t* selected_features_host_ptr =
    //    reinterpret_cast<std::int32_t*>(selected_features_host_ptr_orig);

    auto node_vs_tree_map_list_host = node_vs_tree_map_list.to_host(queue_);

    engine_impl* engines = engine_arr.get_mutable_data();

    if (ctx.selected_ftr_count_ != ctx.column_count_) {
        rng<std::int32_t> rn_gen;
        auto treeMap_ptr = node_vs_tree_map_list_host.get_mutable_data();

        for (Index node = 0; node < node_count; node++) {
            rn_gen.uniform_without_replacement(
                ctx.selected_ftr_count_,
                selected_features_host_ptr + node * ctx.selected_ftr_count_,
                selected_features_host_ptr + (node + 1) * ctx.selected_ftr_count_,
                engines[treeMap_ptr[node]].get_state(),
                0,
                ctx.column_count_);
        }
    }
    else {
        for (Index node = 0; node < node_count; node++) {
            for (Index i = 0; i < ctx.selected_ftr_count_; i++) {
                selected_features_host_ptr[node * ctx.selected_ftr_count_ + i] = i;
            }
        }
    }

    auto event = selected_features_com.assign(queue_,
                                              selected_features_host_ptr,
                                              selected_features_com.get_count());

    return std::tuple{ selected_features_com, event };
}

// Kernel context
template <typename Float, typename Index, typename Task>
struct kernel_context {
    kernel_context(const train_context<Float, Index, Task>& ctx)
            : float_min_(ctx.float_min_),
              index_max_(ctx.index_max_),
              min_observations_in_leaf_node_(ctx.min_observations_in_leaf_node_),
              class_count_(ctx.class_count_),
              impurity_threshold_(ctx.impurity_threshold_) {}

    Float float_min_;
    Index index_max_;
    Index min_observations_in_leaf_node_;
    Index class_count_;
    Float impurity_threshold_;
};

template <typename Float, typename Index, typename Task = task::by_default>
struct imp_data_list_ptr;

template <typename Float, typename Index>
struct imp_data_list_ptr<Float, Index, task::classification> {
    imp_data_list_ptr(const impurity_data<Float, Index, task::classification>& imp_data)
            : imp_list_ptr_(imp_data.imp_list_.get_data()),
              class_hist_list_ptr_(imp_data.class_hist_list_.get_data()) {}
    const Float* imp_list_ptr_;
    const Index* class_hist_list_ptr_;
};

template <typename Float, typename Index>
struct imp_data_list_ptr<Float, Index, task::regression> {
    imp_data_list_ptr(const impurity_data<Float, Index, task::regression>& imp_data)
            : imp_list_ptr_(imp_data.imp_list_.get_data()) {}
    const Float* imp_list_ptr_;
};

template <typename Float, typename Index, typename Task = task::by_default>
struct imp_data_list_ptr_mutable;

template <typename Float, typename Index>
struct imp_data_list_ptr_mutable<Float, Index, task::classification> {
    imp_data_list_ptr_mutable(impurity_data<Float, Index, task::classification>& imp_data)
            : imp_list_ptr_(imp_data.imp_list_.get_mutable_data()),
              class_hist_list_ptr_(imp_data.class_hist_list_.get_mutable_data()) {}

    Float* imp_list_ptr_;
    Index* class_hist_list_ptr_;
};

template <typename Float, typename Index>
struct imp_data_list_ptr_mutable<Float, Index, task::regression> {
    imp_data_list_ptr_mutable(impurity_data<Float, Index, task::regression>& imp_data)
            : imp_list_ptr_(imp_data.imp_list_.get_mutable_data()) {}

    Float* imp_list_ptr_;
};

template <typename Float, typename Index>
inline void add_val_to_hist(
    typename task_types<Float, Index, task::classification>::hist_type_t* hist_ptr,
    Float val) {
    Index classId = static_cast<Index>(val);
    hist_ptr[classId] += 1;
}

template <typename Float, typename Index>
inline void add_val_to_hist(
    typename task_types<Float, Index, task::regression>::hist_type_t* hist_ptr,
    Float val) {
    hist_ptr[0] += Float(1);
    Float invN = Float(1) / hist_ptr[0];
    Float delta = val - hist_ptr[1]; // y[i] - mean
    hist_ptr[1] += delta * invN; // updated mean
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
    for (Index i = 0; i < count; i++) {
        dst[i] += src[i];
    }
}

template <typename Index>
inline void merge_stat(Index* dst, Index* accum, const Index* src, Index count) {
    for (Index i = 0; i < count; i++) {
        *accum += src[i];
        dst[i] += src[i];
    }
}

template <typename Float, typename Index, typename T = enable_if_float_t<Float>>
inline void sub_stat(Float* dst, const Float* src, const Float* mrg, Index elem_count) {
    dst[0] = mrg[0] - src[0];

    dst[1] = dst[0] >= Float(1) ? mrg[1] + (src[0] * (mrg[1] - src[1])) / dst[0] : Float(0);

    Float sum_n1n2 = mrg[0];
    Float mul_n1n2 = src[0] * dst[0];
    Float delta_scl = mul_n1n2 / sum_n1n2;
    Float delta = src[1] - dst[1];

    dst[2] = dst[0] >= Float(1) ? (mrg[2] - src[2] - delta * delta * delta_scl) : Float(0);
}

// classification compute_hist_for_node
template <typename Float, typename Index>
inline void compute_hist_for_node(
    cl::sycl::nd_item<2> item,
    Index ind_start,
    Index ind_end,
    const Float* response_ptr,
    Index* node_ptr,
    const Index* node_tree_order_ptr,
    typename task_types<Float, Index, task::classification>::hist_type_t*
        local_buf_ptr, /* for unification */
    const imp_data_list_ptr_mutable<Float, Index, task::classification>& imp_list_ptr,
    const kernel_context<Float, Index, task::classification>& ctx,
    Index node_id) {
    using task_t = task::classification;
    using impl_const_t = impl_const<Index, task_t>;
    using hist_type_t = typename task_types<Float, Index, task_t>::hist_type_t;

    constexpr Index buff_size = impl_const_t::max_private_class_hist_buff_size;
    hist_type_t private_histogram[buff_size] = { 0 };
    Index* node_histogram_ptr = imp_list_ptr.class_hist_list_ptr_ + node_id * ctx.class_count_;
    Float* node_imp_ptr = imp_list_ptr.imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;
    const Index row_count = node_ptr[1];

    for (Index i = ind_start; i < ind_end; i++) {
        Index id = node_tree_order_ptr[i];
        add_val_to_hist<Float, Index>(private_histogram, response_ptr[id]);
    }

    for (Index cls_idx = 0; cls_idx < ctx.class_count_; cls_idx++) {
        atomic_global_add(node_histogram_ptr + cls_idx, private_histogram[cls_idx]);
    }

    item.barrier(cl::sycl::access::fence_space::local_space);

    Float imp = Float(1);
    Float div = Float(1) / (Float(row_count) * row_count);
    Index max_cls_count = 0;
    Index win_cls = 0;
    Index cls_count = 0;

    for (Index cls_idx = 0; cls_idx < ctx.class_count_; cls_idx++) {
        cls_count = node_histogram_ptr[cls_idx];
        imp -= Float(cls_count) * (cls_count)*div;

        if (cls_count > max_cls_count) {
            max_cls_count = cls_count;
            win_cls = cls_idx;
        }
    }

    node_ptr[5] = win_cls;
    node_imp_ptr[0] = cl::sycl::max(imp, Float(0));
}

// regression compute_hist_for_node
template <typename Float, typename Index>
inline void compute_hist_for_node(
    cl::sycl::nd_item<2> item,
    Index ind_start,
    Index ind_end,
    const Float* response_ptr,
    Index* node_ptr,
    const Index* node_tree_order_ptr,
    typename task_types<Float, Index, task::regression>::hist_type_t* local_buf_ptr,
    const imp_data_list_ptr_mutable<Float, Index, task::regression>& imp_list_ptr,
    const kernel_context<Float, Index, task::regression>& ctx,
    Index node_id) {
    using task_t = task::regression;
    using impl_const_t = impl_const<Index, task_t>;
    using hist_type_t = typename task_types<Float, Index, task_t>::hist_type_t;

    const Index local_id = item.get_local_id()[0];
    const Index local_size = item.get_local_range()[0];

    Float* node_imp_ptr = imp_list_ptr.imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;
    constexpr Index hist_prop_count = impl_const_t::hist_prop_count_;
    hist_type_t private_histogram[hist_prop_count] = { 0 };

    for (Index i = ind_start; i < ind_end; i++) {
        Index id = node_tree_order_ptr[i];
        add_val_to_hist<Float, Index>(private_histogram, response_ptr[id]);
    }

    hist_type_t* local_h_ptr = local_buf_ptr + local_id * hist_prop_count;

    local_h_ptr[0] = private_histogram[0];
    local_h_ptr[1] = private_histogram[1];
    local_h_ptr[2] = private_histogram[2];

    for (Index offset = local_size / 2; offset > 0; offset >>= 1) {
        item.barrier(cl::sycl::access::fence_space::local_space);
        if (local_id < offset) {
            hist_type_t* h_ptr = local_buf_ptr + (local_id + offset) * hist_prop_count;
            merge_stat(local_h_ptr, h_ptr, hist_prop_count);
        }
    }

    if (local_id == 0) {
        node_imp_ptr[0] = local_h_ptr[1]; // store mean
        node_imp_ptr[1] = local_h_ptr[2]; // store sum2cent
    }
}

template <typename Float, typename Bin, typename Index, typename Task>
cl::sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_initial_histogram(
    const pr::ndarray<Float, 1>& response,
    const pr::ndarray<Index, 1>& treeOrder,
    const pr::ndarray<Index, 1>& nodeList,
    imp_data_t& imp_data_list,
    Index node_count,
    const context_t& ctx,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(nodeList.get_count() == node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(imp_data_list.imp_list_.get_count() ==
                  node_count * impl_const_t::node_imp_prop_count_);
    if constexpr (std::is_same_v<Task, task::classification>) {
        ONEDAL_ASSERT(imp_data_list.class_hist_list_.get_count() == node_count * ctx.class_count_);
    }

    const Float* response_ptr = response.get_data();
    const Index* tree_order_ptr = treeOrder.get_data();
    Index* node_list_ptr = nodeList.get_mutable_data();

    // num of split attributes for node
    const Index node_prop_count = impl_const_t::node_prop_count_;
    // num of impurity attributes for node
    const Index node_imp_prop_count = impl_const_t::node_imp_prop_count_;

    cl::sycl::event fill_event = {};
    if constexpr (std::is_same_v<Task, task::classification>) {
        fill_event = imp_data_list.class_hist_list_.fill(queue_, 0, deps);
    }

    imp_data_list_ptr_mutable<Float, Index, Task> imp_list_ptr(imp_data_list);
    const kernel_context<Float, Index, Task> krn_ctx(ctx);

    auto local_size = ctx.preferable_group_size_;
    const cl::sycl::nd_range<2> nd_range =
        be::make_multiple_nd_range_2d({ local_size, node_count }, { local_size, 1 });

    auto event = queue_.submit([&](cl::sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.depends_on(fill_event);
        // local_buf is used for regression only, but need to be present for classification also
        local_accessor_rw_t<hist_type_t> local_buf(local_size * (node_imp_prop_count + 1), cgh);
        cgh.parallel_for(nd_range, [=](cl::sycl::nd_item<2> item) {
            const Index node_id = item.get_global_id()[1];
            const Index local_id = item.get_local_id()[0];
            const Index local_size = item.get_local_range()[0];

            Index* node_ptr = node_list_ptr + node_id * node_prop_count;

            const Index rows_offset = node_ptr[0];
            const Index row_count = node_ptr[1];

            const Index elem_count = row_count / local_size + bool(row_count % local_size);

            const Index ind_start = local_id * elem_count;
            const Index ind_end =
                cl::sycl::min(static_cast<Index>((local_id + 1) * elem_count), row_count);

            const Index* node_tree_order_ptr = &tree_order_ptr[rows_offset];

            hist_type_t* local_buf_ptr = nullptr;
            if constexpr (std::is_same_v<Task, task::regression>) {
                local_buf_ptr = local_buf.get_pointer().get();
            }
            compute_hist_for_node<Float, Index>(item,
                                                ind_start,
                                                ind_end,
                                                response_ptr,
                                                node_ptr,
                                                node_tree_order_ptr,
                                                local_buf_ptr,
                                                imp_list_ptr,
                                                krn_ctx,
                                                node_id);
        });
    });

    event.wait_and_throw();
    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
cl::sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_best_split(
    const pr::ndarray<Bin, 2>& data,
    const pr::ndview<Float, 1>& response,
    const pr::ndarray<Index, 1>& treeOrder,
    const pr::ndarray<Index, 1>& selectedFeatures,
    const pr::ndarray<Index, 1>& binOffsets,
    const imp_data_t& imp_data_list,
    pr::ndarray<Index, 1>& nodeList,
    imp_data_t& left_child_imp_data_list,
    pr::ndarray<Float, 1>& nodeImpDecreaseList,
    bool updateImpDecreaseRequired,
    Index node_count,
    const context_t& ctx,
    const be::event_vector& deps) {
    // no overflow check is required because of ctx.node_group_count_ and ctx.node_group_prop_count_ are small constants
    auto nodesGroups =
        pr::ndarray<Index, 1>::empty(queue_,
                                     { ctx.node_group_count_ * ctx.node_group_prop_count_ },
                                     alloc::device);
    auto nodeIndices = pr::ndarray<Index, 1>::empty(queue_, { node_count }, alloc::device);

    cl::sycl::event last_event;
    last_event =
        train_service_kernels_.split_node_list_on_groups_by_size(nodeList,
                                                                 nodesGroups,
                                                                 nodeIndices,
                                                                 node_count,
                                                                 ctx.node_group_count_,
                                                                 ctx.node_group_prop_count_,
                                                                 deps);
    last_event.wait_and_throw();

    auto nodesGroupsHost_nd = nodesGroups.to_host(queue_, { last_event });
    const Index* nodesGroupsHost = nodesGroupsHost_nd.get_data();

    Index nGroupNodes = 0;
    Index processedNodes = 0;

    for (Index i = 0; i < ctx.node_group_count_; i++, processedNodes += nGroupNodes) {
        nGroupNodes = nodesGroupsHost[i * ctx.node_group_prop_count_ + 0];
        if (0 == nGroupNodes)
            continue;

        Index maxGroupBlocksNum = nodesGroupsHost[i * ctx.node_group_prop_count_ + 1];

        Index groupIndicesOffset = processedNodes;

        if (maxGroupBlocksNum > 1) {
            const Index partHistSize =
                get_part_hist_required_mem_size(ctx.selected_ftr_count_,
                                                ctx.max_bin_count_among_ftrs_,
                                                ctx.class_count_);

            Index nPartialHistograms = maxGroupBlocksNum <= ctx.min_row_block_count_for_one_hist_
                                           ? 1
                                           : ctx.max_part_hist_count_;

            if (nPartialHistograms > 1 &&
                maxGroupBlocksNum < ctx.min_row_block_count_to_use_max_part_hist_count_) {
                while (nPartialHistograms > 1 &&
                       (nPartialHistograms * ctx.min_row_block_count_for_one_hist_ >
                            maxGroupBlocksNum ||
                        nPartialHistograms * partHistSize > ctx.max_part_hist_cumulative_size_)) {
                    nPartialHistograms >>= 1;
                }
            }

            de::check_mul_overflow(nGroupNodes, partHistSize);
            de::check_mul_overflow(nGroupNodes * partHistSize, nPartialHistograms);

            // TODO check sizeof(Float) -> sizeof(hist_type_t)
            const Index maxPHBlockElems = ctx.max_part_hist_cumulative_size_ / sizeof(Float);

            const Index nPHBlockElems = nGroupNodes * nPartialHistograms * partHistSize;
            const Index nPHBlocks =
                nPHBlockElems / maxPHBlockElems
                    ? (nPHBlockElems / maxPHBlockElems + bool(nPHBlockElems % maxPHBlockElems))
                    : 1;

            Index nBlockNodes = nGroupNodes / nPHBlocks + bool(nGroupNodes % nPHBlocks);

            for (Index blockIndicesOffset = groupIndicesOffset;
                 blockIndicesOffset < groupIndicesOffset + nGroupNodes;
                 blockIndicesOffset += nBlockNodes) {
                nBlockNodes =
                    std::min(nBlockNodes, groupIndicesOffset + nGroupNodes - blockIndicesOffset);
                if (1 == nPartialHistograms) {
                    auto nodesHistograms =
                        pr::ndarray<hist_type_t, 1>::empty(queue_,
                                                           { nBlockNodes * partHistSize },
                                                           alloc::device);

                    last_event = compute_partial_histograms(data,
                                                            response,
                                                            treeOrder,
                                                            selectedFeatures,
                                                            binOffsets,
                                                            nodeList,
                                                            nodeIndices,
                                                            blockIndicesOffset,
                                                            nodesHistograms,
                                                            nPartialHistograms,
                                                            nBlockNodes,
                                                            ctx,
                                                            { last_event });

                    last_event = compute_best_split_by_histogram(nodesHistograms,
                                                                 selectedFeatures,
                                                                 binOffsets,
                                                                 imp_data_list,
                                                                 nodeIndices,
                                                                 blockIndicesOffset,
                                                                 nodeList,
                                                                 left_child_imp_data_list,
                                                                 nodeImpDecreaseList,
                                                                 updateImpDecreaseRequired,
                                                                 nBlockNodes,
                                                                 ctx,
                                                                 { last_event });

                    last_event.wait_and_throw();
                }
                else {
                    auto partialHistograms = pr::ndarray<hist_type_t, 1>::empty(
                        queue_,
                        { nBlockNodes * nPartialHistograms * partHistSize },
                        alloc::device);
                    auto nodesHistograms =
                        pr::ndarray<hist_type_t, 1>::empty(queue_,
                                                           { nBlockNodes * partHistSize },
                                                           alloc::device);

                    last_event = compute_partial_histograms(data,
                                                            response,
                                                            treeOrder,
                                                            selectedFeatures,
                                                            binOffsets,
                                                            nodeList,
                                                            nodeIndices,
                                                            blockIndicesOffset,
                                                            partialHistograms,
                                                            nPartialHistograms,
                                                            nBlockNodes,
                                                            ctx,
                                                            { last_event });

                    last_event = reduce_partial_histograms(partialHistograms,
                                                           nodesHistograms,
                                                           nPartialHistograms,
                                                           nBlockNodes,
                                                           ctx,
                                                           { last_event });

                    last_event = compute_best_split_by_histogram(nodesHistograms,
                                                                 selectedFeatures,
                                                                 binOffsets,
                                                                 imp_data_list,
                                                                 nodeIndices,
                                                                 blockIndicesOffset,
                                                                 nodeList,
                                                                 left_child_imp_data_list,
                                                                 nodeImpDecreaseList,
                                                                 updateImpDecreaseRequired,
                                                                 nBlockNodes,
                                                                 ctx,
                                                                 { last_event });

                    last_event.wait_and_throw();
                }
            }
        }
        else {
            last_event = compute_best_split_single_pass(data,
                                                        response,
                                                        treeOrder,
                                                        selectedFeatures,
                                                        binOffsets,
                                                        imp_data_list,
                                                        nodeIndices,
                                                        groupIndicesOffset,
                                                        nodeList,
                                                        left_child_imp_data_list,
                                                        nodeImpDecreaseList,
                                                        updateImpDecreaseRequired,
                                                        nGroupNodes,
                                                        ctx,
                                                        { last_event });
            last_event.wait_and_throw();
        }
    }

    return last_event;
}

template <typename Index>
inline void get_block_borders(Index total_elem_count,
                              Index block_count,
                              Index block_id,
                              Index& ind_start,
                              Index& ind_end) {
    const Index elem_count = total_elem_count / block_count + bool(total_elem_count % block_count);

    ind_start = block_id * elem_count;
    ind_end = cl::sycl::min(static_cast<Index>(block_id + 1) * elem_count, total_elem_count);
}

template <typename Float, typename Bin, typename Index, typename Task>
cl::sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_partial_histograms(
    const pr::ndarray<Bin, 2>& data,
    const pr::ndview<Float, 1>& response,
    const pr::ndarray<Index, 1>& treeOrder,
    const pr::ndarray<Index, 1>& selectedFeatures,
    const pr::ndarray<Index, 1>& binOffsets,
    const pr::ndarray<Index, 1>& nodeList,
    const pr::ndarray<Index, 1>& nodeIndices,
    Index nodeIndicesOffset,
    pr::ndarray<hist_type_t, 1>& partialHistograms,
    Index nPartialHistograms,
    Index node_count,
    const context_t& ctx,
    const be::event_vector& deps) {
    auto fill_event = partialHistograms.fill(queue_, 0, deps);
    fill_event.wait_and_throw();

    const Bin* data_ptr = data.get_data();
    const Float* response_ptr = response.get_data();
    const Index* tree_order_ptr = treeOrder.get_data();
    const Index* selected_ftrs_ptr = selectedFeatures.get_data();
    const Index* node_list_ptr = nodeList.get_data();
    const Index* node_indices_ptr = nodeIndices.get_data();

    Index hist_prop_count = 0;
    if constexpr (std::is_same_v<std::decay_t<Task>, task::classification>) {
        hist_prop_count = ctx.class_count_;
    }
    else {
        hist_prop_count = impl_const<Index, task::regression>::hist_prop_count_;
    }
    const Index nNodeProp = impl_const_t::node_prop_count_; // num of split attributes for node
    const Index max_bin_count_among_ftrs = ctx.max_bin_count_among_ftrs_;
    const Index selected_ftr_count = ctx.selected_ftr_count_;
    const Index column_count = ctx.column_count_;

    hist_type_t* partial_histogram_ptr = partialHistograms.get_mutable_data();

    auto local_size = ctx.preferable_local_size_for_part_hist_kernel_;
    const cl::sycl::nd_range<2> nd_range =
        be::make_multiple_nd_range_2d({ nPartialHistograms * local_size, node_count },
                                      { local_size, 1 });

    auto event = queue_.submit([&](cl::sycl::handler& cgh) {
        cgh.depends_on(fill_event);
        cgh.parallel_for(nd_range, [=](cl::sycl::nd_item<2> item) {
            const Index nodeIdx = item.get_global_id()[1];
            const Index nodeId = node_indices_ptr[nodeIndicesOffset + nodeIdx];
            const Index ftrGrpIdx = item.get_local_id()[0];
            const Index ftrGrpSize = item.get_local_range()[0];
            const Index nPartHist = item.get_group_range(0);
            const Index histIdx = item.get_group().get_id(0);

            const Index rowsOffset = node_list_ptr[nodeId * nNodeProp + 0];
            const Index row_count = node_list_ptr[nodeId * nNodeProp + 1];

            Index ind_start;
            Index ind_end;
            get_block_borders(row_count, nPartHist, histIdx, ind_start, ind_end);

            for (Index i = ind_start; i < ind_end; i++) {
                Index id = tree_order_ptr[rowsOffset + i];
                for (Index featIdx = ftrGrpIdx; featIdx < selected_ftr_count;
                     featIdx += ftrGrpSize) {
                    const Index featId = selected_ftrs_ptr[nodeId * selected_ftr_count + featIdx];

                    hist_type_t* histogram_ptr =
                        partial_histogram_ptr +
                        ((nodeIdx * nPartHist + histIdx) * selected_ftr_count + featIdx) *
                            max_bin_count_among_ftrs * hist_prop_count;

                    Index bin = data_ptr[id * column_count + featId];
                    add_val_to_hist<Float, Index>(histogram_ptr + bin * hist_prop_count,
                                                  response_ptr[id]);
                }
            }
        });
    });

    event.wait_and_throw();
    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
cl::sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::reduce_partial_histograms(
    const pr::ndarray<hist_type_t, 1>& partialHistograms,
    pr::ndarray<hist_type_t, 1>& histograms,
    Index nPartialHistograms,
    Index node_count,
    const context_t& ctx,
    const be::event_vector& deps) {
    Index hist_prop_count = 0;
    if constexpr (std::is_same_v<std::decay_t<Task>, task::classification>) {
        hist_prop_count = ctx.class_count_;
    }
    else {
        hist_prop_count = impl_const<Index, task::regression>::hist_prop_count_;
    }
    const Index max_bin_count_among_ftrs = ctx.max_bin_count_among_ftrs_;
    const Index selected_ftr_count = ctx.selected_ftr_count_;

    const hist_type_t* partial_histogram_ptr = partialHistograms.get_data();
    hist_type_t* histogram_ptr = histograms.get_mutable_data();

    // overflow for nMaxBinsAmongFtrs * nSelectedFeatures should be checked in compute
    const cl::sycl::nd_range<3> nd_range =
        be::make_multiple_nd_range_3d({ max_bin_count_among_ftrs * selected_ftr_count,
                                        ctx.reduce_local_size_part_hist_,
                                        node_count },
                                      { 1, ctx.reduce_local_size_part_hist_, 1 });

    auto event = queue_.submit([&](cl::sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<hist_type_t> buf(ctx.reduce_local_size_part_hist_ * hist_prop_count,
                                             cgh);

        cgh.parallel_for(nd_range, [=](cl::sycl::nd_item<3> item) {
            const Index nodeIdx = item.get_global_id()[2];
            const Index binId = item.get_global_id()[0];
            const Index local_id = item.get_local_id()[1];
            const Index local_size = item.get_local_range()[1];

            hist_type_t* buf_ptr = buf.get_pointer().get();

            for (Index prop = 0; prop < hist_prop_count; prop++) {
                buf_ptr[local_id * hist_prop_count + prop] = 0;
            }

            const hist_type_t* nodePartialHistograms =
                partial_histogram_ptr + nodeIdx * nPartialHistograms * selected_ftr_count *
                                            max_bin_count_among_ftrs * hist_prop_count;
            hist_type_t* nodeHistogram = histogram_ptr + nodeIdx * selected_ftr_count *
                                                             max_bin_count_among_ftrs *
                                                             hist_prop_count;

            for (Index i = local_id; i < nPartialHistograms; i += local_size) {
                Index offset = i * selected_ftr_count * max_bin_count_among_ftrs * hist_prop_count +
                               binId * hist_prop_count;
                merge_stat(buf_ptr + local_id * hist_prop_count,
                           nodePartialHistograms + offset,
                           hist_prop_count);
            }

            for (Index offset = local_size / 2; offset > 0; offset >>= 1) {
                item.barrier(cl::sycl::access::fence_space::local_space);
                if (local_id < offset) {
                    merge_stat(buf_ptr + local_id * hist_prop_count,
                               buf_ptr + (local_id + offset) * hist_prop_count,
                               hist_prop_count);
                }
            }

            if (local_id == 0) {
                merge_stat(nodeHistogram + binId * hist_prop_count,
                           buf_ptr + local_id * hist_prop_count,
                           hist_prop_count);
            }
        });
    });
    return event;
}
//////////////////////////////////////////// Best split kernels
template <typename Float>
inline bool float_eq(Float a, Float b) {
    return cl::sycl::fabs(a - b) <= float_accuracy<Float>::val;
}

template <typename Float>
inline bool float_gt(Float a, Float b) {
    return (a - b) > float_accuracy<Float>::val;
}

template <typename Index>
void mark_bin_processed(std::uint64_t* bin_map, Index bin_idx) {
    std::uint64_t mask = 1ul << (bin_idx % 64);
    bin_map[bin_idx / 64] = bin_map[bin_idx / 64] & mask;
}

template <typename Index>
bool is_bin_processed(const std::uint64_t* bin_map, Index bin_idx) {
    std::uint64_t mask = 1ul << (bin_idx % 64);
    return bin_map[bin_idx / 64] & mask;
}

template <typename Float, typename Index, typename Task>
class split;

template <typename Float, typename Index>
class split<Float, Index, task::classification> {
    using task_t = task::classification;
    using impl_const_t = impl_const<Index, task_t>;
    using kernel_context_t = kernel_context<Float, Index, task_t>;

public:
    split(const kernel_context_t& ctx) : ctx_(ctx), imp_dec_(ctx.float_min_) {}
    split(Index ftr_id, const kernel_context_t& ctx)
            : ctx_(ctx),
              ftr_id_(ftr_id),
              imp_dec_(ctx.float_min_) {}
    split(Index ftr_id, Index ftr_bin, const kernel_context_t& ctx)
            : ctx_(ctx),
              ftr_id_(ftr_id),
              ftr_bin_(ftr_bin),
              imp_dec_(ctx.float_min_) {}

    void add_val(Index obs_bin, Float obs_response) {
        Index class_id = static_cast<Index>(obs_response);

        left_count_ += Index(obs_bin <= ftr_bin_);
        left_class_hist_[class_id] += Index(obs_bin <= ftr_bin_);
    }

    void merge_bin_hist(Index bin, const Index* bin_hist_ptr) {
        ftr_bin_ = bin;
        merge_stat(&left_class_hist_[0], &left_count_, bin_hist_ptr, ctx_.class_count_);
    }

    void calc_imp_dec(Index* node_ptr,
                      const imp_data_list_ptr<Float, Index, task_t>& imp_list_ptr,
                      const kernel_context_t& ctx,
                      Index node_id) {
        Index node_row_count = node_ptr[1];
        const Float* node_imp_ptr =
            imp_list_ptr.imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;
        Float node_imp = node_imp_ptr[0];
        const Index* node_class_hist_ptr =
            imp_list_ptr.class_hist_list_ptr_ + node_id * ctx.class_count_;

        right_count_ = node_row_count - left_count_;

        const Float divL =
            (0 < left_count_) ? Float(1) / (Float(left_count_) * Float(left_count_)) : Float(0);
        const Float divR =
            (0 < right_count_) ? Float(1) / (Float(right_count_) * Float(right_count_)) : Float(0);

        left_imp_ = Float(1);
        right_imp_ = Float(1);

        for (Index class_id = 0; class_id < ctx.class_count_; class_id++) {
            left_imp_ -=
                Float(left_class_hist_[class_id]) * Float(left_class_hist_[class_id]) * divL;
            right_imp_ -= Float(node_class_hist_ptr[class_id] - left_class_hist_[class_id]) *
                          Float(node_class_hist_ptr[class_id] - left_class_hist_[class_id]) * divR;
        }

        left_imp_ = cl::sycl::max(left_imp_, Float(0));
        right_imp_ = cl::sycl::max(right_imp_, Float(0));

        imp_dec_ = node_imp - (Float(left_count_) * left_imp_ + Float(right_count_) * right_imp_) /
                                  Float(node_row_count);
    }

    void choose_best_split(const split& test_split,
                           Index* node_ptr,
                           const imp_data_list_ptr<Float, Index, task_t>& imp_list_ptr,
                           const kernel_context_t& ctx,
                           Index node_id) {
        // TODO move check for imp 0 to node spliti func
        const Float* node_imp_ptr =
            imp_list_ptr.imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;
        Float imp = node_imp_ptr[0];

        if ((Float)0 < test_split.imp_dec_ && !float_eq(imp, (Float)0) &&
            imp >= ctx.impurity_threshold_ &&
            (ftr_bin_ == impl_const_t::leaf_mark_ || float_gt(test_split.imp_dec_, imp_dec_) ||
             (float_eq(test_split.imp_dec_, imp_dec_) &&
              (test_split.ftr_id_ < ftr_id_ ||
               (ftr_id_ == test_split.ftr_id_ && test_split.ftr_bin_ < ftr_bin_)))) &&
            test_split.left_count_ >= ctx.min_observations_in_leaf_node_ &&
            test_split.right_count_ >= ctx.min_observations_in_leaf_node_) {
            ftr_id_ = test_split.ftr_id_;
            ftr_bin_ = test_split.ftr_bin_;
            imp_dec_ = test_split.imp_dec_;

            left_count_ = test_split.left_count_;
            left_imp_ = test_split.left_imp_;
            for (Index class_id = 0; class_id < ctx.class_count_; class_id++) {
                left_class_hist_[class_id] = test_split.left_class_hist_[class_id];
            }
        }
    }

public:
    static inline constexpr Index buff_size = impl_const_t::max_private_class_hist_buff_size;

    const kernel_context_t& ctx_;
    Index ftr_id_ = impl_const_t::leaf_mark_;
    Index ftr_bin_ = impl_const_t::leaf_mark_;
    Float imp_dec_ = 0;
    Index left_count_ = 0;
    Index right_count_ = 0;
    Index left_class_hist_[buff_size] = { 0 };

    Float left_imp_ = Float(0);
    Float right_imp_ = Float(0);
};

template <typename Float, typename Index>
class split<Float, Index, task::regression> {
    using task_t = task::regression;
    using impl_const_t = impl_const<Index, task_t>;
    using kernel_context_t = kernel_context<Float, Index, task_t>;

public:
    split(const kernel_context_t& ctx) : imp_dec_(ctx.float_min_) {}
    split(Index ftr_id, const kernel_context_t& ctx) : ftr_id_(ftr_id), imp_dec_(ctx.float_min_) {}
    split(Index ftr_id, Index ftr_bin, const kernel_context_t& ctx)
            : ftr_id_(ftr_id),
              ftr_bin_(ftr_bin),
              imp_dec_(ctx.float_min_) {}

    void add_val(Index obs_bin, Float obs_response) {
        if (obs_bin <= ftr_bin_) {
            add_val_to_hist<Float, Index>(&left_hist_[0], obs_response);
        }
    }

    void merge_bin_hist(Index bin, const Float* bin_hist_ptr) {
        ftr_bin_ = bin;
        merge_stat(&left_hist_[0], bin_hist_ptr, buff_size);
    }

    void calc_imp_dec(Index* node_ptr,
                      const imp_data_list_ptr<Float, Index, task_t>& imp_list_ptr,
                      const kernel_context_t& ctx,
                      Index node_id) {
        Index node_row_count = node_ptr[1];
        left_count_ = static_cast<Index>(left_hist_[0]);

        const Float* node_imp_ptr =
            imp_list_ptr.imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;

        Float node_hist[buff_size] = { static_cast<Float>(node_row_count),
                                       node_imp_ptr[0],
                                       node_imp_ptr[1] };
        Float right_hist[buff_size] = { 0 };

        // getting hist for right part
        sub_stat<Float, Index, task_t>(&right_hist[0], &left_hist_[0], &node_hist[0], buff_size);

        right_count_ = node_row_count - left_count_;
        imp_dec_ = node_imp_ptr[1] - (left_hist_[2] + right_hist[2]);
    }

    void choose_best_split(const split& test_split,
                           Index* node_ptr,
                           const imp_data_list_ptr<Float, Index, task_t>& imp_list_ptr,
                           const kernel_context_t& ctx,
                           Index node_id) {
        // TODO move check for imp 0 to node split func
        const Float* node_imp_ptr =
            imp_list_ptr.imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;
        Float imp = node_imp_ptr[1];
        imp = imp / Float(test_split.left_count_ + test_split.right_count_);

        if ((Float)0 < test_split.imp_dec_ && !float_eq(imp, (Float)0) &&
            imp >= ctx.impurity_threshold_ &&
            (ftr_bin_ == impl_const_t::leaf_mark_ || float_gt(test_split.imp_dec_, imp_dec_) ||
             (float_eq(test_split.imp_dec_, imp_dec_) &&
              (test_split.ftr_id_ < ftr_id_ ||
               (ftr_id_ == test_split.ftr_id_ && test_split.ftr_bin_ < ftr_bin_)))) &&
            test_split.left_count_ >= ctx.min_observations_in_leaf_node_ &&
            test_split.right_count_ >= ctx.min_observations_in_leaf_node_) {
            ftr_id_ = test_split.ftr_id_;
            ftr_bin_ = test_split.ftr_bin_;
            imp_dec_ = test_split.imp_dec_;

            left_count_ = test_split.left_count_;

            for (Index i = 0; i < buff_size; i++) {
                left_hist_[i] = test_split.left_hist_[i];
            }
        }
    }

public:
    static constexpr Index buff_size = impl_const_t::node_imp_prop_count_ + 1;
    Index ftr_id_ = impl_const_t::leaf_mark_;
    Index ftr_bin_ = impl_const_t::leaf_mark_;
    Float imp_dec_ = 0;
    Index left_count_ = 0;
    Index right_count_ = 0; // need to exclude
    Float left_hist_[buff_size] = { 0 };
};

template <typename Float, typename Index, typename Task>
void update_left_child_imp(const imp_data_list_ptr_mutable<Float, Index, Task>& left_imp_list_ptr,
                           const split<Float, Index, Task>& bs,
                           const kernel_context<Float, Index, Task>& ctx,
                           Index node_id) {
    using impl_const_t = impl_const<Index, Task>;

    Float* left_node_imp_ptr =
        left_imp_list_ptr.imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;

    if constexpr (std::is_same_v<Task, task::classification>) {
        left_node_imp_ptr[0] = bs.left_imp_;
        Index* left_node_class_hist_ptr =
            left_imp_list_ptr.class_hist_list_ptr_ + node_id * ctx.class_count_;

        for (Index class_id = 0; class_id < ctx.class_count_; class_id++) {
            left_node_class_hist_ptr[class_id] = bs.left_class_hist_[class_id];
        }
    }
    else {
        left_node_imp_ptr[0] = bs.left_hist_[1];
        left_node_imp_ptr[1] = bs.left_hist_[2];
    }
}

template <typename Float, typename Index, typename Task>
void choose_best_split_for_sbg(
    cl::sycl::nd_item<2>& item,
    const split<Float, Index, Task>& bs,
    Index* node_ptr,
    Float* node_imp_decr_ptr,
    const imp_data_list_ptr_mutable<Float, Index, Task>& left_imp_list_ptr,
    const kernel_context<Float, Index, Task>& ctx,
    Index node_id,
    bool updateImpDecreaseRequired) {
    using impl_const_t = impl_const<Index, Task>;

    auto sbg = item.get_sub_group();
    if (sbg.get_group_id() > 0) {
        return;
    }

    const Index sub_group_local_id = sbg.get_local_id();
    const Index valNotFound = ctx.index_max_;

    const Float bestImpDec = reduce(sbg, bs.imp_dec_, maximum<Float>());

    const Index impDecIsBest = float_eq(bestImpDec, bs.imp_dec_);

    const Index bestFeatureId =
        reduce(sbg, impDecIsBest ? bs.ftr_id_ : valNotFound, minimum<Index>());
    const Index bestFeatureValue =
        reduce(sbg,
               (bestFeatureId == bs.ftr_id_ && impDecIsBest) ? bs.ftr_bin_ : valNotFound,
               minimum<Index>());

    const bool noneSplitFoundBySubGroup =
        ((impl_const_t::leaf_mark_ == bestFeatureId) && (0 == sub_group_local_id));
    const bool mySplitIsBest = (impl_const_t::leaf_mark_ != bestFeatureId &&
                                bs.ftr_id_ == bestFeatureId && bs.ftr_bin_ == bestFeatureValue);
    if (noneSplitFoundBySubGroup || mySplitIsBest) {
        node_ptr[2] = bs.ftr_id_ == valNotFound ? impl_const_t::leaf_mark_ : bs.ftr_id_;
        node_ptr[3] = bs.ftr_bin_ == valNotFound ? impl_const_t::leaf_mark_ : bs.ftr_bin_;
        node_ptr[4] = bs.left_count_;

        update_left_child_imp<Float, Index, Task>(left_imp_list_ptr, bs, ctx, node_id);

        if (updateImpDecreaseRequired) {
            if constexpr (std::is_same_v<Task, task::classification>) {
                node_imp_decr_ptr[0] = bs.imp_dec_;
            }
            else {
                node_imp_decr_ptr[0] = bs.imp_dec_ / node_ptr[1];
            }
        }
    }
}

template <typename Float, typename Bin, typename Index, typename Task>
cl::sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_best_split_by_histogram(
    const pr::ndarray<hist_type_t, 1>& nodesHistograms,
    const pr::ndarray<Index, 1>& selectedFeatures,
    const pr::ndarray<Index, 1>& binOffsets,
    const imp_data_t& imp_data_list,
    const pr::ndarray<Index, 1>& nodeIndices,
    Index nodeIndicesOffset,
    pr::ndarray<Index, 1>& nodeList,
    imp_data_t& left_child_imp_data_list,
    pr::ndarray<Float, 1>& nodeImpDecreaseList,
    bool updateImpDecreaseRequired,
    Index node_count,
    const context_t& ctx,
    const be::event_vector& deps) {
    using split_t = split<Float, Index, Task>;
    using hist_type_t = typename task_types<Float, Index, Task>::hist_type_t;
    // input asserts is going to be added
    const hist_type_t* node_histogram_ptr = nodesHistograms.get_data();
    const Index* selected_ftrs_ptr = selectedFeatures.get_data();

    const Index* bin_offsets_ptr = binOffsets.get_data();
    imp_data_list_ptr<Float, Index, Task> imp_list_ptr(imp_data_list);

    const Index* node_indices_ptr = nodeIndices.get_data();
    Index* node_list_ptr = nodeList.get_mutable_data();
    Float* node_imp_decr_list_ptr = nodeImpDecreaseList.get_mutable_data();

    imp_data_list_ptr_mutable<Float, Index, Task> left_imp_list_ptr(left_child_imp_data_list);

    const Index nNodeProp = impl_const_t::node_prop_count_;
    const Index max_bin_count_among_ftrs = ctx.max_bin_count_among_ftrs_;
    Index hist_prop_count = 0;
    if constexpr (std::is_same_v<std::decay_t<Task>, task::classification>) {
        hist_prop_count = ctx.class_count_;
    }
    else {
        hist_prop_count = impl_const_t::hist_prop_count_;
    }

    const Index selected_ftr_count = ctx.selected_ftr_count_;

    const kernel_context<Float, Index, Task> krn_ctx(ctx);

    auto local_size = be::device_max_sg_size(queue_);

    const cl::sycl::nd_range<2> nd_range =
        be::make_multiple_nd_range_2d({ local_size, node_count }, { local_size, 1 });

    auto event = queue_.submit([&](cl::sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](cl::sycl::nd_item<2> item) {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id() > 0) {
                return;
            }
            const Index node_idx = item.get_global_id()[1];
            const Index node_id = node_indices_ptr[nodeIndicesOffset + node_idx];
            Index* node_ptr = node_list_ptr + node_id * nNodeProp;

            const Index sub_group_local_id = sbg.get_local_id();
            const Index sub_group_size = sbg.get_local_range()[0];

            split_t bs(krn_ctx);

            for (Index currFtrIdx = sub_group_local_id; currFtrIdx < selected_ftr_count;
                 currFtrIdx += sub_group_size) {
                const hist_type_t* nodeHistogram =
                    node_histogram_ptr +
                    node_idx * selected_ftr_count * max_bin_count_among_ftrs * hist_prop_count;
                const hist_type_t* histogramForFeature =
                    nodeHistogram + currFtrIdx * max_bin_count_among_ftrs * hist_prop_count;

                const Index featId = selected_ftrs_ptr[node_id * selected_ftr_count + currFtrIdx];
                const Index currFtrBins = bin_offsets_ptr[featId + 1] - bin_offsets_ptr[featId];

                split_t test_split(featId, krn_ctx);
                for (Index tbin = 0; tbin < currFtrBins; tbin++) {
                    Index binOffset = tbin * hist_prop_count;
                    test_split.merge_bin_hist(tbin, histogramForFeature + binOffset);

                    test_split.calc_imp_dec(node_ptr, imp_list_ptr, ctx, node_id);

                    bs.choose_best_split(test_split, node_ptr, imp_list_ptr, ctx, node_id);
                } // for tbin
            } // for ftr

            choose_best_split_for_sbg<Float, Index, Task>(item,
                                                          bs,
                                                          node_ptr,
                                                          &node_imp_decr_list_ptr[node_id],
                                                          left_imp_list_ptr,
                                                          ctx,
                                                          node_id,
                                                          updateImpDecreaseRequired);
        });
    });

    event.wait_and_throw();

    return event;
}
/// kernel
template <typename Float, typename Bin, typename Index, typename Task>
cl::sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_best_split_single_pass(
    const pr::ndarray<Bin, 2>& data,
    const pr::ndview<Float, 1>& response,
    const pr::ndarray<Index, 1>& treeOrder,
    const pr::ndarray<Index, 1>& selectedFeatures,
    const pr::ndarray<Index, 1>& binOffsets,
    const imp_data_t& imp_data_list,
    const pr::ndarray<Index, 1>& nodeIndices,
    Index nodeIndicesOffset,
    pr::ndarray<Index, 1>& nodeList,
    imp_data_t& left_child_imp_data_list,
    pr::ndarray<Float, 1>& nodeImpDecreaseList,
    bool updateImpDecreaseRequired,
    Index node_count,
    const context_t& ctx,
    const be::event_vector& deps) {
    using split_t = split<Float, Index, Task>;
    // input asserts is going to be added
    const Bin* data_ptr = data.get_data();
    const Float* response_ptr = response.get_data();
    const Index* tree_order_ptr = treeOrder.get_data();

    const Index* selected_ftrs_ptr = selectedFeatures.get_data();

    imp_data_list_ptr<Float, Index, Task> imp_list_ptr(imp_data_list);

    const Index* node_indices_ptr = nodeIndices.get_data();
    Index* node_list_ptr = nodeList.get_mutable_data();
    Float* node_imp_decr_list_ptr = nodeImpDecreaseList.get_mutable_data();

    imp_data_list_ptr_mutable<Float, Index, Task> left_imp_list_ptr(left_child_imp_data_list);

    const Index nNodeProp = impl_const_t::node_prop_count_;
    const Index column_count = ctx.column_count_;

    const Index selected_ftr_count = ctx.selected_ftr_count_;

    const kernel_context<Float, Index, Task> krn_ctx(ctx);

    auto local_size = be::device_max_sg_size(queue_);

    const cl::sycl::nd_range<2> nd_range =
        be::make_multiple_nd_range_2d({ local_size, node_count }, { local_size, 1 });

    auto event = queue_.submit([&](cl::sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](cl::sycl::nd_item<2> item) {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id() > 0) {
                return;
            }
            const Index node_idx = item.get_global_id()[1];
            const Index node_id = node_indices_ptr[nodeIndicesOffset + node_idx];
            Index* node_ptr = node_list_ptr + node_id * nNodeProp;

            const Index sub_group_local_id = sbg.get_local_id();
            const Index sub_group_size = sbg.get_local_range()[0];

            const Index rowsOffset = node_ptr[0];
            const Index nRows = node_ptr[1];

            split_t bs(krn_ctx);

            for (Index currFtrIdx = sub_group_local_id; currFtrIdx < selected_ftr_count;
                 currFtrIdx += sub_group_size) {
                Index featId = selected_ftrs_ptr[node_id * selected_ftr_count + currFtrIdx];

                std::uint64_t bin_map[4] = { 0 };

                // calculating classes histogram rows count <= bins num
                // add logic for choosing min from rows and bins
                for (Index i = 0; i < nRows; i++) {
                    Index curr_row_id = tree_order_ptr[rowsOffset + i];
                    Index tbin = data_ptr[curr_row_id * column_count + featId];

                    bool bin_not_processed = !is_bin_processed(bin_map, tbin);
                    //bool bin_not_processed = 1;
                    if (bin_not_processed) {
                        split_t test_split(featId, tbin, krn_ctx);

                        for (int row_idx = 0; row_idx < nRows; row_idx++) {
                            Index id = tree_order_ptr[rowsOffset + row_idx];
                            Index bin = data_ptr[id * column_count + featId];
                            test_split.add_val(bin, response_ptr[id]);
                        }

                        test_split.calc_imp_dec(node_ptr, imp_list_ptr, ctx, node_id);

                        bs.choose_best_split(test_split, node_ptr, imp_list_ptr, ctx, node_id);

                        mark_bin_processed(bin_map, tbin);
                    }
                } // for tbin
            } // for ftr

            choose_best_split_for_sbg<Float, Index, Task>(item,
                                                          bs,
                                                          node_ptr,
                                                          &node_imp_decr_list_ptr[node_id],
                                                          left_imp_list_ptr,
                                                          ctx,
                                                          node_id,
                                                          updateImpDecreaseRequired);
        });
    });

    event.wait_and_throw();

    return event;
}

template <typename Float, typename Index, typename Task>
static void do_node_imp_split(const imp_data_list_ptr<Float, Index, Task>& imp_list_ptr,
                              const imp_data_list_ptr<Float, Index, Task>& left_imp_list_ptr,
                              const imp_data_list_ptr_mutable<Float, Index, Task>& imp_list_ptr_new,
                              const Index* nodeP,
                              Index* nodeL,
                              Index* nodeR,
                              const kernel_context<Float, Index, Task>& ctx,
                              Index node_id,
                              Index new_left_node_pos) {
    using impl_const_t = impl_const<Index, Task>;

    if constexpr (std::is_same_v<Task, task::classification>) {
        // assign class hist and compute winner for new nodes
        const Index* class_hist_p = imp_list_ptr.class_hist_list_ptr_ + node_id * ctx.class_count_;
        const Index* left_child_class_hist =
            left_imp_list_ptr.class_hist_list_ptr_ + node_id * ctx.class_count_;
        Index* class_hist_left =
            imp_list_ptr_new.class_hist_list_ptr_ + new_left_node_pos * ctx.class_count_;
        Index* class_hist_right =
            imp_list_ptr_new.class_hist_list_ptr_ + (new_left_node_pos + 1) * ctx.class_count_;

        Index max_cls_count_left = 0;
        Index max_cls_count_right = 0;
        Index win_cls_left = 0;
        Index win_cls_right = 0;

        Index rows_right = nodeR[1];
        Float imp_right = Float(1);
        Float div_right = (0 < rows_right) ? Float(1) / (rows_right * rows_right) : Float(0);

        for (Index class_id = 0; class_id < ctx.class_count_; class_id++) {
            class_hist_left[class_id] = left_child_class_hist[class_id];
            Index class_count_right = class_hist_p[class_id] - left_child_class_hist[class_id];
            class_hist_right[class_id] = class_count_right;

            imp_right -= Float(class_count_right) * (class_count_right)*div_right;

            if (class_hist_left[class_id] > max_cls_count_left) {
                max_cls_count_left = class_hist_left[class_id];
                win_cls_left = class_id;
            }

            if (class_hist_right[class_id] > max_cls_count_right) {
                max_cls_count_right = class_hist_right[class_id];
                win_cls_right = class_id;
            }
        }

        nodeL[5] = win_cls_left;
        nodeR[5] = win_cls_right;

        // assign impurity for new nodes
        const Float* left_child_imp =
            left_imp_list_ptr.imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;
        Float* impL =
            imp_list_ptr_new.imp_list_ptr_ + new_left_node_pos * impl_const_t::node_imp_prop_count_;
        Float* impR = imp_list_ptr_new.imp_list_ptr_ +
                      (new_left_node_pos + 1) * impl_const_t::node_imp_prop_count_;
        impL[0] = left_child_imp[0];
        impR[0] = cl::sycl::max(imp_right, Float(0));
    }
    else {
        constexpr Index buff_size = impl_const_t::node_imp_prop_count_ + 1;
        const Float* left_child_imp =
            left_imp_list_ptr.imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;
        const Float* impP =
            imp_list_ptr.imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;
        Float* impL =
            imp_list_ptr_new.imp_list_ptr_ + new_left_node_pos * impl_const_t::node_imp_prop_count_;
        Float* impR = imp_list_ptr_new.imp_list_ptr_ +
                      (new_left_node_pos + 1) * impl_const_t::node_imp_prop_count_;

        Float node_hist[buff_size] = { static_cast<Float>(nodeP[1]), impP[0], impP[1] };
        Float left_hist[buff_size] = { static_cast<Float>(nodeL[1]),
                                       left_child_imp[0],
                                       left_child_imp[1] };
        Float right_hist[buff_size] = { 0 };

        sub_stat(&right_hist[0], &left_hist[0], &node_hist[0], buff_size);

        impL[0] = left_child_imp[0];
        impL[1] = left_child_imp[1];

        impR[0] = right_hist[1];
        impR[1] = right_hist[2];
    }
}
//////////////////////////// DO NODE SPLIT
template <typename Float, typename Bin, typename Index, typename Task>
cl::sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::do_node_split(
    const pr::ndarray<Index, 1>& node_list,
    const pr::ndarray<Index, 1>& node_vs_tree_map_list,
    const imp_data_t& imp_data_list,
    const imp_data_t& left_child_imp_data_list,
    pr::ndarray<Index, 1>& node_list_new,
    pr::ndarray<Index, 1>& node_vs_tree_map_list_new,
    imp_data_t& imp_data_list_new,
    Index node_count,
    Index node_count_new,
    const context_t& ctx,
    const be::event_vector& deps) {
    // input asserts is going to be added

    const Index nNodeProp = impl_const_t::node_prop_count_; // num of split attributes for node
    const Index bad_val = impl_const_t::bad_val_;

    const Index* node_list_ptr = node_list.get_data();
    const Index* node_vs_tree_map_list_ptr = node_vs_tree_map_list.get_data();

    Index* node_list_new_ptr = node_list_new.get_mutable_data();
    Index* node_vs_tree_map_list_new_ptr = node_vs_tree_map_list_new.get_mutable_data();

    imp_data_list_ptr<Float, Index, Task> imp_list_ptr(imp_data_list);
    imp_data_list_ptr<Float, Index, Task> left_imp_list_ptr(left_child_imp_data_list);

    imp_data_list_ptr_mutable<Float, Index, Task> imp_list_ptr_new(imp_data_list_new);

    const kernel_context<Float, Index, Task> krn_ctx(ctx);

    auto local_size = be::device_max_sg_size(queue_);
    const cl::sycl::nd_range<1> nd_range = be::make_multiple_nd_range_1d(local_size, local_size);

    auto event = queue_.submit([&](cl::sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](cl::sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id() > 0) {
                return;
            }

            const Index local_id = sbg.get_local_id();
            const Index local_size = sbg.get_local_range()[0];

            Index created_node_count = 0;
            for (Index node_id = local_id; node_id < node_count; node_id += local_size) {
                Index splitNode =
                    Index(node_list_ptr[node_id * nNodeProp + 2] != bad_val); // featId != -1
                Index new_left_node_pos =
                    created_node_count + exclusive_scan(sbg, splitNode, plus<Index>()) * 2;
                if (splitNode) {
                    // split parent node on left and right nodes
                    const Index* nodeP = node_list_ptr + node_id * nNodeProp;
                    Index* nodeL = node_list_new_ptr + new_left_node_pos * nNodeProp;
                    Index* nodeR = node_list_new_ptr + (new_left_node_pos + 1) * nNodeProp;

                    nodeL[0] = nodeP[0]; // rows offset
                    nodeL[1] = nodeP[4]; // nRows
                    nodeL[2] = bad_val; // featureId
                    nodeL[3] = bad_val; // featureVal
                    nodeL[4] = nodeP[4]; // num of items in Left part = nRows in new node

                    nodeR[0] = nodeL[0] + nodeL[1];
                    nodeR[1] = nodeP[1] - nodeL[1];
                    nodeR[2] = bad_val;
                    nodeR[3] = bad_val;
                    nodeR[4] = nodeR[1]; // num of items in Left part = nRows in new node

                    node_vs_tree_map_list_new_ptr[new_left_node_pos] =
                        node_vs_tree_map_list_ptr[node_id];
                    node_vs_tree_map_list_new_ptr[new_left_node_pos + 1] =
                        node_vs_tree_map_list_ptr[node_id];

                    do_node_imp_split<Float, Index, Task>(imp_list_ptr,
                                                          left_imp_list_ptr,
                                                          imp_list_ptr_new,
                                                          nodeP,
                                                          nodeL,
                                                          nodeR,
                                                          krn_ctx,
                                                          node_id,
                                                          new_left_node_pos);
                }
                created_node_count += reduce(sbg, splitNode, plus<Index>()) * 2;
            }
        });
    });
    event.wait_and_throw();

    return event;
}

/////////////////////////////////////////////////////////
/// compute results
template <typename Float, typename Bin, typename Index, typename Task>
Float train_kernel_hist_impl<Float, Bin, Index, Task>::compute_oob_error(
    const model_manager_t& model_manager,
    const pr::ndarray<Float, 1>& data_host,
    const pr::ndarray<Float, 1>& response_host,
    const pr::ndarray<Index, 1>& oob_row_list,
    pr::ndarray<hist_type_t, 1>& oob_per_obs_list,
    Index tree_idx,
    Index indicesOffset,
    Index n,
    const context_t& ctx,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(data_host.get_count() == ctx.row_count_ * ctx.column_count_);
    ONEDAL_ASSERT(response_host.get_count() == ctx.column_count_);
    ONEDAL_ASSERT(oob_per_obs_list.get_count() == ctx.row_count_ * ctx.oob_prop_count_);
    // input asserts is going to be added

    auto oob_row_list_host = oob_row_list.to_host(queue_, deps);
    auto oob_per_obs_list_host = oob_per_obs_list.to_host(queue_, deps);

    const Float* data_host_ptr = data_host.get_data();
    const Float* response_host_ptr = response_host.get_data();
    const Index* oob_row_list_host_ptr = oob_row_list_host.get_data();
    hist_type_t* oob_per_obs_list_host_ptr = oob_per_obs_list_host.get_mutable_data();

    //compute prediction error on each OOB row and get its mean online formulae (Welford)

    Float mean = 0;
    for (Index i = 0; i < n; i++) {
        Index row_ind = oob_row_list_host_ptr[indicesOffset + i];
        ONEDAL_ASSERT(row_ind < ctx.row_count_);

        auto prediction =
            model_manager.get_tree_response(tree_idx, &data_host_ptr[row_ind * ctx.column_count_]);

        if constexpr (std::is_same_v<Task, task::classification>) {
            ONEDAL_ASSERT(ctx.class_count_ == ctx.oob_prop_count_);
            Index class_idx = static_cast<Index>(prediction);
            oob_per_obs_list_host_ptr[row_ind * ctx.oob_prop_count_ + class_idx]++;
            mean += Float(class_idx != Index(response_host_ptr[row_ind]));
        }
        else {
            ONEDAL_ASSERT(2 == ctx.oob_prop_count_);
            oob_per_obs_list_host_ptr[row_ind * ctx.oob_prop_count_ + 0] += prediction;
            oob_per_obs_list_host_ptr[row_ind * ctx.oob_prop_count_ + 1] += Float(1);
            mean += (prediction - response_host_ptr[row_ind]) *
                    (prediction - response_host_ptr[row_ind]);
        }
    }

    oob_per_obs_list = oob_per_obs_list_host.to_device(queue_);

    return mean / Float(n);
}

template <typename Float, typename Bin, typename Index, typename Task>
Float train_kernel_hist_impl<Float, Bin, Index, Task>::compute_oob_error_perm(
    const model_manager_t& model_manager,
    const pr::ndarray<Float, 1>& data_host,
    const pr::ndarray<Float, 1>& response_host,
    const pr::ndarray<Index, 1>& oob_row_list,
    const pr::ndarray<Index, 1>& permutation_host,
    Index tree_idx,
    Index indicesOffset,
    Index n,
    Index column_idx,
    const context_t& ctx,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(data_host.get_count() == ctx.row_count_ * ctx.column_count_);
    ONEDAL_ASSERT(response_host.get_count() == ctx.column_count_);
    ONEDAL_ASSERT(permutation_host.get_count() == n);
    ONEDAL_ASSERT(column_idx < ctx.column_count_);
    // input asserts is going to be added

    auto oob_row_list_host = oob_row_list.to_host(queue_, deps);

    const Float* data_host_ptr = data_host.get_data();
    const Float* response_host_ptr = response_host.get_data();
    const Index* oob_row_list_host_ptr = oob_row_list_host.get_data();
    const Index* permutation_ptr = permutation_host.get_data();

    auto buf = pr::ndarray<Float, 1>::empty({ ctx.column_count_ });
    Float* buf_ptr = buf.get_mutable_data();

    Float mean = Float(0);
    for (Index i = 0; i < n; i++) {
        int row_ind = oob_row_list_host_ptr[indicesOffset + i];
        int row_ind_perm = permutation_ptr[i];
        ONEDAL_ASSERT(row_ind < ctx.row_count_);
        ONEDAL_ASSERT(row_ind_perm < ctx.row_count_);

        memcpy(de::default_host_policy{},
               buf_ptr,
               &data_host_ptr[row_ind * ctx.column_count_],
               ctx.column_count_ * sizeof(Float));
        buf_ptr[column_idx] = data_host_ptr[row_ind_perm * ctx.column_count_ + column_idx];
        auto prediction = model_manager.get_tree_response(tree_idx, buf_ptr);
        if constexpr (std::is_same_v<Task, task::classification>) {
            Index class_idx = static_cast<Index>(prediction);
            mean += Float(class_idx != Index(response_host_ptr[row_ind]));
        }
        else {
            mean += (prediction - response_host_ptr[row_ind]) *
                    (prediction - response_host_ptr[row_ind]);
        }
    }

    return mean / Float(n);
}

template <typename Float, typename Bin, typename Index, typename Task>
cl::sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_results(
    const model_manager_t& model_manager,
    const pr::ndarray<Float, 1>& data_host,
    const pr::ndarray<Float, 1>& response_host,
    const pr::ndarray<Index, 1>& oob_row_list,
    const pr::ndarray<Index, 1>& oobRowsNumList,
    pr::ndarray<hist_type_t, 1>& oob_per_obs_list,
    pr::ndarray<Float, 1>& var_imp,
    pr::ndarray<Float, 1>& var_imp_variance,
    const dal::array<engine_impl>& engine_arr,
    Index tree_idx_in_block,
    Index tree_in_block_count,
    Index built_tree_count,
    const context_t& ctx,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(oobRowsNumList.get_count() == tree_in_block_count + 1);
    ONEDAL_ASSERT(var_imp.get_count() == ctx.column_count_);
    ONEDAL_ASSERT(ctx.mda_scaled_required_ ? var_imp_variance.get_count() == ctx.column_count_
                                           : true);

    auto nOOBRowsHost = oobRowsNumList.to_host(queue_, deps);
    const Index* nOOBRowsHost_ptr = nOOBRowsHost.get_data();
    Index oob_indices_offset = nOOBRowsHost_ptr[tree_idx_in_block];
    Index oob_row_count =
        nOOBRowsHost_ptr[tree_idx_in_block + 1] - nOOBRowsHost_ptr[tree_idx_in_block];

    if ((ctx.oob_required_ || ctx.mda_required_) && oob_row_count) {
        const Float oob_err = compute_oob_error(model_manager,
                                                data_host,
                                                response_host,
                                                oob_row_list,
                                                oob_per_obs_list,
                                                tree_idx_in_block,
                                                oob_indices_offset,
                                                oob_row_count,
                                                ctx);

        if (ctx.mda_required_) {
            auto permutation = pr::ndarray<Index, 1>::empty({ oob_row_count });
            Index* permutation_ptr = permutation.get_mutable_data();
            for (Index i = 0; i < oob_row_count; ++i) {
                permutation_ptr[i] = i;
            }

            auto var_imp_host = var_imp.to_host(queue_);
            Float* var_imp_host_ptr = var_imp_host.get_mutable_data();
            Float* var_imp_var_host_ptr = nullptr;
            if (ctx.mda_scaled_required_) {
                auto var_imp_var_host = var_imp_variance.to_host(queue_);
                var_imp_var_host_ptr = var_imp_var_host.get_mutable_data();
            }

            const Float div1 = Float(1) / Float(built_tree_count + tree_idx_in_block + 1);

            rng<Index> rn_gen;

            for (Index column_idx = 0; column_idx < ctx.column_count_; column_idx++) {
                rn_gen
                    .shuffle(oob_row_count,
                             permutation_ptr,
                             engine_arr[built_tree_count + tree_idx_in_block].get_state())
                    .wait_and_throw();
                const Float oob_err_perm = compute_oob_error_perm(model_manager,
                                                                  data_host,
                                                                  response_host,
                                                                  oob_row_list,
                                                                  permutation,
                                                                  tree_idx_in_block,
                                                                  oob_indices_offset,
                                                                  oob_row_count,
                                                                  column_idx,
                                                                  ctx);

                const Float diff = (oob_err_perm - oob_err);
                const Float delta = diff - var_imp_host_ptr[column_idx];
                var_imp_host_ptr[column_idx] += div1 * delta;
                if (var_imp_var_host_ptr) {
                    var_imp_var_host_ptr[column_idx] +=
                        delta * (diff - var_imp_host_ptr[column_idx]);
                }
            }

            var_imp = var_imp_host.to_device(queue_);
            if (ctx.mda_scaled_required_) {
                var_imp_variance = var_imp_variance_host_.to_device(queue_);
            }
        }
    }

    return cl::sycl::event{};
}

template <typename Float, typename Bin, typename Index, typename Task>
cl::sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::finalize_oob_error(
    const pr::ndarray<Float, 1>& response_host,
    pr::ndarray<hist_type_t, 1>& oob_per_obs_list,
    pr::ndarray<Float, 1>& res_oob_err,
    pr::ndarray<Float, 1>& res_oob_err_obs,
    const context_t& ctx,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(oob_per_obs_list.get_count() == ctx.row_count_ * ctx.oob_prop_count_);

    auto oob_per_obs_list_host = oob_per_obs_list.to_host(queue_, deps);

    const Float* response_host_ptr = response_host.get_data();
    const hist_type_t* oob_per_obs_list_host_ptr = oob_per_obs_list_host.get_mutable_data();

    auto res_oob_err_host = pr::ndarray<Float, 1>::empty({ 1 });
    auto res_oob_err_obs_host = pr::ndarray<Float, 1>::empty({ ctx.row_count_ });
    Float* res_oob_err_host_ptr = res_oob_err_host.get_mutable_data();
    Float* res_oob_err_obs_host_ptr = res_oob_err_obs_host.get_mutable_data();

    Index predicted_count = 0;
    Float oob_err = 0;

    for (Index i = 0; i < ctx.row_count_; i++) {
        bool predicted = false;
        hist_type_t prediction = hist_type_t(0);

        if constexpr (std::is_same_v<Task, task::classification>) {
            ONEDAL_ASSERT(ctx.class_count_ == ctx.oob_prop_count_);
            Index max_val = 0;
            for (Index class_idx = 0; class_idx < ctx.class_count_; class_idx++) {
                Index val = oob_per_obs_list_host_ptr[i * ctx.oob_prop_count_ + class_idx];
                if (val > max_val) {
                    max_val = val;
                    prediction = class_idx;
                }
            }

            predicted = bool(0 < max_val);
        }
        else {
            ONEDAL_ASSERT(2 == ctx.oob_prop_count_);
            Float value = oob_per_obs_list_host_ptr[i * ctx.oob_prop_count_ + 0];
            Float count = oob_per_obs_list_host_ptr[i * ctx.oob_prop_count_ + 1];

            predicted = bool(count > Float(0));
            prediction = value / count;
        }

        if (predicted) {
            Float prediction_res = Float(0);
            if constexpr (std::is_same_v<Task, task::classification>) {
                prediction_res = Float(prediction != static_cast<Index>(response_host_ptr[i]));
            }
            else {
                prediction_res =
                    (prediction - response_host_ptr[i]) * (prediction - response_host_ptr[i]);
            }

            if (ctx.oob_err_obs_required_)
                res_oob_err_obs_host_ptr[i] = prediction_res;
            oob_err += prediction_res;
            predicted_count++;
        }
        else if (ctx.oob_err_obs_required_)
            //was not in OOB set of any tree and hence not predicted
            res_oob_err_obs_host_ptr[i] = Float(-1);
    }

    if (ctx.oob_err_required_) {
        *res_oob_err_host_ptr = (0 < predicted_count) ? oob_err / Float(predicted_count) : 0;
        res_oob_err = res_oob_err_host.to_device(queue_);
    }

    if (ctx.oob_err_obs_required_) {
        res_oob_err_obs = res_oob_err_obs_host.to_device(queue_);
    }

    return cl::sycl::event{};
}

template <typename Float, typename Bin, typename Index, typename Task>
cl::sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::finalize_var_imp(
    pr::ndarray<Float, 1>& var_imp,
    pr::ndarray<Float, 1>& var_imp_variance,
    const context_t& ctx,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(var_imp.get_count() == ctx.column_count_);

    auto var_imp_host = var_imp.to_host(queue_);
    Float* var_imp_host_ptr = var_imp_host.get_mutable_data();

    if (ctx.mda_scaled_required_) {
        if (ctx.tree_count_ > 1) {
            ONEDAL_ASSERT(var_imp_variance.get_count() == ctx.column_count_);
            auto var_imp_var_host = var_imp_variance.to_host(queue_);
            Float* var_imp_var_host_ptr = var_imp_var_host.get_mutable_data();

            const Float div = Float(1) / Float(ctx.tree_count_);
            for (Index i = 0; i < ctx.column_count_; i++) {
                var_imp_var_host_ptr[i] *= div;
                if (var_imp_var_host_ptr[i] > Float(0)) {
                    var_imp_host_ptr[i] /= std::sqrt(var_imp_var_host_ptr[i] * div);
                }
            }
            var_imp = var_imp_host.to_device(queue_);
        }
        else {
            var_imp.fill(queue_, 0); // addd deps??
        }
    }
    else if (ctx.mdi_required_) {
        const Float div = Float(1) / ctx.tree_count_;
        for (Index i = 0; i < ctx.column_count_; i++)
            var_imp_host_ptr[i] *= div;
        var_imp = var_imp_host.to_device(queue_);
    }

    return cl::sycl::event{};
}

/////////////////////////////////////////////////////////
/// Main compute
template <typename Float, typename Bin, typename Index, typename Task>
train_result<Task> train_kernel_hist_impl<Float, Bin, Index, Task>::operator()(
    const descriptor_t& desc,
    const table& data,
    const table& responses) {
    using imp_data_mng_t = impurity_data_manager<Float, Index, Task>;
    using tree_level_record_t = tree_level_record<Float, Index, Task>;

    validate_input(desc, data, responses);

    context_t ctx;
    init_params(ctx, desc, data, responses);
    allocate_buffers(ctx);

    /*init engines*/
    de::check_mul_overflow(ctx.tree_count_ - 1, ctx.tree_count_);
    de::check_mul_overflow(ctx.tree_count_ - 1 * ctx.tree_count_, ctx.row_count_);
    de::check_mul_overflow(ctx.tree_count_ - 1 * ctx.tree_count_ * ctx.row_count_,
                           (ctx.selected_ftr_count_ + 1));

    engine_collection collection(ctx.tree_count_);
    dal::array<engine_impl> engine_arr = collection([&](size_t i, size_t& skip) {
        skip = i * ctx.tree_count_ * ctx.row_count_ * (ctx.selected_ftr_count_ + 1);
    });

    pr::ndarray<Float, 1> node_imp_decrease_list;

    model_manager_t model_manager(ctx.tree_count_, ctx.column_count_, ctx);

    cl::sycl::event last_event;

    for (Index iter = 0; iter < ctx.tree_count_; iter += ctx.tree_in_block_) {
        Index nTrees = std::min(ctx.tree_count_ - iter, ctx.tree_in_block_);

        Index node_count = nTrees; // num of potential nodes to split on current tree level
        auto oobRowsNumList = pr::ndarray<Index, 1>::empty(queue_, { nTrees + 1 }, alloc::device);
        pr::ndarray<Index, 1> oobRows;

        std::vector<tree_level_record_t> level_records;
        // lists of nodes int props(rowsOffset, rows, ftrId, ftrVal ... )
        std::vector<pr::ndarray<Index, 1>> levelNodeLists;

        imp_data_mng_t imp_data_holder(queue_, ctx);
        // initilizing imp_list and class_hist_list (for classification)
        imp_data_holder.init_new_level(node_count);

        de::check_mul_overflow(node_count, impl_const_t::node_prop_count_);
        de::check_mul_overflow(node_count, impl_const_t::node_imp_prop_count_);
        auto node_vs_tree_map_list_host = pr::ndarray<Index, 1>::empty({ node_count });
        auto levelNodeLists_init_host =
            pr::ndarray<Index, 1>::empty({ node_count * impl_const_t::node_prop_count_ });

        auto treeMap = node_vs_tree_map_list_host.get_mutable_data();
        auto node_list_ptr = levelNodeLists_init_host.get_mutable_data();

        for (Index node = 0; node < node_count; node++) {
            Index* node_ptr = node_list_ptr + node * impl_const_t::node_prop_count_;
            treeMap[node] = iter + node;
            node_ptr[0] = ctx.selected_row_count_ * node; // rows offset
            node_ptr[1] = ctx.selected_row_count_; // num of rows
        }

        auto node_vs_tree_map_list = node_vs_tree_map_list_host.to_device(queue_);
        levelNodeLists.push_back(levelNodeLists_init_host.to_device(queue_));

        if (ctx.bootstrap_) {
            engine_impl* engines = engine_arr.get_mutable_data();
            Index* selected_rows_ptr = selected_rows_host_.get_mutable_data();

            for (Index tree_idx = 0; tree_idx < nTrees; tree_idx++) {
                rng<Index> rn_gen;
                rn_gen.uniform(ctx.selected_row_count_,
                               selected_rows_ptr + ctx.selected_row_count_ * tree_idx,
                               engines[iter + tree_idx].get_state(),
                               0,
                               ctx.row_count_);
            }

            cl::sycl::event event =
                tree_order_lev_.assign(queue_, selected_rows_ptr, selected_rows_host_.get_count());
            event.wait_and_throw();
        }
        else {
            cl::sycl::event event =
                train_service_kernels_.initialize_tree_order(tree_order_lev_,
                                                             nTrees,
                                                             ctx.selected_row_count_);
            event.wait_and_throw();
        }

        last_event = compute_initial_histogram(response_nd_,
                                               tree_order_lev_,
                                               levelNodeLists[0],
                                               imp_data_holder.get_mutable_data(0),
                                               node_count,
                                               ctx,
                                               { last_event });
        last_event.wait_and_throw();

        if (ctx.oob_required_) {
            cl::sycl::event event = train_service_kernels_.get_oob_row_list(
                tree_order_lev_,
                oobRowsNumList,
                oobRows,
                ctx.selected_row_count_,
                nTrees); // oobRowsNumList and oobRows are the output
            event.wait_and_throw();
        }

        for (Index level = 0; node_count > 0; level++) {
            auto nodeList = levelNodeLists[level];

            imp_data_t left_child_imp_data(queue_, ctx, node_count);

            auto [selectedFeaturesCom, event] =
                gen_features(node_count, node_vs_tree_map_list, engine_arr, ctx);
            event.wait_and_throw();

            if (ctx.mdi_required_) {
                node_imp_decrease_list =
                    pr::ndarray<Float, 1>::empty(queue_, { node_count }, alloc::device);
            }

            last_event = compute_best_split(full_data_nd_,
                                            response_nd_,
                                            tree_order_lev_,
                                            selectedFeaturesCom,
                                            ftr_bin_offsets_nd_,
                                            imp_data_holder.get_data(level),
                                            nodeList,
                                            left_child_imp_data,
                                            node_imp_decrease_list,
                                            ctx.mdi_required_,
                                            node_count,
                                            ctx,
                                            { last_event });
            last_event.wait_and_throw();

            tree_level_record_t level_record(queue_,
                                             nodeList,
                                             imp_data_holder.get_data(level),
                                             node_count,
                                             ctx);
            level_records.push_back(level_record);

            if (ctx.mdi_required_) {
                //mdi is calculated only on split nodes and is not calculated on last level
                last_event =
                    train_service_kernels_.update_mdi_var_importance(nodeList,
                                                                     node_imp_decrease_list,
                                                                     res_var_imp_,
                                                                     ctx.column_count_,
                                                                     node_count,
                                                                     { last_event });
            }

            Index node_count_new;
            last_event = train_service_kernels_.get_split_node_count(nodeList,
                                                                     node_count,
                                                                     node_count_new,
                                                                     { last_event });
            last_event.wait_and_throw();

            if (node_count_new) {
                //there are split nodes -> next level is required
                node_count_new *= 2;

                de::check_mul_overflow(node_count_new, impl_const_t::node_prop_count_);
                auto node_list_new = pr::ndarray<Index, 1>::empty(
                    queue_,
                    { node_count_new * impl_const_t::node_prop_count_ },
                    alloc::device);

                imp_data_holder.init_new_level(node_count_new);

                auto node_vs_tree_map_list_new =
                    pr::ndarray<Index, 1>::empty(queue_, { node_count_new }, alloc::device);

                last_event = do_node_split(nodeList,
                                           node_vs_tree_map_list,
                                           imp_data_holder.get_data(level),
                                           left_child_imp_data,
                                           node_list_new,
                                           node_vs_tree_map_list_new,
                                           imp_data_holder.get_mutable_data(level + 1),
                                           node_count,
                                           node_count_new,
                                           ctx,
                                           { last_event });
                last_event.wait_and_throw();

                if (ctx.max_tree_depth_ > 0 && ctx.max_tree_depth_ == level) {
                    tree_level_record_t level_record(queue_,
                                                     node_list_new,
                                                     imp_data_holder.get_data(level + 1),
                                                     node_count_new,
                                                     ctx);
                    level_records.push_back(level_record);
                    node_count_new = 0;
                }
                else {
                    levelNodeLists.push_back(node_list_new);

                    node_vs_tree_map_list = node_vs_tree_map_list_new;

                    last_event =
                        train_service_kernels_.do_level_partition_by_groups(full_data_nd_,
                                                                            nodeList,
                                                                            tree_order_lev_,
                                                                            tree_order_lev_buf_,
                                                                            ctx.column_count_,
                                                                            node_count,
                                                                            { last_event });
                }
            }

            node_count = node_count_new;
        }

        model_manager.add_tree_block(level_records, bin_borders_host_, nTrees);

        for (Index tree_idx = 0; tree_idx < nTrees; tree_idx++) {
            compute_results(model_manager,
                            data_host_,
                            response_host_,
                            oobRows,
                            oobRowsNumList,
                            oob_per_obs_list_,
                            res_var_imp_,
                            var_imp_variance_host_,
                            engine_arr,
                            tree_idx,
                            nTrees,
                            iter,
                            ctx,
                            { last_event })
                .wait_and_throw();
        }
    }

    result_t res;

    // Finalize results
    if (ctx.oob_err_required_ || ctx.oob_err_obs_required_) {
        pr::ndarray<Float, 1> res_oob_err;
        pr::ndarray<Float, 1> res_oob_err_obs;

        finalize_oob_error(response_host_, oob_per_obs_list_, res_oob_err, res_oob_err_obs, ctx)
            .wait_and_throw();

        if (ctx.oob_err_required_) {
            auto res_oob_err_host = res_oob_err.to_host(queue_);
            res.set_oob_err(homogen_table::wrap(res_oob_err_host.flatten(), 1, 1));
        }

        if (ctx.oob_err_obs_required_) {
            auto res_oob_err_obs_host = res_oob_err_obs.to_host(queue_);
            res.set_oob_err_per_observation(
                homogen_table::wrap(res_oob_err_obs_host.flatten(), ctx.row_count_, 1));
        }
    }

    if (ctx.mdi_required_ || ctx.mda_required_) {
        finalize_var_imp(res_var_imp_, var_imp_variance_host_, ctx).wait_and_throw();
        auto res_var_imp_host = res_var_imp_.to_host(queue_);
        res.set_var_importance(
            homogen_table::wrap(res_var_imp_host.flatten(), 1, ctx.column_count_));
    }

    return res.set_model(model_manager.get_model());
}

#define INSTANTIATE(F, B, I, T) template class train_kernel_hist_impl<F, B, I, T>;

//INSTANTIATE(ONEDAL_FLOAT, std::uint32_t, std::int32_t, task::classification);
//INSTANTIATE(ONEDAL_FLOAT, std::uint32_t, std::int32_t, task::regression);

} // namespace oneapi::dal::decision_forest::backend
