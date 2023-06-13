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
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_helpers.hpp"

#ifdef ONEDAL_DATA_PARALLEL

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_splitter_impl.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_splitter_helpers.hpp"

namespace oneapi::dal::decision_forest::backend {

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

using alloc = sycl::usm::alloc;
using address = sycl::access::address_space;

using sycl::ext::oneapi::plus;
using sycl::ext::oneapi::minimum;
using sycl::ext::oneapi::maximum;

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_splitter_impl<Float, Bin, Index, Task>::random_split(
    sycl::queue& queue,
    const context_t& ctx,
    const pr::ndview<Bin, 2>& data,
    const pr::ndview<Float, 1>& response,
    const pr::ndview<Index, 1>& tree_order,
    const pr::ndview<Index, 1>& selected_ftr_list,
    const pr::ndview<Float, 1>& random_bins_com,
    const pr::ndview<Index, 1>& bin_offset_list,
    const imp_data_t& imp_data_list,
    pr::ndview<Index, 1>& node_list,
    imp_data_t& left_child_imp_data_list,
    pr::ndview<Float, 1>& node_imp_dec_list,
    bool update_imp_dec_required,
    Index node_count,
    const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(random_split, queue);
    using split_smp_t = split_smp<Float, Index, Task>;
    using split_info_t = split_info<Float, Index, Task>;

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
    ONEDAL_ASSERT(node_list.get_count() >= node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(left_child_imp_data_list.imp_list_.get_count() >=
                  node_count * impl_const_t::node_imp_prop_count_);
    if constexpr (std::is_same_v<Task, task::classification>) {
        ONEDAL_ASSERT(left_child_imp_data_list.class_hist_list_.get_count() >=
                      node_count * ctx.class_count_);
    }

    if (update_imp_dec_required) {
        ONEDAL_ASSERT(node_imp_dec_list.get_count() >= node_count);
    }

    const Bin* const data_ptr = data.get_data();
    const Float* const response_ptr = response.get_data();
    const Index* const tree_order_ptr = tree_order.get_data();

    const Index* const selected_ftr_list_ptr = selected_ftr_list.get_data();

    imp_data_list_ptr<Float, Index, Task> imp_list_ptr(imp_data_list);

    Index* const node_list_ptr = node_list.get_mutable_data();
    Float* const node_imp_decr_list_ptr =
        update_imp_dec_required ? node_imp_dec_list.get_mutable_data() : nullptr;

    imp_data_list_ptr_mutable<Float, Index, Task> left_imp_list_ptr(left_child_imp_data_list);

    const Float* const ftr_rnd_ptr = random_bins_com.get_data();

    const Index column_count = ctx.column_count_;
    const Index selected_ftr_count = ctx.selected_ftr_count_;
    const Index index_max = ctx.index_max_;

    Index local_size = bk::device_max_wg_size(queue);

    const std::size_t local_hist_buf_size = hist_prop_count * 2; // x2 because bs_hist and ts_hist
    const std::size_t local_buf_int_size = local_size * sizeof(Index);
    const std::size_t local_buf_float_size = local_size * sizeof(Float);

    sycl::event last_event;

    const Index class_count = ctx.class_count_;
    const Float imp_threshold = ctx.impurity_threshold_;
    const Index min_obs_leaf = ctx.min_observations_in_leaf_node_;
    const Index max_bin_count_among_ftrs = ctx.max_bin_count_among_ftrs_;

    const Float* node_imp_list_ptr = imp_list_ptr.imp_list_ptr_;
    Float* left_child_imp_list_ptr = left_imp_list_ptr.imp_list_ptr_;

    // following vars are not used for regression, but should present to compile kernel
    const Index* class_hist_list_ptr = imp_list_ptr.get_class_hist_list_ptr_or_null();
    Index* left_child_class_hist_list_ptr = left_imp_list_ptr.get_class_hist_list_ptr_or_null();

    Index node_in_block_count = node_count;

    std::size_t local_buf_byte_size =
        local_buf_int_size + local_buf_float_size + local_hist_buf_size * sizeof(hist_type_t);
    ONEDAL_ASSERT(device_has_enough_local_mem(queue, local_buf_byte_size));

    const auto nd_range =
        bk::make_multiple_nd_range_2d({ local_size, node_in_block_count }, { local_size, 1 });

    last_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<byte_t> local_byte_buf(local_buf_byte_size, cgh);

        cgh.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
            auto sbg = item.get_sub_group();
            const Index node_idx = item.get_global_id(1);
            if (node_idx > (node_count - 1)) {
                return;
            }

            const Index node_id = node_idx;
            Index* node_ptr = node_list_ptr + node_id * impl_const_t::node_prop_count_;

            const Index local_id = item.get_local_id(0);

            const Index row_ofs = node_ptr[impl_const_t::ind_ofs];
            const Index row_count = node_ptr[impl_const_t::ind_lrc];

            split_smp_t sp_hlp;
            // Check node impurity
            if (!sp_hlp.is_valid_impurity(node_imp_list_ptr, node_id, imp_threshold, row_count)) {
                return;
            }
            split_info_t bs;

            // slm pointers declaration
            byte_t* local_byte_buf_ptr = local_byte_buf.get_pointer().get();
            hist_type_t* local_hist_buf_ptr =
                get_buf_ptr<hist_type_t>(&local_byte_buf_ptr, local_hist_buf_size);
            Float* local_buf_float_ptr = get_buf_ptr<Float>(&local_byte_buf_ptr, local_size);

            bs.init_clear(item, local_hist_buf_ptr + 0 * hist_prop_count, hist_prop_count);

            for (Index ftr_idx = 0; ftr_idx < selected_ftr_count; ftr_idx++) {
                split_info_t ts;
                ts.init(local_hist_buf_ptr + 1 * hist_prop_count, hist_prop_count);
                ts.ftr_id = selected_ftr_list_ptr[node_id * selected_ftr_count + ftr_idx];
                const Index id =
                    (local_id < row_count) ? tree_order_ptr[row_ofs + local_id] : index_max;
                const Index bin =
                    (local_id < row_count) ? data_ptr[id * column_count + ts.ftr_id] : index_max;
                const Float response = (local_id < row_count) ? response_ptr[id] : Float(0);
                const Index response_int =
                    (local_id < row_count) ? static_cast<Index>(response) : -1;

                const Index min_bin =
                    sycl::reduce_over_group(item.get_group(),
                                            bin < index_max ? bin : max_bin_count_among_ftrs,
                                            minimum<Index>());
                const Index max_bin =
                    sycl::reduce_over_group(item.get_group(),
                                            bin < max_bin_count_among_ftrs ? bin : 0,
                                            maximum<Index>());

                const Float rand_val = ftr_rnd_ptr[node_id * selected_ftr_count + ftr_idx];
                const Index random_bin_ofs = static_cast<Index>(rand_val * (max_bin - min_bin + 1));
                ts.ftr_bin = min_bin + random_bin_ofs;

                const Index count = (bin <= ts.ftr_bin) ? 1 : 0;

                if constexpr (std::is_same_v<Task, task::classification>) {
                    const Index left_count =
                        sycl::reduce_over_group(item.get_group(), count, plus<Index>());
                    const Index val = (bin <= ts.ftr_bin) ? response_int : -1;
                    Index all_class_count = 0;

                    for (Index class_id = 0; class_id < class_count - 1; class_id++) {
                        Index total_class_count = sycl::reduce_over_group(item.get_group(),
                                                                          Index(class_id == val),
                                                                          plus<Index>());
                        all_class_count += total_class_count;
                        ts.left_hist[class_id] = total_class_count;
                    }

                    ts.left_count = left_count;

                    ts.left_hist[class_count - 1] = ts.left_count - all_class_count;
                }
                else {
                    const Float val = (bin <= ts.ftr_bin) ? response : Float(0);

                    Float left_count = Float(sycl::reduce_over_group(sbg, count, plus<Index>()));
                    Float sum = sycl::reduce_over_group(sbg, val, plus<Float>());

                    Float mean = sum / left_count;

                    const Float val_s2c =
                        (bin <= ts.ftr_bin) ? (val - mean) * (val - mean) : Float(0);

                    Float sum2cent = sycl::reduce_over_group(sbg, val_s2c, plus<Float>());

                    reduce_hist_over_group(item, local_buf_float_ptr, left_count, mean, sum2cent);

                    ts.left_count = Index(left_count);

                    ts.left_hist[0] = left_count;
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
                        sp_hlp.choose_best_split(bs, ts, class_count, min_obs_leaf);
                    }
                    else {
                        sp_hlp.calc_imp_dec(ts, node_ptr, node_imp_list_ptr, node_id);
                        sp_hlp.choose_best_split(bs,
                                                 ts,
                                                 impl_const_t::hist_prop_count_,
                                                 min_obs_leaf);
                    }
                }
            }

            if (local_id == 0) {
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
                    sp_hlp.update_left_child_imp(left_child_imp_list_ptr, bs.left_hist, node_id);
                }
            }
        });
    });

    last_event.wait_and_throw();
    return last_event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_splitter_impl<Float, Bin, Index, Task>::best_split(
    sycl::queue& queue,
    const context_t& ctx,
    const pr::ndview<Bin, 2>& data,
    const pr::ndview<Float, 1>& response,
    const pr::ndview<Index, 1>& tree_order,
    const pr::ndview<Index, 1>& selected_ftr_list,
    const pr::ndview<Index, 1>& bin_offset_list,
    const imp_data_t& imp_data_list,
    pr::ndview<Index, 1>& node_list,
    imp_data_t& left_child_imp_data_list,
    pr::ndview<Float, 1>& node_imp_dec_list,
    bool update_imp_dec_required,
    Index node_count,
    const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(best_split, queue);
    ONEDAL_ASSERT(data.get_count() == ctx.row_count_ * ctx.column_count_);
    ONEDAL_ASSERT(response.get_count() == ctx.row_count_);
    ONEDAL_ASSERT(tree_order.get_count() == ctx.tree_in_block_ * ctx.selected_row_total_count_);
    ONEDAL_ASSERT(selected_ftr_list.get_count() >= node_count * ctx.selected_ftr_count_);
    ONEDAL_ASSERT(bin_offset_list.get_count() == ctx.column_count_ + 1);
    ONEDAL_ASSERT(imp_data_list.imp_list_.get_count() >=
                  node_count * impl_const_t::node_imp_prop_count_);

    const Bin* const data_ptr = data.get_data();
    const Float* const response_ptr = response.get_data();
    const Index* const tree_order_ptr = tree_order.get_data();

    const Index* const selected_ftr_list_ptr = selected_ftr_list.get_data();

    imp_data_list_ptr<Float, Index, Task> imp_list_ptr(imp_data_list);

    Index* node_list_ptr = node_list.get_mutable_data();
    Float* node_imp_decr_list_ptr =
        update_imp_dec_required ? node_imp_dec_list.get_mutable_data() : nullptr;

    imp_data_list_ptr_mutable<Float, Index, Task> left_imp_list_ptr(left_child_imp_data_list);

    const Float* node_imp_list_ptr = imp_list_ptr.imp_list_ptr_;
    Float* left_child_imp_list_ptr = left_imp_list_ptr.imp_list_ptr_;

    using split_smp_t = split_smp<Float, Index, Task>;
    using split_scalar_t = split_scalar<Float, Index, Task>;
    using split_info_t = split_info<Float, Index, Task>;

    Index hist_prop_count = 0;
    if constexpr (std::is_same_v<std::decay_t<Task>, task::classification>) {
        hist_prop_count = ctx.class_count_;
    }
    else {
        hist_prop_count = impl_const<Index, task::regression>::hist_prop_count_;
    }

    const Index class_count = ctx.class_count_;
    const Index column_count = ctx.column_count_;
    const Float imp_threshold = ctx.impurity_threshold_;
    const Index min_obs_leaf = ctx.min_observations_in_leaf_node_;
    const Index index_max = ctx.index_max_;

    // following vars are not used for regression, but should present to compile kernel
    const Index* class_hist_list_ptr = imp_list_ptr.get_class_hist_list_ptr_or_null();
    Index* left_child_class_hist_list_ptr = left_imp_list_ptr.get_class_hist_list_ptr_or_null();

    const Index selected_ftr_count = ctx.selected_ftr_count_;
    const Index max_bin_size = ctx.max_bin_count_among_ftrs_;
    const Index node_in_block_count = node_count;

    sycl::event last_event;
    auto device = queue.get_device();
    std::int64_t device_local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
    const std::int64_t bin_local_mem_size =
        (2 * hist_prop_count * sizeof(hist_type_t) + sizeof(split_scalar_t));
    const std::int64_t common_local_data_size = hist_prop_count * sizeof(hist_type_t);
    std::int64_t bin_cnt_per_krn =
        (device_local_mem_size - common_local_data_size) / bin_local_mem_size;
    const Index all_bin_count = selected_ftr_count * max_bin_size;
    bin_cnt_per_krn = std::min<std::int64_t>(bin_cnt_per_krn, all_bin_count);
    const std::int64_t local_hist_size = bin_cnt_per_krn * hist_prop_count;
    const std::int64_t local_splits_size = bin_cnt_per_krn;
    const Index local_size = std::min<Index>(bk::device_max_wg_size(queue), bin_cnt_per_krn);

    const auto nd_range =
        bk::make_multiple_nd_range_2d({ local_size, node_in_block_count }, { local_size, 1 });
    last_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<hist_type_t> local_hist(2 * local_hist_size, cgh);
        local_accessor_rw_t<split_scalar_t> scalars_buf(local_splits_size, cgh);
        local_accessor_rw_t<hist_type_t> best_split_hist(hist_prop_count, cgh);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
            // Load common data
            const Index node_id = item.get_global_id(1);
            Index* node_ptr = node_list_ptr + node_id * impl_const_t::node_prop_count_;

            const Index local_id = item.get_local_id(0);

            const Index row_ofs = node_ptr[impl_const_t::ind_ofs];
            const Index row_count = node_ptr[impl_const_t::ind_lrc];
            split_smp_t sp_hlp;
            // Check node impurity
            if (!sp_hlp.is_valid_impurity(node_imp_list_ptr, node_id, imp_threshold, row_count)) {
                return;
            }
            split_info_t global_bs;
            if (local_id == 0) {
                global_bs.init_clear(item, best_split_hist.get_pointer().get(), hist_prop_count);
            }
            // Due to local memory limitations need to proccess smaller blocks of features
            for (Index gbin_ofs = 0; gbin_ofs < all_bin_count; gbin_ofs += bin_cnt_per_krn) {
                hist_type_t* local_hist_ptr = local_hist.get_pointer().get();
                hist_type_t* tmp_hist_ptr = local_hist_ptr + local_hist_size;
                split_scalar_t* scalars_buf_ptr = scalars_buf.get_pointer().get();

                const Index ftr_ofs = gbin_ofs / max_bin_size;
                const Index bin_ofs = gbin_ofs % max_bin_size;

                const Index real_bin_count =
                    sycl::min<Index>(all_bin_count - gbin_ofs, bin_cnt_per_krn);
                // Clear local memory before use
                for (Index work_bin = local_id; work_bin < bin_cnt_per_krn;
                     work_bin += local_size) {
                    scalars_buf_ptr[work_bin].clear();
                    for (Index prop_idx = 0; prop_idx < hist_prop_count; ++prop_idx) {
                        local_hist_ptr[work_bin * hist_prop_count + prop_idx] = hist_type_t(0);
                        tmp_hist_ptr[work_bin * hist_prop_count + prop_idx] = hist_type_t(0);
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
                // Calculate histogram
                Index working_items =
                    sycl::max<Index>(1, real_bin_count / max_bin_size) * row_count;
                Index rows_per_item = working_items / local_size + bool(working_items % local_size);
                for (Index idx = local_id * rows_per_item;
                     idx < (local_id + 1) * rows_per_item && idx < working_items;
                     ++idx) {
                    const Index local_ftr_idx = idx / row_count;
                    const Index global_ftr_idx = ftr_ofs + local_ftr_idx;

                    const Index ts_ftr_id =
                        selected_ftr_list_ptr[node_id * selected_ftr_count + global_ftr_idx];
                    const Index row_idx = idx % row_count;
                    const Index id = tree_order_ptr[row_ofs + row_idx];
                    const Index bin = data_ptr[id * column_count + ts_ftr_id] - bin_ofs;
                    if (bin < 0 || bin >= bin_cnt_per_krn) {
                        continue;
                    }
                    const Float response = response_ptr[id];
                    const Index response_int = static_cast<Index>(response);

                    const Index cur_hist_pos =
                        local_hist_size + (local_ftr_idx * max_bin_size + bin) * hist_prop_count;

                    if constexpr (std::is_same_v<Task, task::classification>) {
                        sycl::atomic_ref<Index,
                                         sycl::memory_order_relaxed,
                                         sycl::memory_scope_work_group,
                                         sycl::access::address_space::local_space>
                            hist_resp(local_hist[cur_hist_pos + response_int]);
                        hist_resp += 1;
                    }
                    else {
                        sycl::atomic_ref<Float,
                                         sycl::memory_order_relaxed,
                                         sycl::memory_scope_work_group,
                                         sycl::access::address_space::local_space>
                            hist_count(local_hist[cur_hist_pos + 0]);
                        hist_count += 1;

                        sycl::atomic_ref<Float,
                                         sycl::memory_order_relaxed,
                                         sycl::memory_scope_work_group,
                                         sycl::access::address_space::local_space>
                            hist_sum(local_hist[cur_hist_pos + 1]);
                        hist_sum += response;
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
                // Case for regression: collect MSE, based on collected sum and counts
                if constexpr (std::is_same_v<Task, task::regression>) {
                    for (Index idx = local_id * rows_per_item;
                         idx < (local_id + 1) * rows_per_item && idx < working_items;
                         ++idx) {
                        const Index local_ftr_idx = idx / row_count;
                        const Index global_ftr_idx = ftr_ofs + local_ftr_idx;
                        const Index ts_ftr_id =
                            selected_ftr_list_ptr[node_id * selected_ftr_count + global_ftr_idx];
                        const Index row_idx = idx % row_count;
                        const Index id = tree_order_ptr[row_ofs + row_idx];
                        const Index bin = data_ptr[id * column_count + ts_ftr_id] - bin_ofs;
                        if (bin >= bin_cnt_per_krn) {
                            continue;
                        }
                        Float response = response_ptr[id];
                        const Index cur_hist_pos =
                            local_hist_size +
                            (local_ftr_idx * max_bin_size + bin) * hist_prop_count;
                        hist_type_t* cur_hist = local_hist_ptr + cur_hist_pos;
                        Float count = cur_hist[0];
                        Float resp_sum = cur_hist[1];
                        Float mean = resp_sum / count;
                        sycl::atomic_ref<Float,
                                         sycl::memory_order_relaxed,
                                         sycl::memory_scope_work_group,
                                         sycl::access::address_space::local_space>
                            hist_s2c(local_hist[cur_hist_pos + 2]);
                        hist_s2c += (response - mean) * (response - mean);
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                split_info_t ts;
                // Finilize histograms
                for (Index work_bin = local_id; work_bin < real_bin_count; work_bin += local_size) {
                    Index local_ftr_idx = work_bin / max_bin_size;
                    Index global_ftr_idx = ftr_ofs + local_ftr_idx;

                    Index cur_bin = work_bin % max_bin_size;
                    const Index cur_hist_pos = work_bin * hist_prop_count;
                    hist_type_t* cur_hist = local_hist_ptr + cur_hist_pos;
                    hist_type_t* ftr_hist = tmp_hist_ptr + (work_bin - cur_bin) * hist_prop_count;
                    // Collect all hists on the left side of the current bin
                    Index left_count = 0;
                    if constexpr (std::is_same_v<Task, task::classification>) {
                        for (Index bin_idx = 0; bin_idx <= cur_bin; ++bin_idx) {
                            for (Index cls = 0; cls < hist_prop_count; ++cls) {
                                Index bin_class_count = ftr_hist[bin_idx * hist_prop_count + cls];
                                cur_hist[cls] += bin_class_count;
                                left_count += bin_class_count;
                            }
                        }
                    }
                    else {
                        for (Index bin_idx = 0; bin_idx <= cur_bin; ++bin_idx) {
                            hist_type_t* iter_left_hist = ftr_hist + bin_idx * hist_prop_count;
                            if (iter_left_hist[0] <= Float(0)) {
                                continue;
                            }
                            Float source_mean = iter_left_hist[1] / iter_left_hist[0];
                            Float sum_n1n2 = cur_hist[0] + iter_left_hist[0];
                            Float mul_n1n2 = cur_hist[0] * iter_left_hist[0];
                            Float delta_scl = mul_n1n2 / sum_n1n2;
                            Float mean_scl = Float(1) / sum_n1n2;
                            Float delta = source_mean - cur_hist[1];

                            cur_hist[2] =
                                cur_hist[2] + iter_left_hist[2] + delta * delta * delta_scl;
                            cur_hist[1] =
                                (cur_hist[1] * cur_hist[0] + iter_left_hist[1]) * mean_scl;
                            cur_hist[0] = sum_n1n2;
                        }
                        left_count += Index(cur_hist[0]);
                    }
                    // Calculate impurity decrease for current bin
                    ts.init(cur_hist, hist_prop_count);
                    ts.ftr_id =
                        selected_ftr_list_ptr[node_id * selected_ftr_count + global_ftr_idx];
                    ts.ftr_bin = cur_bin + bin_ofs;
                    ts.left_count = left_count;
                    if constexpr (std::is_same_v<Task, task::classification>) {
                        sp_hlp.calc_imp_dec(ts,
                                            node_ptr,
                                            node_imp_list_ptr,
                                            class_hist_list_ptr,
                                            class_count,
                                            node_id);
                    }
                    else {
                        sp_hlp.calc_imp_dec(ts, node_ptr, node_imp_list_ptr, node_id);
                    }
                    if (ts.left_count > min_obs_leaf) {
                        ts.store_scalar(scalars_buf_ptr[work_bin]);
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
                // Init best and current split info
                split_info_t bs;
                bs.init(local_hist_ptr + local_id * hist_prop_count, hist_prop_count);
                bs.clear_scalar();
                // Select best bin among one working-item
                for (Index work_item = local_id; work_item < real_bin_count;
                     work_item += local_size) {
                    ts.init(local_hist_ptr + work_item * hist_prop_count, hist_prop_count);
                    ts.load_scalar(scalars_buf_ptr[work_item]);
                    sp_hlp.choose_best_split(bs, ts, hist_prop_count, min_obs_leaf);
                }
                bs.store_scalar(scalars_buf_ptr[local_id]);
                // Tree reduction and selecting best among work-group
                for (Index i = local_size / 2; i > 0; i /= 2) {
                    item.barrier(sycl::access::fence_space::local_space);
                    if (local_id < i && (local_id + i) < real_bin_count) {
                        ts.init(local_hist_ptr + (local_id + i) * hist_prop_count, hist_prop_count);
                        ts.load_scalar(scalars_buf_ptr[local_id + i]);
                        sp_hlp.choose_best_split(bs, ts, hist_prop_count, min_obs_leaf);
                        bs.store_scalar(scalars_buf_ptr[local_id]);
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
                // Update bs among all features
                if (local_id == 0) {
                    sp_hlp.choose_best_split(global_bs, bs, hist_prop_count, min_obs_leaf);
                }
            }
            // Update global split info
            if (local_id == 0) {
                sp_hlp.update_node_bs_info(global_bs,
                                           node_ptr,
                                           node_imp_decr_list_ptr,
                                           node_id,
                                           index_max,
                                           update_imp_dec_required);
                if constexpr (std::is_same_v<Task, task::classification>) {
                    sp_hlp.update_left_child_imp(left_child_imp_list_ptr,
                                                 left_child_class_hist_list_ptr,
                                                 global_bs.left_imp,
                                                 global_bs.left_hist,
                                                 node_id,
                                                 class_count);
                }
                else {
                    sp_hlp.update_left_child_imp(left_child_imp_list_ptr,
                                                 global_bs.left_hist,
                                                 node_id);
                }
            }
        });
    });
    last_event.wait_and_throw();
    return last_event;
}

#define INSTANTIATE(F, B, I, T) template class train_splitter_impl<F, B, I, T>;

INSTANTIATE(float, std::uint32_t, std::int32_t, task::classification);
INSTANTIATE(float, std::uint32_t, std::int32_t, task::regression);

INSTANTIATE(double, std::uint32_t, std::int32_t, task::classification);
INSTANTIATE(double, std::uint32_t, std::int32_t, task::regression);

} // namespace oneapi::dal::decision_forest::backend

#endif
