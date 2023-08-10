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
    using split_scalar_t = split_scalar<Float, Index, Task>;
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

    const std::size_t bins_per_wg = 2; // 2 for best and current split
    const std::size_t hist_size = hist_prop_count * bins_per_wg;

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

    std::size_t local_buf_byte_size = local_size * sizeof(Float) + hist_size * sizeof(hist_type_t);
    ONEDAL_ASSERT(device_has_enough_local_mem(queue, local_buf_byte_size));

    const auto nd_range =
        bk::make_multiple_nd_range_2d({ local_size, node_in_block_count }, { local_size, 1 });

    sycl::event last_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<hist_type_t> local_hist_buf(hist_size, cgh);
        local_accessor_rw_t<Float> local_float_buf(local_size, cgh);
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
#if __SYCL_COMPILER_VERSION >= 20230828
            hist_type_t* hist_ptr = local_hist_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            Float* local_buf_float_ptr = local_float_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
#else
            hist_type_t* hist_ptr = local_hist_buf.get_pointer().get();
            Float* local_buf_float_ptr = local_float_buf.get_pointer().get();
#endif

            bs.init_clear(hist_ptr + 0 * hist_prop_count, hist_prop_count);
            split_scalar_t& bs_scal = bs.scalars;

            for (Index ftr_idx = 0; ftr_idx < selected_ftr_count; ftr_idx++) {
                split_info_t ts;
                ts.init(hist_ptr + 1 * hist_prop_count, hist_prop_count);
                split_scalar_t& ts_scal = ts.scalars;
                ts_scal.ftr_id = selected_ftr_list_ptr[node_id * selected_ftr_count + ftr_idx];
                const Index id =
                    (local_id < row_count) ? tree_order_ptr[row_ofs + local_id] : index_max;
                const Index bin = (local_id < row_count)
                                      ? data_ptr[id * column_count + ts_scal.ftr_id]
                                      : index_max;
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
                ts_scal.ftr_bin = min_bin + random_bin_ofs;

                const Index count = Index(bin <= ts_scal.ftr_bin);

                if constexpr (std::is_same_v<Task, task::classification>) {
                    const Index left_count =
                        sycl::reduce_over_group(item.get_group(), count, plus<Index>());
                    const Index val = (bin <= ts_scal.ftr_bin) ? response_int : -1;
                    Index all_class_count = 0;

                    for (Index class_id = 0; class_id < class_count - 1; ++class_id) {
                        Index total_class_count = sycl::reduce_over_group(item.get_group(),
                                                                          Index(class_id == val),
                                                                          plus<Index>());
                        all_class_count += total_class_count;
                        ts.left_hist[class_id] = total_class_count;
                    }

                    ts_scal.left_count = left_count;

                    ts.left_hist[class_count - 1] = ts_scal.left_count - all_class_count;
                }
                else {
                    const Float val = (bin <= ts_scal.ftr_bin) ? response : Float(0);

                    Float left_count = Float(sycl::reduce_over_group(sbg, count, plus<Index>()));
                    Float sum = sycl::reduce_over_group(sbg, val, plus<Float>());

                    Float mean = sum / left_count;

                    const Float val_s2c =
                        (bin <= ts_scal.ftr_bin) ? (val - mean) * (val - mean) : Float(0);

                    Float sum2cent = sycl::reduce_over_group(sbg, val_s2c, plus<Float>());

                    reduce_hist_over_group(item, local_buf_float_ptr, left_count, mean, sum2cent);

                    ts_scal.left_count = Index(left_count);

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
                                                 bs_scal.left_imp,
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

/// Best splitter kernel
/// It utilizes local memory in order to calculate histograms for each bin of data.
/// The number of simultaneously processed bins is batch_size. And total number of bins is all_bin_count = ftr_count * bin_count.
/// Each batch processing stage is consist of steps:
/// 1. Clean local data
/// 2. Load global bin data and responses
/// 3. Calculate histogram for batch_size bins.
/// 4. Collect left part of histogram in split for all possible splits. Right part of histogram is calculated by node_hist - left_hist.
/// 5. Calculate impurity for batch_size bins.
/// 6. Select best among batch_size bins and update global best bin data.
///
/// This stage is repeated for all batches in the node.
///
/// The corner case is when batch consist of part of bins (the case when all bins of one feature is not fitted to the local memory).
/// In this case it is important to save last processed bin from previous batch in order to calculate left histogram.
/// Also the last bin index should be saved (last_bin_prev_batch, last_bin_index).
///
/// This kernel also can be considered as a non-overlapping window function among all possible splits, which calculates histogram and impurity during
/// the steps and selecting best split in terms of impurity.
///
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

    const Float* const node_imp_list_ptr = imp_list_ptr.imp_list_ptr_;
    Float* const left_child_imp_list_ptr = left_imp_list_ptr.imp_list_ptr_;

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
    const Index bin_count = ctx.max_bin_count_among_ftrs_;
    ONEDAL_ASSERT(bin_count > 1);
    const Index node_in_block_count = node_count;

    auto device = queue.get_device();
    std::int64_t device_local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
    // Compute memory requirements depending on task and device specs
    const std::int64_t bin_local_mem_size =
        2 * hist_prop_count * sizeof(hist_type_t) + sizeof(split_scalar_t);
    const std::int64_t common_local_data_size = 2 * hist_prop_count * sizeof(hist_type_t);
    std::int64_t batch_size = (device_local_mem_size - common_local_data_size) / bin_local_mem_size;
    const Index all_bin_count = selected_ftr_count * bin_count;
    batch_size = std::min<std::int64_t>(batch_size, all_bin_count);
    const std::int64_t local_hist_size = batch_size * hist_prop_count;
    const std::int64_t local_splits_size = batch_size;
    const Index local_size = bk::device_max_wg_size(queue);
    const auto nd_range =
        bk::make_multiple_nd_range_2d({ local_size, node_in_block_count }, { local_size, 1 });
    sycl::event last_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<hist_type_t> local_hist(local_hist_size, cgh);
        local_accessor_rw_t<hist_type_t> buf_hist(local_hist_size, cgh);
        local_accessor_rw_t<split_scalar_t> scalars_buf(local_splits_size, cgh);
        local_accessor_rw_t<hist_type_t> best_split_hist(hist_prop_count, cgh);
        local_accessor_rw_t<hist_type_t> last_bin_prev_batch(hist_prop_count, cgh);
        local_accessor_rw_t<Index> last_bin_index(1, cgh);
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
#if __SYCL_COMPILER_VERSION >= 20230828
            hist_type_t* const best_split_hist_ptr = best_split_hist.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            hist_type_t* const local_hist_ptr = local_hist.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            hist_type_t* const buf_hist_ptr = buf_hist.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            hist_type_t* const last_bin = last_bin_prev_batch.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            split_scalar_t* const scalars_buf_ptr = scalars_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            Index* const last_bin_idx_ptr = last_bin_index.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
#else
            hist_type_t* const best_split_hist_ptr = best_split_hist.get_pointer().get();
            hist_type_t* const local_hist_ptr = local_hist.get_pointer().get();
            hist_type_t* const buf_hist_ptr = buf_hist.get_pointer().get();
            hist_type_t* const last_bin = last_bin_prev_batch.get_pointer().get();
            split_scalar_t* const scalars_buf_ptr = scalars_buf.get_pointer().get();
            Index* const last_bin_idx_ptr = last_bin_index.get_pointer().get();
#endif
            if (local_id == 0) {
                global_bs.init_clear(best_split_hist_ptr, hist_prop_count);
            }
            // Due to local memory limitations need to proccess smaller blocks of features
            for (Index batch_idx = 0; batch_idx < all_bin_count; batch_idx += batch_size) {
                const Index real_bin_count =
                    sycl::min<Index>(all_bin_count - batch_idx, batch_size);
                const Index ftr_ofs = batch_idx / bin_count;
                const Index bin_ofs = Index(batch_idx > 0) * last_bin_idx_ptr[0];
                const Index new_bin_ofs = (real_bin_count + bin_ofs) % bin_count;

                // Clear local memory before use
                for (Index work_bin = local_id; work_bin < real_bin_count; work_bin += local_size) {
                    scalars_buf_ptr[work_bin].clear();
                    for (Index prop_idx = 0; prop_idx < hist_prop_count; ++prop_idx) {
                        local_hist_ptr[work_bin * hist_prop_count + prop_idx] = hist_type_t(0);
                    }
                    for (Index prop_idx = 0; prop_idx < hist_prop_count; ++prop_idx) {
                        buf_hist_ptr[work_bin * hist_prop_count + prop_idx] = hist_type_t(0);
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
                // Calculate histogram
                const Index batch_ftr_count = sycl::max(1, real_bin_count / bin_count) + 2;
                const Index working_items = batch_ftr_count * row_count;
                const Index rows_per_item =
                    working_items / local_size + bool(working_items % local_size);
                for (Index idx = local_id * rows_per_item;
                     idx < (local_id + 1) * rows_per_item && idx < working_items;
                     ++idx) {
                    const Index local_ftr_idx = idx / row_count;
                    const Index global_ftr_idx = ftr_ofs + local_ftr_idx;

                    const Index ts_ftr_id =
                        selected_ftr_list_ptr[node_id * selected_ftr_count + global_ftr_idx];
                    const Index row_idx = idx % row_count;
                    const Index id = tree_order_ptr[row_ofs + row_idx];
                    // Shift the real bin index by bin offset and check if it should
                    // be processed in current batch
                    const Index bin = data_ptr[id * column_count + ts_ftr_id] -
                                      Index(local_ftr_idx == 0) * bin_ofs;

                    const Index cur_hist_pos = (local_ftr_idx * bin_count + bin) * hist_prop_count;
                    if (bin >= 0 && (cur_hist_pos + hist_prop_count) < local_hist_size) {
                        const Float response = response_ptr[id];
                        if constexpr (std::is_same_v<Task, task::classification>) {
                            const Index response_int = static_cast<Index>(response);
                            sycl::atomic_ref<Index,
                                             sycl::memory_order_relaxed,
                                             sycl::memory_scope_work_group,
                                             sycl::access::address_space::local_space>
                                hist_resp(buf_hist[cur_hist_pos + response_int]);
                            hist_resp += 1;
                        }
                        else {
                            sycl::atomic_ref<Float,
                                            sycl::memory_order_relaxed,
                                            sycl::memory_scope_work_group,
                                            sycl::access::address_space::local_space>
                                hist_count(buf_hist[cur_hist_pos + 0]);
                            sycl::atomic_ref<Float,
                                            sycl::memory_order_relaxed,
                                            sycl::memory_scope_work_group,
                                            sycl::access::address_space::local_space>
                                hist_sum(buf_hist[cur_hist_pos + 1]);
                            hist_count += 1;
                            hist_sum += response;
                        }
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
                        // Shift the real bin index by bin offset and check if it should
                        // be processed in current batch
                        const Index bin = data_ptr[id * column_count + ts_ftr_id] -
                                          Index(local_ftr_idx == 0) * bin_ofs;
                        const Index cur_hist_pos =
                            (local_ftr_idx * bin_count + bin) * hist_prop_count;
                        if (bin >= 0 && (cur_hist_pos + hist_prop_count) < local_hist_size) {
                            const Float response = response_ptr[id];
                            hist_type_t* cur_hist = buf_hist_ptr + cur_hist_pos;
                            const Float count = cur_hist[0];
                            const Float resp_sum = cur_hist[1];
                            const Float mean = resp_sum / count;
                            const Float mse = (response - mean) * (response - mean);
                            sycl::atomic_ref<Float,
                                            sycl::memory_order_relaxed,
                                            sycl::memory_scope_work_group,
                                            sycl::access::address_space::local_space>
                                hist_mse(buf_hist[cur_hist_pos + 2]);
                            hist_mse += mse;
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                split_info_t ts;
                // Finilize histograms
                for (Index work_bin = local_id; work_bin < real_bin_count; work_bin += local_size) {
                    const Index local_ftr_idx = (work_bin + bin_ofs) / bin_count;
                    const Index global_ftr_idx = ftr_ofs + local_ftr_idx;
                    const Index rest_bins = bin_count - bin_ofs;
                    const Index cur_bin =
                        (work_bin - Index(local_ftr_idx > 0 && bin_ofs > 0) * rest_bins) %
                        bin_count;
                    const Index cur_hist_pos = work_bin * hist_prop_count;
                    hist_type_t* const cur_hist = local_hist_ptr + cur_hist_pos;
                    hist_type_t* const ftr_hist =
                        buf_hist_ptr + (work_bin - cur_bin) * hist_prop_count;
                    Index left_count = 0;
                    // Load last bin if it is required
                    if (local_ftr_idx == 0 && bin_ofs > 0) {
                        for (Index cls = 0; cls < hist_prop_count; ++cls) {
                            cur_hist[cls] = last_bin[cls];
                            left_count += Index(cur_hist[cls]);
                        }
                    }
                    // Collect all hists on the left side of the current bin
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
                            const Float source_mean = iter_left_hist[1] / iter_left_hist[0];
                            const Float sum_n1n2 = cur_hist[0] + iter_left_hist[0];
                            const Float mul_n1n2 = cur_hist[0] * iter_left_hist[0];
                            const Float delta_scl = mul_n1n2 / sum_n1n2;
                            const Float mean_scl = Float(1) / sum_n1n2;
                            const Float delta = source_mean - cur_hist[1];

                            cur_hist[2] =
                                cur_hist[2] + iter_left_hist[2] + delta * delta * delta_scl;
                            cur_hist[1] =
                                (cur_hist[1] * cur_hist[0] + iter_left_hist[1]) * mean_scl;
                            cur_hist[0] = sum_n1n2;
                        }
                        left_count += Index(cur_hist[0]);
                    }
                    // Save last bin to proccess rest bins in next batch
                    if (work_bin == (real_bin_count - 1)) {
                        for (Index cls = 0; cls < hist_prop_count; ++cls) {
                            last_bin[cls] = cur_hist[cls];
                        }
                        last_bin_idx_ptr[0] = new_bin_ofs;
                    }
                    // Calculate impurity decrease for current bin
                    ts.init(cur_hist, hist_prop_count);
                    split_scalar_t& ts_scal = ts.scalars;
                    ts_scal.ftr_id =
                        selected_ftr_list_ptr[node_id * selected_ftr_count + global_ftr_idx];
                    ts_scal.ftr_bin = cur_bin + Index(local_ftr_idx == 0) * bin_ofs;
                    ts_scal.left_count = left_count;
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
                    if (ts_scal.left_count > min_obs_leaf && ts_scal.right_count > min_obs_leaf) {
                        scalars_buf_ptr[work_bin] = ts.scalars;
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
                    ts.scalars = scalars_buf_ptr[work_item];
                    sp_hlp.choose_best_split(bs, ts, hist_prop_count, min_obs_leaf);
                }
                scalars_buf_ptr[local_id] = bs.scalars;
                // Tree reduction and selecting best among work-group
                for (Index i = local_size / 2; i > 0; i /= 2) {
                    item.barrier(sycl::access::fence_space::local_space);
                    if (local_id < i && (local_id + i) < real_bin_count) {
                        ts.init(local_hist_ptr + (local_id + i) * hist_prop_count, hist_prop_count);
                        ts.scalars = scalars_buf_ptr[local_id + i];
                        sp_hlp.choose_best_split(bs, ts, hist_prop_count, min_obs_leaf);
                        scalars_buf_ptr[local_id] = bs.scalars;
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
                                                 global_bs.scalars.left_imp,
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
