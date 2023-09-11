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

#pragma once

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/backend/primitives/rng/rng_engine_collection.hpp"
#include "oneapi/dal/algo/decision_forest/train_types.hpp"

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_misc_structs.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_impurity_data.hpp"

namespace oneapi::dal::decision_forest::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = bk::primitives;

template <typename Float,
          typename Bin = std::uint32_t,
          typename Index = std::int32_t,
          typename Task = task::by_default,
          bool use_private_mem = true>
class train_splitter_impl {
    using result_t = train_result<Task>;
    using impl_const_t = impl_const<Index, Task>;
    using descriptor_t = detail::descriptor_base<Task>;
    using context_t = train_context<Float, Index, Task>;
    using imp_data_t = impurity_data<Float, Index, Task>;
    using msg = de::error_messages;
    using hist_type_t = typename task_types<Float, Index, Task>::hist_type_t;

public:
    train_splitter_impl() = default;
    ~train_splitter_impl() = default;

    /// Computing the best split for the `node_count` nodes using `selected_ftr_list`
    /// for splitting and choosing the best split in terms of impurity.
    ///
    /// @param[in] queue                    SYCL parallel queue
    /// @param[in] ctx                      a training context structure for the GPU backend
    /// @param[in] node_hist_list           a pre-calculated histogram for each node
    /// @param[in] selected_ftr_list        a subset of feature indices selected for each node
    /// @param[in] bin_offset_list          an array of offsets for each bin
    /// @param[in] imp_data_list            an array of impurity data for each node
    /// @param[in] node_ind_list            an array of node indices
    /// @param[in] node_ind_ofs             global offset for node indices
    /// @param[in] node_list                a node structure containing split information
    /// @param[in] left_child_imp_data_list an array of left-child impurity values
    /// @param[in] node_imp_dec_list        an array of node impurity decrease values
    /// @param[in] update_imp_dec_required  boolean indicator to update impurity decrease structure
    /// @param[in] node_count               number of the nodes to compute in the current step
    /// @param[in] deps                     a set of SYCL events this method depends on
    static sycl::event compute_best_split_by_histogram(
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
        const bk::event_vector& deps = {});

    /// Computing random split for `node_count` nodes using `selected_ftr_list`
    /// and `random_bins_com` values for splitting. Computes best split among randomly
    /// selected thresholds for each node.
    ///
    /// @param[in] queue                    SYCL parallel queue
    /// @param[in] ctx                      a training context structure for the GPU backend
    /// @param[in] node_hist_list           a pre-calculated histogram for each node
    /// @param[in] selected_ftr_list        a subset of feature indices selected for each node
    /// @param[in] random_bins_com          a set of random (uniformly distributed) thresholds for each selected feature scaled at [0.0, 1.0]
    /// @param[in] bin_offset_list          an array of offsets for each bin
    /// @param[in] imp_data_list            an array of impurity data for each node
    /// @param[in] node_ind_list            an array of node indices
    /// @param[in] node_ind_ofs             global offset for node indices
    /// @param[in] node_list                a node structure containing split information
    /// @param[in] left_child_imp_data_list an array of left-child impurity values
    /// @param[in] node_imp_dec_list        an array of node impurity decrease values
    /// @param[in] update_imp_dec_required  boolean indicator to update impurity decrease structure
    /// @param[in] node_count               number of the nodes to compute in the current step
    /// @param[in] deps                     a set of SYCL events this method depends on
    static sycl::event compute_random_split_by_histogram(
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
        const bk::event_vector& deps = {});
};

#endif

} // namespace oneapi::dal::decision_forest::backend
