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
//#include "oneapi/dal/algo/decision_forest/common.hpp"
#include "oneapi/dal/algo/decision_forest/train_types.hpp"

#include "oneapi/dal/algo/decision_forest/backend/gpu/helper_rng_engine.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_auxiliary_structs.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_impurity_data.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_service_kernels.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_feature_type.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_model_manager.hpp"

namespace oneapi::dal::decision_forest::backend {

#ifdef ONEDAL_DATA_PARALLEL

// task_types
template <typename Float, typename Index, typename Task = task::by_default>
struct task_types;

template <typename Float, typename Index>
struct task_types<Float, Index, task::classification> {
    using hist_type_t = Index;
};

template <typename Float, typename Index>
struct task_types<Float, Index, task::regression> {
    using hist_type_t = Float;
};

// imp_info
// main impl
template <typename Float,
          typename Bin = std::uint32_t,
          typename Index = std::int32_t,
          typename Task = task::by_default>
class train_kernel_hist_impl {
    using result_t = train_result<Task>;
    using train_service_kernels_t = train_service_kernels<Float, Bin, Index, Task>;
    using impl_const_t = impl_const<Index, Task>;
    using descriptor_t = detail::descriptor_base<Task>;
    using model_manager_t = train_model_manager<Float, Index, Task>;
    using hist_type_t = typename task_types<Float, Index, Task>::hist_type_t;
    using context_t = df_train_context<Float, Index, Task>;
    using imp_data_t = impurity_data<Float, Index, Task>;

public:
    train_kernel_hist_impl(cl::sycl::queue& q) : queue_(q), train_service_kernels_(q) {}
    ~train_kernel_hist_impl() = default;

    result_t operator()(const descriptor_t& desc, const table& data, const table& labels);

private:
    std::uint64_t get_part_hist_required_mem_size(Index selected_ftr_count,
                                                  Index max_bin_count_among_ftrs,
                                                  Index class_count) const;

    void validate_input(const descriptor_t& desc, const table& data, const table& labels) const;

    void init_params(const descriptor_t& desc, const table& data, const table& labels);
    void allocate_buffers();

    dal::backend::primitives::ndarray<Index, 1> gen_features(
        Index node_count,
        const dal::backend::primitives::ndarray<Index, 1>& node_vs_tree_map,
        dal::array<engine_impl>& engines);

    cl::sycl::event compute_initial_histogram(
        const dal::backend::primitives::ndarray<Float, 1>& response,
        const dal::backend::primitives::ndarray<Index, 1>& treeOrder,
        const dal::backend::primitives::ndarray<Index, 1>& nodeList,
        imp_data_t& imp_data_list,
        Index node_count,
        const dal::backend::event_vector& deps);

    cl::sycl::event do_node_split(
        const dal::backend::primitives::ndarray<Index, 1>& node_list,
        const dal::backend::primitives::ndarray<Index, 1>& node_vs_tree_map,
        const imp_data_t& imp_data_list,
        const imp_data_t& left_child_imp_data_list,
        dal::backend::primitives::ndarray<Index, 1>& node_list_new,
        dal::backend::primitives::ndarray<Index, 1>& node_vs_tree_map_new,
        imp_data_t& imp_data_list_new,
        Index node_count,
        Index node_count_new,
        const dal::backend::event_vector& deps);

    cl::sycl::event computeBestSplit(
        const dal::backend::primitives::ndarray<Bin, 2>& data,
        const dal::backend::primitives::ndview<Float, 1>& response,
        const dal::backend::primitives::ndarray<Index, 1>& treeOrder,
        const dal::backend::primitives::ndarray<Index, 1>& selectedFeatures,
        const dal::backend::primitives::ndarray<Index, 1>& binOffsets,
        const imp_data_t& imp_data_list,
        dal::backend::primitives::ndarray<Index, 1>& nodeList,
        imp_data_t& left_child_imp_data_list,
        dal::backend::primitives::ndarray<Float, 1>& nodeImpDecreaseList,
        bool updateImpDecreaseRequired,
        Index nNodes,
        const dal::backend::event_vector& deps = {});

    cl::sycl::event compute_partial_histograms(
        const dal::backend::primitives::ndarray<Bin, 2>& data,
        const dal::backend::primitives::ndview<Float, 1>& response,
        const dal::backend::primitives::ndarray<Index, 1>& treeOrder,
        const dal::backend::primitives::ndarray<Index, 1>& selectedFeatures,
        const dal::backend::primitives::ndarray<Index, 1>& binOffsets,
        const dal::backend::primitives::ndarray<Index, 1>& nodeList,
        const dal::backend::primitives::ndarray<Index, 1>& nodeIndices,
        Index nodeIndicesOffset,
        dal::backend::primitives::ndarray<hist_type_t, 1>& partialHistograms,
        Index nPartialHistograms,
        Index node_count,
        const dal::backend::event_vector& deps = {});

    cl::sycl::event reduce_partial_histograms(
        const dal::backend::primitives::ndarray<hist_type_t, 1>& partialHistograms,
        dal::backend::primitives::ndarray<hist_type_t, 1>& histograms,
        Index nPartialHistograms,
        Index node_count,
        Index reduce_local_size,
        const dal::backend::event_vector& deps = {});

    cl::sycl::event compute_best_split_by_histogram(
        const dal::backend::primitives::ndarray<hist_type_t, 1>& nodesHistograms,
        const dal::backend::primitives::ndarray<Index, 1>& selectedFeatures,
        const dal::backend::primitives::ndarray<Index, 1>& binOffsets,
        const imp_data_t& imp_data_list,
        const dal::backend::primitives::ndarray<Index, 1>& nodeIndices,
        Index nodeIndicesOffset,
        dal::backend::primitives::ndarray<Index, 1>& nodeList,
        imp_data_t& left_child_imp_data_list,
        dal::backend::primitives::ndarray<Float, 1>& nodeImpDecreaseList,
        bool updateImpDecreaseRequired,
        Index node_count,
        const dal::backend::event_vector& deps = {});

    cl::sycl::event compute_best_split_single_pass(
        const dal::backend::primitives::ndarray<Bin, 2>& data,
        const dal::backend::primitives::ndview<Float, 1>& response,
        const dal::backend::primitives::ndarray<Index, 1>& treeOrder,
        const dal::backend::primitives::ndarray<Index, 1>& selectedFeatures,
        const dal::backend::primitives::ndarray<Index, 1>& binOffsets,
        const imp_data_t& imp_data_list,
        const dal::backend::primitives::ndarray<Index, 1>& nodeIndices,
        Index nodeIndicesOffset,
        dal::backend::primitives::ndarray<Index, 1>& nodeList,
        imp_data_t& left_child_imp_data_list,
        dal::backend::primitives::ndarray<Float, 1>& nodeImpDecreaseList,
        bool updateImpDecreaseRequired,
        Index node_count,
        const dal::backend::event_vector& deps = {});

    Float compute_oob_error(const model_manager_t& model_manager,
                            const dal::backend::primitives::ndarray<Float, 1>& data_host,
                            const dal::backend::primitives::ndarray<Float, 1>& response_host,
                            const dal::backend::primitives::ndarray<Index, 1>& oob_row_list,
                            dal::backend::primitives::ndarray<hist_type_t, 1>& oob_per_obs_list,
                            Index tree_idx,
                            Index indicesOffset,
                            Index n,
                            const dal::backend::event_vector& deps = {});
    Float compute_oob_error_perm(
        const model_manager_t& model_manager,
        const dal::backend::primitives::ndarray<Float, 1>& data_host,
        const dal::backend::primitives::ndarray<Float, 1>& response_host,
        const dal::backend::primitives::ndarray<Index, 1>& oob_row_list,
        const dal::backend::primitives::ndarray<Index, 1>& permutation_host,
        Index tree_idx,
        Index indicesOffset,
        Index n,
        Index column_idx,
        const dal::backend::event_vector& deps = {});

    cl::sycl::event compute_results(
        const model_manager_t& model_manager,
        const dal::backend::primitives::ndarray<Float, 1>& data_host,
        const dal::backend::primitives::ndarray<Float, 1>& response_host,
        const dal::backend::primitives::ndarray<Index, 1>& oob_row_list,
        const dal::backend::primitives::ndarray<Index, 1>& oobRowsNumList,
        dal::backend::primitives::ndarray<hist_type_t, 1>& oob_per_obs_list,
        dal::backend::primitives::ndarray<Float, 1>& var_imp,
        dal::backend::primitives::ndarray<Float, 1>& var_imp_variance,
        const dal::array<engine_impl>& engine_arr,
        Index tree_idx,
        Index tree_in_block,
        Index built_tree_count,
        const dal::backend::event_vector& deps = {});

    cl::sycl::event finalize_oob_error(
        const dal::backend::primitives::ndarray<Float, 1>& response_host,
        dal::backend::primitives::ndarray<hist_type_t, 1>& oob_per_obs_list,
        dal::backend::primitives::ndarray<Float, 1>& res_oob_err,
        dal::backend::primitives::ndarray<Float, 1>& res_oob_err_obs,
        const dal::backend::event_vector& deps = {});

    cl::sycl::event finalize_var_imp(dal::backend::primitives::ndarray<Float, 1>& var_imp,
                                     dal::backend::primitives::ndarray<Float, 1>& var_imp_variance,
                                     const dal::backend::event_vector& deps = {});

private:
    context_t ctx_;
    cl::sycl::queue queue_;

    train_service_kernels_t train_service_kernels_;

    dal::backend::primitives::ndarray<Bin, 2> full_data_nd_;
    dal::backend::primitives::ndarray<Index, 1> ftr_bin_offsets_nd_;
    std::vector<dal::backend::primitives::ndarray<Float, 1>> bin_borders_host_;
    dal::backend::primitives::ndarray<Float, 1> response_nd_;
    dal::backend::primitives::ndarray<Float, 1> response_host_;
    dal::backend::primitives::ndarray<Float, 1> data_host_;

    dal::backend::primitives::ndarray<Index, 1> selected_rows_host_;
    dal::backend::primitives::ndarray<Index, 1> tree_order_lev_;
    dal::backend::primitives::ndarray<Index, 1> tree_order_lev_buf_;

    dal::backend::primitives::ndarray<Float, 1> node_imp_decr_list_;

    dal::backend::primitives::ndarray<hist_type_t, 1> oob_per_obs_list_;
    dal::backend::primitives::ndarray<Float, 1> var_imp_variance_host_;

    Index class_count_ = 0;
    Index row_count_ = 0;
    Index column_count_ = 0;
    Index total_bins_ = 0;
    Index tree_count_ = 0;

    Index selected_ftr_count_ = 0;
    Index selected_row_count_ = 0;
    Index min_observations_in_leaf_node_ = 0;
    Index max_tree_depth_ = 0;

    Float impurity_threshold_;

    bool mda_required_ = false;
    bool mda_scaled_required_ = false;
    bool mdi_required_ = false;
    bool oob_required_ = false;
    bool oob_err_required_ = false;
    bool oob_err_obs_required_ = false;
    bool bootstrap_ = false;

    Index total_bin_count_ = 0;
    Index max_bin_count_among_ftrs_ = 0;

    Index tree_in_block_ = 0;
    Index preferable_local_size_for_part_hist_kernel_ = 0;
    Index max_part_hist_cumulative_size_ = 0;
    Index oob_prop_count_ = 0;

    // part of free global mem which can be used for processing block of tree
    static constexpr inline double global_mem_fraction_for_tree_block_ = 0.6;
    // part of free global mem which can be used for partial histograms
    static constexpr inline double global_mem_fraction_for_part_hist_ = 0.2;

    // 1 Gb it showed better efficiency than using just platform info.maxMemAllocSize
    static constexpr inline std::uint64_t max_mem_alloc_size_for_algo_ = 1073741824;
    static constexpr inline Index _minRowsBlocksForMaxPartHistNum = 16384;
    static constexpr inline Index _minRowsBlocksForOneHist = 128;
    static constexpr inline Index _maxLocalHistograms = 256;
    static constexpr inline Index reduce_local_size_part_hist_ = 64;

    static constexpr inline Index min_preferable_local_size_for_part_hist_kernel_ = 32;

    // update _nNode naming
    static constexpr inline Index _nNodesGroups =
        3; // all nodes are split on groups (big, medium, small)
    static constexpr inline Index _nodeGroupProps =
        2; // each nodes Group contains props: numOfNodes, maxNumOfBlocks

    static constexpr inline Index preferable_group_size_ = 256;
    static constexpr inline Index preferable_sbg_size_ = 16;
    static constexpr inline Index max_local_block_count_ = 1024;
};

#endif

} // namespace oneapi::dal::decision_forest::backend
