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

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_cls_hist_aux_props.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/helper_cls_tree_level_build.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/helper_rng_engine.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/helper_feature_type.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/helper_tree_build.hpp"

namespace oneapi::dal::decision_forest::backend {

#ifdef ONEDAL_DATA_PARALLEL

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

template <typename Float, typename Bin = std::uint32_t, typename Index = std::int32_t>
class df_cls_train_hist_impl {
    using tree_level_build_helper_t = helper_cls_tree_level_build<Float, Bin, Index>;
    using impl_const_t = impl_const<task::classification, Index>;
    using result_t = train_result<task::classification>;
    using model_builder_t = model_builder_interop<Float, Index, task::classification>;

public:
    df_cls_train_hist_impl(cl::sycl::queue& q) : queue_(q), tree_level_build_helper_(q) {}
    ~df_cls_train_hist_impl() = default;

    // ?? return train_result
    result_t operator()(const detail::descriptor_base<task::classification>& desc,
                        const table& data,
                        const table& labels);

private:
    std::uint64_t get_part_hist_required_mem_size(Index selected_ftr_count,
                                                  Index max_bin_count_among_ftrs,
                                                  Index class_count) const;

    void validate_input(const detail::descriptor_base<task::classification>& desc,
                        const table& data,
                        const table& labels) const;

    void init_params(const detail::descriptor_base<task::classification>& desc,
                     const table& data,
                     const table& labels);
    void allocate_buffers();

    dal::backend::primitives::ndarray<Index, 1> gen_features(
        Index node_count,
        const dal::backend::primitives::ndarray<Index, 1>& node_vs_tree_map,
        dal::array<engine_impl>& engines);

    cl::sycl::event compute_class_histogram(
        const dal::backend::primitives::ndarray<Float, 1>& response,
        const dal::backend::primitives::ndarray<Index, 1>& treeOrder,
        const dal::backend::primitives::ndarray<Index, 1>& nodeList,
        const dal::backend::primitives::ndarray<Float, 1>& impList,
        dal::backend::primitives::ndarray<Index, 1>& class_histogram,
        Index node_count,
        const dal::backend::event_vector& deps = {});

    cl::sycl::event do_node_split(
        const dal::backend::primitives::ndarray<Index, 1>& node_list,
        const dal::backend::primitives::ndarray<Index, 1>& class_hist_list,
        const dal::backend::primitives::ndarray<Float, 1>& left_child_imp_list,
        const dal::backend::primitives::ndarray<Index, 1>& left_child_class_hist_list,
        const dal::backend::primitives::ndarray<Index, 1>& node_vs_tree_map,
        dal::backend::primitives::ndarray<Index, 1>& node_list_new,
        dal::backend::primitives::ndarray<Float, 1>& imp_list_new,
        dal::backend::primitives::ndarray<Index, 1>& class_hist_list_new,
        dal::backend::primitives::ndarray<Index, 1>& node_vs_tree_map_new,
        Index node_count,
        Index node_count_new,
        const dal::backend::event_vector& deps);

    cl::sycl::event computeBestSplit(
        const dal::backend::primitives::ndarray<Bin, 2>& data,
        const dal::backend::primitives::ndview<Float, 1>& response,
        const dal::backend::primitives::ndarray<Index, 1>& treeOrder,
        const dal::backend::primitives::ndarray<Index, 1>& selectedFeatures,
        const dal::backend::primitives::ndarray<Index, 1>& binOffsets,
        const dal::backend::primitives::ndarray<Float, 1>& impList,
        const dal::backend::primitives::ndarray<Index, 1>& class_hist_list,
        dal::backend::primitives::ndarray<Index, 1>& nodeList,
        dal::backend::primitives::ndarray<Float, 1>& left_child_imp_list,
        dal::backend::primitives::ndarray<Index, 1>& left_child_class_hist_list,
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
        dal::backend::primitives::ndarray<Index, 1>& partialHistograms,
        Index nPartialHistograms,
        Index node_count,
        const dal::backend::event_vector& deps = {});

    cl::sycl::event reduce_partial_histograms(
        const dal::backend::primitives::ndarray<Index, 1>& partialHistograms,
        dal::backend::primitives::ndarray<Index, 1>& histograms,
        Index nPartialHistograms,
        Index node_count,
        Index reduce_local_size,
        const dal::backend::event_vector& deps = {});

    cl::sycl::event compute_best_split_by_histogram(
        const dal::backend::primitives::ndarray<Index, 1>& nodesHistograms,
        const dal::backend::primitives::ndarray<Index, 1>& selectedFeatures,
        const dal::backend::primitives::ndarray<Index, 1>& binOffsets,
        const dal::backend::primitives::ndarray<Float, 1>& impList,
        const dal::backend::primitives::ndarray<Index, 1>& class_hist_list,
        const dal::backend::primitives::ndarray<Index, 1>& nodeIndices,
        Index nodeIndicesOffset,
        dal::backend::primitives::ndarray<Index, 1>& nodeList,
        dal::backend::primitives::ndarray<Float, 1>& left_child_imp_list,
        dal::backend::primitives::ndarray<Index, 1>& left_child_class_hist_list,
        dal::backend::primitives::ndarray<Float, 1>& nodeImpDecreaseList,
        bool updateImpDecreaseRequired,
        Index node_count,
        const dal::backend::event_vector& deps);

    cl::sycl::event compute_best_split_single_pass(
        const dal::backend::primitives::ndarray<Bin, 2>& data,
        const dal::backend::primitives::ndview<Float, 1>& response,
        const dal::backend::primitives::ndarray<Index, 1>& treeOrder,
        const dal::backend::primitives::ndarray<Index, 1>& selectedFeatures,
        const dal::backend::primitives::ndarray<Index, 1>& binOffsets,
        const dal::backend::primitives::ndarray<Float, 1>& impList,
        const dal::backend::primitives::ndarray<Index, 1>& class_hist_list,
        const dal::backend::primitives::ndarray<Index, 1>& nodeIndices,
        Index nodeIndicesOffset,
        dal::backend::primitives::ndarray<Index, 1>& nodeList,
        dal::backend::primitives::ndarray<Float, 1>& left_child_imp_list,
        dal::backend::primitives::ndarray<Index, 1>& left_child_class_hist_list,
        dal::backend::primitives::ndarray<Float, 1>& nodeImpDecreaseList,
        bool updateImpDecreaseRequired,
        Index node_count,
        const dal::backend::event_vector& deps);

    Float compute_oob_error(const model_builder_t& model_builder,
                            const dal::backend::primitives::ndarray<Float, 1>& data_host,
                            const dal::backend::primitives::ndarray<Float, 1>& response_host,
                            const dal::backend::primitives::ndarray<Index, 1>& oob_row_list,
                            dal::backend::primitives::ndarray<Index, 1>& oobBuf,
                            Index tree_idx,
                            Index indicesOffset,
                            Index n,
                            const dal::backend::event_vector& deps = {});
    Float compute_oob_error_perm(
        const model_builder_t& model_builder,
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
        const model_builder_t& model_builder,
        const dal::backend::primitives::ndarray<Float, 1>& data_host,
        const dal::backend::primitives::ndarray<Float, 1>& response_host,
        const dal::backend::primitives::ndarray<Index, 1>& oob_row_list,
        const dal::backend::primitives::ndarray<Index, 1>& oobRowsNumList,
        dal::backend::primitives::ndarray<Index, 1>& oobBuf,
        dal::backend::primitives::ndarray<Float, 1>& var_imp,
        dal::backend::primitives::ndarray<Float, 1>& var_imp_variance,
        const dal::array<engine_impl>& engine_arr,
        Index tree_idx,
        Index tree_in_block,
        Index built_tree_count,
        const dal::backend::event_vector& deps = {});

    cl::sycl::event finalize_oob_error(
        const dal::backend::primitives::ndarray<Float, 1>& response_host,
        dal::backend::primitives::ndarray<Index, 1>& oob_per_obs_list,
        dal::backend::primitives::ndarray<Float, 1>& res_oob_err,
        dal::backend::primitives::ndarray<Float, 1>& res_oob_err_obs,
        const dal::backend::event_vector& deps = {});
    cl::sycl::event finalize_var_imp(dal::backend::primitives::ndarray<Float, 1>& var_imp,
                                     dal::backend::primitives::ndarray<Float, 1>& var_imp_variance,
                                     const dal::backend::event_vector& deps = {});

private:
    cl::sycl::queue queue_;

    /// for dbg
    //dal::backend::primitives::ndarray<Index, 1> buf_;

    tree_level_build_helper_t tree_level_build_helper_;
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

    dal::backend::primitives::ndarray<Index, 1> oob_per_obs_list_;
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

    static constexpr inline double global_mem_fraction_for_tree_block_ =
        0.6; // part of free global mem which can be used for processing block of tree
    static constexpr inline double global_mem_fraction_for_part_hist_ =
        0.2; // part of free global mem which can be used for partial histograms

    static constexpr inline std::uint64_t max_mem_alloc_size_for_algo_ =
        1073741824; // 1 Gb it showed better efficiency than using just platform info.maxMemAllocSize
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
