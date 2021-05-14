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

#include "oneapi/dal/algo/decision_forest/backend/gpu/df_cls_train_hist_impl.hpp"

namespace oneapi::dal::decision_forest::backend {

namespace de = dal::detail;
namespace be = dal::backend;
namespace pr = dal::backend::primitives;

using result_t = train_result<task::classification>;
using descriptor_t = detail::descriptor_base<task::classification>;
using alloc = cl::sycl::usm::alloc;
using address = cl::sycl::access::address_space;

using cl::sycl::ONEAPI::broadcast;
using cl::sycl::ONEAPI::reduce;
using cl::sycl::ONEAPI::plus;
using cl::sycl::ONEAPI::minimum;
using cl::sycl::ONEAPI::maximum;
using cl::sycl::ONEAPI::exclusive_scan;

template <typename T>
inline T atomic_global_add(T* ptr, T operand) {
    return cl::sycl::atomic_fetch_add<T, address::global_space>(
        { cl::sycl::multi_ptr<T, address::global_space>{ ptr } },
        operand);
}

template <typename Float, typename Bin, typename Index>
std::uint64_t df_cls_train_hist_impl<Float, Bin, Index>::get_part_hist_required_mem_size(
    Index selected_ftr_count,
    Index max_bin_count_among_ftrs,
    Index class_count) const {
    // mul overflow for nSelectedFeatures * max_bin_count_among_ftrs_ and for nHistBins * _nHistProps were checked before kernel call in compute
    const std::uint64_t hist_bin_count = selected_ftr_count * max_bin_count_among_ftrs;
    return hist_bin_count * class_count;
}

template <typename Float, typename Bin, typename Index>
void df_cls_train_hist_impl<Float, Bin, Index>::validate_input(const descriptor_t& desc,
                                                               const table& data,
                                                               const table& labels) const {
    //_P("vld inp ");
    //_P("cls %ld, rc %ld, clm %ld, tr_c %ld, ", desc. get_class_count(), data.get_row_count(), data.get_column_count(), desc.get_tree_count());
    if (data.get_row_count() > de::limits<Index>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_range_of_rows());
    }
    if (data.get_column_count() > de::limits<Index>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_range_of_columns());
    }
    if (desc.get_tree_count() > de::limits<Index>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_number_of_trees());
    }
    /*
    DAAL_CHECK_EX((par.nClasses <= _int32max), ErrorIncorrectParameter, ParameterName, nClassesStr());
    DAAL_CHECK_EX((par.minObservationsInLeafNode <= _int32max), ErrorIncorrectParameter, ParameterName, minObservationsInLeafNodeStr());
    DAAL_CHECK_EX((par.featuresPerNode <= _int32max), ErrorIncorrectParameter, ParameterName, featuresPerNodeStr());
    DAAL_CHECK_EX((par.maxBins <= _int32max), ErrorIncorrectParameter, ParameterName, maxBinsStr());
    DAAL_CHECK_EX((par.minBinSize <= _int32max), ErrorIncorrectParameter, ParameterName, minBinSizeStr());
    DAAL_CHECK_EX((par.nTrees <= _int32max), ErrorIncorrectParameter, ParameterName, nTreesStr());
    */
    //_P("vld inp done");
}

template <typename Float, typename Bin, typename Index>
void df_cls_train_hist_impl<Float, Bin, Index>::init_params(const descriptor_t& desc,
                                                            const table& data,
                                                            const table& responses) {
    //_P("init_params");
    class_count_ = de::integral_cast<Index>(desc.get_class_count());
    row_count_ = de::integral_cast<Index>(data.get_row_count());
    column_count_ = de::integral_cast<Index>(data.get_column_count());

    tree_count_ = de::integral_cast<Index>(desc.get_tree_count());

    bootstrap_ = desc.get_bootstrap();
    max_tree_depth_ = desc.get_max_tree_depth();

    selected_ftr_count_ =
        desc.get_features_per_node() ? desc.get_features_per_node() : std::sqrt(column_count_);
    selected_row_count_ = desc.get_observations_per_tree_fraction() * row_count_;

    min_observations_in_leaf_node_ = desc.get_min_observations_in_leaf_node();
    impurity_threshold_ = desc.get_impurity_threshold();

    if (0 >= selected_row_count_) {
        //throw domain_error(dal::detail::error_messages::invalid_range_of_columns());
        //DAAL_CHECK_EX((_nSelectedRows > 0), ErrorIncorrectParameter, ParameterName, observationsPerTreeFractionStr());
    }

    preferable_local_size_for_part_hist_kernel_ = preferable_group_size_;
    while (preferable_local_size_for_part_hist_kernel_ >
           std::max(selected_ftr_count_, min_preferable_local_size_for_part_hist_kernel_)) {
        preferable_local_size_for_part_hist_kernel_ >>= 1;
    }

    auto vimp = desc.get_variable_importance_mode();
    mda_required_ =
        (variable_importance_mode::mda_raw == vimp || variable_importance_mode::mda_scaled == vimp);
    mdi_required_ = variable_importance_mode::mdi == vimp;
    mda_scaled_required_ = (variable_importance_mode::mda_scaled == vimp);

    auto emm = desc.get_error_metric_mode();
    oob_required_ = (check_mask_flag(emm, error_metric_mode::out_of_bag_error) ||
                     check_mask_flag(emm, error_metric_mode::out_of_bag_error_per_observation) ||
                     mda_required_);
    oob_err_required_ = check_mask_flag(emm, error_metric_mode::out_of_bag_error);
    oob_err_obs_required_ =
        check_mask_flag(emm, error_metric_mode::out_of_bag_error_per_observation);

    // init ftr -> bins map and related params
    indexed_features<Float, Bin, Index> ind_ftrs(queue_,
                                                 desc.get_min_bin_size(),
                                                 desc.get_max_bins());
    ind_ftrs(data).wait_and_throw();

    total_bin_count_ = ind_ftrs.get_total_bin_count();
    full_data_nd_ = ind_ftrs.get_full_data();
    ftr_bin_offsets_nd_ = ind_ftrs.get_bin_offsets();

    bin_borders_host_.resize(column_count_);
    for (Index i = 0; i < column_count_; i++) {
        ////_P("i %d, arr_count %d, bin_count %d", i, static_cast<std::int32_t>(ind_ftrs.get_bin_borders(i).get_count()), ind_ftrs.get_bin_count(i));
        //ONEDAL_ASSERT(ind_ftrs.get_bin_borders(i).get_count() == static_cast<std::int64_t>(ind_ftrs.get_bin_count(i)));
        bin_borders_host_[i] = ind_ftrs.get_bin_borders(i).to_host(queue_);
    }

    data_host_ =
        pr::flatten_table_1d<Float, row_accessor>(queue_, data, alloc::device).to_host(queue_);

    response_nd_ = pr::flatten_table_1d<Float, row_accessor>(queue_, responses, alloc::device);

    response_host_ = response_nd_.to_host(queue_);

    // calculating the maximal number of bins for feature among all features
    max_bin_count_among_ftrs_ = 0;
    for (Index clmn_idx = 0; clmn_idx < column_count_; clmn_idx++) {
        auto ftr_bins = ind_ftrs.get_bin_count(clmn_idx);
        max_bin_count_among_ftrs_ =
            (max_bin_count_among_ftrs_ < ftr_bins) ? ftr_bins : max_bin_count_among_ftrs_;
    }

    // DAAL_CHECK_STATUS_VAR(_treeLevelBuildHelper.init(buildOptions.c_str(), impl_const_t::node_prop_count_));

    // define number of trees which can be build in parallel
    const std::uint64_t device_global_mem_size =
        queue_.get_device().get_info<cl::sycl::info::device::global_mem_size>();
    const std::uint64_t device_max_mem_alloc_size =
        queue_.get_device().get_info<cl::sycl::info::device::max_mem_alloc_size>();

    const auto part_hist_size = get_part_hist_required_mem_size(selected_ftr_count_,
                                                                max_bin_count_among_ftrs_,
                                                                class_count_);
    const auto max_mem_alloc_size =
        std::min(device_max_mem_alloc_size, std::uint64_t(max_mem_alloc_size_for_algo_));

    std::uint64_t used_mem_size =
        sizeof(Float) * row_count_ * (column_count_ + 1); // input table size + response
    used_mem_size += ind_ftrs.get_required_mem_size(row_count_, column_count_, desc.get_max_bins());
    used_mem_size += oob_required_ ? sizeof(Float) * row_count_ * class_count_ : 0;
    used_mem_size += part_hist_size; // space at least for one part hist

    std::uint64_t available_global_mem_size =
        device_global_mem_size > used_mem_size ? device_global_mem_size - used_mem_size : 0;

    std::uint64_t available_mem_size_for_tree_block =
        std::min(max_mem_alloc_size,
                 static_cast<std::uint64_t>(available_global_mem_size *
                                            global_mem_fraction_for_tree_block_));

    std::uint64_t required_mem_size_for_one_tree =
        oob_required_ ? tree_level_build_helper_.get_oob_rows_required_mem_size(
                            row_count_,
                            1 /* for 1 tree */,
                            desc.get_observations_per_tree_fraction())
                      : 0;

    required_mem_size_for_one_tree += sizeof(Index) * selected_row_count_ *
                                      2; // main tree order and auxilliary one used for partitioning

    tree_in_block_ = de::integral_cast<Index>(available_mem_size_for_tree_block /
                                              required_mem_size_for_one_tree);

    if (tree_in_block_ <= 0) {
        // not enough memory even for one tree
        //throw domain_error(dal::detail::error_messages::invalid_range_of_columns());
        //return services::Status(services::ErrorMemoryAllocationFailed);
    }

    tree_in_block_ = std::min(tree_count_, tree_in_block_);

    available_global_mem_size =
        available_global_mem_size > (tree_in_block_ * required_mem_size_for_one_tree)
            ? available_global_mem_size - (tree_in_block_ * required_mem_size_for_one_tree)
            : 0;
    // size for one part hist was already reserved, add some more if there is available mem
    max_part_hist_cumulative_size_ = std::min(
        max_mem_alloc_size,
        static_cast<std::uint64_t>(part_hist_size +
                                   available_global_mem_size * global_mem_fraction_for_part_hist_));

    //_P("init_params done");
}

template <typename Float, typename Bin, typename Index>
void df_cls_train_hist_impl<Float, Bin, Index>::allocate_buffers() {
    // DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, selected_row_count_, tree_in_block_);
    //selected_rows_host_ = pr::ndarray<Index, 1>::empty(queue_, {selected_row_count_ * tree_in_block_}, alloc::device);
    selected_rows_host_ = pr::ndarray<Index, 1>::empty({ selected_row_count_ * tree_in_block_ });

    tree_order_lev_ = pr::ndarray<Index, 1>::empty(queue_,
                                                   { selected_row_count_ * tree_in_block_ },
                                                   alloc::device);
    tree_order_lev_buf_ = pr::ndarray<Index, 1>::empty(queue_,
                                                       { selected_row_count_ * tree_in_block_ },
                                                       alloc::device);

    if (oob_required_) {
        // oobBufferPerObs contains nClassed counters for all out of bag observations for all trees
        //DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, _nRows, _nClasses);
        auto [oob_per_obs_list, event] =
            pr::ndarray<Index, 1>::zeros(queue_, { row_count_ * class_count_ }, alloc::device);
        oob_per_obs_list_ = oob_per_obs_list;
        event.wait_and_throw();
    }

    /* blocks for MDA scaled error calculation */
    if (mda_scaled_required_) {
        var_imp_variance_host_ = pr::ndarray<Float, 1>::zeros({ column_count_ });
    }
}

template <typename Float, typename Bin, typename Index>
pr::ndarray<Index, 1> df_cls_train_hist_impl<Float, Bin, Index>::gen_features(
    Index node_count,
    const pr::ndarray<Index, 1>& node_vs_tree_map,
    dal::array<engine_impl>& engine_arr) {
    //DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, (node_count + 1), nSelectedFeatures);
    auto selectedFeaturesHost = pr::ndarray<Index, 1>::empty(
        { (node_count + 1) *
          selected_ftr_count_ }); // first part is used features indices, +1 - part for generator
    auto selectedFeaturesCom =
        pr::ndarray<Index, 1>::empty(queue_, { node_count * selected_ftr_count_ }, alloc::device);

    auto selectedFeaturesHost_ptr_orig = selectedFeaturesHost.get_mutable_data();
    std::int32_t* selectedFeaturesHost_ptr =
        reinterpret_cast<std::int32_t*>(selectedFeaturesHost_ptr_orig);

    auto nodeVsTreeMap_host = node_vs_tree_map.to_host(queue_);

    engine_impl* engines = engine_arr.get_mutable_data();

    if (selected_ftr_count_ != column_count_) {
        rng<std::int32_t> rn_gen;
        auto treeMap_ptr = nodeVsTreeMap_host.get_mutable_data();

        for (Index node = 0; node < node_count; node++) {
            rn_gen.uniform_without_replacement(
                selected_ftr_count_,
                selectedFeaturesHost_ptr + node * selected_ftr_count_,
                selectedFeaturesHost_ptr + (node + 1) * selected_ftr_count_,
                engines[treeMap_ptr[node]].get_state(),
                0,
                column_count_);
        }
    }
    else {
        for (Index node = 0; node < node_count; node++) {
            for (Index i = 0; i < selected_ftr_count_; i++) {
                selectedFeaturesHost_ptr[node * selected_ftr_count_ + i] = i;
            }
        }
    }

    // TODO migrate tu return tuple;
    selectedFeaturesCom
        .assign(queue_, selectedFeaturesHost_ptr_orig, selectedFeaturesCom.get_count())
        .wait_and_throw();
    return selectedFeaturesCom;
}

template <typename Float, typename Bin, typename Index>
class compute_class_histogram_krn;

template <typename Float, typename Bin, typename Index>
cl::sycl::event df_cls_train_hist_impl<Float, Bin, Index>::compute_class_histogram(
    const pr::ndarray<Float, 1>& response,
    const pr::ndarray<Index, 1>& treeOrder,
    const pr::ndarray<Index, 1>& nodeList,
    const pr::ndarray<Float, 1>& imp_list,
    pr::ndarray<Index, 1>& class_histogram,
    Index node_count,
    const be::event_vector& deps) {
    //_P("compute_class_histogram 01");
    //DAAL_ITTNOTIFY_SCOPED_TASK(compute.compute_class_histogram);

    ONEDAL_ASSERT(nodeList.get_count() == node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(class_histogram.get_count() == node_count * class_count_);

    auto fill_event = class_histogram.fill(queue_, 0, deps);

    const Float* response_ptr = response.get_data();
    const Index* tree_order_ptr = treeOrder.get_data();
    Index* node_list_ptr = nodeList.get_mutable_data();
    Float* imp_list_ptr = imp_list.get_mutable_data();

    const Index nNodeProp = impl_const_t::node_prop_count_; // num of split attributes for node
    const Index node_imp_prop_count =
        impl_const_t::node_imp_prop_count_; // num of split attributes for node
    const Index class_count = class_count_;

    Index* class_histogram_ptr = class_histogram.get_mutable_data();

    auto local_size = preferable_group_size_;
    const cl::sycl::nd_range<2> nd_range =
        be::make_multiple_nd_range_2d({ local_size, node_count }, { local_size, 1 });

    auto event = queue_.submit([&](cl::sycl::handler& cgh) {
        cgh.depends_on(fill_event);
        cgh.parallel_for<class compute_class_histogram_krn<Float, Bin, Index>>(
            nd_range, [=](cl::sycl::nd_item<2> item) {
            const Index node_id = item.get_global_id()[1];
            const Index local_id = item.get_local_id()[0];
            const Index local_size = item.get_local_range()[0];

            Index* node_ptr = node_list_ptr + node_id * nNodeProp;
            Float* node_imp_ptr = imp_list_ptr + node_id * node_imp_prop_count;

            const Index rows_offset = node_ptr[0];
            const Index row_count = node_ptr[1];

            const Index elem_count = row_count / local_size + bool(row_count % local_size);

            Index ind_start = local_id * elem_count;
            Index ind_end =
                cl::sycl::min(static_cast<Index>((local_id + 1) * elem_count), row_count);

            Index* histogram_ptr = class_histogram_ptr + node_id * class_count;

            constexpr Index buff_size =
                16; // class_count, find a way to specialize it in kernel build time
            Index private_histogram[buff_size] = { 0 };

            for (Index i = ind_start; i < ind_end; i++) {
                Index id = tree_order_ptr[rows_offset + i];
                Index classId = static_cast<Index>(response_ptr[id]);

                private_histogram[classId] += 1;
            }

            for (Index cls_idx = 0; cls_idx < class_count; cls_idx++) {
                atomic_global_add(histogram_ptr + cls_idx, private_histogram[cls_idx]);
            }

            item.barrier(cl::sycl::access::fence_space::local_space);

            Float imp = Float(1);
            Float div = Float(1) / (row_count * row_count);
            Index max_cls_count = 0;
            Index win_cls = 0;
            Index cls_count = 0;

            for (Index cls_idx = 0; cls_idx < class_count; cls_idx++) {
                cls_count = histogram_ptr[cls_idx];
                imp -= Float(cls_count) * (cls_count)*div;

                if (cls_count > max_cls_count) {
                    max_cls_count = cls_count;
                    win_cls = cls_idx;
                }
            }

            node_ptr[5] = win_cls;
            node_imp_ptr[0] = cl::sycl::max(imp, Float(0));
        });
    });

    return event;
}
template <typename Float, typename Bin, typename Index>
cl::sycl::event df_cls_train_hist_impl<Float, Bin, Index>::computeBestSplit(
    const pr::ndarray<Bin, 2>& data,
    const pr::ndview<Float, 1>& response,
    const pr::ndarray<Index, 1>& treeOrder,
    const pr::ndarray<Index, 1>& selectedFeatures,
    const pr::ndarray<Index, 1>& binOffsets,
    const pr::ndarray<Float, 1>& impList,
    const pr::ndarray<Index, 1>& class_hist_list,
    pr::ndarray<Index, 1>& nodeList,
    pr::ndarray<Float, 1>& left_child_imp_list,
    pr::ndarray<Index, 1>& left_child_class_hist_list,
    pr::ndarray<Float, 1>& nodeImpDecreaseList,
    bool updateImpDecreaseRequired,
    Index nNodes,
    const be::event_vector& deps) {
    [[maybe_unused]] Index nSelectedFeatures = selected_ftr_count_;
    [[maybe_unused]] Index nFeatures = column_count_;
    [[maybe_unused]] Index minObservationsInLeafNode = min_observations_in_leaf_node_;
    [[maybe_unused]] Float impurityThreshold = impurity_threshold_;

    // no overflow check is required because of _nNodesGroups and _nodeGroupProps are small constants
    [[maybe_unused]] auto nodesGroups =
        pr::ndarray<Index, 1>::empty(queue_, { _nNodesGroups * _nodeGroupProps }, alloc::device);
    [[maybe_unused]] auto nodeIndices =
        pr::ndarray<Index, 1>::empty(queue_, { nNodes }, alloc::device);

    cl::sycl::event last_event;
    last_event = tree_level_build_helper_.split_node_list_on_groups_by_size(nodeList,
                                                                            nodesGroups,
                                                                            nodeIndices,
                                                                            nNodes,
                                                                            _nNodesGroups,
                                                                            _nodeGroupProps,
                                                                            deps);
    last_event.wait_and_throw();

    [[maybe_unused]] auto nodesGroupsHost_nd = nodesGroups.to_host(queue_, { last_event });
    [[maybe_unused]] const Index* nodesGroupsHost = nodesGroupsHost_nd.get_data();

    [[maybe_unused]] Index nGroupNodes = 0;
    [[maybe_unused]] Index processedNodes = 0;

    ////_P("selected features :");
    //print_nd_arr(queue_, selectedFeatures, nNodes, selected_ftr_count_);
    //const auto row_count = data.get_dimension(0);
    //const auto column_count = data.get_dimension(1);
    //print_nd_arr(queue_, data, row_count, column_count);

    ////_P(" best split calc");
    //print_nd_arr(queue_, nodesGroups, _nNodesGroups, _nodeGroupProps);

    for (Index i = 0; i < _nNodesGroups; i++, processedNodes += nGroupNodes) {
        nGroupNodes = nodesGroupsHost[i * _nodeGroupProps + 0];
        if (0 == nGroupNodes)
            continue;

        Index maxGroupBlocksNum = nodesGroupsHost[i * _nodeGroupProps + 1];

        Index groupIndicesOffset = processedNodes;

        if (maxGroupBlocksNum > 1) {
            const Index partHistSize = get_part_hist_required_mem_size(selected_ftr_count_,
                                                                       max_bin_count_among_ftrs_,
                                                                       class_count_);

            Index nPartialHistograms =
                maxGroupBlocksNum <= _minRowsBlocksForOneHist ? 1 : _maxLocalHistograms;

            if (nPartialHistograms > 1 && maxGroupBlocksNum < _minRowsBlocksForMaxPartHistNum) {
                while (nPartialHistograms > 1 &&
                       (nPartialHistograms * _minRowsBlocksForOneHist > maxGroupBlocksNum ||
                        nPartialHistograms * partHistSize > max_part_hist_cumulative_size_)) {
                    nPartialHistograms >>= 1;
                }
            }

            //DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(Index, nGroupNodes, partHistSize);
            //DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(Index, nGroupNodes * partHistSize, nPartialHistograms);

            const Index maxPHBlockElems = max_part_hist_cumulative_size_ / sizeof(Float);

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
                        pr::ndarray<Index, 1>::empty(queue_,
                                                     { nBlockNodes * partHistSize },
                                                     alloc::device);

                    //auto [buf_, ev] = pr::ndarray<Index, 1>::zeros(queue_, {(nBlockNodes + 1) * selected_ftr_count_ * buf_prop_}, alloc::device);
                    //ev.wait_and_throw();
                    ////_P("ngrp_nodes = %d", nGroupNodes);

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
                                                            { last_event });

                    //last_event.wait_and_throw();
                    //last_event.wait_and_throw();

                    last_event = compute_best_split_by_histogram(nodesHistograms,
                                                                 selectedFeatures,
                                                                 binOffsets,
                                                                 impList,
                                                                 class_hist_list,
                                                                 nodeIndices,
                                                                 blockIndicesOffset,
                                                                 nodeList,
                                                                 left_child_imp_list,
                                                                 left_child_class_hist_list,
                                                                 nodeImpDecreaseList,
                                                                 updateImpDecreaseRequired,
                                                                 nBlockNodes,
                                                                 { last_event });
                    ////_P("ngrp_nodes = %d", nGroupNodes);

                    last_event.wait_and_throw();
                    ////_P("1 hist compute best split done");

                    /*
                    //_P("left_child_imp :");
                    print_nd_arr(queue_, left_child_imp_list, nBlockNodes, 1);
                    //_P("left_child_class hist :");
                    print_nd_arr(queue_, left_child_class_hist_list, nBlockNodes, class_count_);
*/
                }
                else {
                    //_P("%d hist", static_cast<std::int32_t>(nPartialHistograms));
                    auto partialHistograms = pr::ndarray<Index, 1>::empty(
                        queue_,
                        { nBlockNodes * nPartialHistograms * partHistSize },
                        alloc::device);
                    auto nodesHistograms =
                        pr::ndarray<Index, 1>::empty(queue_,
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
                                                            { last_event });

                    last_event = reduce_partial_histograms(partialHistograms,
                                                           nodesHistograms,
                                                           nPartialHistograms,
                                                           nBlockNodes,
                                                           reduce_local_size_part_hist_,
                                                           { last_event });

                    //auto [buf, ev] = pr::ndarray<Index, 1>::zeros(queue_, {nBlockNodes * selected_ftr_count_ * impl_const_t::node_prop_count_}, alloc::device);
                    //ev.wait_and_throw();

                    last_event = compute_best_split_by_histogram(nodesHistograms,
                                                                 selectedFeatures,
                                                                 binOffsets,
                                                                 impList,
                                                                 class_hist_list,
                                                                 nodeIndices,
                                                                 blockIndicesOffset,
                                                                 nodeList,
                                                                 left_child_imp_list,
                                                                 left_child_class_hist_list,
                                                                 nodeImpDecreaseList,
                                                                 updateImpDecreaseRequired,
                                                                 nBlockNodes,
                                                                 { last_event });

                    last_event.wait_and_throw();
                }
            }
        }
        else {
            ////_P("single pass ngrp_nodes = %d", nGroupNodes);
            last_event = compute_best_split_single_pass(data,
                                                        response,
                                                        treeOrder,
                                                        selectedFeatures,
                                                        binOffsets,
                                                        impList,
                                                        class_hist_list,
                                                        nodeIndices,
                                                        groupIndicesOffset,
                                                        nodeList,
                                                        left_child_imp_list,
                                                        left_child_class_hist_list,
                                                        nodeImpDecreaseList,
                                                        updateImpDecreaseRequired,
                                                        nGroupNodes,
                                                        { last_event });
            ////_P("compute best split SINGLE PASS done");
            last_event.wait_and_throw();
            ////_P("node_list :");
            //print_nd_arr(queue_, nodeList, nNodes, impl_const_t::node_prop_count_);
            /*
            static int irr = 0;
            if(irr < 3) {
                irr++;
                //Index nodes_to_print = 10;
                ////_P("sp node_list :");
                //print_nd_arr(queue_, nodeList, nodes_to_print, impl_const_t::node_prop_count_);

                ////_P("sp left_child_imp :");
                //print_nd_arr(queue_, left_child_imp_list, nodes_to_print, 1);
                ////_P("sp left_child_class hist :");
                //print_nd_arr(queue_, left_child_class_hist_list, nodes_to_print, class_count_);
            }
*/
        }
    }

    return last_event;
}

/////     TODO seems no need for bin offsets here
template <typename Float, typename Bin, typename Index>
cl::sycl::event df_cls_train_hist_impl<Float, Bin, Index>::compute_partial_histograms(
    const pr::ndarray<Bin, 2>& data,
    const pr::ndview<Float, 1>& response,
    const pr::ndarray<Index, 1>& treeOrder,
    const pr::ndarray<Index, 1>& selectedFeatures,
    const pr::ndarray<Index, 1>& binOffsets,
    const pr::ndarray<Index, 1>& nodeList,
    const pr::ndarray<Index, 1>& nodeIndices,
    Index nodeIndicesOffset,
    pr::ndarray<Index, 1>& partialHistograms,
    Index nPartialHistograms,
    Index node_count,
    const be::event_vector& deps) {
    ////_P("compute_partial_histograms");
    //DAAL_ITTNOTIFY_SCOPED_TASK(compute.compute_partial_histograms);
    //DAAL_ASSERT(nSelectedFeatures <= _int32max);
    //DAAL_ASSERT(nodeIndicesOffset <= _int32max);
    //DAAL_ASSERT(nMaxBinsAmongFtrs <= _int32max);
    //DAAL_ASSERT(nFeatures <= _int32max);
    //DAAL_ASSERT(response.size() == _nRows);

    // TODO check why failed
    //ONEDAL_ASSERT(nodeList.get_count() == node_count * impl_const_t::node_prop_count_);
    //ONEDAL_ASSERT(nodeIndices.get_count() == node_count);

    //DAAL_ASSERT_UNIVERSAL_BUFFER(data, uint32_t, _nRows * _nFeatures);
    //DAAL_ASSERT_UNIVERSAL_BUFFER(treeOrder, int32_t, _nSelectedRows);
    //DAAL_ASSERT_UNIVERSAL_BUFFER(selectedFeatures, int32_t, node_count * nSelectedFeatures);
    //DAAL_ASSERT_UNIVERSAL_BUFFER(binOffsets, uint32_t, _nFeatures + 1);
    //DAAL_ASSERT_UNIVERSAL_BUFFER(nodeList, int32_t, node_count * TreeLevelRecord<Float>::_nNodeSplitProps);
    //DAAL_ASSERT_UNIVERSAL_BUFFER(nodeIndices, int32_t, node_count);
    //DAAL_ASSERT_UNIVERSAL_BUFFER(partialHistograms, Float,
    //                             node_count * nPartialHistograms * nSelectedFeatures * _nMaxBinsAmongFtrs * _nClasses);

    auto fill_event = partialHistograms.fill(queue_, 0, deps);
    fill_event.wait_and_throw();

    const Bin* data_ptr = data.get_data();
    const Float* response_ptr = response.get_data();
    const Index* tree_order_ptr = treeOrder.get_data();
    const Index* selected_ftrs_ptr = selectedFeatures.get_data();
    const Index* node_list_ptr = nodeList.get_data();
    const Index* node_indices_ptr = nodeIndices.get_data();

    const Index nProp = class_count_; // num of split attributes for node
    const Index nNodeProp = impl_const_t::node_prop_count_; // num of split attributes for node
    const Index max_bin_count_among_ftrs = max_bin_count_among_ftrs_;
    const Index selected_ftr_count = selected_ftr_count_;
    const Index column_count = column_count_;

    Index* partial_histogram_ptr = partialHistograms.get_mutable_data();

    auto local_size = preferable_local_size_for_part_hist_kernel_;
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
            const Index nRows = node_list_ptr[nodeId * nNodeProp + 1];

            const Index nElementsForGroup = nRows / nPartHist + bool(nRows % nPartHist);

            Index iStart = histIdx * nElementsForGroup;
            Index iEnd = (histIdx + 1) * nElementsForGroup;

            iEnd = (iEnd > nRows) ? nRows : iEnd;

            for (Index i = iStart; i < iEnd; i++) {
                Index id = tree_order_ptr[rowsOffset + i];
                for (Index featIdx = ftrGrpIdx; featIdx < selected_ftr_count;
                     featIdx += ftrGrpSize) {
                    const Index featId = selected_ftrs_ptr[nodeId * selected_ftr_count + featIdx];

                    Index* histogram_ptr =
                        partial_histogram_ptr +
                        ((nodeIdx * nPartHist + histIdx) * selected_ftr_count + featIdx) *
                            max_bin_count_among_ftrs * nProp;

                    Index bin = data_ptr[id * column_count + featId];
                    Index classId = static_cast<Index>(response_ptr[id]);

                    histogram_ptr[bin * nProp + classId] += 1;
                }
            }
        });
    });

    event.wait_and_throw();
    return event;
}

template <typename Float, typename Bin, typename Index>
cl::sycl::event df_cls_train_hist_impl<Float, Bin, Index>::reduce_partial_histograms(
    const pr::ndarray<Index, 1>& partialHistograms,
    pr::ndarray<Index, 1>& histograms,
    Index nPartialHistograms,
    Index node_count,
    Index reduce_local_size,
    const be::event_vector& deps) {
    //DAAL_ITTNOTIFY_SCOPED_TASK(compute.reducePartialHistograms);
    //DAAL_ASSERT(nPartialHistograms <= _int32max);
    //DAAL_ASSERT(nSelectedFeatures <= _int32max);
    //DAAL_ASSERT(nMaxBinsAmongFtrs <= _int32max);

    //DAAL_ASSERT_UNIVERSAL_BUFFER(partialHistograms, Float,
    //                             nNodes * nPartialHistograms * nSelectedFeatures * _nMaxBinsAmongFtrs * _nClasses);
    //DAAL_ASSERT_UNIVERSAL_BUFFER(histograms, Float, nNodes * nSelectedFeatures * _nMaxBinsAmongFtrs * _nClasses);

    //_P("reduce part hist");
    const Index LOCAL_BUFFER_SIZE = 256; // num of split attributes for node
    const Index nProp = class_count_; // num of split attributes for node
    const Index max_bin_count_among_ftrs = max_bin_count_among_ftrs_;
    const Index selected_ftr_count = selected_ftr_count_;

    const Index* partial_histogram_ptr = partialHistograms.get_data();
    Index* histogram_ptr = histograms.get_mutable_data();

    // overflow for nMaxBinsAmongFtrs * nSelectedFeatures should be checked in compute
    const cl::sycl::nd_range<3> nd_range = be::make_multiple_nd_range_3d(
        { max_bin_count_among_ftrs * selected_ftr_count, reduce_local_size, node_count },
        { 1, reduce_local_size, 1 });

    auto event = queue_.submit([&](cl::sycl::handler& cgh) {
        cgh.depends_on(deps);
        cl::sycl::
            accessor<Index, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>
                buf(LOCAL_BUFFER_SIZE * nProp, cgh);

        cgh.parallel_for(nd_range, [=](cl::sycl::nd_item<3> item) {
            const Index nodeIdx = item.get_global_id()[2];
            const Index binId = item.get_global_id()[0];
            const Index local_id = item.get_local_id()[1];
            const Index local_size = item.get_local_range()[1];

            for (Index prop = 0; prop < nProp; prop++) {
                buf[local_id * nProp + prop] = 0;
            }

            const Index* nodePartialHistograms =
                partial_histogram_ptr + nodeIdx * nPartialHistograms * selected_ftr_count *
                                            max_bin_count_among_ftrs * nProp;
            Index* nodeHistogram =
                histogram_ptr + nodeIdx * selected_ftr_count * max_bin_count_among_ftrs * nProp;

            for (Index i = local_id; i < nPartialHistograms; i += local_size) {
                Index offset =
                    i * selected_ftr_count * max_bin_count_among_ftrs * nProp + binId * nProp;
                for (Index prop = 0; prop < nProp; prop++) {
                    buf[local_id * nProp + prop] += nodePartialHistograms[offset + prop];
                }
            }

            for (Index offset = local_size / 2; offset > 0; offset >>= 1) {
                item.barrier(cl::sycl::access::fence_space::local_space);
                if (local_id < offset) {
                    for (Index prop = 0; prop < nProp; prop++) {
                        buf[local_id * nProp + prop] += buf[(local_id + offset) * nProp + prop];
                    }
                }
            }

            if (local_id == 0) {
                for (Index prop = 0; prop < nProp; prop++) {
                    nodeHistogram[binId * nProp + prop] = buf[local_id + prop];
                }
            }
        });
    });
    return event;
}
//////////////////////////////////////////// Best split kernels
template <typename Float, typename Bin, typename Index>
class compute_best_split_by_histogram_krn;

template <typename Float>
inline bool float_eq(Float a, Float b) {
    return cl::sycl::fabs(a - b) <= float_accuracy<Float>::val;
}

template <typename Float>
inline bool float_gt(Float a, Float b) {
    return (a - b) > float_accuracy<Float>::val;
}
//// By histogram///////////////////////////////////
template <typename Float, typename Bin, typename Index>
cl::sycl::event df_cls_train_hist_impl<Float, Bin, Index>::compute_best_split_by_histogram(
    const pr::ndarray<Index, 1>& nodesHistograms,
    const pr::ndarray<Index, 1>& selectedFeatures,
    const pr::ndarray<Index, 1>& binOffsets,
    const pr::ndarray<Float, 1>& impList,
    const pr::ndarray<Index, 1>& class_hist_list,
    const pr::ndarray<Index, 1>& nodeIndices,
    Index nodeIndicesOffset,
    pr::ndarray<Index, 1>& nodeList,
    pr::ndarray<Float, 1>& left_child_imp_list,
    pr::ndarray<Index, 1>& left_child_class_hist_list,
    pr::ndarray<Float, 1>& nodeImpDecreaseList,
    bool updateImpDecreaseRequired,
    Index node_count,
    const be::event_vector& deps) {
    //Index node_count, pr::ndarray<Index, 1> & buf, const be::event_vector& deps) {
    //DAAL_ITTNOTIFY_SCOPED_TASK(compute.reducePartialHistograms);
    //    DAAL_ASSERT_UNIVERSAL_BUFFER(nodeHistogramList, Float, nNodes * nSelectedFeatures * _nMaxBinsAmongFtrs * _nClasses);
    //    DAAL_ASSERT_UNIVERSAL_BUFFER(selectedFeatures, int32_t, nNodes * nSelectedFeatures);
    //    DAAL_ASSERT_UNIVERSAL_BUFFER(nodeList, int32_t, nNodes * TreeLevelRecord<Float>::_nNodeSplitProps);
    //    DAAL_ASSERT_UNIVERSAL_BUFFER(nodeIndices, int32_t, nNodes);
    //    DAAL_ASSERT_UNIVERSAL_BUFFER(binOffsets, uint32_t, _nFeatures + 1);
    //    DAAL_ASSERT_UNIVERSAL_BUFFER(impList, Float, nNodes * (TreeLevelRecord<Float>::_nNodeImpProps + _nClasses));
    //    if (updateImpDecreaseRequired) DAAL_ASSERT_UNIVERSAL_BUFFER(nodeImpDecreaseList, Float, nNodes);

    ////_P("compute best split by hist");

    const Index* node_histogram_ptr = nodesHistograms.get_data();
    const Index* selected_ftrs_ptr = selectedFeatures.get_data();
    const Index* bin_offsets_ptr = binOffsets.get_data();
    [[maybe_unused]] const Float* node_imp_ptr = impList.get_data();
    const Index* class_hist_list_ptr = class_hist_list.get_data();
    const Index* node_indices_ptr = nodeIndices.get_data();
    Index* node_list_ptr = nodeList.get_mutable_data();
    Float* left_child_imp_list_ptr = left_child_imp_list.get_mutable_data();
    Index* left_child_class_hist_list_ptr = left_child_class_hist_list.get_mutable_data();
    Float* node_imp_decr_list_ptr = nodeImpDecreaseList.get_mutable_data();

    const Index nProp = class_count_; // num of split attributes for node
    const Index nNodeProp = impl_const_t::node_prop_count_;
    const Index nImpProp = impl_const_t::node_imp_prop_count_;
    const Index leafMark = impl_const_t::bad_val_;
    const Index nMaxBinsAmongFtrs = max_bin_count_among_ftrs_;
    const Float impurityThreshold = impurity_threshold_;
    const Index minObservationsInLeafNode = min_observations_in_leaf_node_;
    const Float minImpDec = de::limits<Float>::min();
    const Index valNotFound = de::limits<Index>::max();

    //[[maybe_unused]] Index buf_prop = buf_prop_;
    //auto [buf, ev] = pr::ndarray<Index, 1>::zeros(queue_, {(node_count + 1) * selected_ftr_count_ * buf_prop_}, alloc::device);
    //ev.wait_and_throw();
    //Index* buf_ptr = buf.get_mutable_data();

    const Index selected_ftr_count = selected_ftr_count_;

    auto local_size = be::device_max_sg_size(queue_);

    const cl::sycl::nd_range<2> nd_range =
        be::make_multiple_nd_range_2d({ local_size, node_count }, { local_size, 1 });

    auto event = queue_.submit([&](cl::sycl::handler& cgh) {
        //cl::sycl::stream out(1024, 256, cgh);
        cgh.depends_on(deps);
        cgh.parallel_for<class compute_best_split_by_histogram_krn<Float, Bin, Index>>(nd_range, [=](cl::sycl::nd_item<2> item)
        {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id() > 0) {
                return;
            }
            const Index nodeIdx = item.get_global_id()[1];
            const Index nodeId = node_indices_ptr[nodeIndicesOffset + nodeIdx];

            const Index sub_group_local_id = sbg.get_local_id();
            const Index sub_group_size = sbg.get_local_range()[0];

            const Index nRows = node_list_ptr[nodeId * nNodeProp + 1];

            Index curFeatureValue = leafMark;
            Index curFeatureId = leafMark;

            Float curImpDec = minImpDec;

            // seems we don't set it here as it is done in node split
            node_list_ptr[nodeId * nNodeProp + 2] = curFeatureId;
            node_list_ptr[nodeId * nNodeProp + 3] = curFeatureValue;
            node_list_ptr[nodeId * nNodeProp + 4] = nRows;

            Index mrgN = nRows;

            constexpr Index buff_size =
                16; // class_count, find a way to specialize it in kernel build time
            //Float mrgLN = (Float)0;

            const Index* mrgCls = class_hist_list_ptr + nodeId * nProp;

            Index bestLN = 0;
            Index bestLCls[buff_size] = { 0 };

            Float bestImpL = (Float)0;
            Float imp = (Float)1;

            //[[maybe_unused]]Index* bufaux = buf_ptr;

            for (Index currFtrIdx = sub_group_local_id; currFtrIdx < selected_ftr_count;
                 currFtrIdx += sub_group_size) {
                const Index* nodeHistogram =
                    node_histogram_ptr + nodeIdx * selected_ftr_count * nMaxBinsAmongFtrs * nProp;
                const Index* histogramForFeature =
                    nodeHistogram + currFtrIdx * nMaxBinsAmongFtrs * nProp;

                Index featId = selected_ftrs_ptr[nodeId * selected_ftr_count + currFtrIdx];
                Index currFtrBins = bin_offsets_ptr[featId + 1] - bin_offsets_ptr[featId];

                //[[maybe_unused]]Index* bufst = buf_ptr + ((nodeIdx + 1) * selected_ftr_count + currFtrIdx)  * buf_prop;

                Index mrgLRN[2] = { 0 };
                Index mrgLRCls[buff_size] = { 0 };

                //bufst[9] = 13;
                //bufst[2] = currFtrBins;
                for (Index tbin = 0; tbin < currFtrBins; tbin++) {
                    //bufaux[0] = mrgN;
                    Index binOffset = tbin * nProp;
                    for (Index prop = 0; prop < nProp; prop++) {
                        mrgLRN[0] += histogramForFeature[binOffset + prop];
                        mrgLRCls[prop] += histogramForFeature[binOffset + prop];
                    }

                    mrgLRN[1] = mrgN - mrgLRN[0];

                    //bufst[0] = mrgLRN[0];
                    //bufst[1] = mrgLRN[1];

                    imp = Float(1);
                    Float impL = Float(1);
                    Float impR = Float(1);
                    Float div = Float(1) / (Float(mrgN) * Float(mrgN));
                    Float divL = (0 < mrgLRN[0]) ? Float(1) / (Float(mrgLRN[0]) * Float(mrgLRN[0]))
                                                 : Float(0);
                    Float divR = (0 < mrgLRN[1]) ? Float(1) / (Float(mrgLRN[1]) * Float(mrgLRN[1]))
                                                 : Float(0);

                    for (Index prop = 0; prop < nProp; prop++) {
                        impL -= Float(mrgLRCls[prop]) * Float(mrgLRCls[prop]) * divL;
                        impR -= Float(mrgCls[prop] - mrgLRCls[prop]) *
                                Float(mrgCls[prop] - mrgLRCls[prop]) * divR;
                        imp -= Float(mrgCls[prop]) * Float(mrgCls[prop]) * div;
                    }

                    impL = Float(0) < impL ? impL : Float(0);
                    impR = Float(0) < impR ? impR : Float(0);
                    imp = Float(0) < imp ? imp : Float(0);

                    Float impDec =
                        imp - (Float(mrgLRN[0]) * impL + Float(mrgLRN[1]) * impR) / Float(mrgN);
                    if ((Float)0 < impDec && !float_eq(imp, (Float)0) && imp >= impurityThreshold &&
                        (curFeatureValue == leafMark || float_gt(impDec, curImpDec) ||
                         (float_eq(impDec, curImpDec) && (curFeatureId < featId))) &&
                        mrgLRN[0] >= minObservationsInLeafNode &&
                        mrgLRN[1] >= minObservationsInLeafNode) {
                        curFeatureId = featId;
                        curFeatureValue = tbin;
                        curImpDec = impDec;

                        bestLN = mrgLRN[0];
                        bestImpL = impL;
                        for (Index prop = 0; prop < nProp; prop++) {
                            bestLCls[prop] = mrgLRCls[prop];
                        }
                    }
                } // for tbin

                Float bestImpDec = reduce(sbg, curImpDec, maximum<Float>());

                Index impDecIsBest = float_eq(bestImpDec, curImpDec);

                Index bestFeatureId =
                    reduce(sbg, impDecIsBest ? curFeatureId : valNotFound, minimum<Index>());
                Index bestFeatureValue = reduce(
                    sbg,
                    (bestFeatureId == curFeatureId && impDecIsBest) ? curFeatureValue : valNotFound,
                    minimum<Index>());

                bool noneSplitFoundBySubGroup =
                    ((leafMark == bestFeatureId) && (0 == sub_group_local_id));
                bool mySplitIsBest = (leafMark != bestFeatureId && curFeatureId == bestFeatureId &&
                                      curFeatureValue == bestFeatureValue);
                if (noneSplitFoundBySubGroup || mySplitIsBest) {
                    Float* left_child_imp_info = left_child_imp_list_ptr + nodeId * nImpProp;
                    Index* left_child_class_hist_info =
                        left_child_class_hist_list_ptr + nodeId * nProp;

                    left_child_imp_info[0] = bestImpL;
                    for (Index i = 0; i < nProp; i++) {
                        left_child_class_hist_info[i] = bestLCls[i];
                    }

                    node_list_ptr[nodeId * nNodeProp + 2] =
                        curFeatureId == valNotFound ? leafMark : curFeatureId;
                    node_list_ptr[nodeId * nNodeProp + 3] =
                        curFeatureValue == valNotFound ? leafMark : curFeatureValue;
                    node_list_ptr[nodeId * nNodeProp + 4] = bestLN;

                    if (updateImpDecreaseRequired)
                        node_imp_decr_list_ptr[nodeId] = bestImpDec;
                }
            }
        });
    });

    event.wait_and_throw();

    //print_nd_arr(queue_, buf, (node_count + 1) * selected_ftr_count_, buf_prop_);

    return event;
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

//// Best split single pass ///////////////////////////////////
// TODO seems no need for binOffsets here
template <typename Float, typename Bin, typename Index>
class compute_best_split_single_pass_krn;

template <typename Float, typename Bin, typename Index>
cl::sycl::event df_cls_train_hist_impl<Float, Bin, Index>::compute_best_split_single_pass(
    const pr::ndarray<Bin, 2>& data,
    const pr::ndview<Float, 1>& response,
    const pr::ndarray<Index, 1>& treeOrder,
    const pr::ndarray<Index, 1>& selectedFeatures,
    const pr::ndarray<Index, 1>& binOffsets,
    const pr::ndarray<Float, 1>& impList,
    const pr::ndarray<Index, 1>& class_hist_list,
    const pr::ndarray<Index, 1>& nodeIndices,
    Index nodeIndicesOffset,
    pr::ndarray<Index, 1>& nodeList,
    pr::ndarray<Float, 1>& left_child_imp_list,
    pr::ndarray<Index, 1>& left_child_class_hist_list,
    pr::ndarray<Float, 1>& nodeImpDecreaseList,
    bool updateImpDecreaseRequired,
    Index node_count,
    const be::event_vector& deps) {
    //DAAL_ITTNOTIFY_SCOPED_TASK(compute.reducePartialHistograms);
    //    DAAL_ASSERT_UNIVERSAL_BUFFER(nodeHistogramList, Float, nNodes * nSelectedFeatures * _nMaxBinsAmongFtrs * _nClasses);
    //    DAAL_ASSERT_UNIVERSAL_BUFFER(selectedFeatures, int32_t, nNodes * nSelectedFeatures);
    //    DAAL_ASSERT_UNIVERSAL_BUFFER(nodeList, int32_t, nNodes * TreeLevelRecord<Float>::_nNodeSplitProps);
    //    DAAL_ASSERT_UNIVERSAL_BUFFER(nodeIndices, int32_t, nNodes);
    //    DAAL_ASSERT_UNIVERSAL_BUFFER(binOffsets, uint32_t, _nFeatures + 1);
    //    DAAL_ASSERT_UNIVERSAL_BUFFER(impList, Float, nNodes * (TreeLevelRecord<Float>::_nNodeImpProps + _nClasses));
    //    if (updateImpDecreaseRequired) DAAL_ASSERT_UNIVERSAL_BUFFER(nodeImpDecreaseList, Float, nNodes);
    const Bin* data_ptr = data.get_data();
    const Float* response_ptr = response.get_data();
    const Index* tree_order_ptr = treeOrder.get_data();

    ////_P("selected features :");
    //print_nd_arr(queue_, selectedFeatures, (node_count), selected_ftr_count_);
    const Index* selected_ftrs_ptr = selectedFeatures.get_data();

    //const Index* bin_offsets_ptr = binOffsets.get_data();
    [[maybe_unused]] const Float* node_imp_ptr = impList.get_data();
    const Index* class_hist_list_ptr = class_hist_list.get_data();
    const Index* node_indices_ptr = nodeIndices.get_data();
    Index* node_list_ptr = nodeList.get_mutable_data();
    Float* left_child_imp_list_ptr = left_child_imp_list.get_mutable_data();
    Index* left_child_class_hist_list_ptr = left_child_class_hist_list.get_mutable_data();
    Float* node_imp_decr_list_ptr = nodeImpDecreaseList.get_mutable_data();

    const Index nProp = class_count_; // num of split attributes for node
    const Index nNodeProp = impl_const_t::node_prop_count_;
    const Index nImpProp = impl_const_t::node_imp_prop_count_;
    const Index leafMark = impl_const_t::bad_val_;
    //const Index nMaxBinsAmongFtrs = max_bin_count_among_ftrs_;
    const Float impurityThreshold = impurity_threshold_;
    const Index minObservationsInLeafNode = min_observations_in_leaf_node_;
    const Index column_count = column_count_;

    const Index selected_ftr_count = selected_ftr_count_;
    const Float minImpDec = de::limits<Float>::min();
    const Index valNotFound = de::limits<Index>::max();

    //[[maybe_unused]] Index buf_prop = buf_prop_;
    //auto [buf, ev] = pr::ndarray<Float, 1>::zeros(queue_, {(node_count + 1) * selected_ftr_count_ * buf_prop_}, alloc::device);
    //ev.wait_and_throw();
    //auto* buf_ptr = buf.get_mutable_data();

    auto local_size = be::device_max_sg_size(queue_);

    ////_P("compute best split single pass");

    const cl::sycl::nd_range<2> nd_range =
        be::make_multiple_nd_range_2d({ local_size, node_count }, { local_size, 1 });

    auto event = queue_.submit([&](cl::sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for<class compute_best_split_single_pass_krn<Float, Bin, Index>>(nd_range, [=](cl::sycl::nd_item<2> item)
        {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id() > 0) {
                return;
            }
            const Index nodeIdx = item.get_global_id()[1];
            const Index nodeId = node_indices_ptr[nodeIndicesOffset + nodeIdx];

            const Index sub_group_local_id = sbg.get_local_id();
            const Index sub_group_size = sbg.get_local_range()[0];

            const Index rowsOffset = node_list_ptr[nodeId * nNodeProp + 0];
            const Index nRows = node_list_ptr[nodeId * nNodeProp + 1];

            Index curFeatureValue = leafMark;
            Index curFeatureId = leafMark;

            Float curImpDec = minImpDec;

            // seems we don't need to set it here as it is done in node aplit
            node_list_ptr[nodeId * nNodeProp + 2] = curFeatureId;
            node_list_ptr[nodeId * nNodeProp + 3] = curFeatureValue;
            node_list_ptr[nodeId * nNodeProp + 4] = nRows;

            Index mrgN = nRows;

            constexpr Index buff_size =
                16; // class_count, find a way to specialize it in kernel build time

            const Index* mrgCls = class_hist_list_ptr + nodeId * nProp;

            Index bestLN = 0;
            Index bestLCls[buff_size] = { 0 };

            Float bestImpL = Float(0);
            Float imp = Float(1);

            for (Index currFtrIdx = sub_group_local_id; currFtrIdx < selected_ftr_count;
                 currFtrIdx += sub_group_size) {
                Index featId = selected_ftrs_ptr[nodeId * selected_ftr_count + currFtrIdx];

                [[maybe_unused]] std::uint64_t bin_map[4] = { 0 };

                // calculating classes histogram rows count <= bins num
                for (Index i = 0; i < nRows; i++) {
                    Index curr_row_id = tree_order_ptr[rowsOffset + i];
                    Index tbin = data_ptr[curr_row_id * column_count + featId];

                    bool bin_not_processed = !is_bin_processed(bin_map, tbin);
                    //bool bin_not_processed = 1;
                    if (bin_not_processed) {
                        Index mrgLRN[2] = { 0 };
                        Index mrgLRCls[buff_size] = { 0 };
                        // calculating classes histogram
                        for (int row_idx = 0; row_idx < nRows; row_idx++) {
                            Index id = tree_order_ptr[rowsOffset + row_idx];
                            Index bin = data_ptr[id * column_count + featId];
                            Index classId = static_cast<Index>(response_ptr[id]);

                            mrgLRN[0] += Index(bin <= tbin);
                            mrgLRCls[classId] += Index(bin <= tbin);
                        }

                        mark_bin_processed(bin_map, tbin);

                        mrgLRN[1] = mrgN - mrgLRN[0];

                        imp = Float(1);
                        Float impL = Float(1);
                        Float impR = Float(1);
                        Float div = Float(1) / (Float(mrgN) * Float(mrgN));
                        Float divL = (0 < mrgLRN[0])
                                         ? Float(1) / (Float(mrgLRN[0]) * Float(mrgLRN[0]))
                                         : Float(0);
                        Float divR = (0 < mrgLRN[1])
                                         ? Float(1) / (Float(mrgLRN[1]) * Float(mrgLRN[1]))
                                         : Float(0);

                        for (Index prop = 0; prop < nProp; prop++) {
                            impL -= Float(mrgLRCls[prop]) * Float(mrgLRCls[prop]) * divL;
                            impR -= Float(mrgCls[prop] - mrgLRCls[prop]) *
                                    Float(mrgCls[prop] - mrgLRCls[prop]) * divR;
                            imp -= Float(mrgCls[prop]) * Float(mrgCls[prop]) * div;
                        }

                        impL = Float(0) < impL ? impL : Float(0);
                        impR = Float(0) < impR ? impR : Float(0);
                        imp = Float(0) < imp ? imp : Float(0);

                        Float impDec =
                            imp - (Float(mrgLRN[0]) * impL + Float(mrgLRN[1]) * impR) / Float(mrgN);

                        if ((Float)0 < impDec && !float_eq(imp, (Float)0) &&
                            imp >= impurityThreshold &&
                            (curFeatureValue == leafMark || float_gt(impDec, curImpDec) ||
                             (float_eq(impDec, curImpDec) &&
                              (curFeatureId < featId ||
                               (curFeatureId == featId && tbin < curFeatureValue)))) &&
                            mrgLRN[0] >= minObservationsInLeafNode &&
                            mrgLRN[1] >= minObservationsInLeafNode) {
                            curFeatureId = featId;
                            curFeatureValue = tbin;
                            curImpDec = impDec;

                            bestLN = mrgLRN[0];
                            bestImpL = impL;
                            for (Index prop = 0; prop < nProp; prop++) {
                                bestLCls[prop] = mrgLRCls[prop];
                            }
                        }
                    }
                } // for tbin
            }

            Float bestImpDec = reduce(sbg, curImpDec, maximum<Float>());

            Index impDecIsBest = float_eq(bestImpDec, curImpDec);

            Index bestFeatureId =
                reduce(sbg, impDecIsBest ? curFeatureId : valNotFound, minimum<Index>());
            Index bestFeatureValue = reduce(
                sbg,
                (bestFeatureId == curFeatureId && impDecIsBest) ? curFeatureValue : valNotFound,
                minimum<Index>());

            bool noneSplitFoundBySubGroup =
                ((leafMark == bestFeatureId) && (0 == sub_group_local_id));
            bool mySplitIsBest = (leafMark != bestFeatureId && curFeatureId == bestFeatureId &&
                                  curFeatureValue == bestFeatureValue);
            if (noneSplitFoundBySubGroup || mySplitIsBest) {
                Float* left_child_imp_info = left_child_imp_list_ptr + nodeId * nImpProp;
                Index* left_child_class_hist_info = left_child_class_hist_list_ptr + nodeId * nProp;

                left_child_imp_info[0] = bestImpL;
                for (Index i = 0; i < nProp; i++) {
                    left_child_class_hist_info[i] = bestLCls[i];
                }

                node_list_ptr[nodeId * nNodeProp + 2] =
                    curFeatureId == valNotFound ? leafMark : curFeatureId;
                node_list_ptr[nodeId * nNodeProp + 3] =
                    curFeatureValue == valNotFound ? leafMark : curFeatureValue;
                node_list_ptr[nodeId * nNodeProp + 4] = bestLN;

                if (updateImpDecreaseRequired)
                    node_imp_decr_list_ptr[nodeId] = bestImpDec;
            }
        });
    });

    event.wait_and_throw();
    //print_nd_arr(queue_, buf, (node_count + 1) * selected_ftr_count_, buf_prop_);

    ////_P("after sp node list");
    //print_nd_arr(queue_, nodeList, (node_count), impl_const_t::node_prop_count_);

    return event;
}
//////////////////////////// DO NODE SPLIT
template <typename Float, typename Bin, typename Index>
class do_node_split_krn;

template <typename Float, typename Bin, typename Index>
cl::sycl::event df_cls_train_hist_impl<Float, Bin, Index>::do_node_split(
    const pr::ndarray<Index, 1>& node_list,
    const pr::ndarray<Index, 1>& class_hist_list,
    const pr::ndarray<Float, 1>& left_child_imp_list,
    const pr::ndarray<Index, 1>& left_child_class_hist_list,
    const pr::ndarray<Index, 1>& node_vs_tree_map,
    pr::ndarray<Index, 1>& node_list_new,
    pr::ndarray<Float, 1>& imp_list_new,
    pr::ndarray<Index, 1>& class_hist_list_new,
    pr::ndarray<Index, 1>& node_vs_tree_map_new,
    Index node_count,
    Index node_count_new,
    const be::event_vector& deps) {
    //DAAL_ITTNOTIFY_SCOPED_TASK(compute.doNodesSplit);

    //ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);
    //ONEDAL_ASSERT(class_hist_list.get_count() == node_count * class_count_);
    //ONEDAL_ASSERT(left_child_imp_list.get_count() == node_count * impl_const_t::node_imp_prop_count_);
    //ONEDAL_ASSERT(left_child_class_hist_list.get_count() == node_count * class_count_);
    //ONEDAL_ASSERT(node_vs_tree_map.get_count() == node_count);

    //ONEDAL_ASSERT(node_list_new.get_count() == node_count_new * impl_const_t::node_prop_count_);
    //ONEDAL_ASSERT(imp_list_new.get_count() == node_count_new * impl_const_t::node_imp_prop_count_);
    //ONEDAL_ASSERT(class_hist_list_new.get_count() == node_count_new * class_count_);
    //ONEDAL_ASSERT(node_vs_tree_map_new.get_count() == node_count_new);

    const Index nNodeProp = impl_const_t::node_prop_count_; // num of split attributes for node
    const Index node_imp_prop_count = impl_const_t::node_imp_prop_count_;
    const Index class_count = class_count_;
    const Index bad_val = impl_const_t::bad_val_;

    const Index* node_list_ptr = node_list.get_data();
    const Index* class_hist_list_ptr = class_hist_list.get_data();
    const Float* left_child_imp_list_ptr = left_child_imp_list.get_data();
    const Index* left_child_class_hist_list_ptr = left_child_class_hist_list.get_data();
    const Index* node_vs_tree_map_ptr = node_vs_tree_map.get_data();

    Index* node_list_new_ptr = node_list_new.get_mutable_data();
    Float* imp_list_new_ptr = imp_list_new.get_mutable_data();
    Index* class_hist_list_new_ptr = class_hist_list_new.get_mutable_data();
    Index* node_vs_tree_map_new_ptr = node_vs_tree_map_new.get_mutable_data();

    auto local_size = preferable_sbg_size_;
    const cl::sycl::nd_range<1> nd_range = be::make_multiple_nd_range_1d(local_size, local_size);

    auto event = queue_.submit([&](cl::sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for<class do_node_split_krn<Float, Bin, Index>>(nd_range, [=](cl::sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id() > 0) {
                return;
            }

            const Index local_id = sbg.get_local_id();
            const Index local_size = sbg.get_local_range()[0];

            Index nCreatedNodes = 0;
            for (Index i = local_id; i < node_count; i += local_size) {
                Index splitNode =
                    Index(node_list_ptr[i * nNodeProp + 2] != bad_val); // featId != -1
                Index newLeftNodePos =
                    nCreatedNodes + exclusive_scan(sbg, splitNode, plus<Index>()) * 2;
                if (splitNode) {
                    // split parent node on left and right nodes
                    const Index* nodeP = node_list_ptr + i * nNodeProp;
                    Index* nodeL = node_list_new_ptr + newLeftNodePos * nNodeProp;
                    Index* nodeR = node_list_new_ptr + (newLeftNodePos + 1) * nNodeProp;

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

                    node_vs_tree_map_new_ptr[newLeftNodePos] = node_vs_tree_map_ptr[i];
                    node_vs_tree_map_new_ptr[newLeftNodePos + 1] = node_vs_tree_map_ptr[i];

                    // assign class hist and compute winner for new nodes
                    const Index* class_hist_p = class_hist_list_ptr + i * class_count;
                    const Index* left_child_class_hist =
                        left_child_class_hist_list_ptr + i * class_count;
                    Index* class_hist_left = class_hist_list_new_ptr + newLeftNodePos * class_count;
                    Index* class_hist_right =
                        class_hist_list_new_ptr + (newLeftNodePos + 1) * class_count;

                    Index max_cls_count_left = 0;
                    Index max_cls_count_right = 0;
                    Index win_cls_left = 0;
                    Index win_cls_right = 0;

                    Index rows_right = nodeR[1];
                    Float imp_right = Float(1);
                    Float div_right =
                        (0 < rows_right) ? Float(1) / (rows_right * rows_right) : Float(0);

                    for (Index i = 0; i < class_count; i++) {
                        class_hist_left[i] = left_child_class_hist[i];
                        Index class_count_right = class_hist_p[i] - left_child_class_hist[i];
                        class_hist_right[i] = class_count_right;

                        imp_right -= Float(class_count_right) * (class_count_right)*div_right;

                        if (class_hist_left[i] > max_cls_count_left) {
                            max_cls_count_left = class_hist_left[i];
                            win_cls_left = i;
                        }

                        if (class_hist_right[i] > max_cls_count_right) {
                            max_cls_count_right = class_hist_right[i];
                            win_cls_right = i;
                        }
                    }

                    nodeL[5] = win_cls_left;
                    nodeR[5] = win_cls_right;

                    // assign impurity for new nodes
                    const Float* left_child_imp = left_child_imp_list_ptr + i * node_imp_prop_count;
                    Float* impL = imp_list_new_ptr + newLeftNodePos * node_imp_prop_count;
                    Float* impR = imp_list_new_ptr + (newLeftNodePos + 1) * node_imp_prop_count;
                    impL[0] = left_child_imp[0];
                    impR[0] = cl::sycl::max(imp_right, Float(0));
                }
                nCreatedNodes += reduce(sbg, splitNode, plus<Index>()) * 2;
            }
        });
    });
    event.wait_and_throw();

    return event;
}

/////////////////////////////////////////////////////////
/// compute results
template <typename Float, typename Bin, typename Index>
Float df_cls_train_hist_impl<Float, Bin, Index>::compute_oob_error(
    const model_builder_t& model_builder,
    const pr::ndarray<Float, 1>& data_host,
    const pr::ndarray<Float, 1>& response_host,
    const pr::ndarray<Index, 1>& oob_row_list,
    pr::ndarray<Index, 1>& oob_per_obs_list,
    Index tree_idx,
    Index indicesOffset,
    Index n,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(data_host.get_count() == row_count_ * column_count_);
    ONEDAL_ASSERT(response_host.get_count() == column_count_);
    //ONEDAL_ASSERT(indices.get_count() == column_count_);
    ONEDAL_ASSERT(oob_per_obs_list.get_count() == row_count_ * class_count_);

    auto oob_row_list_host = oob_row_list.to_host(queue_, deps);
    auto oob_per_obs_list_host = oob_per_obs_list.to_host(queue_, deps);

    const Float* data_host_ptr = data_host.get_data();
    const Float* response_host_ptr = response_host.get_data();
    const Index* oob_row_list_host_ptr = oob_row_list_host.get_data();
    Index* oob_per_obs_list_host_ptr = oob_per_obs_list_host.get_mutable_data();

    //compute prediction error on each OOB row and get its mean online formulae (Welford)

    Index mean = 0;
    ////_P("tree idx %d, ind_ofst %d, n %d, oob_row_list count %ld", tree_idx, indicesOffset, n, oob_row_list_host.get_count());
    for (Index i = 0; i < n; i++) {
        int row_ind = oob_row_list_host_ptr[indicesOffset + i];
        ////_P("row_ind %d", row_ind);
        ONEDAL_ASSERT(row_ind < row_count_);

        Index class_idx = static_cast<Index>(
            model_builder.get_tree_response(tree_idx, &data_host_ptr[row_ind * column_count_]));
        ////_P(" class_idx %d", class_idx);
        oob_per_obs_list_host_ptr[row_ind * class_count_ + class_idx]++;
        ////_P(" ---");
        mean += Index(class_idx != Index(response_host_ptr[row_ind]));
    }

    oob_per_obs_list = oob_per_obs_list_host.to_device(queue_);

    return Float(mean) / n;
}

template <typename Float, typename Bin, typename Index>
Float df_cls_train_hist_impl<Float, Bin, Index>::compute_oob_error_perm(
    const model_builder_t& model_builder,
    const pr::ndarray<Float, 1>& data_host,
    const pr::ndarray<Float, 1>& response_host,
    const pr::ndarray<Index, 1>& oob_row_list,
    const pr::ndarray<Index, 1>& permutation_host,
    Index tree_idx,
    Index indicesOffset,
    Index n,
    Index column_idx,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(data_host.get_count() == row_count_ * column_count_);
    ONEDAL_ASSERT(response_host.get_count() == column_count_);
    //ONEDAL_ASSERT(indices.get_count() == column_count_);
    ONEDAL_ASSERT(permutation_host.get_count() == n);
    ONEDAL_ASSERT(column_idx < column_count_);
    //DAAL_ASSERT_UNIVERSAL_BUFFER(indices, int32_t, indicesOffset + n);

    auto oob_row_list_host = oob_row_list.to_host(queue_, deps);

    const Float* data_host_ptr = data_host.get_data();
    const Float* response_host_ptr = response_host.get_data();
    const Index* oob_row_list_host_ptr = oob_row_list_host.get_data();
    const Index* permutation_ptr = permutation_host.get_data();

    auto buf = pr::ndarray<Float, 1>::empty({ column_count_ });
    Float* buf_ptr = buf.get_mutable_data();

    Index mean = 0;
    for (Index i = 0; i < n; i++) {
        int row_ind = oob_row_list_host_ptr[indicesOffset + i];
        int row_ind_perm = permutation_ptr[i];
        ONEDAL_ASSERT(row_ind < row_count_);
        ONEDAL_ASSERT(row_ind_perm < row_count_);

        //services::internal::tmemcpy<algorithmFPType, sse2>(buf.get(), &x[row_ind * nFeatures], nFeatures);
        memcpy(de::default_host_policy{},
               buf_ptr,
               &data_host_ptr[row_ind * column_count_],
               column_count_ * sizeof(Float));
        buf_ptr[column_idx] = data_host_ptr[row_ind_perm * column_count_ + column_idx];
        Index class_idx = static_cast<Index>(model_builder.get_tree_response(tree_idx, buf_ptr));
        mean += Index(class_idx != Index(response_host_ptr[row_ind]));
    }

    return Float(mean) / n;
}

template <typename Float, typename Bin, typename Index>
cl::sycl::event df_cls_train_hist_impl<Float, Bin, Index>::compute_results(
    const model_builder_t& model_builder,
    const pr::ndarray<Float, 1>& data_host,
    const pr::ndarray<Float, 1>& response_host,
    const pr::ndarray<Index, 1>& oob_row_list,
    const pr::ndarray<Index, 1>& oobRowsNumList,
    pr::ndarray<Index, 1>& oob_per_obs_list,
    pr::ndarray<Float, 1>& var_imp,
    pr::ndarray<Float, 1>& var_imp_variance,
    const dal::array<engine_impl>& engine_arr,
    Index tree_idx_in_block,
    Index tree_in_block_count,
    Index built_tree_count,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(oobRowsNumList.get_count() == tree_in_block_count + 1);
    ONEDAL_ASSERT(var_imp.get_count() == column_count_);
    ONEDAL_ASSERT(mda_scaled_required_ ? var_imp_variance.get_count() == column_count_ : true);

    auto nOOBRowsHost = oobRowsNumList.to_host(queue_, deps);
    const Index* nOOBRowsHost_ptr = nOOBRowsHost.get_data();
    Index oob_indices_offset = nOOBRowsHost_ptr[tree_idx_in_block];
    Index oob_row_count =
        nOOBRowsHost_ptr[tree_idx_in_block + 1] - nOOBRowsHost_ptr[tree_idx_in_block];

    if ((oob_required_ || mda_required_) && oob_row_count) {
        [[maybe_unused]] const Float oob_err = compute_oob_error(model_builder,
                                                                 data_host,
                                                                 response_host,
                                                                 oob_row_list,
                                                                 oob_per_obs_list,
                                                                 tree_idx_in_block,
                                                                 oob_indices_offset,
                                                                 oob_row_count);

        if (mda_required_) {
            auto permutation = pr::ndarray<Index, 1>::empty({ oob_row_count });
            Index* permutation_ptr = permutation.get_mutable_data();
            for (Index i = 0; i < oob_row_count; ++i) {
                permutation_ptr[i] = i;
            }

            auto var_imp_host = var_imp.to_host(queue_);
            Float* var_imp_host_ptr = var_imp_host.get_mutable_data();
            Float* var_imp_var_host_ptr = nullptr;
            if (mda_scaled_required_) {
                auto var_imp_var_host = var_imp_variance.to_host(queue_);
                var_imp_var_host_ptr = var_imp_var_host.get_mutable_data();
            }

            const Float div1 = Float(1) / Float(built_tree_count + tree_idx_in_block + 1);

            rng<Index> rn_gen;

            for (Index column_idx = 0; column_idx < column_count_; column_idx++) {
                rn_gen
                    .shuffle(oob_row_count,
                             permutation_ptr,
                             engine_arr[built_tree_count + tree_idx_in_block].get_state())
                    .wait_and_throw();
                const Float oob_err_perm = compute_oob_error_perm(model_builder,
                                                                  data_host,
                                                                  response_host,
                                                                  oob_row_list,
                                                                  permutation,
                                                                  tree_idx_in_block,
                                                                  oob_indices_offset,
                                                                  oob_row_count,
                                                                  column_idx);

                const Float diff = (oob_err_perm - oob_err);
                const Float delta = diff - var_imp_host_ptr[column_idx];
                var_imp_host_ptr[column_idx] += div1 * delta;
                if (var_imp_var_host_ptr) {
                    var_imp_var_host_ptr[column_idx] +=
                        delta * (diff - var_imp_host_ptr[column_idx]);
                }
            }

            var_imp = var_imp_host.to_device(queue_);
            if (mda_scaled_required_) {
                var_imp_variance = var_imp_variance_host_.to_device(queue_);
            }
        }
    }

    return cl::sycl::event{};
}

template <typename Float, typename Bin, typename Index>
cl::sycl::event df_cls_train_hist_impl<Float, Bin, Index>::finalize_oob_error(
    const pr::ndarray<Float, 1>& response_host,
    pr::ndarray<Index, 1>& oob_per_obs_list,
    pr::ndarray<Float, 1>& res_oob_err,
    pr::ndarray<Float, 1>& res_oob_err_obs,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(oob_per_obs_list.get_count() == row_count_ * class_count_);

    //_P("fin oob 01");
    auto oob_per_obs_list_host = oob_per_obs_list.to_host(queue_, deps);
    //_P("fin oob 02");

    const Float* response_host_ptr = response_host.get_data();
    const Index* oob_per_obs_list_host_ptr = oob_per_obs_list_host.get_mutable_data();

    auto res_oob_err_host = pr::ndarray<Float, 1>::empty({ 1 });
    auto res_oob_err_obs_host = pr::ndarray<Float, 1>::empty({ row_count_ });
    Float* res_oob_err_host_ptr = res_oob_err_host.get_mutable_data();
    Float* res_oob_err_obs_host_ptr = res_oob_err_obs_host.get_mutable_data();

    Index predicted_count = 0;
    Float oob_err = 0;

    for (Index i = 0; i < row_count_; i++) {
        Index prediction = 0;
        Index expectation(response_host_ptr[i]);
        Index max_val = 0;
        for (Index class_idx = 0; class_idx < class_count_; class_idx++) {
            Index val = oob_per_obs_list_host_ptr[i * class_count_ + class_idx];
            if (val > max_val) {
                max_val = val;
                prediction = class_idx;
            }
        }

        if (0 < max_val) {
            Float prediction_res = Float(prediction != expectation);
            if (oob_err_obs_required_)
                res_oob_err_obs_host_ptr[i] = prediction_res;
            oob_err += prediction_res;
            predicted_count++;
        }
        else if (oob_err_obs_required_)
            res_oob_err_obs_host_ptr[i] =
                Float(-1); //was not in OOB set of any tree and hence not predicted
    }

    if (oob_err_required_) {
        *res_oob_err_host_ptr = (0 < predicted_count) ? oob_err / Float(predicted_count) : 0;
        res_oob_err = res_oob_err_host.to_device(queue_);
    }

    if (oob_err_obs_required_) {
        res_oob_err_obs = res_oob_err_obs_host.to_device(queue_);
    }

    //_P("fin oob done");

    return cl::sycl::event{};
}

template <typename Float, typename Bin, typename Index>
cl::sycl::event df_cls_train_hist_impl<Float, Bin, Index>::finalize_var_imp(
    pr::ndarray<Float, 1>& var_imp,
    pr::ndarray<Float, 1>& var_imp_variance,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(var_imp.get_count() == column_count_);

    auto var_imp_host = var_imp.to_host(queue_);
    Float* var_imp_host_ptr = var_imp_host.get_mutable_data();

    if (mda_scaled_required_) {
        if (tree_count_ > 1) {
            ONEDAL_ASSERT(var_imp_variance.get_count() == column_count_);
            auto var_imp_var_host = var_imp_variance.to_host(queue_);
            Float* var_imp_var_host_ptr = var_imp_var_host.get_mutable_data();

            const Float div = Float(1) / Float(tree_count_);
            for (Index i = 0; i < column_count_; i++) {
                var_imp_var_host_ptr[i] *= div;
                if (var_imp_var_host_ptr[i] > Float(0)) {
                    var_imp_host_ptr[i] /= std::sqrt(var_imp_var_host_ptr[i] * div);
                }
            }
            var_imp = var_imp_host.to_device(queue_);
        }
        else {
            var_imp.fill(queue_, 0); // addd deps??
            //var_imp = pr::ndarray<Float, 1>::zeros(queue_, {column_count_}, alloc::device);
            //for (size_t i = 0; i < nFeatures; i++)
            //{
            //    var_imp_host_ptr[i] = algorithmFPType(0);
            //}
        }
    }
    else if (mdi_required_) {
        const Float div = Float(1) / tree_count_;
        for (Index i = 0; i < column_count_; i++)
            var_imp_host_ptr[i] *= div;
        var_imp = var_imp_host.to_device(queue_);
    }

    return cl::sycl::event{};
}

/////////////////////////////////////////////////////////
/// Main compute
template <typename Float, typename Bin, typename Index>
result_t df_cls_train_hist_impl<Float, Bin, Index>::operator()(const descriptor_t& desc,
                                                               const table& data,
                                                               const table& responses) {
    using tree_level_record_t = tree_level_record<Float, Index, task::classification>;

    //_P("compute");
    validate_input(desc, data, responses);
    //_P("input done");
    init_params(desc, data, responses);
    //_P("init params done");
    allocate_buffers();
    //_P("alloc buff done");

    pr::ndarray<Float, 1> res_var_imp;

    if (mdi_required_ || mda_required_) {
        res_var_imp = pr::ndarray<Float, 1>::empty(queue_, { column_count_ }, alloc::device);
        res_var_imp.fill(queue_, 0); // addd deps??
    }

    /*init engines*/
    //DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, par.nTrees - 1, par.nTrees);
    //DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, (par.nTrees - 1) * par.nTrees, _nRows);
    //DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, (par.nTrees - 1) * par.nTrees * _nRows, (par.featuresPerNode + 1));
    engine_collection collection(tree_count_);
    dal::array<engine_impl> engine_arr = collection([&](size_t i, size_t& skip) {
        skip = i * tree_count_ * row_count_ * (selected_ftr_count_ + 1);
    });
    //skip = i * tree_count_ * row_count_ * (desc.get_features_per_node() + 1);});

    pr::ndarray<Float, 1> node_imp_decrease_list;

    model_builder_interop<Float, Index, task::classification> model_builder(tree_count_,
                                                                            column_count_,
                                                                            class_count_);

    cl::sycl::event last_event;

    //for (Index iter = 0; (iter < tree_count_) && !algorithms::internal::isCancelled(status, pHostApp); iter += treeBlock)
    for (Index iter = 0; iter < tree_count_; iter += tree_in_block_) {
        Index nTrees = std::min(tree_count_ - iter, tree_in_block_);

        Index nNodes = nTrees; // num of potential nodes to split on current tree level
        auto oobRowsNumList = pr::ndarray<Index, 1>::empty(queue_, { nTrees + 1 }, alloc::device);
        pr::ndarray<Index, 1> oobRows;

        std::vector<tree_level_record_t> DFTreeRecords;
        std::vector<pr::ndarray<Index, 1>>
            levelNodeLists; // lists of nodes int props(rowsOffset, rows, ftrId, ftrVal ... )
        std::vector<pr::ndarray<Float, 1>>
            levelNodeImpLists; // lists of nodes int props(rowsOffset, rows, ftrId, ftrVal ... )
        std::vector<pr::ndarray<Index, 1>> level_node_class_hist_list; // lists of nodes class hists

        //DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nNodes, impl_const_t::node_prop_count_);
        //DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nNodes, TreeLevel::_nNodeImpProps);
        //auto nodeVsTreeMap = pr::ndarray<Index, 1>::empty(queue_, {nNodes}, alloc::device);
        auto nodeVsTreeMap_host = pr::ndarray<Index, 1>::empty({ nNodes });
        auto levelNodeLists_init_host =
            pr::ndarray<Index, 1>::empty({ nNodes * impl_const_t::node_prop_count_ });
        //levelNodeLists.push_back(pr::ndarray<Index, 1>::empty(queue_, {nNodes * impl_const_t::node_prop_count_}, alloc::device));
        levelNodeImpLists.push_back(
            pr::ndarray<Float, 1>::empty(queue_,
                                         { nNodes * impl_const_t::node_imp_prop_count_ },
                                         alloc::device));
        level_node_class_hist_list.push_back(
            pr::ndarray<Index, 1>::empty(queue_, { nNodes * class_count_ }, alloc::device));

        auto treeMap = nodeVsTreeMap_host.get_mutable_data();
        auto rootNode = levelNodeLists_init_host.get_mutable_data();

        for (Index node = 0; node < nNodes; node++) {
            treeMap[node] =
                iter + node; // check for par.nTrees less than int32 was done at the beggining
            rootNode[node * impl_const_t::node_prop_count_ + 0] =
                selected_row_count_ * node; // rows offset
            rootNode[node * impl_const_t::node_prop_count_ + 1] =
                selected_row_count_; // num of rows
        }

        auto nodeVsTreeMap = nodeVsTreeMap_host.to_device(queue_);
        levelNodeLists.push_back(levelNodeLists_init_host.to_device(queue_));

        if (bootstrap_) {
            //DAAL_ITTNOTIFY_SCOPED_TASK(compute.RNG);

            engine_impl* engines = engine_arr.get_mutable_data();
            Index* selected_rows_ptr_index = selected_rows_host_.get_mutable_data();
            std::int32_t* selected_rows_ptr =
                reinterpret_cast<std::int32_t*>(selected_rows_ptr_index);

            for (Index tree_idx = 0; tree_idx < nTrees; tree_idx++) {
                rng<std::int32_t> rn_gen;
                rn_gen.uniform(selected_row_count_,
                               selected_rows_ptr + selected_row_count_ * tree_idx,
                               engines[iter + tree_idx].get_state(),
                               0,
                               row_count_);
            }

            cl::sycl::event event = tree_order_lev_.assign(queue_,
                                                           selected_rows_ptr_index,
                                                           selected_rows_host_.get_count());
            event.wait_and_throw();
        }
        else {
            cl::sycl::event event =
                tree_level_build_helper_.initialize_tree_order(tree_order_lev_,
                                                               nTrees,
                                                               selected_row_count_);
            event.wait_and_throw();
        }

        last_event = compute_class_histogram(response_nd_,
                                             tree_order_lev_,
                                             levelNodeLists[0],
                                             levelNodeImpLists[0],
                                             level_node_class_hist_list[0],
                                             nNodes,
                                             { last_event });

        if (oob_required_) {
            //_P("get oob rows");
            cl::sycl::event event = tree_level_build_helper_.get_oob_row_list(
                tree_order_lev_,
                oobRowsNumList,
                oobRows,
                selected_row_count_,
                nTrees); // oobRowsNumList and oobRows are the output
            event.wait_and_throw();
        }

        //_P("tree build started");
        for (Index level = 0; nNodes > 0; level++) {
            //_P("level %d, nNodes = %d", static_cast<std::int32_t>(level), nNodes);
            auto nodeList = levelNodeLists[level];
            auto impList = levelNodeImpLists[level];
            auto class_hist_list = level_node_class_hist_list[level];

            auto left_child_imp_list =
                pr::ndarray<Float, 1>::empty(queue_,
                                             { nNodes * impl_const_t::node_imp_prop_count_ },
                                             alloc::device);
            auto left_child_class_hist_list =
                pr::ndarray<Index, 1>::empty(queue_, { nNodes * class_count_ }, alloc::device);

            auto selectedFeaturesCom = gen_features(nNodes, nodeVsTreeMap, engine_arr);

            if (mdi_required_) {
                node_imp_decrease_list =
                    pr::ndarray<Float, 1>::empty(queue_, { nNodes }, alloc::device);
            }

            ////_P("before split class hist list :");
            //print_nd_arr(queue_, class_hist_list, nNodes, class_count_);

            last_event = computeBestSplit(full_data_nd_,
                                          response_nd_,
                                          tree_order_lev_,
                                          selectedFeaturesCom,
                                          ftr_bin_offsets_nd_,
                                          impList,
                                          class_hist_list,
                                          nodeList,
                                          left_child_imp_list,
                                          left_child_class_hist_list,
                                          node_imp_decrease_list,
                                          mdi_required_,
                                          nNodes,
                                          { last_event });
            last_event.wait_and_throw();

            tree_level_record_t level_record(queue_,
                                             nodeList,
                                             impList,
                                             class_hist_list,
                                             nNodes,
                                             class_count_);
            DFTreeRecords.push_back(level_record);

            if (mdi_required_) {
                //mdi is calculated only on split nodes and is not calculated on last level
                last_event =
                    tree_level_build_helper_.update_mdi_var_importance(nodeList,
                                                                       node_imp_decrease_list,
                                                                       res_var_imp,
                                                                       column_count_,
                                                                       nNodes,
                                                                       { last_event });
            }

            Index node_count_new;
            last_event = tree_level_build_helper_.get_split_node_count(nodeList,
                                                                       nNodes,
                                                                       node_count_new,
                                                                       { last_event });
            last_event.wait_and_throw();

            ////_P("after bs node_list new Nodes = %d", node_count_new);
            //print_nd_arr(queue_, nodeList, 5, impl_const_t::node_prop_count_);

            if (node_count_new) {
                //there are split nodes -> next level is required
                node_count_new *= 2;

                //DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nNodesNewLevel, impl_const_t::node_prop_count_);
                //DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nNodesNewLevel, (TreeLevel::_nNodeImpProps + _nClasses));
                auto node_list_new = pr::ndarray<Index, 1>::empty(
                    queue_,
                    { node_count_new * impl_const_t::node_prop_count_ },
                    alloc::device);
                auto imp_list_new = pr::ndarray<Float, 1>::empty(
                    queue_,
                    { node_count_new * impl_const_t::node_imp_prop_count_ },
                    alloc::device);
                auto class_hist_list_new =
                    pr::ndarray<Index, 1>::empty(queue_,
                                                 { node_count_new * class_count_ },
                                                 alloc::device);
                auto node_vs_tree_map_new =
                    pr::ndarray<Index, 1>::empty(queue_, { node_count_new }, alloc::device);

                last_event = do_node_split(nodeList,
                                           class_hist_list,
                                           left_child_imp_list,
                                           left_child_class_hist_list,
                                           nodeVsTreeMap,
                                           node_list_new,
                                           imp_list_new,
                                           class_hist_list_new,
                                           node_vs_tree_map_new,
                                           nNodes,
                                           node_count_new,
                                           { last_event });
                last_event.wait_and_throw();

                //print_nd_arr(queue_, node_list_new, node_count_new, impl_const_t::node_prop_count_);

                ////_P("after split node_list :");
                //print_nd_arr(queue_, node_list_new, 5, impl_const_t::node_prop_count_);

                if (max_tree_depth_ > 0 && max_tree_depth_ == level) {
                    tree_level_record_t level_record(queue_,
                                                     node_list_new,
                                                     imp_list_new,
                                                     class_hist_list_new,
                                                     node_count_new,
                                                     class_count_);
                    DFTreeRecords.push_back(level_record);
                    node_count_new = 0;
                }
                else {
                    levelNodeLists.push_back(node_list_new);
                    levelNodeImpLists.push_back(imp_list_new);
                    level_node_class_hist_list.push_back(class_hist_list_new);

                    nodeVsTreeMap = node_vs_tree_map_new;

                    last_event =
                        tree_level_build_helper_.do_level_partition_by_groups(full_data_nd_,
                                                                              nodeList,
                                                                              tree_order_lev_,
                                                                              tree_order_lev_buf_,
                                                                              column_count_,
                                                                              nNodes,
                                                                              { last_event });
                }
            }

            nNodes = node_count_new;
        }

        model_builder.add_tree_block(DFTreeRecords, bin_borders_host_, nTrees);

        for (Index tree_idx = 0; tree_idx < nTrees; tree_idx++) {
            compute_results(model_builder,
                            data_host_,
                            response_host_,
                            oobRows,
                            oobRowsNumList,
                            oob_per_obs_list_,
                            res_var_imp,
                            var_imp_variance_host_,
                            engine_arr,
                            tree_idx,
                            nTrees,
                            iter,
                            { last_event })
                .wait_and_throw();
        }
    }

    result_t res;

    // Finalize results
    if (oob_err_required_ || oob_err_obs_required_) {
        pr::ndarray<Float, 1> res_oob_err; // = pr::ndarray<Float, 1>::empty({1});
        pr::ndarray<Float, 1> res_oob_err_obs; // = pr::ndarray<Float, 1>::empty({row_count_});

        finalize_oob_error(response_host_, oob_per_obs_list_, res_oob_err, res_oob_err_obs)
            .wait_and_throw();

        if (oob_err_required_) {
            auto res_oob_err_host = res_oob_err.to_host(queue_);
            res.set_oob_err(homogen_table::wrap(res_oob_err_host.flatten(), 1, 1));
        }

        if (oob_err_obs_required_) {
            auto res_oob_err_obs_host = res_oob_err_obs.to_host(queue_);
            res.set_oob_err_per_observation(
                homogen_table::wrap(res_oob_err_obs_host.flatten(), row_count_, 1));
        }
    }

    if (mdi_required_ || mda_required_) {
        finalize_var_imp(res_var_imp, var_imp_variance_host_).wait_and_throw();
        ////_P("res_var_imp = %d, cc %d ", static_cast<std::int32_t>(res_var_imp.get_count()), column_count_);
        auto res_var_imp_host = res_var_imp.to_host(queue_);
        //res.set_var_importance(homogen_table::wrap(res_var_imp_host.flatten(queue_), 1, column_count_));
        res.set_var_importance(homogen_table::wrap(res_var_imp_host.flatten(), 1, column_count_));
    }

    return res.set_model(model_builder.get_model());
}

#define INSTANTIATE(F, B) template class df_cls_train_hist_impl<F, B>;

} // namespace oneapi::dal::decision_forest::backend
