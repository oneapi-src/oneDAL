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

#include <daal/include/services/daal_defines.h>
#include <daal/include/algorithms/tree_utils/tree_utils_classification.h>
#include <daal/include/algorithms/tree_utils/tree_utils_regression.h>

#include "oneapi/dal/algo/decision_tree/common.hpp"
#include "oneapi/dal/algo/decision_tree/backend/node_info_impl.hpp"

namespace oneapi::dal::decision_tree {

template <typename Task>
struct daal_tree_map;

template <>
struct daal_tree_map<task::classification> {
    using daal_visitor_t = daal::algorithms::tree_utils::classification::TreeNodeVisitor;
    using daal_leaf_desc_t = daal::algorithms::tree_utils::classification::LeafNodeDescriptor;
    using daal_split_desc_t = daal::algorithms::tree_utils::classification::SplitNodeDescriptor;
};

template <>
struct daal_tree_map<task::regression> {
    using daal_visitor_t = daal::algorithms::tree_utils::regression::TreeNodeVisitor;
    using daal_leaf_desc_t = daal::algorithms::tree_utils::regression::LeafNodeDescriptor;
    using daal_split_desc_t = daal::algorithms::tree_utils::regression::SplitNodeDescriptor;
};

template <typename Task>
class visitor_interop : public daal_tree_map<Task>::daal_visitor_t {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;
    using daal_leaf_t = typename daal_tree_map<task_t>::daal_leaf_desc_t;
    using daal_split_t = typename daal_tree_map<task_t>::daal_split_desc_t;
    using leaf_t = detail::leaf_node_info_impl<task_t>;
    using split_t = detail::split_node_info_impl<task_t>;
    using visitor_t = std::shared_ptr<typename detail::node_visitor_iface<task_t>>;

    visitor_interop() = default;
    visitor_interop(const visitor_interop&) = delete;
    visitor_interop& operator=(const visitor_interop&) = delete;

    template <typename T = Task, typename = detail::enable_if_regression_t<T>>
    explicit visitor_interop(visitor_t&& vis) : _visitor(std::move(vis)) {}

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    explicit visitor_interop(visitor_t&& vis, std::int64_t class_count)
            : _visitor(std::move(vis)),
              _leaf_info(class_count) {}

    bool onLeafNode(const daal_leaf_t& desc) override {
        auto& leaf_info = dal::detail::cast_impl<leaf_t>(_leaf_info);
        leaf_info.level = desc.level;

        if constexpr (std::is_same_v<Task, task::classification>) {
            leaf_info.response = desc.label;
            leaf_info.prob = desc.prob;
        }
        else if constexpr (std::is_same_v<Task, task::regression>) {
            leaf_info.response = desc.response;
        }

        leaf_info.impurity = desc.impurity;
        leaf_info.sample_count = desc.nNodeSampleCount;

        return _visitor->on_leaf_node(_leaf_info);
    }

    bool onSplitNode(const daal_split_t& desc) override {
        auto& split_info = dal::detail::cast_impl<split_t>(_split_info);
        split_info.level = desc.level;
        split_info.impurity = desc.impurity;
        split_info.sample_count = desc.nNodeSampleCount;
        split_info.feature_index = desc.featureIndex;
        split_info.feature_value = desc.featureValue;

        return _visitor->on_split_node(_split_info);
    }

private:
    visitor_t _visitor;
    split_node_info<task_t> _split_info;
    leaf_node_info<task_t> _leaf_info;
};

} // namespace oneapi::dal::decision_tree
