/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "oneapi/dal/algo/decision_forest/common.hpp"
#include "oneapi/dal/algo/decision_forest/backend/model_interop.hpp"

namespace oneapi::dal::decision_forest {

template <typename Task>
struct daal_model_map;

template <>
struct daal_model_map<task::classification> {
    using daal_visitor_t = daal::algorithms::tree_utils::classification::TreeNodeVisitor;
    using daal_leaf_desc_t = daal::algorithms::tree_utils::classification::LeafNodeDescriptor;
    using daal_split_desc_t = daal::algorithms::tree_utils::classification::SplitNodeDescriptor;
    using daal_model_t = backend::model_interop_cls;
};

template <>
struct daal_model_map<task::regression> {
    using daal_visitor_t = daal::algorithms::tree_utils::regression::TreeNodeVisitor;
    using daal_leaf_desc_t = daal::algorithms::tree_utils::regression::LeafNodeDescriptor;
    using daal_split_desc_t = daal::algorithms::tree_utils::regression::SplitNodeDescriptor;
    using daal_model_t = backend::model_interop_reg;
};

template <typename Task>
class interop_visitor : public daal_model_map<Task>::daal_visitor_t {
public:
    using task_t = Task;
    using visitor_t = std::shared_ptr<detail::tree_node_visitor_iface<task_t>>;

    interop_visitor() = default;
    interop_visitor(const interop_visitor&) = delete;
    interop_visitor& operator=(const interop_visitor&) = delete;

    explicit interop_visitor(visitor_t&& vis) : _visitor(std::forward<visitor_t>(vis)) {}
    bool onLeafNode(const typename daal_model_map<Task>::daal_leaf_desc_t& desc) override {
        leaf_node_descriptor<Task> leaf_desc;
        leaf_desc.level = desc.level;
        leaf_desc.impurity = desc.impurity;
        leaf_desc.node_sample_count = desc.nNodeSampleCount;
        if constexpr (std::is_same_v<Task, task::classification>) {
            leaf_desc.label = desc.label;
            leaf_desc.prob = desc.prob;
        }
        else if constexpr (std::is_same_v<Task, task::regression>) {
            leaf_desc.label = desc.response;
        }
        else {
            static_assert("Unknown task");
        }

        return _visitor->on_leaf_node(leaf_desc);
    }

    bool onSplitNode(const typename daal_model_map<Task>::daal_split_desc_t& desc) override {
        split_node_descriptor split_desc;
        split_desc.level = desc.level;
        split_desc.impurity = desc.impurity;
        split_desc.node_sample_count = desc.nNodeSampleCount;
        split_desc.feature_index = desc.featureIndex;
        split_desc.feature_value = desc.featureValue;

        return _visitor->on_split_node(split_desc);
    }

private:
    visitor_t _visitor;
};

template <typename Task>
class detail::v1::model_impl : public base {
public:
    using task_t = Task;
    using visitor_t = std::shared_ptr<detail::tree_node_visitor_iface<task_t>>;

    model_impl() = default;
    model_impl(const model_impl&) = delete;
    model_impl& operator=(const model_impl&) = delete;

    model_impl(backend::model_interop* interop) : interop_(interop) {
        if (!interop_) {
            throw dal::internal_error(
                dal::detail::error_messages::input_model_does_not_match_kernel_function());
        }
    }

    virtual ~model_impl() {
        delete interop_;
        interop_ = nullptr;
    }

    backend::model_interop* get_interop() const {
        return interop_;
    }

    void traverse_dfs_impl(std::int64_t tree_idx, visitor_t&& visitor) const {
        auto daal_model =
            static_cast<const typename daal_model_map<Task>::daal_model_t*>(interop_)->get_model();
        interop_visitor<task_t> vis(std::forward<visitor_t>(visitor));
        daal_model->traverseDFS(static_cast<size_t>(tree_idx), vis);
    }

    void traverse_bfs_impl(std::int64_t tree_idx, visitor_t&& visitor) const {
        auto daal_model =
            static_cast<const typename daal_model_map<Task>::daal_model_t*>(interop_)->get_model();
        interop_visitor<task_t> vis(std::forward<visitor_t>(visitor));
        daal_model->traverseBFS(static_cast<size_t>(tree_idx), vis);
    }

    std::int64_t tree_count = 0;
    std::int64_t class_count = 0;

private:
    backend::model_interop* interop_ = nullptr;
};

namespace backend {

using model_impl_cls = detail::model_impl<task::classification>;
using model_impl_reg = detail::model_impl<task::regression>;

} // namespace backend
} // namespace oneapi::dal::decision_forest
