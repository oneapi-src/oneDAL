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
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::decision_forest::detail {
namespace v1 {

template <typename Task>
class tree_node_visitor_iface {
public:
    using task_t = Task;
    using leaf_t = leaf_node_descriptor<task_t>;
    using split_t = split_node_descriptor;

    virtual ~tree_node_visitor_iface() {}
    virtual bool on_leaf_node(const leaf_t& desc) = 0;
    virtual bool on_split_node(const split_t& desc) = 0;
};

template <typename Task, typename Op>
class tree_node_visitor_impl : public base, public tree_node_visitor_iface<Task> {
public:
    using task_t = Task;
    using leaf_t = leaf_node_descriptor<task_t>;
    using split_t = split_node_descriptor;

    tree_node_visitor_impl() = delete;
    tree_node_visitor_impl(const tree_node_visitor_impl&) = delete;
    tree_node_visitor_impl& operator=(const tree_node_visitor_impl&) = delete;

    explicit tree_node_visitor_impl(Op&& op) : _op(std::forward<Op>(op)) {}

    virtual bool on_leaf_node(const leaf_t& desc) override {
        return _op(desc);
    }
    virtual bool on_split_node(const split_t& desc) override {
        return _op(desc);
    }

private:
    Op _op;
};

template <typename Task, typename Op>
std::shared_ptr<tree_node_visitor_iface<Task>> get_tree_node_visitor(Op&& op) {
    return std::shared_ptr<tree_node_visitor_iface<Task>>(
        new tree_node_visitor_impl<Task, Op>(std::forward<Op>(op)));
}

} // namespace v1

using v1::tree_node_visitor_impl;

} // namespace oneapi::dal::decision_forest::detail
