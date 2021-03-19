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

#include "oneapi/dal/algo/decision_tree/common.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::decision_tree::detail {
namespace v1 {

template <typename Task>
class node_visitor_iface {
public:
    using task_t = Task;
    using leaf_t = leaf_node_info<task_t>;
    using split_t = split_node_info<task_t>;

    virtual ~node_visitor_iface() {}
    virtual bool on_leaf_node(const leaf_t& desc) = 0;
    virtual bool on_split_node(const split_t& desc) = 0;
};

template <typename Task, typename Visitor>
class node_visitor_impl : public base, public node_visitor_iface<Task> {
public:
    using task_t = Task;
    using leaf_t = leaf_node_info<task_t>;
    using split_t = split_node_info<task_t>;

    node_visitor_impl() = delete;
    node_visitor_impl(const node_visitor_impl&) = delete;
    node_visitor_impl& operator=(const node_visitor_impl&) = delete;

    explicit node_visitor_impl(Visitor&& visitor) : _visitor(std::move(visitor)) {}

    bool on_leaf_node(const leaf_t& desc) override {
        return _visitor(desc);
    }
    bool on_split_node(const split_t& desc) override {
        return _visitor(desc);
    }

private:
    Visitor _visitor;
};

template <typename Task, typename Visitor>
std::shared_ptr<node_visitor_iface<Task>> make_node_visitor(Visitor&& visitor) {
    return std::shared_ptr<node_visitor_iface<Task>>(
        new node_visitor_impl<Task, Visitor>(std::move(visitor)));
}

} // namespace v1

using v1::node_visitor_iface;
using v1::node_visitor_impl;
using v1::make_node_visitor;

} // namespace oneapi::dal::decision_tree::detail
