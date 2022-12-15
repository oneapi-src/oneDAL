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

/// @file
/// Contains the definition of the input and output for the Connected Components
/// algorithm

#pragma once

#include "oneapi/dal/algo/connected_components/common.hpp"
#include "oneapi/dal/algo/connected_components/detail/vertex_partitioning_types.hpp"

namespace oneapi::dal::preview::connected_components {

/// Class for the description of the input parameters of the Connected Components
/// algorithm
///
/// @tparam Graph  Type of the input graph
template <typename Graph, typename Task = task::by_default>
class vertex_partitioning_input : public base {
    static_assert(detail::is_valid_task<Task>);

public:
    using task_t = Task;
    static_assert(detail::is_valid_graph<Graph>,
                  "Only undirected_adjacency_vector_graph is supported.");
    /// Constructs the algorithm input initialized with the graph
    ///
    /// @param [in]   g  The input graph
    vertex_partitioning_input(const Graph& g);

    /// Returns the constant reference to the input graph
    const Graph& get_graph() const;

    /// Sets the input graph
    auto& set_graph(const Graph& g);

private:
    dal::detail::pimpl<detail::vertex_partitioning_input_impl<Graph, Task>> impl_;
};

/// Class for the description of the result of the Connected Components algorithm
template <typename Task = task::by_default>
class vertex_partitioning_result {
    static_assert(detail::is_valid_task<Task>);

public:
    using task_t = Task;
    /// Constructs the empty result
    vertex_partitioning_result();

    /// The table of size [vertex_count x 1] with computed component ids for each vertex
    const table& get_labels() const {
        return get_labels_impl();
    }

    /// The number of connected components
    // represented as std::int64_t
    std::int64_t get_component_count() const {
        return get_component_count_impl();
    }

    /// Sets the table with computed component ids for each vertex
    auto& set_labels(const table& value) {
        set_labels_impl(value);
        return *this;
    }

    /// Sets the number of connected components
    auto& set_component_count(std::int64_t value) {
        set_component_count_impl(value);
        return *this;
    }

private:
    const table& get_labels_impl() const;
    std::int64_t get_component_count_impl() const;
    void set_labels_impl(const table& value);
    void set_component_count_impl(std::int64_t value);
    dal::detail::pimpl<detail::vertex_partitioning_result_impl> impl_;
};

template <typename Graph, typename Task>
vertex_partitioning_input<Graph, Task>::vertex_partitioning_input(const Graph& data)
        : impl_(new detail::vertex_partitioning_input_impl<Graph, Task>(data)) {}

template <typename Graph, typename Task>
const Graph& vertex_partitioning_input<Graph, Task>::get_graph() const {
    return impl_->graph_data;
}

} // namespace oneapi::dal::preview::connected_components
