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
/// Contains the definition of the input and output for the Shortest Paths
/// algorithm

#pragma once

#include "oneapi/dal/algo/shortest_paths/common.hpp"
#include "oneapi/dal/algo/shortest_paths/detail/traverse_types.hpp"

namespace oneapi::dal::preview::shortest_paths {

/// Class for the description of the input parameters of the Shortest Paths
/// algorithm
///
/// @tparam Graph  Type of the input graph
template <typename Graph, typename Task = task::by_default>
class traverse_input : public base {
    static_assert(detail::is_valid_task<Task>);

public:
    using task_t = Task;
    static_assert(detail::is_valid_graph<Graph>,
                  "Only directed_adjacency_vector_graph is supported.");
    /// Constructs the algorithm input initialized with the graph
    ///
    /// @param [in]   g  The input graph
    traverse_input(const Graph& g);

    /// Returns the constant reference to the input graph
    const Graph& get_graph() const;

    /// Sets the input graph
    auto& set_graph(const Graph& g);

private:
    dal::detail::pimpl<detail::traverse_input_impl<Graph, Task>> impl_;
};

/// Class for the description of the result of the Shortest Paths algorithm
template <typename Task = task::by_default>
class traverse_result {
    static_assert(detail::is_valid_task<Task>);

public:
    using task_t = Task;
    /// Constructs the empty result
    traverse_result();

    /// Returns the table with computed distances from the source to each vertex
    /// represented as the type of weights of the graph (std::int32_t or double)
    const table& get_distances() const {
        return get_distances_impl();
    }

    /// Returns the table with computed predecessors from the source to each vertex
    // represented as std::int32_t
    const table& get_predecessors() const {
        return get_predecessors_impl();
    }

    /// Sets the table with computed distances from the source to each vertex
    auto& set_distances(const table& value) {
        set_distances_impl(value);
        return *this;
    }

    /// Sets the table with computed predecessors from the source to each vertex
    auto& set_predecessors(const table& value) {
        set_predecessors_impl(value);
        return *this;
    }

private:
    const table& get_distances_impl() const;
    const table& get_predecessors_impl() const;
    void set_distances_impl(const table& value);
    void set_predecessors_impl(const table& value);
    dal::detail::pimpl<detail::traverse_result_impl> impl_;
};

template <typename Graph, typename Task>
traverse_input<Graph, Task>::traverse_input(const Graph& data)
        : impl_(new detail::traverse_input_impl<Graph, Task>(data)) {}

template <typename Graph, typename Task>
const Graph& traverse_input<Graph, Task>::get_graph() const {
    return impl_->graph_data;
}

} // namespace oneapi::dal::preview::shortest_paths
