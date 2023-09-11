/*******************************************************************************
* Copyright 2020 Intel Corporation
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
/// Contains the definition of the input and output for Triangle Counting
/// algorithm

#pragma once

#include "oneapi/dal/algo/triangle_counting/common.hpp"
#include "oneapi/dal/algo/triangle_counting/detail/vertex_ranking_types.hpp"

namespace oneapi::dal::preview::triangle_counting {

/// Class for the description of the input parameters of the Triangle Counting
/// algorithm
///
/// @tparam Graph  Type of the input graph
template <typename Graph, typename Task = task::by_default>
class vertex_ranking_input : public base {
    static_assert(detail::is_valid_task<Task>);

public:
    using task_t = Task;
    static_assert(detail::is_valid_graph<Graph>,
                  "Only undirected_adjacency_vector_graph is supported.");
    /// Constructs the algorithm input initialized with the graph
    ///
    /// @param [in]   g  The input graph
    vertex_ranking_input(const Graph& g);

    /// Returns the constant reference to the input graph
    const Graph& get_graph() const;

    /// Sets the input graph
    auto& set_graph(const Graph& g);

private:
    dal::detail::pimpl<detail::vertex_ranking_input_impl<Graph, Task>> impl_;
};

/// Class for the description of the result of the Triangle Counting algorithm
template <typename Task = task::by_default>
class vertex_ranking_result {
    static_assert(detail::is_valid_task<Task>);

public:
    using task_t = Task;
    /// Constructs the empty result
    vertex_ranking_result();

    /// Returns the table with computed number of local triangles for each vertex
    /// represented as std::int64_t values
    template <typename T = Task, typename = detail::enable_if_local_t<T>>
    const table& get_ranks() const {
        return get_ranks_impl();
    }

    /// Returns the total number of triangles in the graph
    template <typename T = Task, typename = detail::enable_if_global_t<T>>
    std::int64_t get_global_rank() const {
        return get_global_rank_impl();
    }

    /// Sets the table with computed number of local triangles for each vertex
    template <typename T = Task, typename = detail::enable_if_local_t<T>>
    auto& set_ranks(const table& value) {
        set_ranks_impl(value);
        return *this;
    }

    /// Sets the total number of triangles in the graph
    template <typename T = Task, typename = detail::enable_if_global_t<T>>
    auto& set_global_rank(std::int64_t value) {
        set_global_rank_impl(value);
        return *this;
    }

private:
    const table& get_ranks_impl() const;
    std::int64_t get_global_rank_impl() const;
    void set_ranks_impl(const table& value);
    void set_global_rank_impl(std::int64_t value);
    dal::detail::pimpl<detail::vertex_ranking_result_impl> impl_;
};

template <typename Graph, typename Task>
vertex_ranking_input<Graph, Task>::vertex_ranking_input(const Graph& data)
        : impl_(new detail::vertex_ranking_input_impl<Graph, Task>(data)) {}

template <typename Graph, typename Task>
const Graph& vertex_ranking_input<Graph, Task>::get_graph() const {
    return impl_->graph_data;
}

} // namespace oneapi::dal::preview::triangle_counting
