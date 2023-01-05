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
/// Contains the definition of the input and output for the Louvain
/// algorithm

#pragma once

#include "oneapi/dal/algo/louvain/common.hpp"
#include "oneapi/dal/algo/louvain/detail/vertex_partitioning_types.hpp"

namespace oneapi::dal::preview::louvain {

/// Class for the description of the input parameters of the Louvain
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
    /// and the initial partition
    ///
    /// @param [in]   g                 The input graph
    /// @param [in]   initial_partition The initial partition of verteces
    vertex_partitioning_input(const Graph &g, const table &initial_partition = table{});

    /// Returns the constant reference to the input graph
    const Graph &get_graph() const;

    /// Returns the constant reference to the initial partition
    const table &get_initial_partition() const;

    /// Set the initial partition
    auto &set_initial_partition(const table &labels);

private:
    dal::detail::pimpl<detail::vertex_partitioning_input_impl<Graph, Task>> impl_;
};

/// Class for the description of the result of the Louvain algorithm
template <typename Task = task::by_default>
class vertex_partitioning_result {
    static_assert(detail::is_valid_task<Task>);

public:
    using task_t = Task;
    /// Constructs the empty result
    vertex_partitioning_result();

    /// Returns the table of size [vertex_count x 1] with community
    /// labels for each vertex
    const table &get_labels() const {
        return get_labels_impl();
    }

    /// Returns the modularity value for computed vertex partition
    double get_modularity() const {
        return get_modularity_impl();
    }

    /// Returns the number of computed communities
    std::int64_t get_community_count() const {
        return get_community_count_impl();
    }

    /// Sets the table with computed community labels for each vertex
    auto &set_labels(const table &value) {
        set_labels_impl(value);
        return *this;
    }

    /// Sets the modularity value for the computed vertex partition
    auto &set_modularity(double value) {
        set_modularity_impl(value);
        return *this;
    }

    /// Sets the number of computed communities
    auto &set_community_count(std::int64_t value) {
        set_community_count_impl(value);
        return *this;
    }

private:
    const table &get_labels_impl() const;
    double get_modularity_impl() const;
    std::int64_t get_community_count_impl() const;
    void set_labels_impl(const table &value);
    void set_modularity_impl(double value);
    void set_community_count_impl(std::int64_t value);
    dal::detail::pimpl<detail::vertex_partitioning_result_impl> impl_;
};

template <typename Graph, typename Task>
vertex_partitioning_input<Graph, Task>::vertex_partitioning_input(const Graph &g,
                                                                  const table &initial_partition)
        : impl_(new detail::vertex_partitioning_input_impl<Graph, Task>(g, initial_partition)) {}

template <typename Graph, typename Task>
const Graph &vertex_partitioning_input<Graph, Task>::get_graph() const {
    return impl_->graph_data;
}

template <typename Graph, typename Task>
const table &vertex_partitioning_input<Graph, Task>::get_initial_partition() const {
    return impl_->labels_data;
}

template <typename Graph, typename Task>
auto &vertex_partitioning_input<Graph, Task>::set_initial_partition(const table &labels) {
    impl_->labels_data = labels;
    return *this;
}

} // namespace oneapi::dal::preview::louvain
