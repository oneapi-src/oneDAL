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

#pragma once

#include "oneapi/dal/graph/detail/graph_container.hpp"
#include "oneapi/dal/graph/graph_common.hpp"

namespace oneapi::dal::preview::detail {

template <typename Graph>
ONEAPI_DAL_EXPORT auto get_vertex_count_impl(const Graph &graph) noexcept
    -> vertex_size_type<Graph>;

template <typename Graph>
ONEAPI_DAL_EXPORT auto get_edge_count_impl(const Graph &graph) noexcept -> edge_size_type<Graph>;

template <typename Graph>
ONEAPI_DAL_EXPORT auto get_vertex_degree_impl(const Graph &graph,
                                              const vertex_type<Graph> &vertex) noexcept
    -> edge_size_type<Graph>;

template <typename Graph>
ONEAPI_DAL_EXPORT auto get_vertex_neighbors_impl(const Graph &graph,
                                                 const vertex_type<Graph> &vertex) noexcept
    -> const_edge_range_type<Graph>;
template <typename Graph>

ONEAPI_DAL_EXPORT auto get_vertex_count_impl(const Graph &graph) noexcept
    -> vertex_size_type<Graph> {
    const auto &layout = detail::get_impl(graph);
    return layout->_vertex_count;
}

template <typename Graph>
ONEAPI_DAL_EXPORT auto get_edge_count_impl(const Graph &graph) noexcept -> edge_size_type<Graph> {
    const auto &layout = detail::get_impl(graph);
    return layout->_edge_count;
}

template <typename Graph>
ONEAPI_DAL_EXPORT auto get_vertex_degree_impl(const Graph &graph,
                                              const vertex_type<Graph> &vertex) noexcept
    -> edge_size_type<Graph> {
    const auto &layout = detail::get_impl(graph);
    return layout->_degrees[vertex];
}

template <typename Graph>
ONEAPI_DAL_EXPORT auto get_vertex_neighbors_impl(const Graph &graph,
                                                 const vertex_type<Graph> &vertex) noexcept
    -> const_edge_range_type<Graph> {
    const auto &layout = detail::get_impl(graph);
    const_edge_iterator_type<Graph> vertex_neighbors_begin =
        layout->_vertex_neighbors.begin() + layout->_edge_offsets[vertex];
    const_edge_iterator_type<Graph> vertex_neighbors_end =
        layout->_vertex_neighbors.begin() + layout->_edge_offsets[vertex + 1];
    return std::make_pair(vertex_neighbors_begin, vertex_neighbors_end);
}
} // namespace oneapi::dal::preview::detail
