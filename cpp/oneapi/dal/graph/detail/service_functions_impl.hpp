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

#include "oneapi/dal/graph/common.hpp"
#include "oneapi/dal/graph/detail/container.hpp"

namespace oneapi::dal::preview::detail {

template <typename Topology>
constexpr std::int64_t get_topology_vertex_count(const Topology &_topology);

template <typename Topology>
constexpr std::int64_t get_topology_edge_count(const Topology &_topology);

template <typename Topology>
constexpr auto get_topology_vertex_degree(const Topology &_topology,
                                          const typename Topology::vertex_type &vertex) noexcept ->
    typename Topology::vertex_edge_size_type;

template <typename Topology>
constexpr auto get_topology_vertex_neighbors(const Topology &_topology,
                                             const typename Topology::vertex_type &vertex) noexcept
    -> typename Topology::const_vertex_edge_range;

template <typename Graph>
constexpr auto get_vertex_count_impl(const Graph &graph) noexcept -> vertex_size_type<Graph> {
    const auto &layout = dal::detail::get_impl(graph).get_topology();
    return get_topology_vertex_count(layout);
}

template <typename Graph>
constexpr auto get_edge_count_impl(const Graph &graph) noexcept -> edge_size_type<Graph> {
    const auto &layout = dal::detail::get_impl(graph).get_topology();
    return get_topology_edge_count(layout);
}

template <typename Graph>
constexpr auto get_vertex_degree_impl(const Graph &graph, const vertex_type<Graph> &vertex) noexcept
    -> vertex_edge_size_type<Graph> {
    const auto &layout = dal::detail::get_impl(graph).get_topology();
    return get_topology_vertex_degree(layout, vertex);
}

template <typename Graph>
constexpr auto get_vertex_neighbors_impl(const Graph &graph,
                                         const vertex_type<Graph> &vertex) noexcept
    -> const_vertex_edge_range_type<Graph> {
    const auto &layout = dal::detail::get_impl(graph).get_topology();
    return get_topology_vertex_neighbors /*<typename graph_traits<Graph>::impl_type::topology_type>*/
        (layout, vertex);
}

} // namespace oneapi::dal::preview::detail
