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
ONEDAL_EXPORT auto get_vertex_count_impl(const Graph &graph) noexcept -> vertex_size_type<Graph>;

template <typename Graph>
ONEDAL_EXPORT auto get_edge_count_impl(const Graph &graph) noexcept -> edge_size_type<Graph>;

template <typename Graph>
ONEDAL_EXPORT auto get_vertex_degree_impl(const Graph &graph,
                                          const vertex_type<Graph> &vertex) noexcept
    -> edge_size_type<Graph>;

template <typename Graph>
ONEDAL_EXPORT auto get_vertex_neighbors_impl(const Graph &graph,
                                             const vertex_type<Graph> &vertex) noexcept
    -> const_edge_range_type<Graph>;
} // namespace oneapi::dal::preview::detail
