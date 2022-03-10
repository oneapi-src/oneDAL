/* file: directed_adjacency_vector_graph_topology_builder.hpp */
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
/// Contains functionality to construct topology of directed_adjacency_vector_graph

#pragma once

#include "oneapi/dal/graph/common.hpp"
#include "oneapi/dal/graph/directed_adjacency_vector_graph.hpp"

namespace oneapi::dal::preview::detail {

// service to construct required csr for an algorithm
template <typename Graph>
struct csr_topology_builder;

template <typename VertexValue,
          typename EdgeValue,
          typename GraphValue,
          typename IndexType,
          typename Allocator>
struct csr_topology_builder<
    directed_adjacency_vector_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>> {
    const topology<IndexType> &operator()(const directed_adjacency_vector_graph<VertexValue,
                                                                                EdgeValue,
                                                                                GraphValue,
                                                                                IndexType,
                                                                                Allocator> &graph) {
        return dal::detail::get_impl(graph).get_topology();
    }
};

} // namespace oneapi::dal::preview::detail
