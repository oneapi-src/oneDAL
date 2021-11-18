/* file: directed_adjacency_vector_graph_builder.hpp */
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
/// Contains functionality to construct topology of undirected_adjacency_vector_graph

#pragma once

#include "oneapi/dal/graph/common.hpp"
#include "oneapi/dal/graph/directed_adjacency_vector_graph.hpp"
#include "oneapi/dal/detail/memory.hpp"

namespace oneapi::dal::preview::detail {

template <typename VertexValue = empty_value,
          typename EdgeValue = empty_value,
          typename GraphValue = empty_value,
          typename IndexType = std::int32_t,
          typename Allocator = std::allocator<char>>
class directed_adjacency_vector_graph_builder {
public:
    using graph_type =
        directed_adjacency_vector_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>;

    directed_adjacency_vector_graph_builder(std::int64_t vertex_count,
                                            std::int64_t edge_count,
                                            const std::int64_t* rows,
                                            const IndexType* cols,
                                            const EdgeValue* vals)
            : g() {
        auto& graph_impl = oneapi::dal::detail::get_impl(g);
        using vertex_set_t = typename graph_traits<graph_type>::vertex_set;

        rebinded_allocator ra(graph_impl._vertex_allocator);
        auto [degrees_array, degrees] = ra.template allocate_array<vertex_set_t>(vertex_count);

        for (std::int64_t u = 0; u < vertex_count; u++) {
            degrees[u] = rows[u + 1] - rows[u];
        }

        graph_impl.set_topology(vertex_count, edge_count, rows, cols, edge_count, degrees);
        graph_impl.get_topology()._degrees = degrees_array;
        graph_impl.set_edge_values(vals, edge_count);
    }

    const graph_type& get_graph() const {
        return g;
    }

private:
    graph_type g;
};

} // namespace oneapi::dal::preview::detail
