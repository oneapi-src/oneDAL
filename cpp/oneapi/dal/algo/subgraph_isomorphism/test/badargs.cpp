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

#include <initializer_list>

#include "oneapi/dal/algo/subgraph_isomorphism.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::algo::subgraph_isomorphism::test {

typedef dal::preview::subgraph_isomorphism::kind isomorphism_kind;

class subgraph_isomorphism_badarg_test {
public:
    auto create_graph() {
        oneapi::dal::preview::undirected_adjacency_vector_graph<std::int32_t> my_graph;
        auto &graph_impl = oneapi::dal::detail::get_impl(my_graph);
        auto &vertex_allocator = graph_impl._vertex_allocator;
        auto &edge_allocator = graph_impl._edge_allocator;

        const std::int64_t vertex_count = 7;
        const std::int64_t edge_count = 8;
        const std::int64_t cols_count = edge_count * 2;
        const std::int64_t rows_count = vertex_count + 1;

        std::int32_t *degrees_ =
            std::allocator_traits<std::allocator<char>>::rebind_traits<std::int32_t>::allocate(
                vertex_allocator,
                vertex_count);
        std::int32_t *cols_ =
            std::allocator_traits<std::allocator<char>>::rebind_traits<std::int32_t>::allocate(
                vertex_allocator,
                cols_count);
        std::int64_t *rows_ =
            std::allocator_traits<std::allocator<char>>::rebind_traits<std::int64_t>::allocate(
                edge_allocator,
                rows_count);
        std::int32_t *rows_vertex_ =
            std::allocator_traits<std::allocator<char>>::rebind_traits<std::int32_t>::allocate(
                vertex_allocator,
                rows_count);

        std::int32_t *degrees = new (degrees_) std::int32_t[vertex_count]{ 1, 3, 4, 2, 3, 1, 2 };
        std::int32_t *cols =
            new (cols_) std::int32_t[cols_count]{ 1, 0, 2, 4, 1, 3, 4, 5, 2, 6, 1, 2, 6, 2, 3, 4 };
        std::int64_t *rows = new (rows_) std::int64_t[rows_count]{ 0, 1, 4, 8, 10, 13, 14, 16 };
        std::int32_t *rows_vertex =
            new (rows_vertex_) std::int32_t[rows_count]{ 0, 1, 4, 8, 10, 13, 14, 16 };

        graph_impl.set_topology(vertex_count, edge_count, rows, cols, degrees);
        graph_impl.get_topology()._rows_vertex =
            oneapi::dal::preview::detail::container<std::int32_t>::wrap(rows_vertex, rows_count);

        return my_graph;
    }

    void check_subgraph_isomorphism(bool semantic_match,
                                    std::int64_t max_match_count,
                                    isomorphism_kind kind) {
        const auto target_graph = create_graph();
        const auto pattern_graph = create_graph();

        std::allocator<char> alloc;
        const auto subgraph_isomorphism_desc =
            dal::preview::subgraph_isomorphism::descriptor<>(alloc)
                .set_kind(kind)
                .set_semantic_match(semantic_match)
                .set_max_match_count(max_match_count);

        const auto result =
            dal::preview::graph_matching(subgraph_isomorphism_desc, target_graph, pattern_graph);
    }
};

#define SUBGRAPH_ISOMORPHISM_BADARG_TEST(name) \
    TEST_M(subgraph_isomorphism_badarg_test, name, "[subgraph_isomorphism][badarg]")

SUBGRAPH_ISOMORPHISM_BADARG_TEST("positive check") {
    REQUIRE_NOTHROW(this->check_subgraph_isomorphism(false, 100, isomorphism_kind::induced));
}

SUBGRAPH_ISOMORPHISM_BADARG_TEST("throws if match count is negative") {
    REQUIRE_THROWS_AS(this->check_subgraph_isomorphism(false, -1, isomorphism_kind::induced),
                      invalid_argument);
}

SUBGRAPH_ISOMORPHISM_BADARG_TEST("throws if semantic match is true") {
    REQUIRE_THROWS_AS(this->check_subgraph_isomorphism(true, 100, isomorphism_kind::induced),
                      invalid_argument);
}

// TODO: Add empty graph case after implementation

} // namespace oneapi::dal::algo::subgraph_isomorphism::test
