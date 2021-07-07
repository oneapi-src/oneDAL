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

#include <initializer_list>

#include "oneapi/dal/algo/subgraph_isomorphism/graph_matching.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/graph/service_functions.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::algo::subgraph_isomorphism::test {

typedef dal::preview::subgraph_isomorphism::kind isomorphism_kind;

class graph_base_data {
public:
    graph_base_data() = default;

    std::int64_t get_vertex_count() const {
        return vertex_count;
    }

    std::int64_t get_edge_count() const {
        return edge_count;
    }

    std::int64_t get_cols_count() const {
        return cols_count;
    }

    std::int64_t get_rows_count() const {
        return rows_count;
    }

protected:
    std::int64_t vertex_count;
    std::int64_t edge_count;
    std::int64_t cols_count;
    std::int64_t rows_count;
};

/*
  O------O---
 /|\    /|\  \
O | O--O-+-O--O
 \|/    \|/  /
  O------O---
*/
class double_triangle_target_type : public graph_base_data {
public:
    double_triangle_target_type() {
        vertex_count = 9;
        edge_count = 16;
        cols_count = 32;
        rows_count = 10;
    }
    std::array<std::int32_t, 9> degrees = { 2, 4, 3, 4, 4, 5, 3, 5, 2 };
    std::array<std::int32_t, 32> cols = { 1, 3, 0, 2, 5, 3, 4, 1, 3, 7, 0, 2, 1, 2, 6, 5,
                                          7, 1, 8, 7, 4, 6, 4, 7, 5, 3, 5, 6, 4, 8, 5, 7 };
    std::array<std::int64_t, 10> rows = { 0, 2, 6, 9, 13, 17, 22, 25, 30, 32 };
    std::array<std::int32_t, 9> labels = { 1, 0, 1, 0, 0, 1, 0, 1, 0 };
};

class k_6_type : public graph_base_data {
public:
    k_6_type() {
        vertex_count = 6;
        edge_count = 15;
        cols_count = 30;
        rows_count = 7;
    }
    std::array<std::int32_t, 6> degrees = { 5, 5, 5, 5, 5, 5 };
    std::array<std::int32_t, 30> cols = { 1, 4, 3, 2, 5, 0, 2, 5, 4, 3, 4, 1, 3, 0, 5,
                                          4, 0, 2, 1, 5, 2, 0, 3, 1, 5, 1, 4, 0, 2, 3 };
    std::array<std::int64_t, 7> rows = { 0, 5, 10, 15, 20, 25, 30 };
    std::array<std::int32_t, 6> labels = { 0, 0, 0, 0, 0, 0 };
};

class empty_graph_type : public graph_base_data {
public:
    empty_graph_type() {
        vertex_count = 0;
        edge_count = 0;
        cols_count = 0;
        rows_count = 1;
    }
    std::array<std::int32_t, 0> degrees = {};
    std::array<std::int32_t, 0> cols = {};
    std::array<std::int64_t, 1> rows = { 0 };
    std::array<std::int32_t, 0> labels = {};
};

class subgraph_isomorphism_badarg_test {
public:
    using my_graph_type = dal::preview::undirected_adjacency_vector_graph<std::int32_t>;

    template <typename GraphType>
    auto create_graph() {
        GraphType graph_data;
        my_graph_type my_graph;

        auto &graph_impl = oneapi::dal::detail::get_impl(my_graph);
        auto &vertex_allocator = graph_impl._vertex_allocator;
        auto &edge_allocator = graph_impl._edge_allocator;

        const std::int64_t vertex_count = graph_data.get_vertex_count();
        const std::int64_t edge_count = graph_data.get_edge_count();
        const std::int64_t cols_count = graph_data.get_cols_count();
        const std::int64_t rows_count = graph_data.get_rows_count();

        typedef std::allocator_traits<std::allocator<char>>::rebind_traits<std::int32_t>
            int32_traits_t;
        typedef std::allocator_traits<std::allocator<char>>::rebind_traits<std::int64_t>
            int64_traits_t;
        std::int32_t *degrees = int32_traits_t::allocate(vertex_allocator, vertex_count);
        std::int32_t *cols = int32_traits_t::allocate(vertex_allocator, cols_count);
        std::int64_t *rows = int64_traits_t::allocate(edge_allocator, rows_count);
        std::int32_t *rows_vertex = int32_traits_t::allocate(vertex_allocator, rows_count);

        for (int i = 0; i < vertex_count; i++) {
            degrees[i] = graph_data.degrees[i];
        }

        for (int i = 0; i < cols_count; i++) {
            cols[i] = graph_data.cols[i];
        }
        for (int i = 0; i < rows_count; i++) {
            rows[i] = graph_data.rows[i];
            rows_vertex[i] = graph_data.rows[i];
        }
        graph_impl.set_topology(vertex_count, edge_count, rows, cols, cols_count, degrees);
        graph_impl.get_topology()._rows_vertex =
            oneapi::dal::preview::detail::container<std::int32_t>::wrap(rows_vertex, rows_count);
        return my_graph;
    }

    template <typename TargetGraphType, typename PatternGraphType>
    void check_subgraph_isomorphism(bool semantic_match,
                                    isomorphism_kind kind,
                                    std::int64_t max_match_count) {
        TargetGraphType target_graph_data;
        PatternGraphType pattern_graph_data;
        const auto target_graph = create_graph<TargetGraphType>();
        const auto pattern_graph = create_graph<PatternGraphType>();

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

SUBGRAPH_ISOMORPHISM_BADARG_TEST("Positive check") {
    REQUIRE_NOTHROW(
        this->check_subgraph_isomorphism<double_triangle_target_type, double_triangle_target_type>(
            false,
            isomorphism_kind::induced,
            0));
}

SUBGRAPH_ISOMORPHISM_BADARG_TEST("Empty target graph") {
    REQUIRE_THROWS_AS(
        (this->check_subgraph_isomorphism<empty_graph_type, double_triangle_target_type>(
            false,
            isomorphism_kind::induced,
            0)),
        invalid_argument);
}

SUBGRAPH_ISOMORPHISM_BADARG_TEST("Empty pattern graph") {
    REQUIRE_THROWS_AS(
        (this->check_subgraph_isomorphism<double_triangle_target_type, empty_graph_type>(
            false,
            isomorphism_kind::induced,
            0)),
        invalid_argument);
}

SUBGRAPH_ISOMORPHISM_BADARG_TEST("Throws if match count is negative") {
    REQUIRE_THROWS_AS(
        (this->check_subgraph_isomorphism<double_triangle_target_type, double_triangle_target_type>(
            false,
            isomorphism_kind::induced,
            -1)),
        invalid_argument);
}

SUBGRAPH_ISOMORPHISM_BADARG_TEST("Throws if target graph is smaller than pattern graph") {
    REQUIRE_THROWS_AS((this->check_subgraph_isomorphism<k_6_type, double_triangle_target_type>(
                          false,
                          isomorphism_kind::induced,
                          1)),
                      invalid_argument);
}

// SUBGRAPH_ISOMORPHISM_BADARG_TEST("Throws if semantic match is true") {
//     REQUIRE_THROWS_AS(
//         (this->check_subgraph_isomorphism<double_triangle_target_type, double_triangle_target_type>(
//             true,
//             isomorphism_kind::induced,
//             0)),
//         invalid_argument);
// }

} // namespace oneapi::dal::algo::subgraph_isomorphism::test
