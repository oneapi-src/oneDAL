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

#include "oneapi/dal/algo/jaccard/vertex_similarity.hpp"

#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::algo::jaccard::test {

class jaccard_badarg_test {
public:
    auto create_graph() {
        oneapi::dal::preview::undirected_adjacency_vector_graph<> my_graph;
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

    void check_vertex_similarity(const std::int64_t row_range_begin,
                                 const std::int64_t row_range_end,
                                 const std::int64_t column_range_begin,
                                 const std::int64_t column_range_end) {
        const auto jaccard_desc = dal::preview::jaccard::descriptor<>().set_block(
            { row_range_begin, row_range_end },
            { column_range_begin, column_range_end });
        const auto my_graph = create_graph();

        dal::preview::jaccard::caching_builder builder;

        const auto result_vertex_similarity =
            oneapi::dal::preview::vertex_similarity(jaccard_desc, my_graph, builder);
    }
};

#define JACCARD_BADARG_TEST(name) TEST_M(jaccard_badarg_test, name, "[jaccard][badarg]")

JACCARD_BADARG_TEST("accepts non-negative block ranges") {
    REQUIRE_NOTHROW(this->check_vertex_similarity(0, 2, 0, 3));
}

JACCARD_BADARG_TEST("throws if row_range_begin is negative") {
    REQUIRE_THROWS_AS(this->check_vertex_similarity(-2, 0, 0, 3), invalid_argument);
}

JACCARD_BADARG_TEST("throws if column_range_begin is negative") {
    REQUIRE_THROWS_AS(this->check_vertex_similarity(0, 2, -3, 0), invalid_argument);
}

JACCARD_BADARG_TEST("throws if row_range_begin is greater than row_range_end") {
    REQUIRE_THROWS_AS(this->check_vertex_similarity(2, 0, 0, 3), invalid_argument);
}

JACCARD_BADARG_TEST("throws if column_range_begin is greater than column_range_end") {
    REQUIRE_THROWS_AS(this->check_vertex_similarity(0, 2, 3, 0), invalid_argument);
}

JACCARD_BADARG_TEST("throws if row count is equal to zero") {
    REQUIRE_THROWS_AS(this->check_vertex_similarity(0, 0, 0, 3), domain_error);
}

JACCARD_BADARG_TEST("throws if column count is equal to zero") {
    REQUIRE_THROWS_AS(this->check_vertex_similarity(0, 2, 0, 0), domain_error);
}

JACCARD_BADARG_TEST("throws if block ranges is greater than vertex count") {
    REQUIRE_THROWS_AS(this->check_vertex_similarity(0, 8, 0, 8), out_of_range);
}

} // namespace oneapi::dal::algo::jaccard::test
