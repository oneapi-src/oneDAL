#include <initializer_list>

#include "oneapi/dal/algo/jaccard/vertex_similarity.hpp"

#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::algo::jaccard::test {

class jaccard_badarg_test {
public:
    auto create_graph() {
        dal::preview::undirected_adjacency_vector_graph<> my_graph;
        auto &graph_impl = oneapi::dal::detail::get_impl(my_graph);

        const std::int64_t vertex_count = 7;
        const std::int64_t edge_count = 8;
        const std::int64_t rows_count = vertex_count + 1;
        const std::int64_t cols_count = edge_count * 2;

        std::int32_t *degrees = new std::int32_t[vertex_count]{ 1, 3, 4, 2, 3, 1, 2 };
        std::int32_t *cols =
            new std::int32_t[cols_count]{ 1, 0, 2, 4, 1, 3, 4, 5, 2, 6, 1, 2, 6, 2, 3, 4 };
        std::int32_t *rows = new std::int32_t[rows_count]{ 0, 1, 4, 8, 10, 13, 14, 16 };

        graph_impl.set_topology(vertex_count, edge_count, rows, cols, degrees);
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

#define JACCARD_BADARG_TEST(name) TEST_M(jaccard_badarg_test, name)

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
