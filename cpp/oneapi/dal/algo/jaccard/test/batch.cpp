#include <array>

#include "oneapi/dal/algo/jaccard/vertex_similarity.hpp"

#include "oneapi/dal/test/engine/common.hpp"
namespace oneapi::dal::algo::jaccard::test {

class jaccard_batch_test {
public:
    static constexpr std::int64_t nonzero_coeff_count = 8;
    static constexpr std::int64_t rows_count = 12;

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

    auto get_jaccard_descriptor(const std::int64_t row_range_begin,
                                const std::int64_t row_range_end,
                                const std::int64_t column_range_begin,
                                const std::int64_t column_range_end) {
        const auto jaccard_desc = dal::preview::jaccard::descriptor<>().set_block(
            { row_range_begin, row_range_end },
            { column_range_begin, column_range_end });
        return jaccard_desc;
    }

    template <typename VertexPairsDataType>
    void check_vertex_pairs_data(const VertexPairsDataType &vertex_pairs_data) {
        int correct_pair_num = 0;
        for (std::int64_t i = 0; i < nonzero_coeff_count; i++) {
            if (vertex_pairs_data[i] == vertex_pairs[i] &&
                vertex_pairs_data[i + rows_count] == vertex_pairs[i + rows_count])
                correct_pair_num++;
        }
        REQUIRE(correct_pair_num == nonzero_coeff_count);
    }

    template <typename JaccardCoeffsDataType>

    void check_jaccard_coeffs_data(const JaccardCoeffsDataType &jaccard_coeffs_data) {
        int correct_element_num = 0;
        for (std::int64_t i = 0; i < nonzero_coeff_count; i++) {
            if (Approx(jaccard_coeffs_data[i]) == jaccard_coeffs[i])
                correct_element_num++;
        }
        REQUIRE(correct_element_num == nonzero_coeff_count);
    }

private:
    static constexpr std::array<std::int64_t, 20> vertex_pairs = { 0, 0, 1, 1, 1, 2, 2, 2, 0, 0,
                                                                   0, 0, 0, 2, 1, 2, 3, 0, 1, 2 };
    static constexpr std::array<float, 8> jaccard_coeffs = { 1.0,  0.25, 1.0,      0.166667,
                                                             0.25, 0.25, 0.166667, 1.0 };
};

TEST_M(jaccard_batch_test, "jaccard batch test") {
    const auto jaccard_desc = this->get_jaccard_descriptor(0, 3, 0, 4);
    const auto my_graph = this->create_graph();
    dal::preview::jaccard::caching_builder builder;
    const auto result_vertex_similarity =
        dal::preview::vertex_similarity(jaccard_desc, my_graph, builder);

    INFO("check nonzero_coeff_count")
    const std::int64_t nonzero_coeff_count = result_vertex_similarity.get_nonzero_coeff_count();
    REQUIRE(nonzero_coeff_count == this->nonzero_coeff_count);

    INFO("check rows_count")
    const auto rows_count = result_vertex_similarity.get_vertex_pairs().get_row_count();
    REQUIRE(rows_count == this->rows_count);

    INFO("check vertex_pairs_data")
    auto vertex_pairs_table = result_vertex_similarity.get_vertex_pairs();
    homogen_table &vertex_pairs = static_cast<homogen_table &>(vertex_pairs_table);
    const auto vertex_pairs_data = vertex_pairs.get_data<int>();
    this->check_vertex_pairs_data(vertex_pairs_data);

    INFO("check jaccard_coeffs_data")
    auto coeffs_table = result_vertex_similarity.get_coeffs();
    homogen_table &coeffs = static_cast<homogen_table &>(coeffs_table);
    const auto jaccard_coeffs_data = coeffs.get_data<float>();
    this->check_jaccard_coeffs_data(jaccard_coeffs_data);
}
} // namespace oneapi::dal::algo::jaccard::test
