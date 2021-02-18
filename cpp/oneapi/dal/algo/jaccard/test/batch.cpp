
#include "oneapi/dal/test/engine/common.hpp"

#include <fstream>

#include "oneapi/dal/algo/jaccard/vertex_similarity.hpp"
#include "oneapi/dal/io/load_graph.hpp"

namespace oneapi::dal::algo::jaccard::test {

class graph_base_data {
public:
    graph_base_data() = default;

    void write_test_data() {
        std::ofstream outf(filename);
        if (outf.is_open()) {
            outf << file_content;
        }
        else {
            throw invalid_argument(dal::detail::error_messages::file_not_found());
        }
        outf.close();
    }

    void delete_test_data() {
        std::remove(filename.c_str());
    }

    const std::string get_filename() const {
        return filename;
    }

    std::int64_t get_nonzero_coeff_count() const {
        return nonzero_coeff_count;
    }

    std::int64_t get_rows_count() const {
        return rows_count;
    }

    const std::array<std::int64_t, 18> &get_vertex_pairs_data() const {
        return vertex_pairs_data;
    }

    const std::array<float, 6> &get_jaccard_coeffs_data() const {
        return jaccard_coeffs_data;
    }

private:
    const std::string filename = "test_graph.csv";
    const std::string file_content = "3 5 0 1 2 3 3 7 0 4 2 6 0 2";
    const std::array<std::int64_t, 18> vertex_pairs_data = { 0, 0, 1, 1, 2, 2, 0, 0, 0,
                                                             0, 0, 0, 0, 3, 1, 2, 1, 2 };
    const std::array<float, 6> jaccard_coeffs_data = { 1.0, 0.2, 1.0, 0.33333, 0.33333, 1.0 };
    const std::int64_t nonzero_coeff_count = 6;
    const std::int64_t rows_count = 12;
};

class jaccard_batch_test {
public:
    using my_graph_type = dal::preview::undirected_adjacency_vector_graph<>;

    template <typename GraphType>
    auto create_graph() {
        GraphType graph_data;
        graph_data.write_test_data();
        const dal::preview::graph_csv_data_source ds(graph_data.get_filename());
        const dal::preview::load_graph::descriptor<dal::preview::edge_list<int32_t>, my_graph_type>
            desc;
        auto my_graph = dal::preview::load_graph::load(desc, ds);
        graph_data.delete_test_data();
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

    template <typename VertexPairsDataType, std::size_t SIZE>
    void check_vertex_pairs_data(const VertexPairsDataType vertex_pairs_data,
                                 const std::array<std::int64_t, SIZE> &correct_vertex_pairs_data,
                                 std::int64_t element_count,
                                 std::int64_t pair_count) {
        int correct_pair_num = 0;
        for (std::int64_t i = 0; i < pair_count; i++) {
            if (vertex_pairs_data[i] == correct_vertex_pairs_data[i] &&
                vertex_pairs_data[i + element_count] ==
                    correct_vertex_pairs_data[i + element_count])
                correct_pair_num++;
        }
        REQUIRE(correct_pair_num == pair_count);
    }

    template <typename JaccardCoeffsDataType, std::size_t SIZE>
    void check_jaccard_coeffs_data(const JaccardCoeffsDataType jaccard_coeffs_data,
                                   const std::array<float, SIZE> &correct_jaccard_coeffs_data,
                                   std::int64_t element_count) {
        int correct_element_num = 0;
        for (std::int64_t i = 0; i < element_count; i++) {
            if (Approx(jaccard_coeffs_data[i]) == correct_jaccard_coeffs_data[i])
                correct_element_num++;
        }
        REQUIRE(correct_element_num == element_count);
    }
};

TEST_M(jaccard_batch_test, "jaccard batch test") {
    auto jaccard_desc = this->get_jaccard_descriptor(0, 3, 0, 4);
    graph_base_data graph;
    const auto my_graph = this->create_graph<graph_base_data>();
    dal::preview::jaccard::caching_builder builder;

    const auto result_vertex_similarity =
        dal::preview::vertex_similarity(jaccard_desc, my_graph, builder);

    INFO("check nonzero_coeff_count")
    const std::int64_t nonzero_coeff_count = result_vertex_similarity.get_nonzero_coeff_count();
    REQUIRE(nonzero_coeff_count == graph.get_nonzero_coeff_count());

    INFO("check rows_count")
    const auto rows_count = result_vertex_similarity.get_vertex_pairs().get_row_count();
    REQUIRE(rows_count == graph.get_rows_count());

    INFO("check vertex_pairs_data")
    auto vertex_pairs_table = result_vertex_similarity.get_vertex_pairs();
    homogen_table &vertex_pairs = static_cast<homogen_table &>(vertex_pairs_table);
    const auto vertex_pairs_data = vertex_pairs.get_data<int>();
    this->check_vertex_pairs_data(vertex_pairs_data,
                                  graph.get_vertex_pairs_data(),
                                  rows_count,
                                  nonzero_coeff_count);

    INFO("check jaccard_coeffs_data")
    auto coeffs_table = result_vertex_similarity.get_coeffs();
    homogen_table &coeffs = static_cast<homogen_table &>(coeffs_table);
    const auto jaccard_coeffs_data = coeffs.get_data<float>();

    this->check_jaccard_coeffs_data(jaccard_coeffs_data,
                                    graph.get_jaccard_coeffs_data(),
                                    nonzero_coeff_count);
}
} // namespace oneapi::dal::algo::jaccard::test