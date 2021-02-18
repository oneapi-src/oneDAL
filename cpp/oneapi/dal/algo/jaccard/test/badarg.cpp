#include <fstream>

#include "oneapi/dal/algo/jaccard/vertex_similarity.hpp"
#include "oneapi/dal/io/load_graph.hpp"

#include "oneapi/dal/test/engine/common.hpp"

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

private:
    const std::string filename = "test_graph.csv";
    const std::string file_content = "3 5 0 1 2 3 3 7 0 4 2 6 0 2";
    ;
};

class jaccard_badarg_test {
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
};

TEST_M(jaccard_badarg_test, "jaccard badarg test") {
    graph_base_data graph;
    const auto my_graph = this->create_graph<graph_base_data>();
    dal::preview::jaccard::caching_builder builder;

    auto jaccard_desc = this->get_jaccard_descriptor(-2, 0, 0, 3);
    REQUIRE_THROWS_WITH(oneapi::dal::preview::vertex_similarity(jaccard_desc, my_graph, builder),
                        "Negative interval");

    jaccard_desc = this->get_jaccard_descriptor(0, 2, -3, 0);
    REQUIRE_THROWS_WITH(oneapi::dal::preview::vertex_similarity(jaccard_desc, my_graph, builder),
                        "Negative interval");

    jaccard_desc = this->get_jaccard_descriptor(2, 0, 0, 3);
    REQUIRE_THROWS_WITH(oneapi::dal::preview::vertex_similarity(jaccard_desc, my_graph, builder),
                        "Row begin is greater than row end");

    jaccard_desc = this->get_jaccard_descriptor(0, 2, 3, 0);
    REQUIRE_THROWS_WITH(oneapi::dal::preview::vertex_similarity(jaccard_desc, my_graph, builder),
                        "Column begin is greater than column end");

    jaccard_desc = this->get_jaccard_descriptor(0, 0, 0, 3);
    REQUIRE_THROWS_WITH(oneapi::dal::preview::vertex_similarity(jaccard_desc, my_graph, builder),
                        "Row count is lower than or equal to zero");

    jaccard_desc = this->get_jaccard_descriptor(0, 2, 0, 0);
    REQUIRE_THROWS_WITH(oneapi::dal::preview::vertex_similarity(jaccard_desc, my_graph, builder),
                        "Row count is lower than or equal to zero"); // Column expected

    jaccard_desc = this->get_jaccard_descriptor(0, 9, 0, 9);
    REQUIRE_THROWS_WITH(oneapi::dal::preview::vertex_similarity(jaccard_desc, my_graph, builder),
                        "Interval is greater than vertex count");
}
} // namespace oneapi::dal::algo::jaccard::test