#include <fstream>
#include <array>

#include "oneapi/dal/graph/service_functions.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/load_graph.hpp"

#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::graph::test {

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

    std::int64_t get_correct_vertex_count() const {
        return vertex_count;
    }

    std::int64_t get_correct_edge_count() const {
        return edge_count;
    }

    std::int64_t get_neighbors_count() const {
        return neighbors_count;
    }

protected:
    std::string filename;
    std::string file_content;
    std::int64_t vertex_count;
    std::int64_t edge_count;
    std::int64_t neighbors_count;
};

class two_vertices_graph_type : public graph_base_data {
public:
    two_vertices_graph_type() {
        filename = "test_two_vertices_graph.csv";
        file_content = "0 1";
        vertex_count = 2;
        edge_count = 1;
        neighbors_count = 2;
    }

    std::array<std::int64_t, 2> degrees = { 1, 1 };
    std::array<std::int64_t, 2> neighbors = { 1, 0 };
};

class acyclic_graph_type : public graph_base_data {
public:
    acyclic_graph_type() {
        filename = "test_acyclic_graph.csv";
        file_content = "0 1 0 2 0 4 2 3 2 6 3 5 3 7";
        vertex_count = 8;
        edge_count = 7;
        neighbors_count = 14;
    }

    std::array<std::int64_t, 8> degrees = { 3, 1, 3, 3, 1, 1, 1, 1 };
    std::array<std::int64_t, 14> neighbors = { 1, 2, 4, 0, 0, 3, 6, 2, 5, 7, 0, 3, 2, 3 };
};

class complete_graph_type : public graph_base_data {
public:
    complete_graph_type() {
        filename = "test_complete_graph.csv";
        file_content = "0 1 0 2 0 3 0 4 1 2 1 3 1 4 2 3 2 4 3 4";
        vertex_count = 5;
        edge_count = 10;
        neighbors_count = 20;
    }

    std::array<std::int64_t, 5> degrees = { 4, 4, 4, 4, 4 };
    std::array<std::int64_t, 20> neighbors = { 1, 2, 3, 4, 0, 2, 3, 4, 0, 1,
                                               3, 4, 0, 1, 2, 4, 0, 1, 2, 3 };
};

class pseudograph_type : public graph_base_data {
public:
    pseudograph_type() {
        filename = "test_pseudograph.csv";
        file_content = "0 0 1 1 2 2 3 3";
        vertex_count = 4;
        edge_count = 0;
        neighbors_count = 0;
    }

    std::array<std::int64_t, 4> degrees = { 0, 0, 0, 0 };
};

class service_functions_test {
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

    template <typename GraphType, std::size_t SIZE>
    void check_vertex_degree(const GraphType &graph,
                             const std::array<std::int64_t, SIZE> &correct_degrees,
                             std::int64_t vertex_count) {
        int correct_degree_num = 0;
        for (std::int64_t i = 0; i < vertex_count; i++) {
            if (dal::preview::get_vertex_degree(graph, i) == correct_degrees[i])
                correct_degree_num++;
        }
        REQUIRE(correct_degree_num == vertex_count);
    }

    template <typename GraphType, std::size_t SIZE>
    void check_vertex_neighbors(const GraphType &graph,
                                const std::array<std::int64_t, SIZE> &correct_neighbors,
                                std::int64_t vertex_count,
                                std::int64_t neighbors_count) {
        int correct_neighbors_num = 0;
        std::int64_t neighbor_index = 0;
        for (std::int64_t j = 0; j < vertex_count; j++) {
            const auto [start, end] = dal::preview::get_vertex_neighbors(graph, j);
            for (auto i = start; i != end; ++i) {
                if (*i == correct_neighbors[neighbor_index])
                    correct_neighbors_num++;
                neighbor_index++;
            }
        }
        REQUIRE(correct_neighbors_num == neighbors_count);
    }

    void general_checks() {
        two_vertices_graph_type two_vertices_graph;
        acyclic_graph_type acyclic_graph;
        complete_graph_type complete_graph;
        pseudograph_type pseudograph;

        const auto my_two_vertices_graph = create_graph<two_vertices_graph_type>();
        const auto my_acyclic_graph = create_graph<acyclic_graph_type>();
        const auto my_complete_graph = create_graph<complete_graph_type>();
        const auto my_pseudograph = create_graph<pseudograph_type>();
        my_graph_type my_empty_graph;

        INFO("check get_vertex_count")
        REQUIRE(dal::preview::get_vertex_count(my_two_vertices_graph) ==
                two_vertices_graph.get_correct_vertex_count());
        REQUIRE(dal::preview::get_vertex_count(my_acyclic_graph) ==
                acyclic_graph.get_correct_vertex_count());
        REQUIRE(dal::preview::get_vertex_count(my_complete_graph) ==
                complete_graph.get_correct_vertex_count());
        REQUIRE(dal::preview::get_vertex_count(my_pseudograph) ==
                pseudograph.get_correct_vertex_count());
        REQUIRE(dal::preview::get_vertex_count(my_empty_graph) == 0);

        INFO("check get_edge_count")
        REQUIRE(dal::preview::get_edge_count(my_two_vertices_graph) ==
                two_vertices_graph.get_correct_edge_count());
        REQUIRE(dal::preview::get_edge_count(my_acyclic_graph) ==
                acyclic_graph.get_correct_edge_count());
        REQUIRE(dal::preview::get_edge_count(my_complete_graph) ==
                complete_graph.get_correct_edge_count());
        REQUIRE(dal::preview::get_edge_count(my_pseudograph) ==
                pseudograph.get_correct_edge_count());
        REQUIRE(dal::preview::get_edge_count(my_empty_graph) == 0);

        INFO("check get_vertex_degree")
        check_vertex_degree(my_two_vertices_graph,
                            two_vertices_graph.degrees,
                            two_vertices_graph.get_correct_vertex_count());
        check_vertex_degree(my_acyclic_graph,
                            acyclic_graph.degrees,
                            acyclic_graph.get_correct_vertex_count());
        check_vertex_degree(my_complete_graph,
                            complete_graph.degrees,
                            complete_graph.get_correct_vertex_count());
        check_vertex_degree(my_pseudograph,
                            pseudograph.degrees,
                            pseudograph.get_correct_vertex_count());
        REQUIRE_THROWS_WITH(dal::preview::get_vertex_degree(my_empty_graph, 0),
                            "Vertex index is out of range, expect index in [0, vertex_count)");

        INFO("check get_vertex_neighbors")
        check_vertex_neighbors(my_two_vertices_graph,
                               two_vertices_graph.neighbors,
                               two_vertices_graph.get_correct_vertex_count(),
                               two_vertices_graph.get_neighbors_count());
        check_vertex_neighbors(my_acyclic_graph,
                               acyclic_graph.neighbors,
                               acyclic_graph.get_correct_vertex_count(),
                               acyclic_graph.get_neighbors_count());
        check_vertex_neighbors(my_complete_graph,
                               complete_graph.neighbors,
                               complete_graph.get_correct_vertex_count(),
                               complete_graph.get_neighbors_count());
        for (std::int64_t j = 0; j < pseudograph.get_correct_vertex_count(); j++) {
            const auto [start, end] = dal::preview::get_vertex_neighbors(my_pseudograph, j);
            REQUIRE(start == end);
        }
        REQUIRE_THROWS_WITH(dal::preview::get_vertex_neighbors(my_empty_graph, 0),
                            "Vertex index is out of range, expect index in [0, vertex_count)");
    }
};

TEST_M(service_functions_test, "graph service functions test") {
    this->general_checks();
}

} // namespace oneapi::dal::graph::test