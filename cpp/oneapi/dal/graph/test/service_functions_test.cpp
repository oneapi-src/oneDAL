#include "oneapi/dal/test/engine/common.hpp"

#include <fstream>
#include <array>

#include "oneapi/dal/graph/service_functions.hpp"
#include "oneapi/dal/io/load_graph.hpp"

namespace oneapi::dal::graph::test {

  class graph_base_data {
  public:
    graph_base_data() = default;

    void write_test_data() {
      std::ofstream outf(filename);
      if (outf.is_open()) {
        outf << file_content;
      } else {
        throw invalid_argument(dal::detail::error_messages::file_not_found());
      }
      outf.close();
    }

    void delete_test_data() { std::remove(filename.c_str()); }

    const std::string get_filename() const { return filename; }

    std::int64_t get_correct_vertex_count() const { return vertex_count; }

    std::int64_t get_correct_edge_count() const { return edge_count; }

  protected:
    std::string filename;
    std::string file_content;
    std::int64_t vertex_count;
    std::int64_t edge_count;
  };

  class two_vertices_graph_type : public graph_base_data {
  public:
    two_vertices_graph_type() {
      filename = "test_two_vertices_graph.csv";
      file_content = "0 1";
      vertex_count = 2;
      edge_count = 1;
    }

    static const std::int64_t correct_vertex_count = 2;
    static const std::int64_t neighbors_count = 2;
    std::array<std::int64_t, correct_vertex_count> degrees = {1, 1};
    std::array<std::int64_t, neighbors_count> neighbors = {1, 0};
  };

  class acyclic_graph_type : public graph_base_data {
  public:
    acyclic_graph_type() {
      filename = "test_acyclic_graph.csv";
      file_content = "0 1 0 2 0 4 2 3 2 6 3 5 3 7";
      vertex_count = 8;
      edge_count = 7;
    }

    static const std::int64_t correct_vertex_count = 8;
    static const std::int64_t neighbors_count = 14;
    std::array<std::int64_t, correct_vertex_count> degrees = {3, 1, 3, 3,
                                                              1, 1, 1, 1};
    std::array<std::int64_t, neighbors_count> neighbors = {1, 2, 4, 0, 0, 3, 6,
                                                           2, 5, 7, 0, 3, 2, 3};
  };

  class complete_graph_type : public graph_base_data {
  public:
    complete_graph_type() {
      filename = "test_complete_graph.csv";
      file_content = "0 1 0 2 0 3 0 4 1 2 1 3 1 4 2 3 2 4 3 4";
      vertex_count = 5;
      edge_count = 10;
    }

    static const std::int64_t correct_vertex_count = 5;
    static const std::int64_t neighbors_count = 20;
    std::array<std::int64_t, correct_vertex_count> degrees = {4, 4, 4, 4, 4};
    std::array<std::int64_t, neighbors_count> neighbors = {
        1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3};
  };

  class service_functions_test {
  public:
    using my_graph_type = dal::preview::undirected_adjacency_vector_graph<>;

    template <typename GraphType> auto create_graph() {
      GraphType graph_data;
      graph_data.write_test_data();
      const dal::preview::graph_csv_data_source ds(graph_data.get_filename());
      const dal::preview::load_graph::descriptor<
          dal::preview::edge_list<int32_t>, my_graph_type> desc;
      auto my_graph = dal::preview::load_graph::load(desc, ds);
      graph_data.delete_test_data();
      return my_graph;
    }

    template <typename GraphType>
    void check_vertex_degree(const GraphType &graph,
                             std::int64_t *correct_degrees,
                             std::int64_t vertex_count) {
      int correct_degree_num = 0;
      for (std::int64_t i = 0; i < vertex_count; i++) {
        if (dal::preview::get_vertex_degree(graph, i) == correct_degrees[i])
          correct_degree_num++;
      }
      REQUIRE(correct_degree_num == vertex_count);
    }

    template <typename GraphType>
    void check_vertex_neighbors(const GraphType &graph,
                                std::int64_t *correct_neighbors,
                                std::int64_t vertex_count,
                                std::int64_t neighbors_count) {
      int correct_neighbors_num = 0;
      std::int64_t neighbor_index = 0;
      for (std::int64_t j = 0; j < vertex_count; j++) {
        const auto[ start, end ] = dal::preview::get_vertex_neighbors(graph, j);
        for (auto i = start; i != end; ++i) {
          if (*i == correct_neighbors[neighbor_index])
            correct_neighbors_num++;
          neighbor_index++;
        }
      }
      REQUIRE(correct_neighbors_num == neighbors_count);
    }

    void general_checks() {
      two_vertices_graph_type tvg;
      acyclic_graph_type ag;
      complete_graph_type cg;

      const auto two_vertices_graph = create_graph<two_vertices_graph_type>();
      const auto acyclic_graph = create_graph<acyclic_graph_type>();
      const auto complete_graph = create_graph<complete_graph_type>();

      SECTION("check get_vertex_count") {
        REQUIRE(dal::preview::get_vertex_count(two_vertices_graph) ==
                tvg.get_correct_vertex_count());
        REQUIRE(dal::preview::get_vertex_count(acyclic_graph) ==
                ag.get_correct_vertex_count());
        REQUIRE(dal::preview::get_vertex_count(complete_graph) ==
                cg.get_correct_vertex_count());
      }

      SECTION("check get_edge_count") {
        REQUIRE(dal::preview::get_edge_count(two_vertices_graph) ==
                tvg.get_correct_edge_count());
        REQUIRE(dal::preview::get_edge_count(acyclic_graph) ==
                ag.get_correct_edge_count());
        REQUIRE(dal::preview::get_edge_count(complete_graph) ==
                cg.get_correct_edge_count());
      }

      SECTION("check get_vertex_degree") {
        check_vertex_degree(two_vertices_graph, tvg.degrees.data(),
                            tvg.get_correct_vertex_count());
        check_vertex_degree(acyclic_graph, ag.degrees.data(),
                            ag.get_correct_vertex_count());
        check_vertex_degree(complete_graph, cg.degrees.data(),
                            cg.get_correct_vertex_count());
      }

      SECTION("check get_vertex_neighbors") {
        check_vertex_neighbors(two_vertices_graph, tvg.neighbors.data(),
                               tvg.get_correct_vertex_count(),
                               tvg.neighbors_count);
        check_vertex_neighbors(acyclic_graph, ag.neighbors.data(),
                               ag.get_correct_vertex_count(),
                               ag.neighbors_count);
        check_vertex_neighbors(complete_graph, cg.neighbors.data(),
                               cg.get_correct_vertex_count(),
                               cg.neighbors_count);
      }
    }
  };

  TEST_M(service_functions_test, "graph service functions test") {
    this->general_checks();
  }

} // namespace oneapi::dal::graph::test