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

#include <array>

#include "oneapi/dal/algo/triangle_counting/backend/cpu/vertex_ranking_default_kernel_scalar.hpp"
#include "oneapi/dal/algo/triangle_counting/vertex_ranking.hpp"

#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::algo::triangle_counting::test {
class graph_base_data {
public:
    graph_base_data() = default;

    std::int64_t get_correct_vertex_count() const {
        return vertex_count;
    }

    std::int64_t get_correct_edge_count() const {
        return edge_count;
    }

    std::int64_t get_correct_triangle_count() const {
        return global_triangle_count;
    }

protected:
    std::int64_t vertex_count;
    std::int64_t edge_count;
    std::int64_t global_triangle_count;
};
class complete_graph_5_type : public graph_base_data {
public:
    complete_graph_5_type() {
        vertex_count = 5;
        edge_count = 10;
        global_triangle_count = 10;
    }

    std::array<std::int32_t, 5> degrees = { 4, 4, 4, 4, 4 };
    std::array<std::int32_t, 20> cols = {
        1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3
    };
    std::array<std::int64_t, 6> rows = { 0, 4, 8, 12, 16, 20 };
    std::array<std::int64_t, 5> local_triangles = { 6, 6, 6, 6, 6 };
};

class complete_graph_9_type : public graph_base_data {
public:
    complete_graph_9_type() {
        vertex_count = 9;
        edge_count = 36;
        global_triangle_count = 84;
    }

    std::array<std::int32_t, 9> degrees = { 8, 8, 8, 8, 8, 8, 8, 8, 8 };
    std::array<std::int32_t, 72> cols = { 1, 2, 3, 4, 5, 6, 7, 8, 0, 2, 3, 4, 5, 6, 7, 8, 0, 1,
                                          3, 4, 5, 6, 7, 8, 0, 1, 2, 4, 5, 6, 7, 8, 0, 1, 2, 3,
                                          5, 6, 7, 8, 0, 1, 2, 3, 4, 6, 7, 8, 0, 1, 2, 3, 4, 5,
                                          7, 8, 0, 1, 2, 3, 4, 5, 6, 8, 0, 1, 2, 3, 4, 5, 6, 7 };
    std::array<std::int64_t, 10> rows = { 0, 8, 16, 24, 32, 40, 48, 56, 64, 72 };
    std::array<std::int64_t, 9> local_triangles = { 28, 28, 28, 28, 28, 28, 28, 28, 28 };
};

class acyclic_graph_type : public graph_base_data {
public:
    acyclic_graph_type() {
        vertex_count = 8;
        edge_count = 7;
        global_triangle_count = 0;
    }

    std::array<std::int32_t, 8> degrees = { 3, 1, 3, 3, 1, 1, 1, 1 };
    std::array<std::int32_t, 14> cols = { 1, 2, 4, 0, 0, 3, 6, 2, 5, 7, 0, 3, 2, 3 };
    std::array<std::int64_t, 9> rows = { 0, 3, 4, 7, 10, 11, 12, 13, 14 };
    std::array<std::int64_t, 8> local_triangles = { 0, 0, 0, 0, 0, 0, 0, 0 };
};

class two_vertices_graph_type : public graph_base_data {
public:
    two_vertices_graph_type() {
        vertex_count = 2;
        edge_count = 1;
        global_triangle_count = 0;
    }

    std::array<std::int32_t, 2> degrees = { 1, 1 };
    std::array<std::int32_t, 2> cols = { 1, 0 };
    std::array<std::int64_t, 3> rows = { 0, 1, 2 };
    std::array<std::int64_t, 2> local_triangles = { 0, 0 };
};

class cycle_graph_type : public graph_base_data {
public:
    cycle_graph_type() {
        vertex_count = 9;
        edge_count = 9;
        global_triangle_count = 0;
    }

    std::array<std::int32_t, 9> degrees = { 2, 2, 2, 2, 2, 2, 2, 2, 2 };
    std::array<std::int32_t, 18> cols = { 1, 8, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 0, 7 };
    std::array<std::int64_t, 10> rows = { 0, 2, 4, 6, 8, 10, 12, 14, 16, 18 };
    std::array<std::int64_t, 9> local_triangles = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
};

class triangle_graph_type : public graph_base_data {
public:
    triangle_graph_type() {
        vertex_count = 3;
        edge_count = 3;
        global_triangle_count = 1;
    }

    std::array<std::int32_t, 3> degrees = { 2, 2, 2 };
    std::array<std::int32_t, 6> cols = { 1, 2, 0, 2, 0, 1 };
    std::array<std::int64_t, 4> rows = { 0, 2, 4, 6 };
    std::array<std::int64_t, 3> local_triangles = { 1, 1, 1 };
};

class wheel_graph_type : public graph_base_data {
public:
    wheel_graph_type() {
        vertex_count = 6;
        edge_count = 10;
        global_triangle_count = 5;
    }

    std::array<std::int32_t, 6> degrees = { 5, 3, 3, 3, 3, 3 };
    std::array<std::int32_t, 20> cols = {
        1, 2, 3, 4, 5, 0, 2, 5, 0, 1, 3, 0, 2, 4, 0, 3, 5, 0, 1, 4
    };
    std::array<std::int64_t, 7> rows = { 0, 5, 8, 11, 14, 17, 20 };
    std::array<std::int64_t, 6> local_triangles = { 5, 2, 2, 2, 2, 2 };
};

class triangle_counting_test {
public:
    using my_graph_type = dal::preview::undirected_adjacency_vector_graph<>;

    template <typename GraphType>
    auto create_graph() {
        GraphType graph_data;
        my_graph_type my_graph;

        auto &graph_impl = oneapi::dal::detail::get_impl(my_graph);
        auto &vertex_allocator = graph_impl._vertex_allocator;
        auto &edge_allocator = graph_impl._edge_allocator;

        const std::int64_t vertex_count = graph_data.get_correct_vertex_count();
        const std::int64_t edge_count = graph_data.get_correct_edge_count();
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

        std::int32_t *degrees = new (degrees_) std::int32_t[vertex_count];
        std::int32_t *cols = new (cols_) std::int32_t[cols_count];
        std::int64_t *rows = new (rows_) std::int64_t[rows_count];
        std::int32_t *rows_vertex = new (rows_vertex_) std::int32_t[rows_count];

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
        graph_impl.set_topology(vertex_count, edge_count, rows, cols, degrees);
        graph_impl.get_topology()._rows_vertex =
            oneapi::dal::preview::detail::container<std::int32_t>::wrap(rows_vertex, rows_count);
        return my_graph;
    }

    template <typename Graph, std::size_t SIZE>
    void check_triangle_counting(const Graph &my_graph,
                                 const std::array<std::int64_t, SIZE> &correct_triangles,
                                 std::int64_t vertex_count,
                                 std::int64_t global_triangle_count) {
        std::allocator<char> alloc;
        const auto tc_desc = dal::preview::triangle_counting::descriptor<
            float,
            dal::preview::triangle_counting::method::ordered_count,
            dal::preview::triangle_counting::task::local_and_global,
            std::allocator<char>>(alloc);
        const auto result_vertex_ranking = dal::preview::vertex_ranking(tc_desc, my_graph);

        auto local_triangles_table = result_vertex_ranking.get_ranks();
        const auto &local_triangles =
            static_cast<const dal::homogen_table &>(local_triangles_table);
        const auto local_triangles_data = local_triangles.get_data<std::int64_t>();

        REQUIRE(result_vertex_ranking.get_global_rank() == global_triangle_count);
        REQUIRE(local_triangles_table.get_row_count() == vertex_count);

        int correct_local_triangle_count = 0;
        for (std::int64_t i = 0; i < vertex_count; i++) {
            if (local_triangles_data[i] == correct_triangles[i])
                correct_local_triangle_count++;
        }
        REQUIRE(correct_local_triangle_count == vertex_count);
    }

    template <typename GraphType>
    void check_triangle_counting_global_scalar_() {
        GraphType graph_data;
        REQUIRE(dal::preview::triangle_counting::backend::triangle_counting_global_scalar_<
                    dal::backend::cpu_dispatch_default>(graph_data.cols.data(),
                                                        graph_data.rows.data(),
                                                        graph_data.degrees.data(),
                                                        graph_data.get_correct_vertex_count(),
                                                        graph_data.get_correct_edge_count()) ==
                graph_data.get_correct_triangle_count());
    }

    template <typename GraphType>
    void check_triangle_counting_global_vector_() {
        GraphType graph_data;
        REQUIRE(dal::preview::triangle_counting::backend::triangle_counting_global_vector_<
                    dal::backend::cpu_dispatch_default>(graph_data.cols.data(),
                                                        graph_data.rows.data(),
                                                        graph_data.degrees.data(),
                                                        graph_data.get_correct_vertex_count(),
                                                        graph_data.get_correct_edge_count()) ==
                graph_data.get_correct_triangle_count());
    }

    template <typename GraphType>
    void check_triangle_counting_global_vector_relabel_() {
        GraphType graph_data;
        REQUIRE(dal::preview::triangle_counting::backend::triangle_counting_global_vector_relabel_<
                    dal::backend::cpu_dispatch_default>(graph_data.cols.data(),
                                                        graph_data.rows.data(),
                                                        graph_data.degrees.data(),
                                                        graph_data.get_correct_vertex_count(),
                                                        graph_data.get_correct_edge_count()) ==
                graph_data.get_correct_triangle_count());
    }
};

TEST_M(triangle_counting_test, "Average_degree < 4") {
    complete_graph_5_type complete_graph;
    const auto graph = create_graph<complete_graph_5_type>();
    this->check_triangle_counting(graph,
                                  complete_graph.local_triangles,
                                  complete_graph.get_correct_vertex_count(),
                                  complete_graph.get_correct_triangle_count());
}

TEST_M(triangle_counting_test, "Average_degree = 4") {
    complete_graph_9_type complete_graph;
    const auto my_graph = create_graph<complete_graph_9_type>();
    this->check_triangle_counting(my_graph,
                                  complete_graph.local_triangles,
                                  complete_graph.get_correct_vertex_count(),
                                  complete_graph.get_correct_triangle_count());
}

TEST_M(triangle_counting_test, "Check triangle_counting_global_scalar_") {
    this->check_triangle_counting_global_scalar_<complete_graph_5_type>();
    this->check_triangle_counting_global_scalar_<complete_graph_9_type>();
    this->check_triangle_counting_global_scalar_<acyclic_graph_type>();
    this->check_triangle_counting_global_scalar_<two_vertices_graph_type>();
    this->check_triangle_counting_global_scalar_<cycle_graph_type>();
    this->check_triangle_counting_global_scalar_<triangle_graph_type>();
    this->check_triangle_counting_global_scalar_<wheel_graph_type>();
}

TEST_M(triangle_counting_test, "Check triangle_counting_global_vector_") {
    this->check_triangle_counting_global_vector_<complete_graph_5_type>();
    this->check_triangle_counting_global_vector_<complete_graph_9_type>();
    this->check_triangle_counting_global_vector_<acyclic_graph_type>();
    this->check_triangle_counting_global_vector_<two_vertices_graph_type>();
    this->check_triangle_counting_global_vector_<cycle_graph_type>();
    this->check_triangle_counting_global_vector_<triangle_graph_type>();
    this->check_triangle_counting_global_vector_<wheel_graph_type>();
}

TEST_M(triangle_counting_test, "Check triangle_counting_global_vector_relabel_") {
    this->check_triangle_counting_global_vector_relabel_<complete_graph_5_type>();
    this->check_triangle_counting_global_vector_relabel_<complete_graph_9_type>();
    this->check_triangle_counting_global_vector_relabel_<acyclic_graph_type>();
    this->check_triangle_counting_global_vector_relabel_<two_vertices_graph_type>();
    this->check_triangle_counting_global_vector_relabel_<cycle_graph_type>();
    this->check_triangle_counting_global_vector_relabel_<triangle_graph_type>();
    this->check_triangle_counting_global_vector_relabel_<wheel_graph_type>();
}

TEST_M(triangle_counting_test, "Check triangle counting in a acyclic graph") {
    acyclic_graph_type acyclic_graph;
    const auto my_acyclic_graph = create_graph<acyclic_graph_type>();
    this->check_triangle_counting(my_acyclic_graph,
                                  acyclic_graph.local_triangles,
                                  acyclic_graph.get_correct_vertex_count(),
                                  acyclic_graph.get_correct_triangle_count());
}

TEST_M(triangle_counting_test, "Check triangle counting in a graph with 2 vertices") {
    two_vertices_graph_type two_vertices_graph;
    const auto my_two_vertices_graph = create_graph<two_vertices_graph_type>();
    this->check_triangle_counting(my_two_vertices_graph,
                                  two_vertices_graph.local_triangles,
                                  two_vertices_graph.get_correct_vertex_count(),
                                  two_vertices_graph.get_correct_triangle_count());
}

TEST_M(triangle_counting_test, "Check triangle counting in a cycle graph") {
    cycle_graph_type cycle_graph;
    const auto my_cycle_graph = create_graph<cycle_graph_type>();
    this->check_triangle_counting(my_cycle_graph,
                                  cycle_graph.local_triangles,
                                  cycle_graph.get_correct_vertex_count(),
                                  cycle_graph.get_correct_triangle_count());
}

TEST_M(triangle_counting_test, "Check triangle counting in a triangle graph") {
    triangle_graph_type triangle_graph;
    const auto my_triangle_graph = create_graph<triangle_graph_type>();
    this->check_triangle_counting(my_triangle_graph,
                                  triangle_graph.local_triangles,
                                  triangle_graph.get_correct_vertex_count(),
                                  triangle_graph.get_correct_triangle_count());
}

TEST_M(triangle_counting_test, "Check triangle counting in a wheel graph") {
    wheel_graph_type wheel_graph;
    const auto my_wheel_graph = create_graph<wheel_graph_type>();
    this->check_triangle_counting(my_wheel_graph,
                                  wheel_graph.local_triangles,
                                  wheel_graph.get_correct_vertex_count(),
                                  wheel_graph.get_correct_triangle_count());
}

} // namespace oneapi::dal::algo::triangle_counting::test
