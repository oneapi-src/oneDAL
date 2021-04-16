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

    template <typename GraphType>
    void check_local_task() {
        GraphType graph_data;
        const auto graph = create_graph<GraphType>();
        std::int64_t vertex_count = graph_data.get_correct_vertex_count();

        std::allocator<char> alloc;
        const auto tc_desc = dal::preview::triangle_counting::descriptor<
            float,
            dal::preview::triangle_counting::method::ordered_count,
            dal::preview::triangle_counting::task::local,
            std::allocator<char>>(alloc);

        const auto result_vertex_ranking = dal::preview::vertex_ranking(tc_desc, graph);

        auto local_triangles_table = result_vertex_ranking.get_ranks();
        const auto &local_triangles =
            static_cast<const dal::homogen_table &>(local_triangles_table);
        const auto local_triangles_data = local_triangles.get_data<std::int64_t>();

        REQUIRE(local_triangles_table.get_row_count() == vertex_count);

        int correct_local_triangle_count = 0;
        for (std::int64_t i = 0; i < vertex_count; i++) {
            if (local_triangles_data[i] == graph_data.local_triangles[i]) {
                correct_local_triangle_count++;
            }
        }
        REQUIRE(correct_local_triangle_count == vertex_count);
    }

    template <typename GraphType>
    void check_local_and_global_task() {
        GraphType graph_data;
        const auto graph = create_graph<GraphType>();
        std::int64_t vertex_count = graph_data.get_correct_vertex_count();
        std::int64_t global_triangle_count = graph_data.get_correct_triangle_count();

        std::allocator<char> alloc;
        const auto tc_desc = dal::preview::triangle_counting::descriptor<
            float,
            dal::preview::triangle_counting::method::ordered_count,
            dal::preview::triangle_counting::task::local_and_global,
            std::allocator<char>>(alloc);
        const auto result_vertex_ranking = dal::preview::vertex_ranking(tc_desc, graph);

        auto local_triangles_table = result_vertex_ranking.get_ranks();
        const auto &local_triangles =
            static_cast<const dal::homogen_table &>(local_triangles_table);
        const auto local_triangles_data = local_triangles.get_data<std::int64_t>();

        REQUIRE(result_vertex_ranking.get_global_rank() == global_triangle_count);
        REQUIRE(local_triangles_table.get_row_count() == vertex_count);

        int correct_local_triangle_count = 0;
        for (std::int64_t i = 0; i < vertex_count; i++) {
            if (local_triangles_data[i] == graph_data.local_triangles[i])
                correct_local_triangle_count++;
        }
        REQUIRE(correct_local_triangle_count == vertex_count);
    }

    template <typename GraphType>
    void check_global_task_relabeled() {
        GraphType graph_data;
        const auto graph = create_graph<GraphType>();
        std::int64_t global_triangle_count = graph_data.get_correct_triangle_count();

        std::allocator<char> alloc;
        auto tc_desc = dal::preview::triangle_counting::descriptor<
            float,
            dal::preview::triangle_counting::method::ordered_count,
            dal::preview::triangle_counting::task::global,
            std::allocator<char>>(alloc);

        const auto relabel = tc_desc.get_relabel();
        if (relabel == dal::preview::triangle_counting::relabel::yes) {
            const auto result_vertex_ranking = dal::preview::vertex_ranking(tc_desc, graph);
            REQUIRE(result_vertex_ranking.get_global_rank() == global_triangle_count);
        }
    }

    template <typename GraphType>
    void check_global_task_not_relabeled() {
        GraphType graph_data;
        const auto graph = create_graph<GraphType>();
        std::int64_t global_triangle_count = graph_data.get_correct_triangle_count();

        std::allocator<char> alloc;
        auto tc_desc = dal::preview::triangle_counting::descriptor<
            float,
            dal::preview::triangle_counting::method::ordered_count,
            dal::preview::triangle_counting::task::global,
            std::allocator<char>>(alloc);

        tc_desc.set_relabel(dal::preview::triangle_counting::relabel::no);
        const auto relabel = tc_desc.get_relabel();
        if (relabel == dal::preview::triangle_counting::relabel::no) {
            const auto result_vertex_ranking = dal::preview::vertex_ranking(tc_desc, graph);
            REQUIRE(result_vertex_ranking.get_global_rank() == global_triangle_count);
        }
    }
};

TEST_M(triangle_counting_test,
       "local task for graphs with average_degree < average_degree_sparsity_boundary") {
    this->check_local_task<complete_graph_5_type>();
    this->check_local_task<acyclic_graph_type>();
    this->check_local_task<two_vertices_graph_type>();
    this->check_local_task<cycle_graph_type>();
    this->check_local_task<triangle_graph_type>();
    this->check_local_task<wheel_graph_type>();
}

TEST_M(triangle_counting_test,
       "local task for graph with average_degree = average_degree_sparsity_boundary") {
    this->check_local_task<complete_graph_9_type>();
}

TEST_M(triangle_counting_test,
       "local_and_global task for graphs with average_degree < average_degree_sparsity_boundary") {
    this->check_local_and_global_task<complete_graph_5_type>();
    this->check_local_and_global_task<acyclic_graph_type>();
    this->check_local_and_global_task<two_vertices_graph_type>();
    this->check_local_and_global_task<cycle_graph_type>();
    this->check_local_and_global_task<triangle_graph_type>();
    this->check_local_and_global_task<wheel_graph_type>();
}

TEST_M(triangle_counting_test,
       "local_and_global task for graph with average_degree = average_degree_sparsity_boundary") {
    this->check_local_and_global_task<complete_graph_9_type>();
}

TEST_M(triangle_counting_test,
       "global task for relabeled graphs with average_degree < average_degree_sparsity_boundary") {
    this->check_global_task_relabeled<complete_graph_5_type>();
    this->check_global_task_relabeled<acyclic_graph_type>();
    this->check_global_task_relabeled<two_vertices_graph_type>();
    this->check_global_task_relabeled<cycle_graph_type>();
    this->check_global_task_relabeled<triangle_graph_type>();
    this->check_global_task_relabeled<wheel_graph_type>();
}

TEST_M(triangle_counting_test,
       "global task for relabeled graph with average_degree = average_degree_sparsity_boundary") {
    this->check_global_task_relabeled<complete_graph_9_type>();
}

TEST_M(
    triangle_counting_test,
    "global task for not relabeled graphs with average_degree < average_degree_sparsity_boundary") {
    this->check_global_task_not_relabeled<complete_graph_5_type>();
    this->check_global_task_not_relabeled<acyclic_graph_type>();
    this->check_global_task_not_relabeled<two_vertices_graph_type>();
    this->check_global_task_not_relabeled<cycle_graph_type>();
    this->check_global_task_not_relabeled<triangle_graph_type>();
    this->check_global_task_not_relabeled<wheel_graph_type>();
}

TEST_M(
    triangle_counting_test,
    "global task for not relabeled graph with average_degree = average_degree_sparsity_boundary") {
    this->check_global_task_not_relabeled<complete_graph_9_type>();
}

} // namespace oneapi::dal::algo::triangle_counting::test
