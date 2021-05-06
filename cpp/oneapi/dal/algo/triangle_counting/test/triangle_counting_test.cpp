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
#include "oneapi/dal/graph/test_data/test_graphs.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::algo::triangle_counting::test {

namespace gt = dal::graph::test;

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

        const std::int64_t vertex_count = graph_data.get_vertex_count();
        const std::int64_t edge_count = graph_data.get_edge_count();
        const std::int64_t cols_count = graph_data.get_cols_count();
        const std::int64_t rows_count = graph_data.get_rows_count();

        std::int32_t *degrees =
            oneapi::dal::preview::detail::allocate(vertex_allocator, vertex_count);
        std::int32_t *cols = oneapi::dal::preview::detail::allocate(vertex_allocator, cols_count);
        std::int64_t *rows = oneapi::dal::preview::detail::allocate(edge_allocator, rows_count);
        std::int32_t *rows_vertex =
            oneapi::dal::preview::detail::allocate(vertex_allocator, rows_count);

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
        std::int64_t vertex_count = graph_data.get_vertex_count();

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
        std::int64_t vertex_count = graph_data.get_vertex_count();
        std::int64_t global_triangle_count = graph_data.get_triangle_count();

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
            if (local_triangles_data[i] == graph_data.local_triangles[i]) {
                correct_local_triangle_count++;
            }
        }
        REQUIRE(correct_local_triangle_count == vertex_count);
    }

    template <typename GraphType>
    void check_global_task_relabeled() {
        GraphType graph_data;
        const auto graph = create_graph<GraphType>();
        std::int64_t global_triangle_count = graph_data.get_triangle_count();

        std::allocator<char> alloc;
        auto tc_desc = dal::preview::triangle_counting::descriptor<
            float,
            dal::preview::triangle_counting::method::ordered_count,
            dal::preview::triangle_counting::task::global,
            std::allocator<char>>(alloc).set_relabel(dal::preview::triangle_counting::relabel::yes);

        const auto relabel = tc_desc.get_relabel();
            const auto result_vertex_ranking = dal::preview::vertex_ranking(tc_desc, graph);
            REQUIRE(result_vertex_ranking.get_global_rank() == global_triangle_count);
    }

    template <typename GraphType>
    void check_global_task_not_relabeled() {
        GraphType graph_data;
        const auto graph = create_graph<GraphType>();
        std::int64_t global_triangle_count = graph_data.get_triangle_count();

        std::allocator<char> alloc;
        auto tc_desc = dal::preview::triangle_counting::descriptor<
            float,
            dal::preview::triangle_counting::method::ordered_count,
            dal::preview::triangle_counting::task::global,
            std::allocator<char>>(alloc).set_relabel(dal::preview::triangle_counting::relabel::no);

        const auto relabel = tc_desc.get_relabel();
            const auto result_vertex_ranking = dal::preview::vertex_ranking(tc_desc, graph);
            REQUIRE(result_vertex_ranking.get_global_rank() == global_triangle_count);
    }
};

TEST_M(triangle_counting_test, "local task for graphs with average_degree < 4") {
    this->check_local_task<gt::complete_graph_5_type>();
    this->check_local_task<gt::acyclic_graph_8_type>();
    this->check_local_task<gt::two_vertices_graph_type>();
    this->check_local_task<gt::cycle_graph_9_type>();
    this->check_local_task<gt::triangle_graph_type>();
    this->check_local_task<gt::wheel_graph_6_type>();
}

TEST_M(triangle_counting_test, "local task for graphs with average_degree >= 4") {
    this->check_local_task<gt::complete_graph_9_type>();
    this->check_local_task<gt::graph_with_isolated_vertex_11_type>();
}

TEST_M(triangle_counting_test, "local_and_global task for graphs with average_degree < 4") {
    this->check_local_and_global_task<gt::complete_graph_5_type>();
    this->check_local_and_global_task<gt::acyclic_graph_8_type>();
    this->check_local_and_global_task<gt::two_vertices_graph_type>();
    this->check_local_and_global_task<gt::cycle_graph_9_type>();
    this->check_local_and_global_task<gt::triangle_graph_type>();
    this->check_local_and_global_task<gt::wheel_graph_6_type>();
}

TEST_M(triangle_counting_test, "local_and_global task for graphs with average_degree >= 4") {
    this->check_local_and_global_task<gt::complete_graph_9_type>();
    this->check_local_and_global_task<gt::graph_with_isolated_vertex_11_type>();
}

TEST_M(triangle_counting_test, "global task for graphs with average_degree < 4") {
    this->check_global_task_relabeled<gt::complete_graph_5_type>();
    this->check_global_task_relabeled<gt::acyclic_graph_8_type>();
    this->check_global_task_relabeled<gt::two_vertices_graph_type>();
    this->check_global_task_relabeled<gt::cycle_graph_9_type>();
    this->check_global_task_relabeled<gt::triangle_graph_type>();
    this->check_global_task_relabeled<gt::wheel_graph_6_type>();
}

TEST_M(triangle_counting_test, "global task for relabeled graph with average_degree >= 4") {
    this->check_global_task_relabeled<gt::complete_graph_9_type>();
    this->check_global_task_relabeled<gt::graph_with_isolated_vertex_11_type>();
}

TEST_M(triangle_counting_test, "global task for not relabeled graph with average_degree >= 4") {
    this->check_global_task_not_relabeled<gt::complete_graph_9_type>();
    this->check_global_task_not_relabeled<gt::graph_with_isolated_vertex_11_type>();
}

} // namespace oneapi::dal::algo::triangle_counting::test
