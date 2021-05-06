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

namespace oneapi::dal::graph::test {
class graph_base_data {
public:
    graph_base_data() = default;

    std::int64_t get_vertex_count() const {
        return vertex_count;
    }

    std::int64_t get_edge_count() const {
        return edge_count;
    }

    std::int64_t get_cols_count() const {
        return cols_count;
    }

    std::int64_t get_rows_count() const {
        return rows_count;
    }

protected:
    std::int64_t vertex_count;
    std::int64_t edge_count;
    std::int64_t cols_count;
    std::int64_t rows_count;
};

class complete_graph_5_type : public graph_base_data {
public:
    complete_graph_5_type() {
        vertex_count = 5;
        edge_count = 10;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
    }

    std::array<std::int32_t, 5> degrees = { 4, 4, 4, 4, 4 };
    std::array<std::int32_t, 20> cols = {
        1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3
    };
    std::array<std::int64_t, 6> rows = { 0, 4, 8, 12, 16, 20 };
};

class complete_graph_9_type : public graph_base_data {
public:
    complete_graph_9_type() {
        vertex_count = 9;
        edge_count = 36;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
    }

    std::array<std::int32_t, 9> degrees = { 8, 8, 8, 8, 8, 8, 8, 8, 8 };
    std::array<std::int32_t, 72> cols = { 1, 2, 3, 4, 5, 6, 7, 8, 0, 2, 3, 4, 5, 6, 7, 8, 0, 1,
                                          3, 4, 5, 6, 7, 8, 0, 1, 2, 4, 5, 6, 7, 8, 0, 1, 2, 3,
                                          5, 6, 7, 8, 0, 1, 2, 3, 4, 6, 7, 8, 0, 1, 2, 3, 4, 5,
                                          7, 8, 0, 1, 2, 3, 4, 5, 6, 8, 0, 1, 2, 3, 4, 5, 6, 7 };
    std::array<std::int64_t, 10> rows = { 0, 8, 16, 24, 32, 40, 48, 56, 64, 72 };
};

class acyclic_graph_8_type : public graph_base_data {
public:
    acyclic_graph_8_type() {
        vertex_count = 8;
        edge_count = 7;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
    }

    std::array<std::int32_t, 8> degrees = { 3, 1, 3, 3, 1, 1, 1, 1 };
    std::array<std::int32_t, 14> cols = { 1, 2, 4, 0, 0, 3, 6, 2, 5, 7, 0, 3, 2, 3 };
    std::array<std::int64_t, 9> rows = { 0, 3, 4, 7, 10, 11, 12, 13, 14 };
};

class two_vertices_graph_type : public graph_base_data {
public:
    two_vertices_graph_type() {
        vertex_count = 2;
        edge_count = 1;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
    }

    std::array<std::int32_t, 2> degrees = { 1, 1 };
    std::array<std::int32_t, 2> cols = { 1, 0 };
    std::array<std::int64_t, 3> rows = { 0, 1, 2 };
};

class cycle_graph_9_type : public graph_base_data {
public:
    cycle_graph_9_type() {
        vertex_count = 9;
        edge_count = 9;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
    }

    std::array<std::int32_t, 9> degrees = { 2, 2, 2, 2, 2, 2, 2, 2, 2 };
    std::array<std::int32_t, 18> cols = { 1, 8, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 0, 7 };
    std::array<std::int64_t, 10> rows = { 0, 2, 4, 6, 8, 10, 12, 14, 16, 18 };
};

class triangle_graph_type : public graph_base_data {
public:
    triangle_graph_type() {
        vertex_count = 3;
        edge_count = 3;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
    }

    std::array<std::int32_t, 3> degrees = { 2, 2, 2 };
    std::array<std::int32_t, 6> cols = { 1, 2, 0, 2, 0, 1 };
    std::array<std::int64_t, 4> rows = { 0, 2, 4, 6 };
};

class wheel_graph_6_type : public graph_base_data {
public:
    wheel_graph_6_type() {
        vertex_count = 6;
        edge_count = 10;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
        // global_triangle_count = 5;
    }

    std::array<std::int32_t, 6> degrees = { 5, 3, 3, 3, 3, 3 };
    std::array<std::int32_t, 20> cols = {
        1, 2, 3, 4, 5, 0, 2, 5, 0, 1, 3, 0, 2, 4, 0, 3, 5, 0, 1, 4
    };
    std::array<std::int64_t, 7> rows = { 0, 5, 8, 11, 14, 17, 20 };
};

class graph_with_isolated_vertices_10_type : public graph_base_data {
public:
    graph_with_isolated_vertices_10_type() {
        vertex_count = 10;
        edge_count = 11;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
    }

    std::array<std::int32_t, 10> degrees = { 5, 3, 2, 0, 3, 4, 0, 2, 0, 3 };
    std::array<std::int32_t, 22> cols = { 1, 2, 4, 5, 7, 0, 5, 9, 0, 7, 0,
                                          5, 9, 0, 1, 4, 9, 0, 2, 1, 4, 5 };
    std::array<std::int64_t, 11> rows = { 0, 5, 8, 10, 10, 13, 17, 17, 19, 19, 22 };
};

class graph_with_isolated_vertex_11_type : public graph_base_data {
public:
    graph_with_isolated_vertex_11_type() {
        vertex_count = 11;
        edge_count = 45;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
    }

    std::array<std::int32_t, 11> degrees = { 9, 9, 9, 9, 9, 0, 9, 9, 9, 9, 9 };
    std::array<std::int32_t, 90> cols = { 1, 2, 3, 4, 6, 7, 8, 9, 10, 0, 2, 3, 4, 6, 7, 8, 9, 10,
                                          0, 1, 3, 4, 6, 7, 8, 9, 10, 0, 1, 2, 4, 6, 7, 8, 9, 10,
                                          0, 1, 2, 3, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 7, 8, 9, 10,
                                          0, 1, 2, 3, 4, 6, 8, 9, 10, 0, 1, 2, 3, 4, 6, 7, 9, 10,
                                          0, 1, 2, 3, 4, 6, 7, 8, 10, 0, 1, 2, 3, 4, 6, 7, 8, 9 };

    std::array<std::int64_t, 12> rows = { 0, 9, 18, 27, 36, 45, 45, 54, 63, 72, 81, 90 };
};

} // namespace oneapi::dal::graph::test
