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

#include <fstream>
#include <array>

#include "oneapi/dal/io/load_graph.hpp"

#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::io::test {

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

protected:
    std::string filename;
    std::string file_content;
    std::int64_t vertex_count;
    std::int64_t edge_count;
};

class complete_graph_type : public graph_base_data {
public:
    complete_graph_type() {
        filename = "complete_graph.csv";
        file_content = "1 4 0 1 3 4 1 3 0 2 1 2 2 3 0 3 2 4 0 4";
        vertex_count = 5;
        edge_count = 10;
    }
    std::array<std::int32_t, 5> degrees = { 4, 4, 4, 4, 4 };
    std::array<std::int32_t, 20> cols = {
        1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3
    };
    std::array<std::int64_t, 6> rows = { 0, 4, 8, 12, 16, 20 };
    std::array<std::int32_t, 6> rows_vertex = { 0, 4, 8, 12, 16, 20 };
};

class empty_graph_type : public graph_base_data {
public:
    empty_graph_type() {
        filename = "empty_graph.csv";
        file_content = " ";
    }
};

class two_vertices_graph_type : public graph_base_data {
public:
    two_vertices_graph_type() {
        filename = "two_vertices_graph.csv";
        file_content = "1 0";
        vertex_count = 2;
        edge_count = 1;
    }
    std::array<std::int32_t, 2> degrees = { 1, 1 };
    std::array<std::int32_t, 2> cols = { 1, 0 };
    std::array<std::int64_t, 3> rows = { 0, 1, 2 };
    std::array<std::int32_t, 3> rows_vertex = { 0, 1, 2 };
};

class one_edge_graph_type : public graph_base_data {
public:
    one_edge_graph_type() {
        filename = "one_edge_graph.csv";
        file_content = "10 0";
        vertex_count = 11;
        edge_count = 1;
    }
    std::array<std::int32_t, 11> degrees = { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
    std::array<std::int32_t, 2> cols = { 10, 0 };
    std::array<std::int64_t, 12> rows = { 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2 };
    std::array<std::int32_t, 12> rows_vertex = { 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2 };
};

class wheel_graph_type : public graph_base_data {
public:
    wheel_graph_type() {
        filename = "wheel_graph.csv";
        file_content = "0 1 0 2 0 3 0 4 0 5 1 2 1 5 2 3 3 4 4 5";
        vertex_count = 6;
        edge_count = 10;
    }

    std::array<std::int32_t, 6> degrees = { 5, 3, 3, 3, 3, 3 };
    std::array<std::int32_t, 20> cols = {
        1, 2, 3, 4, 5, 0, 2, 5, 0, 1, 3, 0, 2, 4, 0, 3, 5, 0, 1, 4
    };
    std::array<std::int64_t, 7> rows = { 0, 5, 8, 11, 14, 17, 20 };
    std::array<std::int32_t, 7> rows_vertex = { 0, 5, 8, 11, 14, 17, 20 };
};

class graph_data_with_letter_type : public graph_base_data {
public:
    graph_data_with_letter_type() {
        filename = "data_with_letter.csv";
        file_content = "0 1 0 2 0 a 4 5 4 3";
    }
};

class odd_number_of_values_data_type : public graph_base_data {
public:
    odd_number_of_values_data_type() {
        filename = "odd_number_of_values_data.csv";
        file_content = "0 1 0 2 0 5 4 5 4";
    }
};

class negative_values_data_type : public graph_base_data {
public:
    negative_values_data_type() {
        filename = "negative_value_data.csv";
        file_content = "0 1 0 -2 0 5 -4 5 4 3";
    }
};

class load_graph_test {
public:
    using my_graph_type = oneapi::dal::preview::undirected_adjacency_vector_graph<>;

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

    template <typename element_type, std::size_t SIZE>
    void compare_data(const oneapi::dal::preview::detail::container<element_type> &actual_data,
                      const std::array<element_type, SIZE> &correct_data,
                      std::int64_t element_count) {
        int correct_element_count = 0;
        for (std::int64_t i = 0; i < element_count; i++) {
            if (actual_data[i] == correct_data[i])
                correct_element_count++;
        }
        REQUIRE(correct_element_count == element_count);
    }

    template <typename GraphType>
    void general_checks() {
        GraphType graph_data;
        const auto graph = create_graph<GraphType>();
        auto &graph_impl = dal::detail::get_impl(graph);

        REQUIRE(graph_impl.get_topology()._vertex_count == graph_data.get_correct_vertex_count());
        INFO("vertex_count value is correct");

        REQUIRE(graph_impl.get_topology()._edge_count == graph_data.get_correct_edge_count());
        INFO("edge_count value is correct");

        compare_data<std::int32_t>(graph_impl.get_topology()._degrees,
                                   graph_data.degrees,
                                   graph_data.get_correct_vertex_count());
        INFO("degrees values are correct");

        compare_data<std::int32_t>(graph_impl.get_topology()._cols,
                                   graph_data.cols,
                                   graph_data.get_correct_edge_count() * 2);
        INFO("cols values are correct");

        compare_data<std::int64_t>(graph_impl.get_topology()._rows,
                                   graph_data.rows,
                                   graph_data.get_correct_vertex_count() + 1);
        INFO("rows values are correct");

        compare_data<std::int32_t>(graph_impl.get_topology()._rows_vertex,
                                   graph_data.rows_vertex,
                                   graph_data.get_correct_vertex_count() + 1);
        INFO("rows_vertex values are correct");
    }
};

TEST_M(load_graph_test, "load a complete graph") {
    this->general_checks<complete_graph_type>();
}

TEST_M(load_graph_test, "load a wheel graph") {
    this->general_checks<wheel_graph_type>();
}

TEST_M(load_graph_test, "load a graph with two vertices") {
    this->general_checks<two_vertices_graph_type>();
}

TEST_M(load_graph_test, "load a graph with one edge") {
    this->general_checks<one_edge_graph_type>();
}

TEST_M(load_graph_test, "throws if input file is empty") {
    REQUIRE_THROWS_AS(create_graph<empty_graph_type>(), invalid_argument);
}

TEST_M(load_graph_test, "throws if input file does not exist") {
    const dal::preview::load_graph::descriptor<dal::preview::edge_list<int32_t>, my_graph_type>
        desc;
    const dal::preview::graph_csv_data_source ds("nonexistfile.csv");
    REQUIRE_THROWS_AS(dal::preview::load_graph::load(desc, ds), invalid_argument);
}

} // namespace oneapi::dal::io::test
