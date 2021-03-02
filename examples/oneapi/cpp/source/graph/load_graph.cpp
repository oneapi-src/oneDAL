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

#include <iostream>

#include "example_util/utils.hpp"
#include "oneapi/dal/graph/service_functions.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/graph_csv_data_source.hpp"
#include "oneapi/dal/io/load_graph.hpp"
#include "oneapi/dal/io/csv.hpp"

namespace dal = oneapi::dal;

using namespace dal;

int main(int argc, char **argv) {
    std::cout << "__________________" << std::endl;
    std::cout << "   Hello World" << std::endl;
    std::cout << "__________________" << std::endl;
    const auto filename = get_data_path("graph.csv");

    read<table>(csv::data_source{ filename });
    read<table>(csv::data_source{ filename }, preview::read_mode::table);
    // read<table>(csv::data_source{ filename }, preview::read_mode::edge_list);

    using graph_t = dal::preview::graph_base;
    read<graph_t>(csv::data_source{ filename });
    read<graph_t>(csv::data_source{ filename }, preview::read_mode::edge_list);
    // read<graph_t>(csv::data_source{ filename }, preview::read_mode::table);

    {
        const auto read_args = csv::read_args<table>{}.set_read_mode(preview::read_mode::table);
        read<table>(csv::data_source{ filename }, read_args);
    }

    {
        std::allocator<int> my_allocator;
        const auto read_args = csv::read_args<graph_t>{}
                                   .set_read_mode(preview::read_mode::edge_list)
                                   .set_allocator(my_allocator);

        read<graph_t>(csv::data_source{ filename }, read_args);
    }

    return 0;
}
