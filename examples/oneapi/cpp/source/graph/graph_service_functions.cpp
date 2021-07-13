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
#include "oneapi/dal/io/csv.hpp"

namespace dal = oneapi::dal;

using namespace dal;

#include <stdlib.h> // size_t, malloc, free
#include <new> // bad_alloc, bad_array_new_length
template <class T>
struct Mallocator {
    typedef T value_type;
    Mallocator() noexcept {} // default ctor not required
    template <class U>
    Mallocator(const Mallocator<U> &) noexcept {}
    template <class U>
    bool operator==(const Mallocator<U> &) const noexcept {
        return true;
    }
    template <class U>
    bool operator!=(const Mallocator<U> &) const noexcept {
        return false;
    }

    T *allocate(const size_t n) const {
        if (n == 0) {
            return nullptr;
        }
        if (n > static_cast<size_t>(-1) / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        void *const pv = malloc(n * sizeof(T));
        if (!pv) {
            throw std::bad_alloc();
        }
        return static_cast<T *>(pv);
    }
    void deallocate(T *const p, size_t) const noexcept {
        free(p);
    }
};

void print_graph_info(preview::undirected_adjacency_vector_graph<> &graph) {
    using graph_t = preview::undirected_adjacency_vector_graph<>;
    std::cout << "Number of vertices: " << dal::preview::get_vertex_count(graph) << std::endl;
    std::cout << "Number of edges: " << dal::preview::get_edge_count(graph) << std::endl;

    dal::preview::vertex_edge_size_type<graph_t> vertex_id = 0;
    std::cout << "Degree of " << vertex_id << ": "
              << dal::preview::get_vertex_degree(graph, vertex_id) << std::endl;

    for (dal::preview::vertex_edge_size_type<graph_t> j = 0;
         j < dal::preview::get_vertex_count(graph);
         ++j) {
        std::cout << "Neighbors of " << j << ": ";
        const auto neigh = dal::preview::get_vertex_neighbors(graph, j);
        for (auto i = neigh.first; i != neigh.second; ++i) {
            std::cout << *i << " ";
        }
        std::cout << std::endl;
    }
    return;
}

int main(int argc, char **argv) {
    const auto filename = get_data_path("graph.csv");

    { const auto x_test = dal::read<dal::table>(dal::csv::data_source{ filename }); }

    using graph_t = preview::undirected_adjacency_vector_graph<>;
    Mallocator<char> mallocator;

    {
        auto read_args = csv::read_args<graph_t, Mallocator<char>>{ mallocator }.set_read_mode(
            preview::read_mode::edge_list);
        auto graph = read<graph_t, csv::data_source, csv::read_args<graph_t, Mallocator<char>>>(
            csv::data_source{ filename },
            std::move(read_args));
        print_graph_info(graph);
    }

    // {
    //     auto graph = read<graph_t, csv::data_source, csv::read_args<graph_t, Mallocator<char>>>(
    //         csv::data_source{ filename },
    //         mallocator,
    //         preview::read_mode::edge_list);
    //     print_graph_info(graph);
    // }

    // {
    //     auto graph = read<graph_t>(csv::data_source{ filename });
    //     print_graph_info(graph);
    // }
}
