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

void print_graph_info(const dal::preview::undirected_adjacency_vector_graph<> &graph) {
    using graph_t = dal::preview::undirected_adjacency_vector_graph<>;
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

    using graph_t = dal::preview::undirected_adjacency_vector_graph<>;

    {
        const auto graph = dal::read<graph_t>(dal::csv::data_source{ filename });
        print_graph_info(graph);
    }

    {
        const auto graph = dal::read<graph_t>(dal::csv::data_source{ filename },
                                              dal::preview::read_mode::edge_list);
        print_graph_info(graph);
    }

    // { Doesn't work - we've decided on meeting that it's OK
    //     const auto graph = dal::read<graph_t, csv::data_source>(dal::csv::data_source{ filename }, mallocator);
    //     print_graph_info(graph);
    // }

    Mallocator<char> mallocator;

    {
        const auto graph = dal::read<graph_t>(dal::csv::data_source{ filename }, mallocator);
        print_graph_info(graph);
    }

    {
        auto read_args =
            dal::preview::csv::read_args<graph_t>{ mallocator, dal::preview::read_mode::edge_list };
        const auto graph =
            dal::read<graph_t>(dal::csv::data_source{ filename }, std::move(read_args));
        print_graph_info(graph);
    }

    {
        // auto read_args = dal::preview::csv::read_args<graph_t>(mallocator); // - partial deduction doesn't work
        // sof: 45528865 trailing-class-template-arguments-not-deduced
        auto read_args = dal::preview::csv::read_args<graph_t>(mallocator);
        const auto graph =
            dal::read<graph_t>(dal::csv::data_source{ filename }, std::move(read_args));
        print_graph_info(graph);
    }

    {
        auto read_args = dal::preview::csv::read_args<graph_t>{ mallocator }.set_read_mode(
            dal::preview::read_mode::edge_list);
        const auto graph =
            dal::read<graph_t, dal::csv::data_source, dal::preview::csv::read_args<graph_t>>(
                dal::csv::data_source{ filename },
                std::move(read_args));
        print_graph_info(graph);
    }

    {
        const auto graph =
            dal::read<graph_t>(dal::csv::data_source{ filename },
                               dal::preview::csv::read_args<graph_t>{ mallocator }.set_read_mode(
                                   dal::preview::read_mode::edge_list));
        print_graph_info(graph);
    }
}
