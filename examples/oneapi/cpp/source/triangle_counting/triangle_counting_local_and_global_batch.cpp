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

#include "example_util/output_helpers_graph.hpp"
#include "example_util/utils.hpp"
#include "oneapi/dal/algo/triangle_counting.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/graph_csv_data_source.hpp"
#include "oneapi/dal/io/load_graph.hpp"
#include "oneapi/dal/table/column_accessor.hpp"
#include "oneapi/dal/table/common.hpp"

template <class T>
struct custom_allocator {
    using value_type = T;
    using pointer = T*;
    custom_allocator() noexcept {}
    template <class U>
    custom_allocator(const custom_allocator<U>&) noexcept {}
    T* allocate(std::size_t n) {
        std::cout << "Allocated" << n * sizeof(T) << "\n";
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }
    void deallocate(T* p, std::size_t n) {
        std::cout << "Deallocated" << n * sizeof(T) << "\n";
        ::delete (p);
    }
};

template <class T, class U>
constexpr bool operator==(const custom_allocator<T>&, const custom_allocator<U>&) noexcept {
    return true;
}

template <class T, class U>
constexpr bool operator!=(const custom_allocator<T>&, const custom_allocator<U>&) noexcept {
    return false;
}

namespace dal = oneapi::dal;
using namespace dal::preview::triangle_counting;

int main(int argc, char** argv) {
    const auto filename = get_data_path("graph.csv");

    // read the graph
    const dal::preview::graph_csv_data_source ds(argv[1]);
    const dal::preview::load_graph::descriptor<> d;
    const auto my_graph = dal::preview::load_graph::load(d, ds);
    custom_allocator<char> alloc;
    // set algorithm parameters
    const auto tc_desc =
        descriptor<float, method::ordered_count, task::local_and_global, custom_allocator<char>>()
            .set_allocator(alloc);

    // compute local and global triangles
    const auto result_vertex_ranking = dal::preview::vertex_ranking(tc_desc, my_graph);

    // extract the result

    std::cout << "Global triangles: " << result_vertex_ranking.get_global_rank() << std::endl;

    return 0;
}
