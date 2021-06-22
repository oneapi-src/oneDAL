/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/algo/shortest_paths/traverse.hpp"
#include "oneapi/dal/graph/detail/directed_adjacency_vector_graph_builder.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/graph/service_functions.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/algo/shortest_paths/test/shortest_paths_graphs.hpp"

namespace oneapi::dal::algo::shortest_paths::test {

namespace dal = oneapi::dal;
namespace te = dal::test::engine;
namespace la = te::linalg;

template <class T>
struct LimitedAllocator {
    typedef T value_type;

    bool is_limited = false;
    size_t max_allocation_size = 0;

    LimitedAllocator(bool is_limited = false, size_t max_allocation_size = 0)
            : is_limited(is_limited),
              max_allocation_size(max_allocation_size) {}

    template <class U>
    LimitedAllocator(const LimitedAllocator<U>& other) noexcept {
        is_limited = other.is_limited;
        max_allocation_size = other.max_allocation_size;
    }

    T* allocate(const size_t n) const {
        if (n == 0 || (is_limited && max_allocation_size < n)) {
            return nullptr;
        }
        if (n > static_cast<size_t>(-1) / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        void* const pv = malloc(n * sizeof(T));
        if (!pv) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(pv);
    }

    void deallocate(T* const p, size_t n) const noexcept {
        free(p);
    }
};

template <typename Type>
Type get_inf_value() {
    return std::numeric_limits<Type>::max();
}

class shortest_paths_test {
public:
    inline auto get_result_type(bool calculate_distances, bool calculate_predecessors) {
        using namespace dal::preview::shortest_paths;
        if (calculate_distances && calculate_predecessors) {
            return optional_results::distances | optional_results::predecessors;
        }
        else if (calculate_distances) {
            return optional_results::distances;
        }
        else if (calculate_predecessors) {
            return optional_results::predecessors;
        }
        return optional_results::distances & optional_results::predecessors;
    }

    inline bool compare_distances(int32_t lhs, int32_t rhs) {
        return lhs == rhs;
    }
    inline bool compare_distances(double lhs, double rhs) {
        const double tol = te::get_tolerance<double>(1e-4, 1e-10);
        return std::abs(lhs - rhs) < tol;
    }

    template <typename T, size_t Size>
    bool check_distances(const std::array<T, Size>& true_distances,
                         const std::vector<T>& distances) {
        if (true_distances.size() != distances.size()) {
            return false;
        }
        for (size_t index = 0; index < true_distances.size(); ++index) {
            if (!compare_distances(true_distances[index], distances[index])) {
                return false;
            }
        }
        return true;
    }

    template <typename DirectedGraphType, typename EdgeValueType, size_t Size>
    bool check_predecessors(const DirectedGraphType& graph,
                            const std::vector<int32_t>& predecessors,
                            const std::array<EdgeValueType, Size>& distances,
                            int32_t source) {
        EdgeValueType unreachable_distance = std::numeric_limits<EdgeValueType>::max();
        if (predecessors.size() != distances.size()) {
            return false;
        }
        if (distances[source] != 0) {
            return false;
        }
        for (size_t index = 0; index < predecessors.size(); ++index) {
            int32_t predecessor = predecessors[index];
            if (predecessor != -1) {
                oneapi::dal::preview::vertex_outward_edge_size_type<DirectedGraphType> from =
                    predecessor;
                oneapi::dal::preview::vertex_outward_edge_size_type<DirectedGraphType> to = index;
                if (!compare_distances(distances[predecessor] +
                                           oneapi::dal::preview::get_edge_value(graph, from, to),
                                       distances[index])) {
                    return false;
                }
            }
            else if (index != static_cast<size_t>(source)) {
                if (!compare_distances(unreachable_distance, distances[index])) {
                    return false;
                }
            }
        }
        return true;
    }

    template <typename T>
    std::vector<T> get_data_from_table(const oneapi::dal::table& table) {
        auto arr = oneapi::dal::row_accessor<const T>(table).pull();
        const auto x = arr.get_data();
        std::vector<T> result(table.get_row_count());
        for (std::int64_t i = 0; i < table.get_row_count(); i++) {
            result[i] = x[i * table.get_column_count()];
        }
        return result;
    }

    template <typename EdgeValueType, typename Allocator, size_t Size>
    void general_shortest_paths_check(
        const oneapi::dal::preview::directed_adjacency_vector_graph<
            int32_t,
            EdgeValueType,
            oneapi::dal::preview::empty_value,
            int,
            std::allocator<char>>& graph,
        double delta,
        int32_t source,
        oneapi::dal::preview::shortest_paths::optional_result_id result_type,
        const std::array<EdgeValueType, Size>& true_distances,
        const Allocator& alloc) {
        using namespace dal::preview::shortest_paths;
        const auto shortest_paths_desc =
            descriptor<float, method::delta_stepping, task::one_to_all, Allocator>(
                source,
                delta,
                result_type,
                Allocator(alloc));
        const auto result_shortest_paths = dal::preview::traverse(shortest_paths_desc, graph);
        if (result_type & optional_results::distances) {
            const std::vector<EdgeValueType> distances =
                get_data_from_table<EdgeValueType>(result_shortest_paths.get_distances());
            REQUIRE(check_distances(true_distances, distances));
        }
        else {
            REQUIRE_THROWS_AS(result_shortest_paths.get_distances(), uninitialized_optional_result);
        }
        if (result_type & optional_results::predecessors) {
            const std::vector<int32_t> predecessors =
                get_data_from_table<int32_t>(result_shortest_paths.get_predecessors());
            REQUIRE(check_predecessors(graph, predecessors, true_distances, source));
        }
        else {
            REQUIRE_THROWS_AS(result_shortest_paths.get_predecessors(),
                              uninitialized_optional_result);
        }
    }

    template <typename DirectedGraphType, typename EdgeValueType>
    void shortest_paths_check(double delta, bool calculate_distances, bool calculate_predecessors) {
        DirectedGraphType graph_data;
        const auto graph_builder = dal::preview::detail::directed_adjacency_vector_graph_builder<
            int32_t,
            EdgeValueType,
            oneapi::dal::preview::empty_value,
            int,
            std::allocator<char>>(graph_data.get_vertex_count(),
                                  graph_data.get_edge_count(),
                                  graph_data.rows.data(),
                                  graph_data.cols.data(),
                                  graph_data.edge_weights.data());
        const auto& graph = graph_builder.get_graph();
        std::allocator<char> alloc;
        const auto result_type = get_result_type(calculate_distances, calculate_predecessors);
        general_shortest_paths_check(graph,
                                     delta,
                                     graph_data.get_source(),
                                     result_type,
                                     graph_data.distances,
                                     alloc);
    }

    template <typename DirectedGraphType, typename EdgeValueType, typename AllocatorType>
    void shortest_paths_custom_allocator_check(double delta,
                                               bool calculate_distances,
                                               bool calculate_predecessors,
                                               AllocatorType alloc) {
        DirectedGraphType graph_data;
        const auto graph_builder = dal::preview::detail::directed_adjacency_vector_graph_builder<
            int32_t,
            EdgeValueType,
            oneapi::dal::preview::empty_value,
            int,
            std::allocator<char>>(graph_data.get_vertex_count(),
                                  graph_data.get_edge_count(),
                                  graph_data.rows.data(),
                                  graph_data.cols.data(),
                                  graph_data.edge_weights.data());
        const auto& graph = graph_builder.get_graph();
        const auto result_type = get_result_type(calculate_distances, calculate_predecessors);
        general_shortest_paths_check(graph,
                                     delta,
                                     graph_data.get_source(),
                                     result_type,
                                     graph_data.distances,
                                     alloc);
    }
};

#define SHORTEST_PATHS_TEST(name) TEST_M(shortest_paths_test, name, "[shortest_paths]")

SHORTEST_PATHS_TEST("Bucket count > number of threads, distances + predecessors") {
    this->shortest_paths_check<d_thread_buckets_graph_type, double>(10, true, true);
}

SHORTEST_PATHS_TEST("Bucket count > number of threads, predecessors") {
    this->shortest_paths_check<d_thread_buckets_graph_type, double>(10, false, true);
}

SHORTEST_PATHS_TEST(
    "Vertex count inside bucket > number of threads && < max_elements_in_bin, distances + predecessors") {
    this->shortest_paths_check<d_thread_bucket_size_graph_type, double>(100, true, true);
}

SHORTEST_PATHS_TEST(
    "Vertex count inside bucket > number of threads && < max_elements_in_bin, predecessors") {
    this->shortest_paths_check<d_thread_bucket_size_graph_type, double>(100, false, true);
}

SHORTEST_PATHS_TEST("Bucket size > max_elemnents_in_bin, distances + predecessors") {
    this->shortest_paths_check<d_max_element_bin_graph_type, double>(1000, true, true);
}

SHORTEST_PATHS_TEST("Bucket size > max_elemnents_in_bin, predecessors") {
    this->shortest_paths_check<d_max_element_bin_graph_type, double>(1000, false, true);
}

SHORTEST_PATHS_TEST("All vertexes are isolated, double edge weights, distances + predecessors") {
    this->shortest_paths_check<d_isolated_vertexes_graph_type, double>(15, true, true);
}

SHORTEST_PATHS_TEST("All vertexes are isolated, double edge weights, predecessors") {
    this->shortest_paths_check<d_isolated_vertexes_graph_type, double>(15, false, true);
}

SHORTEST_PATHS_TEST("All vertexes are isolated, int32_t edge weights, distances + predecessors") {
    this->shortest_paths_check<d_isolated_vertexes_int_graph_type, int32_t>(15, true, true);
}

SHORTEST_PATHS_TEST("All edge weights > delta, distances + predecessors") {
    this->shortest_paths_check<d_graph_3_graph_type, double>(9, true, true);
}

SHORTEST_PATHS_TEST("All edge weights > delta, predecessors") {
    this->shortest_paths_check<d_graph_3_graph_type, double>(9, false, true);
}

SHORTEST_PATHS_TEST("All vertexes inside 1 bucket, distances + predecessors") {
    this->shortest_paths_check<d_one_bucket_graph_type, double>(10, true, true);
}

SHORTEST_PATHS_TEST("All vertexes inside 1 bucket, predecessors") {
    this->shortest_paths_check<d_one_bucket_graph_type, double>(10, false, true);
}

SHORTEST_PATHS_TEST("Custom allocator case") {
    this->shortest_paths_custom_allocator_check<d_graph_3_graph_type, double>(
        0.5,
        true,
        true,
        LimitedAllocator<char>());
}

// SHORTEST_PATHS_TEST("Custom allocator with limited allocations (cannot allocate enough memory)"){
//     this->shortest_paths_custom_allocator_check<d_graph_3_type, double>(0.5, true, true, LimitedAllocator<char>(true, 10));
// }

SHORTEST_PATHS_TEST("Double edge weights)") {
    this->shortest_paths_check<d_net_10_10_double_edges_graph_type, double>(40, true, true);
}

SHORTEST_PATHS_TEST("Int32_t edge weights)") {
    this->shortest_paths_check<d_net_10_10_int_edges_graph_type, int32_t>(40, true, true);
}

SHORTEST_PATHS_TEST("Calculate distances only)") {
    this->shortest_paths_check<d_net_10_10_double_edges_graph_type, double>(40, true, false);
}

SHORTEST_PATHS_TEST("Calculate predecessors only)") {
    this->shortest_paths_check<d_net_10_10_int_edges_graph_type, int32_t>(40, false, true);
}

SHORTEST_PATHS_TEST("Multiple connectivity components)") {
    this->shortest_paths_check<d_multiple_connectivity_components_graph_type, double>(1.5,
                                                                                      true,
                                                                                      true);
}

SHORTEST_PATHS_TEST("Isolated source vertex, distances + predecessors") {
    this->shortest_paths_check<d_source_isolated_vertex_graph_type, double>(3, true, true);
}

SHORTEST_PATHS_TEST("Isolated source vertex, predecessors") {
    this->shortest_paths_check<d_source_isolated_vertex_graph_type, double>(3, false, true);
}

SHORTEST_PATHS_TEST("Non-zero source vertex, distances + predecessors") {
    this->shortest_paths_check<d_k_15_double_edges_source_5_graph_type, double>(50, true, true);
}

SHORTEST_PATHS_TEST("Non-zero source vertex, predecessors") {
    this->shortest_paths_check<d_k_15_double_edges_source_5_graph_type, double>(50, false, true);
}

} // namespace oneapi::dal::algo::shortest_paths::test
