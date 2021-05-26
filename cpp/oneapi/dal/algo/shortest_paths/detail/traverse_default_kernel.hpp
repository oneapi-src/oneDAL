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

#pragma once

#include <queue>
#include <set>
#include <tuple>
#include <vector>
#include <atomic>

#include "oneapi/dal/algo/shortest_paths/common.hpp"
#include "oneapi/dal/algo/shortest_paths/traverse_types.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/graph/detail/directed_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/graph/detail/directed_adjacency_vector_graph_topology_builder.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::preview::shortest_paths::detail {

using namespace dal::preview::detail;
template <typename Method, typename Task, typename Allocator, typename Graph>
struct traverse_kernel_cpu {
    inline traverse_result<Task> operator()(const dal::detail::host_policy& ctx,
                                            const detail::descriptor_base<Task>& desc,
                                            const Allocator& alloc,
                                            const Graph& g) const;
};

template <typename Float, typename Task, typename Topology, typename EdgeValue, typename... Param>
struct delta_stepping {
    traverse_result<Task> operator()(const dal::detail::host_policy& ctx,
                                     const detail::descriptor_base<Task>& desc,
                                     const Topology& t,
                                     const EdgeValue* vals) const;
};

template <typename Float, typename EdgeValue>
struct delta_stepping<Float,
                      task::one_to_all,
                      dal::preview::detail::topology<std::int32_t>,
                      EdgeValue> {
    traverse_result<task::one_to_all> operator()(
        const dal::detail::host_policy& ctx,
        const detail::descriptor_base<task::one_to_all>& desc,
        const dal::preview::detail::topology<std::int32_t>& t,
        const EdgeValue* vals) const;
};

template <typename Allocator, typename Graph>
struct traverse_kernel_cpu<method::delta_stepping, task::one_to_all, Allocator, Graph> {
    inline traverse_result<task::one_to_all> operator()(
        const dal::detail::host_policy& ctx,
        const detail::descriptor_base<task::one_to_all>& desc,
        const Allocator& alloc,
        const Graph& g) const {
        using topology_type = typename graph_traits<Graph>::impl_type::topology_type;
        using value_type = edge_user_value_type<Graph>;
        const auto& t = dal::preview::detail::csr_topology_builder<Graph>()(g);
        const auto vals = dal::detail::get_impl(g).get_edge_values().get_data();

        return delta_stepping<float, task::one_to_all, topology_type, value_type>{}(ctx,
                                                                                    desc,
                                                                                    t,
                                                                                    vals);
    }
};

template <typename Allocator, typename Graph>
struct traverse_kernel_cpu<method::dijkstra, task::one_to_all, Allocator, Graph> {
    inline traverse_result<task::one_to_all> operator()(
        const dal::detail::host_policy& ctx,
        const detail::descriptor_base<task::one_to_all>& desc,
        const Allocator& alloc,
        const Graph& g) const {
        const auto source = desc.get_source();
        const auto& t = dal::preview::detail::csr_topology_builder<Graph>()(g);
        const auto& vals = dal::detail::get_impl(g).get_edge_values();
        using value_type = edge_user_value_type<Graph>;
        using vertex_type = vertex_type<Graph>;
        using value_allocator_type =
            typename std::allocator_traits<Allocator>::template rebind_alloc<value_type>;

        const auto vertex_count = t.get_vertex_count();
        const value_type max_dist = std::numeric_limits<value_type>::max();
        //std::vector<value_type, value_allocator_type> dist(t._vertex_count, max_dist);
        auto dist_arr = array<value_type>::full(vertex_count, max_dist);
        value_type* dist = dist_arr.get_mutable_data();
        dist[source] = 0;

        std::priority_queue<std::pair<value_type, vertex_type>,
                            std::vector<std::pair<value_type, vertex_type>>,
                            std::greater<std::pair<value_type, vertex_type>>>
            pq;
        pq.push(std::make_pair(0, source));
        while (!pq.empty()) {
            const auto curr_source = pq.top().second;
            const auto tentative_distance = pq.top().first;
            pq.pop();
            for (auto u_ = t._rows[curr_source]; u_ < t._rows[curr_source + 1]; u_++) {
                const auto u = t._cols[u_];
                const auto u_w = vals[u_];
                if (tentative_distance + u_w < dist[u]) {
                    dist[u] = tentative_distance + u_w;
                    pq.push(std::make_pair(dist[u], u));
                }
            }
        }

        return traverse_result<task::one_to_all>().set_distances(
            dal::detail::homogen_table_builder{}.reset(dist_arr, t.get_vertex_count(), 1).build());
    }
};

} // namespace oneapi::dal::preview::shortest_paths::detail
