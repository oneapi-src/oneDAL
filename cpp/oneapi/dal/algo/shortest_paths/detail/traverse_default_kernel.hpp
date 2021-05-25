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

template <typename Method, typename Task, typename Allocator, typename Graph>
struct traverse_kernel_cpu {
    inline traverse_result<Task> operator()(const dal::detail::host_policy& ctx,
                                            const detail::descriptor_base<Task>& desc,
                                            const Allocator& alloc,
                                            const Graph& g) const;
};

template <typename Float, typename Task, typename Graph, typename... Param>
struct shortest_paths {
    traverse_result<Task> operator()(const dal::detail::host_policy& ctx,
                                     const detail::descriptor_base<Task>& desc,
                                     const Graph& g) const;
};

template <typename Topology, typename EdgeValue>
inline void relax_edges(const Topology& t,
                        const EdgeValue* vals,
                        typename Topology::vertex_type u,
                        EdgeValue delta,
                        std::atomic<EdgeValue>* dist,
                        std::vector<std::vector<typename Topology::vertex_type>>& local_bins) {
    for (std::int64_t v_ = t._rows_ptr[u]; v_ < t._rows_ptr[u + 1]; v_++) {
        const auto v = t._cols_ptr[v_];
        const auto v_w = vals[v_];
        EdgeValue old_dist = dist[v].load();
        const EdgeValue new_dist = dist[u].load() + v_w;
        while (new_dist < old_dist) {
            if (dist[v].compare_exchange_strong(old_dist, new_dist)) {
                std::int64_t dest_bin = new_dist / delta;
                if (dest_bin >= local_bins.size())
                    local_bins.resize(dest_bin + 1);
                local_bins[dest_bin].push_back(v);
                break;
            }
            old_dist = dist[v].load();
        }
    }
}

template <typename T>
bool find_next_bin_index(std::int64_t& curr_bin_index,
                         const std::vector<std::vector<std::vector<T>>>& local_bins) {
    const std::int64_t kMaxBin = std::numeric_limits<std::int64_t>::max() / 2;
    bool is_queue_empty = true;
    /*
    for (const auto &local_bins_id : local_bins) {
      for (std::int64_t i = curr_bin_index; i < local_bins_id.size(); i++) {
          if (!local_bins_id[i].empty()) {
              curr_bin_index = std::min(kMaxBin, i);
              is_queue_empty = false;
              break;
          }
      }
  }
*/
    // curr_bin_index = kMaxBin;

    auto total = oneapi::dal::detail::parallel_reduce_int32_int64_t(
        (std::int64_t)local_bins.size(),
        (std::int64_t)kMaxBin,
        [&](std::int64_t begin, std::int64_t end, std::int64_t thread_min_index) -> std::int64_t {
            for (std::int64_t id = begin; id < end; ++id) {
                for (std::int64_t i = curr_bin_index; i < local_bins[id].size(); i++) {
                    if (!local_bins[id][i].empty()) {
                        thread_min_index = std::min(kMaxBin, i);
                        break;
                    }
                }
            }
            return thread_min_index;
        },
        [&](std::int64_t x, std::int64_t y) -> std::int64_t {
            return std::min(x, y);
        });

    if (total < kMaxBin) {
        curr_bin_index = total;
        is_queue_empty = false;
    }

    return is_queue_empty;
}

template <typename T>
std::int64_t reduce_to_common_bin(const std::int64_t& curr_bin_index,
                                  std::vector<std::vector<std::vector<T>>>& local_bins,
                                  std::vector<T>& frontier) {
    const std::int64_t kBinSizeThreshold = 1000;
    std::atomic<std::int64_t> curr_frontier_tail = 0;
    dal::detail::threader_for(local_bins.size(), local_bins.size(), [&](std::int64_t i) {
        int thread_id = dal::detail::threader_get_current_thread_index();
        if (curr_bin_index < local_bins[thread_id].size()) {
            std::int64_t copy_start =
                curr_frontier_tail.fetch_add(local_bins[thread_id][curr_bin_index].size());
            std::copy(local_bins[thread_id][curr_bin_index].begin(),
                      local_bins[thread_id][curr_bin_index].end(),
                      frontier.data() + copy_start);
            local_bins[thread_id][curr_bin_index].resize(0);
        }
    });
    return curr_frontier_tail.load();
}

template <typename Allocator, typename Graph>
struct traverse_kernel_cpu<method::delta_stepping, task::one_to_all, Allocator, Graph> {
    inline traverse_result<task::one_to_all> operator()(
        const dal::detail::host_policy& ctx,
        const detail::descriptor_base<task::one_to_all>& desc,
        const Allocator& alloc,
        const Graph& g) const {
        using value_type = edge_user_value_type<Graph>;
        using vertex_type = vertex_type<Graph>;
        using value_allocator_type =
            typename std::allocator_traits<Allocator>::template rebind_alloc<value_type>;

        const auto source = desc.get_source();

        const auto& t = dal::preview::detail::csr_topology_builder<Graph>()(g);
        const auto vals = dal::detail::get_impl(g).get_edge_values().get_data();
        const value_type delta = desc.get_delta();

        //using atomic_value_allocator_type =
        // typename std::allocator_traits<Allocator>::template rebind_alloc<value_type>;

        const value_type kDistInf = std::numeric_limits<value_type>::max() / 2;
        const std::int64_t kMaxBin = std::numeric_limits<std::int64_t>::max() / 2;
        const std::int64_t kBinSizeThreshold = 1000;
        const auto vertex_count = t.get_vertex_count();
        const value_type max_dist = std::numeric_limits<value_type>::max();
        std::atomic<value_type>* dist = new std::atomic<value_type>[vertex_count];

        dist = new (dist) std::atomic<value_type>[vertex_count]();
        dal::detail::threader_for(vertex_count, vertex_count, [&](std::int64_t i) {
            dist[i] = max_dist;
        });
        dist[source].store(0);

        std::vector<vertex_type> frontier(t.get_edge_count());

        frontier[0] = source;
        std::int64_t curr_bin_index = 0;
        std::int64_t curr_frontier_tail = 1;
        bool empty_queue = false;
        std::int64_t thread_cnt = dal::detail::threader_get_max_threads();

        std::vector<std::vector<std::vector<vertex_type>>> local_bins(thread_cnt);

        std::int64_t iter = 0;

        while (curr_bin_index != kMaxBin && iter != kMaxBin && !empty_queue) {
            // processing shared bin  by all threads
            dal::detail::threader_for(curr_frontier_tail, curr_frontier_tail, [&](std::int64_t i) {
                vertex_type u = frontier[i];
                if (dist[u].load() >= delta * static_cast<value_type>(curr_bin_index)) {
                    relax_edges(t,
                                vals,
                                u,
                                delta,
                                dist,
                                local_bins[dal::detail::threader_get_current_thread_index()]);
                }
            });

            // processing local bins.
            dal::detail::threader_for(thread_cnt, thread_cnt, [&](std::int64_t i) {
                int thread_id = dal::detail::threader_get_current_thread_index();
                while (curr_bin_index < local_bins[thread_id].size() &&
                       !local_bins[thread_id][curr_bin_index].empty() &&
                       local_bins[thread_id][curr_bin_index].size() < kBinSizeThreshold) {
                    std::vector<vertex_type> curr_bin_copy = local_bins[thread_id][curr_bin_index];

                    local_bins[thread_id][curr_bin_index].resize(0);
                    for (vertex_type u : curr_bin_copy)
                        relax_edges(t, vals, u, delta, dist, local_bins[thread_id]);
                }
            });

            empty_queue = find_next_bin_index(curr_bin_index, local_bins);

            curr_frontier_tail = reduce_to_common_bin(curr_bin_index, local_bins, frontier);
            iter++;
        }

        auto dist_arr = array<value_type>::empty(vertex_count);
        value_type* dist_ = dist_arr.get_mutable_data();
        dal::detail::threader_for(vertex_count, vertex_count, [&](std::int64_t i) {
            dist_[i] = dist[i].load();
        });

        delete[] dist;
        return traverse_result<task::one_to_all>().set_distances(
            dal::detail::homogen_table_builder{}.reset(dist_arr, t.get_vertex_count(), 1).build());
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
