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

#include <atomic>

#include "oneapi/dal/algo/shortest_paths/common.hpp"
#include "oneapi/dal/algo/shortest_paths/traverse_types.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/graph/detail/container.hpp"

namespace oneapi::dal::preview::shortest_paths::backend {
using namespace oneapi::dal::preview::detail;

template <typename Topology, typename EdgeValue>
inline void relax_edges(
    const Topology& t,
    const EdgeValue* vals,
    typename Topology::vertex_type u,
    EdgeValue delta,
    std::atomic<EdgeValue>* dist,
    vector_container<vector_container<typename Topology::vertex_type>>& local_bins) {
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
inline bool find_next_bin_index(
    std::int64_t& curr_bin_index,
    const vector_container<vector_container<vector_container<T>>>& local_bins) {
    const std::int64_t kMaxBin = std::numeric_limits<std::int64_t>::max() / 2;
    bool is_queue_empty = true;

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
inline std::int64_t reduce_to_common_bin(
    const std::int64_t& curr_bin_index,
    vector_container<vector_container<vector_container<T>>>& local_bins,
    vector_container<T>& frontier) {
    const std::int64_t kBinSizeThreshold = 1000;
    std::atomic<std::int64_t> curr_frontier_tail = 0;
    dal::detail::threader_for(local_bins.size(), local_bins.size(), [&](std::int64_t i) {
        int thread_id = dal::detail::threader_get_current_thread_index();
        if (curr_bin_index < local_bins[thread_id].size()) {
            std::int64_t copy_start =
                curr_frontier_tail.fetch_add(local_bins[thread_id][curr_bin_index].size());
            copy(local_bins[thread_id][curr_bin_index].begin(),
                 local_bins[thread_id][curr_bin_index].end(),
                 frontier.get_mutable_data() + copy_start);
            local_bins[thread_id][curr_bin_index].resize(0);
        }
    });
    return curr_frontier_tail.load();
}

template <typename Cpu, typename EdgeValue>
traverse_result<task::one_to_all> delta_stepping(
    const detail::descriptor_base<task::one_to_all>& desc,
    const dal::preview::detail::topology<std::int32_t>& t,
    const EdgeValue* vals) {
    using value_type = EdgeValue;
    using vertex_type = std::int32_t;
    //using value_allocator_type =
    //typename std::allocator_traits<Allocator>::template rebind_alloc<value_type>;

    const auto source = desc.get_source();

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

    vector_container<vertex_type> frontier(t.get_edge_count());

    frontier[0] = source;
    std::int64_t curr_bin_index = 0;
    std::int64_t curr_frontier_tail = 1;
    bool empty_queue = false;
    std::int64_t thread_cnt = dal::detail::threader_get_max_threads();

    vector_container<vector_container<vector_container<vertex_type>>> local_bins(thread_cnt);

    local_bins[0].reserve(t.get_vertex_degree(source));

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
                vector_container<vertex_type> curr_bin_copy(
                    local_bins[thread_id][curr_bin_index].size());
                copy(local_bins[thread_id][curr_bin_index].begin(),
                     local_bins[thread_id][curr_bin_index].end(),
                     curr_bin_copy.begin());

                local_bins[thread_id][curr_bin_index].resize(0);
                for (std::int64_t j = 0; j < curr_bin_copy.size(); ++j)
                    relax_edges(t, vals, curr_bin_copy[j], delta, dist, local_bins[thread_id]);
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

} // namespace oneapi::dal::preview::shortest_paths::backend
