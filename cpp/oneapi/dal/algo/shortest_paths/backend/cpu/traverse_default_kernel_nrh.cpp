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

#include "oneapi/dal/algo/shortest_paths/backend/cpu/traverse_default_kernel.hpp"

namespace oneapi::dal::preview::shortest_paths::backend {

using namespace oneapi::dal::preview::detail;
using namespace oneapi::dal::preview::backend;

template <typename BinsVector>
inline bool find_next_bin_index_seq(std::int64_t& curr_bin_index, const BinsVector& local_bins) {
    const std::int64_t max_bin_count = std::numeric_limits<std::int64_t>::max() / 2;
    bool is_queue_empty = true;
    for (size_t i = curr_bin_index; i < local_bins[0].size(); i++) {
        if (!local_bins[0][i].empty()) {
            if (i < max_bin_count) {
                curr_bin_index = i;
            }
            else {
                curr_bin_index = max_bin_count;
            }
            is_queue_empty = false;
            break;
        }
    }
    return is_queue_empty;
}

template <typename SharedBinContainer, typename BinsVector>
inline std::int64_t reduce_to_common_bin_seq(const std::int64_t& curr_bin_index,
                                             BinsVector& local_bins,
                                             SharedBinContainer& shared_bin) {
    const std::int64_t max_elements_in_bin = 1000;
    std::atomic<std::int64_t> curr_shared_bin_tail = 0;

    if (curr_bin_index < local_bins[0].size()) {
        std::int64_t copy_start =
            curr_shared_bin_tail.fetch_add(local_bins[0][curr_bin_index].size());
        copy(local_bins[0][curr_bin_index].begin(),
             local_bins[0][curr_bin_index].end(),
             shared_bin.get_mutable_data() + copy_start);
        local_bins[0][curr_bin_index].resize(0);
    }
    return curr_shared_bin_tail.load();
}

template <typename EdgeValue>
traverse_result<task::one_to_all>
delta_stepping<dal::backend::cpu_dispatch_sse2, EdgeValue>::operator()(
    const detail::descriptor_base<task::one_to_all>& desc,
    const dal::preview::detail::topology<std::int32_t>& t,
    const EdgeValue* vals,
    byte_alloc_iface* alloc_ptr) {
    using value_type = EdgeValue;
    using vertex_type = std::int32_t;
    using atomic_value_allocator_type = inner_alloc<std::atomic<value_type>>;
    using vertex_allocator_type = inner_alloc<vertex_type>;

    vertex_allocator_type vertex_allocator(alloc_ptr);
    atomic_value_allocator_type atomic_value_allocator(alloc_ptr);

    const auto source = desc.get_source();

    const value_type delta = desc.get_delta();

    const std::int64_t max_bin_count = std::numeric_limits<std::int64_t>::max() / 2;
    const std::int64_t max_elements_in_bin = 1000;
    const auto vertex_count = t.get_vertex_count();
    const value_type max_dist = std::numeric_limits<value_type>::max();

    std::atomic<value_type>* dist = allocate(atomic_value_allocator, vertex_count);
    dist = new (dist) std::atomic<value_type>[vertex_count]();
    for (std::int64_t i = 0; i < vertex_count; ++i) {
        dist[i] = max_dist;
    }

    dist[source].store(0);

    vector_container<vertex_type, vertex_allocator_type> shared_bin(t.get_edge_count(),
                                                                    vertex_allocator);

    shared_bin[0] = source;
    std::int64_t curr_bin_index = 0;
    std::int64_t curr_shared_bin_tail = 1;
    bool empty_queue = false;

    using v1v_t = vector_container<vertex_type, vertex_allocator_type>;
    using v1a_t = inner_alloc<v1v_t>;

    using v2v_t = vector_container<v1v_t, v1a_t>;
    using v2a_t = inner_alloc<v2v_t>;
    v2a_t v2a(alloc_ptr);

    using v3v_t = vector_container<v2v_t, v2a_t>;

    v3v_t local_bins(1, v2a);

    local_bins[0].resize(0);

    std::int64_t iter = 0;

    while (curr_bin_index != max_bin_count && iter != max_bin_count && !empty_queue) {
        for (std::int64_t i = 0; i < curr_shared_bin_tail; ++i) {
            vertex_type u = shared_bin[i];
            if (dist[u].load() >= delta * static_cast<value_type>(curr_bin_index)) {
                relax_edges(t, vals, u, delta, dist, local_bins[0]);
            }
        }

        while (curr_bin_index < local_bins[0].size() && !local_bins[0][curr_bin_index].empty() &&
               local_bins[0][curr_bin_index].size() < max_elements_in_bin) {
            vector_container<vertex_type> curr_bin_copy(local_bins[0][curr_bin_index].size());
            copy(local_bins[0][curr_bin_index].begin(),
                 local_bins[0][curr_bin_index].end(),
                 curr_bin_copy.begin());

            local_bins[0][curr_bin_index].resize(0);
            for (std::int64_t j = 0; j < curr_bin_copy.size(); ++j) {
                relax_edges(t, vals, curr_bin_copy[j], delta, dist, local_bins[0]);
            }
        }

        empty_queue = find_next_bin_index_seq(curr_bin_index, local_bins);

        curr_shared_bin_tail = reduce_to_common_bin_seq(curr_bin_index, local_bins, shared_bin);

        iter++;
    }

    auto dist_arr = array<value_type>::empty(vertex_count);
    value_type* dist_ = dist_arr.get_mutable_data();
    for (std::int64_t i = 0; i < vertex_count; ++i) {
        dist_[i] = dist[i].load();
    }

    deallocate(atomic_value_allocator, dist, vertex_count);
    return traverse_result<task::one_to_all>().set_distances(
        dal::detail::homogen_table_builder{}.reset(dist_arr, t.get_vertex_count(), 1).build());
}

template <typename EdgeValue>
traverse_result<task::one_to_all>
delta_stepping_with_pred<dal::backend::cpu_dispatch_sse2, EdgeValue>::operator()(
    const detail::descriptor_base<task::one_to_all>& desc,
    const dal::preview::detail::topology<std::int32_t>& t,
    const EdgeValue* vals,
    byte_alloc_iface* alloc_ptr) {
    using value_type = EdgeValue;
    using vertex_type = std::int32_t;
    using atomic_vp_type = std::atomic<dist_pred<value_type, vertex_type>>;
    using atomic_vp_allocator_type = inner_alloc<atomic_vp_type>;
    using vertex_allocator_type = inner_alloc<vertex_type>;

    vertex_allocator_type vertex_allocator(alloc_ptr);
    atomic_vp_allocator_type atomic_vp(alloc_ptr);

    const auto source = desc.get_source();

    const value_type delta = desc.get_delta();

    const std::int64_t max_bin_count = std::numeric_limits<std::int64_t>::max() / 2;
    const std::int64_t max_elements_in_bin = 1000;
    const auto vertex_count = t.get_vertex_count();
    const value_type max_dist = std::numeric_limits<value_type>::max();

    atomic_vp_type* dp = allocate(atomic_vp, vertex_count);
    for (std::int64_t i = 0; i < vertex_count; ++i) {
        new (dp + i) dist_pred<value_type, vertex_type>(max_dist, -1);
    }

    dp[source].store(dist_pred<value_type, vertex_type>(0, -1));

    vector_container<vertex_type, vertex_allocator_type> shared_bin(t.get_edge_count(),
                                                                    vertex_allocator);

    shared_bin[0] = source;
    std::int64_t curr_bin_index = 0;
    std::int64_t curr_shared_bin_tail = 1;
    bool empty_queue = false;

    using v1v_t = vector_container<vertex_type, vertex_allocator_type>;
    using v1a_t = inner_alloc<v1v_t>;

    using v2v_t = vector_container<v1v_t, v1a_t>;
    using v2a_t = inner_alloc<v2v_t>;
    v2a_t v2a(alloc_ptr);

    using v3v_t = vector_container<v2v_t, v2a_t>;

    v3v_t local_bins(1, v2a);

    local_bins[0].resize(0);

    std::int64_t iter = 0;

    while (curr_bin_index != max_bin_count && iter != max_bin_count && !empty_queue) {
        for (std::int64_t i = 0; i < curr_shared_bin_tail; ++i) {
            vertex_type u = shared_bin[i];
            if (dp[u].load().dist >= delta * static_cast<value_type>(curr_bin_index)) {
                relax_edges_with_pred(t, vals, u, delta, dp, local_bins[0]);
            }
        }

        while (curr_bin_index < local_bins[0].size() && !local_bins[0][curr_bin_index].empty() &&
               local_bins[0][curr_bin_index].size() < max_elements_in_bin) {
            vector_container<vertex_type> curr_bin_copy(local_bins[0][curr_bin_index].size());
            copy(local_bins[0][curr_bin_index].begin(),
                 local_bins[0][curr_bin_index].end(),
                 curr_bin_copy.begin());

            local_bins[0][curr_bin_index].resize(0);
            for (std::int64_t j = 0; j < curr_bin_copy.size(); ++j) {
                relax_edges_with_pred(t, vals, curr_bin_copy[j], delta, dp, local_bins[0]);
            }
        }

        empty_queue = find_next_bin_index_seq(curr_bin_index, local_bins);

        curr_shared_bin_tail = reduce_to_common_bin_seq(curr_bin_index, local_bins, shared_bin);

        iter++;
    }

    if (desc.get_optional_results() & optional_results::distances) {
        auto dist_arr = array<value_type>::empty(vertex_count);
        auto pred_arr = array<vertex_type>::empty(vertex_count);
        value_type* dist_ = dist_arr.get_mutable_data();
        vertex_type* pred_ = pred_arr.get_mutable_data();
        for (std::int64_t i = 0; i < vertex_count; ++i) {
            const auto dp_i = dp[i].load();
            dist_[i] = dp_i.dist;
            pred_[i] = dp_i.pred;
        }

        deallocate(atomic_vp, dp, vertex_count);
        return traverse_result<task::one_to_all>()
            .set_distances(dal::detail::homogen_table_builder{}
                               .reset(dist_arr, t.get_vertex_count(), 1)
                               .build())
            .set_predecessors(dal::detail::homogen_table_builder{}
                                  .reset(pred_arr, t.get_vertex_count(), 1)
                                  .build());
    }
    else {
        auto pred_arr = array<vertex_type>::empty(vertex_count);
        vertex_type* pred_ = pred_arr.get_mutable_data();
        for (std::int64_t i = 0; i < vertex_count; ++i) {
            const auto dp_i = dp[i].load();
            pred_[i] = dp_i.pred;
        }

        deallocate(atomic_vp, dp, vertex_count);
        return traverse_result<task::one_to_all>().set_predecessors(
            dal::detail::homogen_table_builder{}.reset(pred_arr, t.get_vertex_count(), 1).build());
    }
}

template struct delta_stepping<dal::backend::cpu_dispatch_sse2, std::int32_t>;

template struct delta_stepping<dal::backend::cpu_dispatch_sse2, double>;

template struct delta_stepping_with_pred<dal::backend::cpu_dispatch_sse2, std::int32_t>;

template struct delta_stepping_with_pred<dal::backend::cpu_dispatch_sse2, double>;

} // namespace oneapi::dal::preview::shortest_paths::backend
