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

#include "oneapi/dal/algo/shortest_paths/backend/cpu/traverse_default_kernel.hpp"
#include "oneapi/dal/algo/shortest_paths/common.hpp"
#include "oneapi/dal/algo/shortest_paths/traverse_types.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/graph/detail/container.hpp"

namespace oneapi::dal::preview::shortest_paths::backend {

using namespace oneapi::dal::preview::detail;
using namespace oneapi::dal::preview::backend;

traverse_result<task::one_to_all>
delta_stepping<dal::backend::cpu_dispatch_sse2, std::int32_t>::operator()(
    const detail::descriptor_base<task::one_to_all>& desc,
    const dal::preview::detail::topology<std::int32_t>& t,
    const std::int32_t* vals,
    byte_alloc_iface* alloc_ptr) {
    using value_type = std::int32_t;
    using vertex_type = std::int32_t;
    using value_allocator_type = inner_alloc<value_type>;
    using vertex_allocator_type = inner_alloc<vertex_type>;

    vertex_allocator_type vertex_allocator(alloc_ptr);
    value_allocator_type value_allocator(alloc_ptr);

    const auto source = dal::detail::integral_cast<std::int32_t>(desc.get_source());
    const value_type delta = desc.get_delta();

    const std::int64_t max_bin_count = std::numeric_limits<std::int64_t>::max() / 2;
    const std::int64_t max_elements_in_bin = 1000;
    const auto vertex_count = t.get_vertex_count();
    const value_type max_dist = std::numeric_limits<value_type>::max();

    value_type* dist = allocate(value_allocator, vertex_count);
    for (std::int64_t i = 0; i < vertex_count; ++i) {
        dist[i] = max_dist;
    }

    dist[source] = 0;

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
    using v3v_t = vector_container<v2v_t, v2a_t>;
    v2a_t v2a(alloc_ptr);
    v3v_t local_bins(1, v2a);

    std::int64_t iter = 0;

    while (curr_bin_index != max_bin_count && iter != max_bin_count && !empty_queue) {
        for (std::int64_t i = 0; i < curr_shared_bin_tail; ++i) {
            vertex_type u = shared_bin[i];
            if (dist[u] >= delta * static_cast<value_type>(curr_bin_index)) {
                relax_edges_seq(t, vals, u, delta, dist, local_bins[0]);
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
                relax_edges_seq(t, vals, curr_bin_copy[j], delta, dist, local_bins[0]);
            }
        }

        empty_queue = find_next_bin_index_seq(curr_bin_index, local_bins);

        curr_shared_bin_tail = reduce_to_common_bin_seq(curr_bin_index, local_bins, shared_bin);

        iter++;
    }

    auto dist_arr = array<value_type>::empty(vertex_count);
    value_type* dist_ = dist_arr.get_mutable_data();
    for (std::int64_t i = 0; i < vertex_count; ++i) {
        dist_[i] = dist[i];
    }

    deallocate(value_allocator, dist, vertex_count);
    return traverse_result<task::one_to_all>().set_distances(
        dal::detail::homogen_table_builder{}.reset(dist_arr, t.get_vertex_count(), 1).build());
}

traverse_result<task::one_to_all>
delta_stepping_with_pred<dal::backend::cpu_dispatch_sse2, std::int32_t>::operator()(
    const detail::descriptor_base<task::one_to_all>& desc,
    const dal::preview::detail::topology<std::int32_t>& t,
    const std::int32_t* vals,
    byte_alloc_iface* alloc_ptr) {
    using value_type = std::int32_t;
    using vertex_type = std::int32_t;
    using vp_type = dist_pred<value_type, vertex_type>;
    using vp_allocator_type = inner_alloc<vp_type>;
    using vertex_allocator_type = inner_alloc<vertex_type>;

    vertex_allocator_type vertex_allocator(alloc_ptr);
    vp_allocator_type vp_allocator(alloc_ptr);

    const auto source = dal::detail::integral_cast<std::int32_t>(desc.get_source());

    const value_type delta = desc.get_delta();

    const std::int64_t max_bin_count = std::numeric_limits<std::int64_t>::max() / 2;
    const std::int64_t max_elements_in_bin = 1000;
    const auto vertex_count = t.get_vertex_count();
    const value_type max_dist = std::numeric_limits<value_type>::max();

    vp_type* dp = allocate(vp_allocator, vertex_count);
    for (std::int64_t i = 0; i < vertex_count; ++i) {
        new (dp + i) dist_pred<value_type, vertex_type>(max_dist, -1);
    }

    dp[source] = dist_pred<value_type, vertex_type>(0, -1);

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
    using v3v_t = vector_container<v2v_t, v2a_t>;

    v2a_t v2a(alloc_ptr);
    v3v_t local_bins(1, v2a);

    std::int64_t iter = 0;

    while (curr_bin_index != max_bin_count && iter != max_bin_count && !empty_queue) {
        for (std::int64_t i = 0; i < curr_shared_bin_tail; ++i) {
            vertex_type u = shared_bin[i];
            if (dp[u].dist >= delta * static_cast<value_type>(curr_bin_index)) {
                relax_edges_with_pred_seq(t, vals, u, delta, dp, local_bins[0]);
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
                relax_edges_with_pred_seq(t, vals, curr_bin_copy[j], delta, dp, local_bins[0]);
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
            const auto dp_i = dp[i];
            dist_[i] = dp_i.dist;
            pred_[i] = dp_i.pred;
        }

        deallocate(vp_allocator, dp, vertex_count);
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
            const auto dp_i = dp[i];
            pred_[i] = dp_i.pred;
        }

        deallocate(vp_allocator, dp, vertex_count);
        return traverse_result<task::one_to_all>().set_predecessors(
            dal::detail::homogen_table_builder{}.reset(pred_arr, t.get_vertex_count(), 1).build());
    }
}

template struct delta_stepping<__CPU_TAG__, std::int32_t>;

template struct delta_stepping<__CPU_TAG__, double>;

template struct delta_stepping_with_pred<__CPU_TAG__, std::int32_t>;

template struct delta_stepping_with_pred<__CPU_TAG__, double>;

} // namespace oneapi::dal::preview::shortest_paths::backend
