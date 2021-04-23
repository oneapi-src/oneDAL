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

#include "oneapi/dal/algo/subgraph_isomorphism/common.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/graph_matching_types.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/graph/detail/container.hpp"

#include "oneapi/dal/algo/subgraph_isomorphism/detail/si.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/detail/debug.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

template <typename Alloc>
struct alloc_connector : public byte_alloc_iface {
    using byte_t = char;
    using t_allocator_traits =
        typename std::allocator_traits<Alloc>::template rebind_traits<byte_t>;
    alloc_connector(Alloc alloc) : _alloc(alloc) {}
    byte_t* allocate(std::int64_t count) override {
        typename t_allocator_traits::pointer ptr = t_allocator_traits::allocate(_alloc, count);
        if (ptr == nullptr) {
            throw host_bad_alloc();
        }
        return ptr;
    };

    void deallocate(byte_t* ptr, std::int64_t count) override {
        if (ptr != nullptr) {
            t_allocator_traits::deallocate(_alloc, ptr, count);
        }
    };

private:
    Alloc _alloc;
};

template <typename Allocator, typename VertexValue, typename EdgeValue>
graph_matching_result call_subgraph_isomorphism_default_kernel(
    const dal::detail::host_policy& ctx,
    const descriptor_base& desc,
    const Allocator& alloc,
    byte_alloc_iface* alloc_ptr,
    const dal::preview::detail::topology<std::int32_t>& t_data,
    const dal::preview::detail::topology<std::int32_t>& p_data,
    const dal::preview::detail::vertex_values<VertexValue>& vv_t,
    const dal::preview::detail::edge_values<EdgeValue>& ev_t,
    const dal::preview::detail::vertex_values<VertexValue>& vv_p,
    const dal::preview::detail::edge_values<EdgeValue>& ev_p) {
    graph pattern(p_data, graph_storage_scheme::bit, alloc_ptr);
    graph target(t_data, graph_storage_scheme::auto_detect, alloc_ptr);

    const auto t_vertex_count = t_data._vertex_count;
    const auto p_vertex_count = p_data._vertex_count;

    inner_alloc local_allocator(alloc_ptr);

    int64_t *t_vertex_attribute = nullptr, *p_vertex_attribute = nullptr;

    if (vv_t.get_count() != 0) {
        t_vertex_attribute = local_allocator.allocate<std::int64_t>(t_vertex_count);
        for (std::int32_t i = 0; i < t_vertex_count; i++) {
            t_vertex_attribute[i] = vv_t[i];
        }
        target.load_vertex_attribute(t_vertex_count, t_vertex_attribute);
    }
    if (vv_p.get_count() != 0) {
        p_vertex_attribute = local_allocator.allocate<std::int64_t>(p_vertex_count);
        for (std::int32_t i = 0; i < p_vertex_count; i++) {
            p_vertex_attribute[i] = vv_p[i];
        }
        pattern.load_vertex_attribute(p_vertex_count, p_vertex_attribute);
    }

    std::uint64_t control_flags = flow_switch_ids::multi_thread_mode;
    solution results = si(pattern, target, desc.get_kind(), control_flags, alloc_ptr);

    if (t_vertex_attribute)
        local_allocator.deallocate<std::int64_t>(t_vertex_attribute, t_vertex_count);
    if (p_vertex_attribute)
        local_allocator.deallocate<std::int64_t>(p_vertex_attribute, p_vertex_count);

    return graph_matching_result(results.export_as_table(), results.get_solution_count());
}

} // namespace oneapi::dal::preview::subgraph_isomorphism::detail
