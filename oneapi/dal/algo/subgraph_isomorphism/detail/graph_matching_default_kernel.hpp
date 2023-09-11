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

#pragma once

#include "oneapi/dal/algo/subgraph_isomorphism/common.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/graph_matching_types.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/graph/detail/container.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/memory.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

using byte_alloc_iface_t = oneapi::dal::preview::detail::byte_alloc_iface;

template <typename Task>
subgraph_isomorphism::graph_matching_result<Task> call_kernel(
    const dal::detail::host_policy& ctx,
    const kind& desc,
    std::int64_t max_match_count,
    byte_alloc_iface_t* alloc_ptr,
    const dal::preview::detail::topology<std::int32_t>& t_data,
    const dal::preview::detail::topology<std::int32_t>& p_data,
    std::int64_t* vv_t = nullptr,
    std::int64_t* vv_p = nullptr);

template <typename Allocator, typename VertexValue, typename EdgeValue>
struct call_subgraph_isomorphism_kernel_cpu {
    graph_matching_result<task::compute> operator()(
        const dal::detail::host_policy& ctx,
        const descriptor_base<task::compute>& desc,
        const Allocator& alloc,
        byte_alloc_iface_t* alloc_ptr,
        const dal::preview::detail::topology<std::int32_t>& t_data,
        const dal::preview::detail::topology<std::int32_t>& p_data,
        const dal::preview::detail::vertex_values<VertexValue>& vv_t,
        const dal::preview::detail::edge_values<EdgeValue>& ev_t,
        const dal::preview::detail::vertex_values<VertexValue>& vv_p,
        const dal::preview::detail::edge_values<EdgeValue>& ev_p) {
        std::int64_t *t_vertex_attribute = nullptr, *p_vertex_attribute = nullptr;

        const auto t_vertex_count = t_data._vertex_count;
        const auto p_vertex_count = p_data._vertex_count;
        if (vv_t.get_count() != 0) {
            t_vertex_attribute = reinterpret_cast<std::int64_t*>(
                alloc_ptr->allocate(t_vertex_count * sizeof(std::int64_t)));
            if (t_vertex_attribute == nullptr) {
                throw oneapi::dal::host_bad_alloc();
            }
            for (std::int32_t i = 0; i < t_vertex_count; i++) {
                t_vertex_attribute[i] = vv_t[i];
            }
        }
        if (vv_p.get_count() != 0) {
            p_vertex_attribute = reinterpret_cast<std::int64_t*>(
                alloc_ptr->allocate(p_vertex_count * sizeof(std::int64_t)));
            if (p_vertex_attribute == nullptr) {
                throw oneapi::dal::host_bad_alloc();
            }
            for (std::int32_t i = 0; i < p_vertex_count; i++) {
                p_vertex_attribute[i] = vv_p[i];
            }
        }
        if (ev_t.get_count() != 0 || ev_p.get_count() != 0) {
            using msg = dal::detail::error_messages;
            throw unimplemented(msg::subgraph_isomorphism_is_not_implemented_for_labeled_edges());
        }
        auto result = call_kernel<task::compute>(ctx,
                                                 desc.get_kind(),
                                                 desc.get_max_match_count(),
                                                 alloc_ptr,
                                                 t_data,
                                                 p_data,
                                                 t_vertex_attribute,
                                                 p_vertex_attribute);
        if (t_vertex_attribute)
            alloc_ptr->deallocate(reinterpret_cast<byte_t*>(t_vertex_attribute),
                                  t_vertex_count * sizeof(std::int64_t));
        if (p_vertex_attribute)
            alloc_ptr->deallocate(reinterpret_cast<byte_t*>(p_vertex_attribute),
                                  p_vertex_count * sizeof(std::int64_t));
        return result;
    }
};

template <typename Allocator>
struct call_subgraph_isomorphism_kernel_cpu<Allocator,
                                            oneapi::dal::preview::empty_value,
                                            oneapi::dal::preview::empty_value> {
    graph_matching_result<task::compute> operator()(
        const dal::detail::host_policy& ctx,
        const descriptor_base<task::compute>& desc,
        const Allocator& alloc,
        byte_alloc_iface_t* alloc_ptr,
        const dal::preview::detail::topology<std::int32_t>& t_data,
        const dal::preview::detail::topology<std::int32_t>& p_data,
        const dal::preview::detail::vertex_values<oneapi::dal::preview::empty_value>& vv_t,
        const dal::preview::detail::edge_values<oneapi::dal::preview::empty_value>& ev_t,
        const dal::preview::detail::vertex_values<oneapi::dal::preview::empty_value>& vv_p,
        const dal::preview::detail::edge_values<oneapi::dal::preview::empty_value>& ev_p) {
        auto result = call_kernel<task::compute>(ctx,
                                                 desc.get_kind(),
                                                 desc.get_max_match_count(),
                                                 alloc_ptr,
                                                 t_data,
                                                 p_data);
        return result;
    }
};

} // namespace oneapi::dal::preview::subgraph_isomorphism::detail
