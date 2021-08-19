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

#include "oneapi/dal/algo/louvain/common.hpp"
#include "oneapi/dal/algo/louvain/vertex_partitioning_types.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_topology_builder.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::preview::louvain::detail {

using namespace dal::preview::detail;

template <typename Method, typename Task, typename Allocator, typename Graph>
struct vertex_partitioning_kernel_cpu {
    inline vertex_partitioning_result<Task> operator()(const dal::detail::host_policy &ctx,
                                                       const detail::descriptor_base<Task> &desc,
                                                       const Allocator &alloc,
                                                       const Graph &g,
                                                       const table &init_partition) const;
};

template <typename Float, typename Task, typename Topology, typename EdgeValue, typename... Param>
struct louvain_kernel {
    vertex_partitioning_result<Task> operator()(const dal::detail::host_policy &ctx,
                                                const detail::descriptor_base<Task> &desc,
                                                const Topology &t,
                                                const EdgeValue *vals,
                                                byte_alloc_iface *alloc) const;
};

template <typename Float, typename EdgeValue>
struct louvain_kernel<Float,
                      task::vertex_partitioning,
                      dal::preview::detail::topology<std::int32_t>,
                      EdgeValue> {
    vertex_partitioning_result<task::vertex_partitioning> operator()(
        const dal::detail::host_policy &ctx,
        const detail::descriptor_base<task::vertex_partitioning> &desc,
        const dal::preview::detail::topology<std::int32_t> &t,
        const std::int32_t *init_partition,
        const EdgeValue *vals,
        byte_alloc_iface *alloc) const;
};

template <typename Allocator, typename Graph>
struct vertex_partitioning_kernel_cpu<method::fast, task::vertex_partitioning, Allocator, Graph> {
    inline vertex_partitioning_result<task::vertex_partitioning> operator()(
        const dal::detail::host_policy &ctx,
        const detail::descriptor_base<task::vertex_partitioning> &desc,
        const Allocator &alloc,
        const Graph &g,
        const table &init_partition) const {
        using topology_type = typename graph_traits<Graph>::impl_type::topology_type;
        using value_type = edge_user_value_type<Graph>;
        const auto &t = dal::preview::detail::csr_topology_builder<Graph>()(g);
        const auto vals = dal::detail::get_impl(g).get_edge_values().get_data();
        auto init_partition_arr =
            oneapi::dal::row_accessor<const std::int32_t>(init_partition).pull();
        const auto init_partition_data = init_partition_arr.get_data();
        alloc_connector<Allocator> alloc_con(alloc);
        return louvain_kernel<float, task::vertex_partitioning, topology_type, value_type>{}(
            ctx,
            desc,
            t,
            init_partition_data,
            vals,
            &alloc_con);
    }
};

} // namespace oneapi::dal::preview::louvain::detail
