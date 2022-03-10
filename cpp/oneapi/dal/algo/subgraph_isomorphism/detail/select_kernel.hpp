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
#include "oneapi/dal/algo/subgraph_isomorphism/detail/graph_matching_default_kernel.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/graph_matching_types.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/detail/memory.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

template <typename Policy,
          typename Descriptor,
          typename Topology,
          typename VertexValue,
          typename EdgeValue>
struct backend_base {
    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = typename Descriptor::method_t;
    using allocator_t = typename Descriptor::allocator_t;

    virtual graph_matching_result<task_t> operator()(
        const Policy &ctx,
        const Descriptor &descriptor,
        const Topology &t_data,
        const Topology &p_data,
        const oneapi::dal::preview::detail::vertex_values<VertexValue> &vv_t,
        const oneapi::dal::preview::detail::edge_values<EdgeValue> &ev_t,
        const oneapi::dal::preview::detail::vertex_values<VertexValue> &vv_p,
        const oneapi::dal::preview::detail::edge_values<EdgeValue> &ev_p) = 0;
    virtual ~backend_base() = default;
};

template <typename Policy,
          typename Descriptor,
          typename Topology,
          typename VertexValue,
          typename EdgeValue>
struct backend_default : public backend_base<Policy, Descriptor, Topology, VertexValue, EdgeValue> {
    static_assert(dal::detail::is_one_of_v<Policy, dal::detail::host_policy>,
                  "Host policy only is supported.");

    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = typename Descriptor::method_t;
    using allocator_t = typename Descriptor::allocator_t;

    virtual graph_matching_result<task_t> operator()(
        const Policy &ctx,
        const Descriptor &descriptor,
        const Topology &t_data,
        const Topology &p_data,
        const oneapi::dal::preview::detail::vertex_values<VertexValue> &vv_t,
        const oneapi::dal::preview::detail::edge_values<EdgeValue> &ev_t,
        const oneapi::dal::preview::detail::vertex_values<VertexValue> &vv_p,
        const oneapi::dal::preview::detail::edge_values<EdgeValue> &ev_p) {
        oneapi::dal::preview::detail::alloc_connector<allocator_t> alloc_con(
            descriptor.get_allocator());
        static auto impl = std::make_shared<
            call_subgraph_isomorphism_kernel_cpu<allocator_t, VertexValue, EdgeValue>>();
        return (*impl)(ctx,
                       descriptor,
                       descriptor.get_allocator(),
                       &alloc_con,
                       t_data,
                       p_data,
                       vv_t,
                       ev_t,
                       vv_p,
                       ev_p);
    }
};

template <typename Policy,
          typename Descriptor,
          typename Topology,
          typename VertexValue,
          typename EdgeValue>
dal::detail::shared<backend_base<Policy, Descriptor, Topology, VertexValue, EdgeValue>> get_backend(
    const Descriptor &desc,
    const Topology &target_data,
    const Topology &pattern_data,
    const oneapi::dal::preview::detail::vertex_values<VertexValue> &vv_t,
    const oneapi::dal::preview::detail::edge_values<EdgeValue> &ev_t) {
    return std::make_shared<
        backend_default<Policy, Descriptor, Topology, VertexValue, EdgeValue>>();
}

} // namespace oneapi::dal::preview::subgraph_isomorphism::detail
