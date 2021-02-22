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
#include "oneapi/dal/algo/subgraph_isomorphism/detail/graph_matching_default_kernel.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/graph_matching_types.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

template <typename Policy, typename Topology>
struct ONEDAL_EXPORT backend_base {
    virtual graph_matching_result operator()(const Policy &ctx,
                                             const descriptor_base &descriptor,
                                             const Topology &t_data,
                                             const Topology &p_data) = 0;
    virtual ~backend_base() {}
};

template <typename Policy, typename Float, typename Method, typename Topology>
struct ONEDAL_EXPORT backend_default : public backend_base<Policy, Topology> {
    virtual graph_matching_result operator()(const Policy &ctx,
                                             const descriptor_base &descriptor,
                                             const Topology &t_data,
                                             const Topology &p_data) {
        return call_subgraph_isomorphism_default_kernel_general(descriptor, t_data, p_data);
    }
    virtual ~backend_default() {}
};

template <typename Float, typename Method>
struct backend_default<dal::detail::host_policy,
                       Float,
                       Method,
                       dal::preview::detail::topology<std::int32_t>>
        : public backend_base<dal::detail::host_policy,
                              dal::preview::detail::topology<std::int32_t>> {
    virtual graph_matching_result operator()(
        const dal::detail::host_policy &ctx,
        const descriptor_base &descriptor,
        const dal::preview::detail::topology<std::int32_t> &target_data,
        const dal::preview::detail::topology<std::int32_t> &pattern_data);
    virtual ~backend_default() {}
};

template <typename Policy, typename Float, class Method, typename Topology>
dal::detail::pimpl<backend_base<Policy, Topology>> get_backend(const descriptor_base &desc,
                                                               const Topology &target_data,
                                                               const Topology &pattern_data) {
    return dal::detail::pimpl<backend_base<Policy, Topology>>(
        new backend_default<Policy, float, method::by_default, Topology>);
}

} // namespace oneapi::dal::preview::subgraph_isomorphism::detail
