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

template <typename Policy, typename Descriptor, typename Topology>
struct backend_base {
    using float_t = typename Descriptor::float_t;
    using method_t = typename Descriptor::method_t;
    using allocator_t = typename Descriptor::allocator_t;

    virtual graph_matching_result operator()(const Policy &ctx,
                                             const Descriptor &descriptor,
                                             const Topology &t_data,
                                             const Topology &p_data) = 0;
    virtual ~backend_base() = default;
};

template <typename Policy, typename Descriptor, typename Topology>
struct backend_default : public backend_base<Policy, Descriptor, Topology> {
    static_assert(dal::detail::is_one_of_v<Policy, dal::detail::host_policy>,
                  "Host policy only is supported.");

    using allocator_t = typename Descriptor::allocator_t;

    virtual graph_matching_result operator()(const Policy &ctx,
                                             const Descriptor &descriptor,
                                             const Topology &t_data,
                                             const Topology &p_data) {
        return call_subgraph_isomorphism_default_kernel(ctx,
                                                        descriptor,
                                                        descriptor.get_allocator(),
                                                        t_data,
                                                        p_data);
    }
    virtual ~backend_default() {}
};

template <typename Policy, typename Descriptor, typename Topology>
dal::detail::shared<backend_base<Policy, Descriptor, Topology>>
get_backend(const Descriptor &desc, const Topology &target_data, const Topology &pattern_data) {
    return std::make_shared<backend_default<Policy, Descriptor, Topology>>();
}

} // namespace oneapi::dal::preview::subgraph_isomorphism::detail
