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

#include "oneapi/dal/algo/triangle_counting/common.hpp"
#include "oneapi/dal/algo/triangle_counting/detail/select_kernel.hpp"
#include "oneapi/dal/algo/triangle_counting/vertex_ranking_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_topology_builder.hpp"

namespace oneapi::dal::preview::triangle_counting::detail {

template <typename Policy, typename Float, class Method, typename Graph>
struct ONEDAL_EXPORT vertex_ranking_ops_dispatcher {
    vertex_ranking_result operator()(const Policy &policy,
                                        const descriptor_base &descriptor,
                                        vertex_ranking_input<Graph> &input) const;
};

template <typename Descriptor, typename Graph>
struct vertex_ranking_ops {
    using float_t = typename Descriptor::float_t;
    using method_t = typename Descriptor::method_t;
    using input_t = vertex_similarity_input<Graph>;
    using result_t = vertex_similarity_result;
    using descriptor_base_t = descriptor_base;

    void check_preconditions(const Descriptor &param, vertex_ranking_input<Graph> &input) const {
        using msg = dal::detail::error_messages;
    }

    template <typename Policy>
    auto operator()(const Policy &policy,
                    const Descriptor &desc,
                    vertex_ranking_input<Graph> &input) const {
        check_preconditions(desc, input);
        return vertex_ranking_ops_dispatcher<Policy, float_t, method_t, Graph>()(policy,
                                                                                    desc,
                                                                                    input);
    }
};

template <typename Policy, typename Float, class Method, typename Graph>
vertex_ranking_result vertex_ranking_ops_dispatcher<Policy, Float, Method, Graph>::operator()(
    const Policy &policy,
    const descriptor_base &desc,
    vertex_ranking_input<Graph> &input) const {
    const auto &csr_topology =
        dal::preview::detail::csr_topology_builder<Graph>()(input.get_graph());
    const detail::kind kind = desc.get_kind();
    const detail::relabel relabel = desc.get_relabel();

    static auto impl = get_backend<Policy, Float, Method>(desc, csr_topology);
    return (*impl)(policy, desc, csr_topology, kind, relabel);
}

} // namespace oneapi::dal::preview::jaccard::detail
