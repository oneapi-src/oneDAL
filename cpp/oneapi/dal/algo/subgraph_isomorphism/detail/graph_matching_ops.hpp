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
#include "oneapi/dal/algo/subgraph_isomorphism/detail/select_kernel.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/graph_matching_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_topology_builder.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

template <typename Policy, typename Float, class Method, typename Graph>
struct ONEDAL_EXPORT graph_matching_ops_dispatcher {
    graph_matching_result operator()(const Policy &policy,
                                     const descriptor_base &descriptor,
                                     graph_matching_input<Graph> &input) const;
};

template <typename Descriptor, typename Graph>
struct graph_matching_ops {
    using float_t           = typename Descriptor::float_t;
    using method_t          = typename Descriptor::method_t;
    using input_t           = graph_matching_input<Graph>;
    using result_t          = graph_matching_result;
    using descriptor_base_t = descriptor_base;

    void check_preconditions(const Descriptor &param, graph_matching_input<Graph> &input) const {
        using msg = dal::detail::error_messages;

        if (param.get_kind() != kind::induced) {
            throw invalid_argument(msg::unsupported_kind());
        }
        if (param.get_max_match_count() < 0) {
            throw invalid_argument(msg::max_match_count_lt_zero());
        }
        if (param.get_semantic_match() != false) {
            throw invalid_argument(msg::unsupported_semantic_match());
        }
    }

    template <typename Policy>
    auto operator()(const Policy &policy,
                    const Descriptor &desc,
                    graph_matching_input<Graph> &input) const {
        check_preconditions(desc, input);
        return graph_matching_ops_dispatcher<Policy, float_t, method_t, Graph>()(policy,
                                                                                 desc,
                                                                                 input);
    }
};

template <typename Policy, typename Float, class Method, typename Graph>
graph_matching_result graph_matching_ops_dispatcher<Policy, Float, Method, Graph>::operator()(
    const Policy &policy,
    const descriptor_base &desc,
    graph_matching_input<Graph> &input) const {
    const auto &csr_target_topology =
        dal::preview::detail::csr_topology_builder<Graph>()(input.get_target_graph());
    const auto &csr_pattern_topology =
        dal::preview::detail::csr_topology_builder<Graph>()(input.get_target_graph());
    static auto impl =
        get_backend<Policy, Float, Method>(desc, csr_target_topology, csr_pattern_topology);
    return (*impl)(policy, desc, csr_target_topology, csr_pattern_topology);
}

} // namespace oneapi::dal::preview::subgraph_isomorphism::detail
