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

#include "oneapi/dal/algo/jaccard/common.hpp"
#include "oneapi/dal/algo/jaccard/detail/select_kernel.hpp"
#include "oneapi/dal/algo/jaccard/vertex_similarity_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_topology_builder.hpp"

namespace oneapi::dal::preview::jaccard::detail {

template <typename Policy, typename Float, class Method, typename Graph>
struct ONEDAL_EXPORT vertex_similarity_ops_dispatcher {
    vertex_similarity_result operator()(const Policy &policy,
                                        const descriptor_base &descriptor,
                                        vertex_similarity_input<Graph> &input) const;
};

template <typename Descriptor, typename Graph>
struct vertex_similarity_ops {
    using float_t = typename Descriptor::float_t;
    using method_t = typename Descriptor::method_t;
    using input_t = vertex_similarity_input<Graph>;
    using result_t = vertex_similarity_result;
    using descriptor_base_t = descriptor_base;

    void check_preconditions(const Descriptor &param, vertex_similarity_input<Graph> &input) const {
        using msg = dal::detail::error_messages;

        const std::int64_t row_begin = param.get_row_range_begin();
        const std::int64_t row_end = param.get_row_range_end();
        const std::int64_t column_begin = param.get_column_range_begin();
        const std::int64_t column_end = param.get_column_range_end();
        if (row_begin < 0 || column_begin < 0) {
            throw invalid_argument(msg::negative_interval());
        }
        if (row_begin > row_end) {
            throw invalid_argument(msg::row_begin_gt_row_end());
        }
        if (column_begin > column_end) {
            throw invalid_argument(msg::column_begin_gt_column_end());
        }
        const std::int64_t vertex_count =
            dal::detail::get_impl(input.get_graph()).get_topology()._vertex_count;
        // Safe conversion as ranges were checked
        if (row_end > vertex_count || column_end > vertex_count) {
            throw out_of_range(msg::interval_gt_vertex_count());
        }
        if (row_end >= dal::detail::limits<std::int32_t>::max() ||
            column_end >= dal::detail::limits<std::int32_t>::max()) {
            throw invalid_argument(msg::range_idx_gt_max_int32());
        }
    }

    template <typename Policy>
    auto operator()(const Policy &policy,
                    const Descriptor &desc,
                    vertex_similarity_input<Graph> &input) const {
        check_preconditions(desc, input);
        return vertex_similarity_ops_dispatcher<Policy, float_t, method_t, Graph>()(policy,
                                                                                    desc,
                                                                                    input);
    }
};

template <typename Policy, typename Float, class Method, typename Graph>
vertex_similarity_result vertex_similarity_ops_dispatcher<Policy, Float, Method, Graph>::operator()(
    const Policy &policy,
    const descriptor_base &desc,
    vertex_similarity_input<Graph> &input) const {
    const auto &csr_topology =
        dal::preview::detail::csr_topology_builder<Graph>()(input.get_graph());
    const std::int64_t row_begin = desc.get_row_range_begin();
    const std::int64_t row_end = desc.get_row_range_end();
    const std::int64_t column_begin = desc.get_column_range_begin();
    const std::int64_t column_end = desc.get_column_range_end();
    const std::int64_t number_elements_in_block =
        get_number_elements_in_block(row_begin, row_end, column_begin, column_end);
    const std::int64_t max_block_size =
        get_max_block_size<Float, vertex_type<Graph>>(number_elements_in_block);
    void *result_ptr = input.get_caching_builder()(max_block_size);
    static auto impl = get_backend<Policy, Float, Method>(desc, csr_topology);
    return (*impl)(policy, desc, csr_topology, result_ptr);
}

} // namespace oneapi::dal::preview::jaccard::detail
