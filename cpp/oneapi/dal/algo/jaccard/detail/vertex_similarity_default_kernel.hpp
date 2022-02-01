/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include "oneapi/dal/algo/jaccard/detail/service.hpp"
#include "oneapi/dal/algo/jaccard/vertex_similarity_types.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"

namespace oneapi::dal::preview::jaccard::detail {

template <typename Float, typename Task, typename Topology, typename... Params>
struct vertex_similarity {
    vertex_similarity_result<Task> operator()(const dal::detail::host_policy& ctx,
                                              const detail::descriptor_base<Task>& desc,
                                              const Topology& t,
                                              void* result_ptr);
};

template <typename Float>
struct vertex_similarity<Float,
                         task::all_vertex_pairs,
                         dal::preview::detail::topology<std::int32_t>> {
    vertex_similarity_result<task::all_vertex_pairs> operator()(
        const dal::detail::host_policy& ctx,
        const detail::descriptor_base<task::all_vertex_pairs>& desc,
        const dal::preview::detail::topology<std::int32_t>& t,
        void* result_ptr);
};

template <typename Float, typename Method, typename Task, typename Topology>
struct vertex_similarity_kernel_cpu {
    vertex_similarity_result<Task> operator()(const dal::detail::host_policy& ctx,
                                              const detail::descriptor_base<Task>& desc,
                                              const Topology& t,
                                              caching_builder& result_builder) const;
};

template <typename Topology>
struct vertex_similarity_kernel_cpu<float, method::fast, task::all_vertex_pairs, Topology> {
    vertex_similarity_result<task::all_vertex_pairs> operator()(
        const dal::detail::host_policy& ctx,
        const detail::descriptor_base<task::all_vertex_pairs>& desc,
        const Topology& t,
        caching_builder& result_builder) const {
        const std::int64_t row_begin = desc.get_row_range_begin();
        const std::int64_t row_end = desc.get_row_range_end();
        const std::int64_t column_begin = desc.get_column_range_begin();
        const std::int64_t column_end = desc.get_column_range_end();
        const std::int64_t number_elements_in_block =
            compute_number_elements_in_block(row_begin, row_end, column_begin, column_end);
        if (number_elements_in_block == 0) {
            return vertex_similarity_result<task::all_vertex_pairs>();
        }
        const std::int64_t max_block_size = compute_max_block_size<
            typename detail::descriptor_base<task::all_vertex_pairs>::float_t,
            std::int32_t>(number_elements_in_block);
        void* result_ptr = result_builder(max_block_size);
        using kernel_t = vertex_similarity<float, task::all_vertex_pairs, Topology>;
        return kernel_t()(ctx, desc, t, result_ptr);
    }
};

} // namespace oneapi::dal::preview::jaccard::detail
