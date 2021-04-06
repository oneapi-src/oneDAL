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
#include "oneapi/dal/algo/jaccard/detail/service.hpp"
#include "oneapi/dal/algo/jaccard/vertex_similarity_types.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"

namespace oneapi::dal::preview::jaccard::detail {

ONEDAL_EXPORT vertex_similarity_result<task::all_vertex_pairs> jaccard(
    const dal::detail::host_policy& ctx,
    const detail::descriptor_base<task::all_vertex_pairs>& desc,
    const dal::preview::detail::topology<std::int32_t>& t,
    void* result_ptr);

inline vertex_similarity_result<task::all_vertex_pairs> jaccard_default_kernel(
    const dal::detail::host_policy& ctx,
    const detail::descriptor_base<task::all_vertex_pairs>& desc,
    const dal::preview::detail::topology<std::int32_t>& t,
    caching_builder& result_builder) {
    const std::int64_t row_begin = desc.get_row_range_begin();
    const std::int64_t row_end = desc.get_row_range_end();
    const std::int64_t column_begin = desc.get_column_range_begin();
    const std::int64_t column_end = desc.get_column_range_end();
    const std::int64_t number_elements_in_block =
        compute_number_elements_in_block(row_begin, row_end, column_begin, column_end);
    const std::int64_t max_block_size =
        compute_max_block_size<typename detail::descriptor_base<task::all_vertex_pairs>::float_t,
                               std::int32_t>(number_elements_in_block);
    void* result_ptr = result_builder(max_block_size);
    return jaccard(ctx, desc, t, result_ptr);
}

} // namespace oneapi::dal::preview::jaccard::detail
