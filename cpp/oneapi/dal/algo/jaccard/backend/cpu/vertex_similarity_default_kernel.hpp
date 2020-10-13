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

#include <memory>

#include "oneapi/dal/algo/jaccard/common.hpp"
#include "oneapi/dal/algo/jaccard/vertex_similarity_types.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/policy.hpp"

namespace oneapi::dal::preview {
namespace jaccard {
namespace detail {

template <typename Graph, typename Cpu>
vertex_similarity_result call_jaccard_default_kernel(const descriptor_base &desc,
                                                     vertex_similarity_input<Graph> &input);

DAAL_FORCEINLINE std::int32_t min(const std::int32_t &a, const std::int32_t &b) {
    return (a >= b) ? b : a;
}

DAAL_FORCEINLINE std::int32_t max(const std::int32_t &a, const std::int32_t &b) {
    return (a <= b) ? b : a;
}

DAAL_FORCEINLINE std::size_t compute_max_block_size(const std::int32_t &row_range_begin,
                                                    const std::int32_t &row_range_end,
                                                    const std::int32_t &column_range_begin,
                                                    const std::int32_t &column_range_end) {
    // compute the number of the vertex pairs in the block of the graph
    auto vertex_pairs_count =
        (row_range_end - row_range_begin) * (column_range_end - column_range_begin);

    // compute the size of the result element for the algorithm
    auto vertex_pair_element_count = 2; // 2 elements in the vertex pair
    auto jaccard_coeff_element_count = 1; // 1 Jaccard coeff for the vertex pair

    auto vertex_pair_size = vertex_pair_element_count * sizeof(std::int32_t); // size in bytes
    auto jaccard_coeff_size = jaccard_coeff_element_count * sizeof(float); // size in bytes

    auto block_result_size = (vertex_pair_size + jaccard_coeff_size) * vertex_pairs_count;
    return block_result_size;
}

} // namespace detail
} // namespace jaccard
} // namespace oneapi::dal::preview
