/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include <algorithm>

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

/// @brief Computes boundaries/offsets for blocks on different ranks
///
/// @param[in] sample_counts Number of samples on each device
/// @param[in] block_size    Number of samples in one block
/// @return                  Tuple of node numbers and corresponding offsets
inline auto get_boundary_indices(const ndview<std::int64_t, 1>& sample_counts,
                                 std::int64_t block_size) {
    ONEDAL_ASSERT(sample_counts.has_data());
    std::vector<std::int64_t> nodes, boundaries;
    std::int64_t global_bias = 0;
    for (std::int64_t i = 0; i < sample_counts.get_dimension(0); ++i) {
        const auto s = sample_counts.at(i);
        ONEDAL_ASSERT(s >= 0);
        const auto block_counting = uniform_blocking(s, block_size);
        const auto block_count = block_counting.get_block_count();
        for (std::int64_t block_index = 0; block_index < block_count; ++block_index) {
            nodes.push_back(i);
            const auto local = std::min(s, block_index * block_size);
            const auto biased = local + global_bias;
            boundaries.push_back(biased);
        }
        global_bias += s;
    }
    boundaries.push_back(global_bias);
    return std::make_tuple(nodes, boundaries);
}

} // namespace oneapi::dal::backend::primitives
