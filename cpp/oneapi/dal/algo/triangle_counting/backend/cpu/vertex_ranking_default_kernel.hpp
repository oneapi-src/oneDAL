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

#include <memory>

#include "oneapi/dal/algo/triangle_counting/common.hpp"
#include "oneapi/dal/algo/triangle_counting/vertex_ranking_types.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/threading.hpp"

namespace oneapi::dal::preview {
namespace triangle_counting {
namespace detail {

template <typename Cpu>
std::int64_t triangle_counting_global_scalar_cpu(const std::int32_t* vertex_neighbors,
                                                 const std::int64_t* edge_offsets,
                                                 const std::int32_t* degrees,
                                                 std::int64_t vertex_count,
                                                 std::int64_t edge_count);

template <typename Cpu>
std::int64_t triangle_counting_global_vector_cpu(const std::int32_t* vertex_neighbors,
                                                 const std::int64_t* edge_offsets,
                                                 const std::int32_t* degrees,
                                                 std::int64_t vertex_count,
                                                 std::int64_t edge_count);

template <typename Cpu>
std::int64_t triangle_counting_global_vector_relabel_cpu(const std::int32_t* vertex_neighbors,
                                                         const std::int64_t* edge_offsets,
                                                         const std::int32_t* degrees,
                                                         std::int64_t vertex_count,
                                                         std::int64_t edge_count);

template <typename Cpu>
array<std::int64_t> triangle_counting_local_cpu(
    const dal::preview::detail::topology<std::int32_t>& data,
    int64_t* triangles_local);

template <typename Cpu>
std::int64_t compute_global_triangles_cpu(const array<std::int64_t>& local_triangles,
                                          std::int64_t vertex_count) {
    std::int64_t total_s = oneapi::dal::detail::parallel_reduce_int32_int64_t(
        vertex_count,
        (std::int64_t)0,
        [&](std::int32_t begin_u, std::int32_t end_u, std::int64_t tc) -> std::int64_t {
            for (auto u = begin_u; u != end_u; ++u) {
                tc += local_triangles[u];
            }
            return tc;
        },
        [&](std::int64_t x, std::int64_t y) -> std::int64_t {
            return x + y;
        });
    total_s /= 3;
    return total_s;
}

DAAL_FORCEINLINE std::int32_t min(const std::int32_t& a, const std::int32_t& b) {
    return (a >= b) ? b : a;
}

DAAL_FORCEINLINE std::int32_t max(const std::int32_t& a, const std::int32_t& b) {
    return (a <= b) ? b : a;
}

} // namespace detail
} // namespace triangle_counting
} // namespace oneapi::dal::preview
