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

#include "oneapi/dal/algo/triangle_counting/backend/cpu/vertex_ranking_default_kernel_avx512.hpp"

namespace oneapi::dal::preview {
namespace triangle_counting {
namespace backend {

template <>
array<std::int64_t> triangle_counting_local<dal::backend::cpu_dispatch_avx512>(
    const dal::preview::detail::topology<std::int32_t>& data,
    int64_t* triangles_local) {
    return triangle_counting_local_<dal::backend::cpu_dispatch_avx512>(data, triangles_local);
}

template <>
std::int64_t triangle_counting_global_scalar<dal::backend::cpu_dispatch_avx512>(
    const std::int32_t* vertex_neighbors,
    const std::int64_t* edge_offsets,
    const std::int32_t* degrees,
    std::int64_t vertex_count,
    std::int64_t edge_count) {
    return triangle_counting_global_scalar_<dal::backend::cpu_dispatch_avx512>(vertex_neighbors,
                                                                               edge_offsets,
                                                                               degrees,
                                                                               vertex_count,
                                                                               edge_count);
}

template <>
std::int64_t triangle_counting_global_vector<dal::backend::cpu_dispatch_avx512>(
    const std::int32_t* vertex_neighbors,
    const std::int64_t* edge_offsets,
    const std::int32_t* degrees,
    std::int64_t vertex_count,
    std::int64_t edge_count) {
    return triangle_counting_global_vector_<dal::backend::cpu_dispatch_avx512>(vertex_neighbors,
                                                                               edge_offsets,
                                                                               degrees,
                                                                               vertex_count,
                                                                               edge_count);
}

template <>
std::int64_t triangle_counting_global_vector_relabel<dal::backend::cpu_dispatch_avx512>(
    const std::int32_t* vertex_neighbors,
    const std::int64_t* edge_offsets,
    const std::int32_t* degrees,
    std::int64_t vertex_count,
    std::int64_t edge_count) {
    return triangle_counting_global_vector_relabel_<dal::backend::cpu_dispatch_avx512>(
        vertex_neighbors,
        edge_offsets,
        degrees,
        vertex_count,
        edge_count);
}

template std::int64_t compute_global_triangles<dal::backend::cpu_dispatch_avx512>(
    const array<std::int64_t>& local_triangles,
    std::int64_t vertex_count);

} // namespace backend
} // namespace triangle_counting
} // namespace oneapi::dal::preview
