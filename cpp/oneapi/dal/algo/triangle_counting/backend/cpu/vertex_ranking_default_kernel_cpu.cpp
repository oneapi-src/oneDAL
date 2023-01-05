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

#include "oneapi/dal/algo/triangle_counting/backend/cpu/vertex_ranking_default_kernel.hpp"

namespace oneapi::dal::preview::triangle_counting::backend {

template array<std::int64_t> triangle_counting_local<__CPU_TAG__>(
    const dal::preview::detail::topology<std::int32_t>& t,
    std::int64_t* triangles_local);

template std::int64_t triangle_counting_global_scalar<__CPU_TAG__>(
    const dal::preview::detail::topology<std::int32_t>& t);

template std::int64_t triangle_counting_global_vector<__CPU_TAG__>(
    const dal::preview::detail::topology<std::int32_t>& t);

template std::int64_t triangle_counting_global_vector_relabel<__CPU_TAG__>(
    const std::int32_t* vertex_neighbors,
    const std::int64_t* edge_offsets,
    const std::int32_t* degrees,
    std::int64_t vertex_count,
    std::int64_t edge_count);

template std::int64_t compute_global_triangles<__CPU_TAG__>(
    const array<std::int64_t>& local_triangles,
    std::int64_t vertex_count);

} // namespace oneapi::dal::preview::triangle_counting::backend
