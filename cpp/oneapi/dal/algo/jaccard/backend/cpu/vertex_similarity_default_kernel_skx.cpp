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

#ifndef __ARM_ARCH
#include <immintrin.h>
#endif

#include "oneapi/dal/algo/jaccard/backend/cpu/vertex_similarity_default_kernel.hpp"
#include "oneapi/dal/algo/jaccard/backend/cpu/vertex_similarity_default_kernel_avx512.hpp"
#include "oneapi/dal/algo/jaccard/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/graph/detail/service_functions_impl.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"

namespace oneapi::dal::preview::jaccard::backend {

#ifdef __ARM_ARCH
template vertex_similarity_result<task::all_vertex_pairs> jaccard_sve<
    dal::backend::cpu_dispatch_sve>(const detail::descriptor_base<task::all_vertex_pairs>& desc,
                                    const dal::preview::detail::topology<std::int32_t>& t,
                                    void* result_ptr);

template <>
vertex_similarity_result<task::all_vertex_pairs> jaccard<dal::backend::cpu_dispatch_sve>(
    const detail::descriptor_base<task::all_vertex_pairs>& desc,
    const dal::preview::detail::topology<std::int32_t>& t,
    void* result_ptr) {
    return jaccard_sve<dal::backend::cpu_dispatch_sve>(desc, t, result_ptr);
}
#else
template vertex_similarity_result<task::all_vertex_pairs> jaccard_avx512<
    dal::backend::cpu_dispatch_avx512>(const detail::descriptor_base<task::all_vertex_pairs>& desc,
                                       const dal::preview::detail::topology<std::int32_t>& t,
                                       void* result_ptr);

template <>
vertex_similarity_result<task::all_vertex_pairs> jaccard<dal::backend::cpu_dispatch_avx512>(
    const detail::descriptor_base<task::all_vertex_pairs>& desc,
    const dal::preview::detail::topology<std::int32_t>& t,
    void* result_ptr) {
    return jaccard_avx512<dal::backend::cpu_dispatch_avx512>(desc, t, result_ptr);
}

#endif
} // namespace oneapi::dal::preview::jaccard::backend
