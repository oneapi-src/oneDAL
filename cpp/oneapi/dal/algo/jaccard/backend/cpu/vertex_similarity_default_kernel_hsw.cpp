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

#include <immintrin.h>

#include "oneapi/dal/algo/jaccard/backend/cpu/vertex_similarity_default_kernel.hpp"
#include "oneapi/dal/algo/jaccard/backend/cpu/vertex_similarity_default_kernel_avx2.hpp"
#include "oneapi/dal/algo/jaccard/common.hpp"
#include "oneapi/dal/algo/jaccard/vertex_similarity_types.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/graph/detail/graph_service_functions_impl.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::preview {
namespace jaccard {
namespace detail {

template vertex_similarity_result
call_jaccard_default_kernel_avx2<oneapi::dal::backend::cpu_dispatch_avx2>(
    const descriptor_base &desc,
    vertex_similarity_input<undirected_adjacency_array_graph<>> &input);

template <>
vertex_similarity_result call_jaccard_default_kernel<undirected_adjacency_array_graph<>,
                                                     oneapi::dal::backend::cpu_dispatch_avx2>(
    const descriptor_base &desc,
    vertex_similarity_input<undirected_adjacency_array_graph<>> &input) {
    return call_jaccard_default_kernel_avx2<oneapi::dal::backend::cpu_dispatch_avx2>(desc, input);
}
} // namespace detail
} // namespace jaccard
} // namespace oneapi::dal::preview
