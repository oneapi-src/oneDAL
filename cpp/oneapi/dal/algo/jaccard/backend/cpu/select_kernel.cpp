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

#include "oneapi/dal/algo/jaccard/backend/cpu/select_kernel.hpp"
#include "oneapi/dal/algo/jaccard/backend/cpu/vertex_similarity_default_kernel.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::preview {
namespace jaccard {
namespace detail {
template <typename Float, typename Method, typename Graph>
vertex_similarity_result backend_default<Float, Method, Graph>::operator()(
    const dal::backend::context_cpu &ctx,
    const descriptor_base &desc,
    vertex_similarity_input<Graph> &input) {
    return dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
        return call_jaccard_default_kernel<Graph, decltype(cpu)>(desc, input);
    });
}

template <>
dal::detail::pimpl<backend_base<undirected_adjacency_array_graph<>>>
get_backend<float, method::by_default, undirected_adjacency_array_graph<>>(
    const descriptor_base &desc,
    vertex_similarity_input<undirected_adjacency_array_graph<>> &input) {
    return dal::detail::pimpl<backend_base<undirected_adjacency_array_graph<>>>(
        new backend_default<float, method::by_default, undirected_adjacency_array_graph<>>);
}
} // namespace detail
} // namespace jaccard
} // namespace oneapi::dal::preview
