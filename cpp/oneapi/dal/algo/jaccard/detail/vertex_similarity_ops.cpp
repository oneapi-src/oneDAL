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

#include "oneapi/dal/algo/jaccard/detail/vertex_similarity_ops.hpp"
#include "oneapi/dal/algo/jaccard/backend/cpu/select_kernel.hpp"

namespace oneapi::dal::preview {
namespace jaccard {
namespace detail {

template <typename Float, class Method, typename Graph>
ONEAPI_DAL_EXPORT similarity_result
vertex_similarity_ops_dispatcher<Float, Method, Graph>::operator()(
    const descriptor_base &desc,
    const similarity_input<Graph> &input) const {
    static auto impl = get_backend<Float, Method>(desc, input);
    return (*impl)(desc, input);
}

#define INSTANTIATE(F, M, G) \
    template struct ONEAPI_DAL_EXPORT vertex_similarity_ops_dispatcher<F, M, G>;

INSTANTIATE(float,
            oneapi::dal::preview::jaccard::method::by_default,
            undirected_adjacency_array<> &)

} // namespace detail
} // namespace jaccard
} // namespace oneapi::dal::preview
