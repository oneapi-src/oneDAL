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

#include "oneapi/dal/algo/jaccard/backend/cpu/vertex_similarity_default_kernel.hpp"
#include "oneapi/dal/algo/jaccard/detail/vertex_similarity_default_kernel.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::preview::jaccard::detail {

template <typename Float>
vertex_similarity_result<task::all_vertex_pairs>
vertex_similarity<Float, task::all_vertex_pairs, dal::preview::detail::topology<std::int32_t>>::
operator()(const dal::detail::host_policy& ctx,
           const detail::descriptor_base<task::all_vertex_pairs>& desc,
           const dal::preview::detail::topology<std::int32_t>& t,
           void* result_ptr) {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ ctx }, [&](auto cpu) {
        return backend::jaccard<decltype(cpu)>(desc, t, result_ptr);
    });
}

template struct ONEDAL_EXPORT
    vertex_similarity<float, task::all_vertex_pairs, dal::preview::detail::topology<std::int32_t>>;

} // namespace oneapi::dal::preview::jaccard::detail
