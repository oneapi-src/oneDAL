/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/algo/subgraph_isomorphism/detail/graph_matching_default_kernel.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/si.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

template <>
ONEDAL_EXPORT subgraph_isomorphism::graph_matching_result<task::compute> call_kernel(
    const dal::detail::host_policy& policy,
    const kind& si_kind,
    std::int64_t max_match_count,
    byte_alloc_iface_t* alloc_ptr,
    const dal::preview::detail::topology<std::int32_t>& t_data,
    const dal::preview::detail::topology<std::int32_t>& p_data,
    std::int64_t* vv_t,
    std::int64_t* vv_p) {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return backend::si_call_kernel<decltype(cpu)>(si_kind,
                                                      max_match_count,
                                                      alloc_ptr,
                                                      t_data,
                                                      p_data,
                                                      vv_t,
                                                      vv_p);
    });
}

} // namespace oneapi::dal::preview::subgraph_isomorphism::detail
