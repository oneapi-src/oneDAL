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

#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/si.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

template oneapi::dal::homogen_table si<__CPU_TAG__>(const graph<__CPU_TAG__>& pattern,
                                                    const graph<__CPU_TAG__>& target,
                                                    kind isomorphism_kind,
                                                    std::int64_t max_match_count,
                                                    byte_alloc_iface_t* alloc_ptr);

template subgraph_isomorphism::graph_matching_result<task::compute> si_call_kernel<__CPU_TAG__>(
    const kind& si_kind,
    std::int64_t max_match_count,
    byte_alloc_iface_t* alloc_ptr,
    const dal::preview::detail::topology<std::int32_t>& t_data,
    const dal::preview::detail::topology<std::int32_t>& p_data,
    std::int64_t* vv_t,
    std::int64_t* vv_p);

} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
