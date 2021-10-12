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

#include "oneapi/dal/algo/shortest_paths/detail/traverse_default_kernel.hpp"
#include "oneapi/dal/algo/shortest_paths/backend/cpu/traverse_default_kernel.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::preview::shortest_paths::detail {

template <typename Float, typename EdgeValue>
traverse_result<task::one_to_all>
delta_stepping<Float, task::one_to_all, dal::preview::detail::topology<std::int32_t>, EdgeValue>::
operator()(const dal::detail::host_policy& policy,
           const detail::descriptor_base<task::one_to_all>& desc,
           const dal::preview::detail::topology<std::int32_t>& t,
           const EdgeValue* vals,
           byte_alloc_iface* alloc_ptr) const {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return backend::delta_stepping<decltype(cpu), EdgeValue, backend::mode::distances>{}(
            desc,
            t,
            vals,
            alloc_ptr);
    });
}

template <typename Float, typename EdgeValue>
traverse_result<task::one_to_all> delta_stepping_with_pred<
    Float,
    task::one_to_all,
    dal::preview::detail::topology<std::int32_t>,
    EdgeValue>::operator()(const dal::detail::host_policy& policy,
                           const detail::descriptor_base<task::one_to_all>& desc,
                           const dal::preview::detail::topology<std::int32_t>& t,
                           const EdgeValue* vals,
                           byte_alloc_iface* alloc_ptr) const {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return backend::delta_stepping<decltype(cpu),
                                       EdgeValue,
                                       backend::mode::distances_predecessors>{}(desc,
                                                                                t,
                                                                                vals,
                                                                                alloc_ptr);
    });
}

template struct ONEDAL_EXPORT delta_stepping<float,
                                             task::one_to_all,
                                             dal::preview::detail::topology<std::int32_t>,
                                             std::int32_t>;

template struct ONEDAL_EXPORT
    delta_stepping<float, task::one_to_all, dal::preview::detail::topology<std::int32_t>, double>;

template struct ONEDAL_EXPORT delta_stepping_with_pred<float,
                                                       task::one_to_all,
                                                       dal::preview::detail::topology<std::int32_t>,
                                                       std::int32_t>;

template struct ONEDAL_EXPORT delta_stepping_with_pred<float,
                                                       task::one_to_all,
                                                       dal::preview::detail::topology<std::int32_t>,
                                                       double>;

} // namespace oneapi::dal::preview::shortest_paths::detail
