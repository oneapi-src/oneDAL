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

#include "oneapi/dal/algo/triangle_counting/detail/select_kernel.hpp"
#include "oneapi/dal/algo/triangle_counting/backend/cpu/vertex_ranking_default_kernel.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::preview::triangle_counting::detail {

template <typename Task, typename Float, typename Method>
vertex_ranking_result<Task> backend_default<Task, dal::detail::host_policy,
                                         Float,
                                         Method,
                                         dal::preview::detail::topology<std::int32_t>>::
operator()(const dal::detail::host_policy &policy,
           const descriptor_base<Task> &desc,
           const dal::preview::detail::topology<std::int32_t> &data,
           void *result_ptr) {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return call_triangle_counting_default_kernel_int32<decltype(cpu)>(desc, data, result_ptr);
    });
}

template struct ONEDAL_EXPORT backend_default<dal::preview::triangle_counting::task::local, dal::detail::host_policy,
                                              float,
                                              dal::preview::triangle_counting::method::ordered_count,
                                              dal::preview::detail::topology<std::int32_t>>;

} // namespace oneapi::dal::preview::triangle_counting::detail
