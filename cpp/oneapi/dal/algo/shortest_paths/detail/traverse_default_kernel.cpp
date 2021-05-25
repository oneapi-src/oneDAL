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

#include "oneapi/dal/algo/shortest_paths/detail/traverse_default_kernel.hpp"
#include "oneapi/dal/algo/shortest_paths/backend/cpu/traverse_default_kernel.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::preview::shortest_paths::detail {
/*
template <typename Float>
array<std::int64_t>
triangle_counting<Float, task::local, dal::preview::detail::topology<std::int32_t>, automatic>::
operator()(const dal::detail::host_policy& policy,
           const dal::preview::detail::topology<std::int32_t>& t,
           std::int64_t* triangles_local) const {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return backend::triangle_counting_local<decltype(cpu)>(t, triangles_local);
    });
}
*/
} // namespace oneapi::dal::preview::shortest_paths::detail
