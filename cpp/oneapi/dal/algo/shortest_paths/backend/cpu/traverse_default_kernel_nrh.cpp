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

#include "oneapi/dal/algo/shortest_paths/backend/cpu/traverse_default_kernel.hpp"
#include "oneapi/dal/algo/shortest_paths/common.hpp"
#include "oneapi/dal/algo/shortest_paths/traverse_types.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/graph/detail/container.hpp"

namespace oneapi::dal::preview::shortest_paths::backend {

using namespace oneapi::dal::preview::detail;
using namespace oneapi::dal::preview::backend;

traverse_result<task::one_to_all>
delta_stepping<dal::backend::cpu_dispatch_sse2, std::int32_t>::operator()(
    const detail::descriptor_base<task::one_to_all>& desc,
    const dal::preview::detail::topology<std::int32_t>& t,
    const std::int32_t* vals,
    byte_alloc_iface* alloc_ptr) {
    return delta_stepping_sequential<dal::backend::cpu_dispatch_sse2, std::int32_t>()(desc,
                                                                                      t,
                                                                                      vals,
                                                                                      alloc_ptr);
}

template struct delta_stepping<dal::backend::cpu_dispatch_sse2, double>;

template struct delta_stepping_with_pred<dal::backend::cpu_dispatch_sse2, std::int32_t>;

template struct delta_stepping_with_pred<dal::backend::cpu_dispatch_sse2, double>;

} // namespace oneapi::dal::preview::shortest_paths::backend
