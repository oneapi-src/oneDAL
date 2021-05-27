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

#include "oneapi/dal/algo/shortest_paths/backend/cpu/traverse_default_kernel.hpp"

namespace oneapi::dal::preview::shortest_paths::backend {

template traverse_result<task::one_to_all> delta_stepping<__CPU_TAG__, std::int32_t>(
    const detail::descriptor_base<task::one_to_all>& desc,
    const dal::preview::detail::topology<std::int32_t>& t,
    const std::int32_t* vals,
    byte_alloc_iface* alloc_ptr);

template traverse_result<task::one_to_all> delta_stepping<__CPU_TAG__, double>(
    const detail::descriptor_base<task::one_to_all>& desc,
    const dal::preview::detail::topology<std::int32_t>& t,
    const double* vals,
    byte_alloc_iface* alloc_ptr);

} // namespace oneapi::dal::preview::shortest_paths::backend
