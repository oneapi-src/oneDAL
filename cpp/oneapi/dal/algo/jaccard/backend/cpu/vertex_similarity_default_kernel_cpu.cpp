/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2023-24 FUJITSU LIMITED
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
#include "oneapi/dal/algo/jaccard/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/graph/detail/service_functions_impl.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::preview::jaccard::backend {

template vertex_similarity_result<task::all_vertex_pairs> jaccard<__CPU_TAG__>(
    const detail::descriptor_base<task::all_vertex_pairs> &desc,
    const dal::preview::detail::topology<std::int32_t> &t,
    void *result_ptr);

} // namespace oneapi::dal::preview::jaccard::backend
