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

#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph_matching_default_kernel.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph_matching_default_kernel_scalar.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/graph/detail/service_functions_impl.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::preview {
namespace subgraph_isomorphism {
namespace detail {

template graph_matching_result
call_subgraph_isomorphism_default_kernel_scalar<__CPU_TAG__, std::int32_t>(
    const descriptor_base &desc,
    const dal::preview::detail::topology<std::int32_t> &t_data,
    const dal::preview::detail::topology<std::int32_t> &p_data);

template graph_matching_result
call_subgraph_isomorphism_default_kernel_scalar<__CPU_TAG__, std::int64_t>(
    const descriptor_base &desc,
    const dal::preview::detail::topology<std::int32_t> &t_data,
    const dal::preview::detail::topology<std::int32_t> &p_data);

template <>
graph_matching_result call_subgraph_isomorphism_default_kernel_int32<__CPU_TAG__>(
    const descriptor_base &desc,
    const dal::preview::detail::topology<std::int32_t> &t_data,
    const dal::preview::detail::topology<std::int32_t> &p_data) {
    return call_subgraph_isomorphism_default_kernel_scalar<__CPU_TAG__, std::int32_t>(desc,
                                                                                      t_data,
                                                                                      p_data);
}

} // namespace detail
} // namespace subgraph_isomorphism
} // namespace oneapi::dal::preview
