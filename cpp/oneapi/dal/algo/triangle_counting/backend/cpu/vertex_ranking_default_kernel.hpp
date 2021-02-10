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

#pragma once

#include <memory>

#include "oneapi/dal/algo/triangle_counting/common.hpp"
#include "oneapi/dal/algo/triangle_counting/vertex_ranking_types.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/policy.hpp"

namespace oneapi::dal::preview {
namespace triangle_counting {
namespace detail {

using input_t = vertex_ranking_input<task::local>;
using descriptor_t = detail::descriptor_base<task::local>;
using result_t = vertex_ranking_result<task::local>;

template <typename Cpu>
result_t call_triangle_counting_default_kernel_int32(
    const descriptor_t &desc,
    const dal::preview::detail::topology<int32_t> &data);

DAAL_FORCEINLINE std::int32_t min(const std::int32_t &a, const std::int32_t &b) {
    return (a >= b) ? b : a;
}

DAAL_FORCEINLINE std::int32_t max(const std::int32_t &a, const std::int32_t &b) {
    return (a <= b) ? b : a;
}

} // namespace detail
} // namespace triangle_counting
} // namespace oneapi::dal::preview
