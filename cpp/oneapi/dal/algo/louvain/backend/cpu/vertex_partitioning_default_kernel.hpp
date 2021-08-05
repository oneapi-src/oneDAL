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

#pragma once

#include "oneapi/dal/algo/louvain/common.hpp"
#include "oneapi/dal/algo/louvain/vertex_partitioning_types.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::preview::louvain::backend {
using namespace oneapi::dal::preview::detail;
using namespace oneapi::dal::preview::backend;

template <typename Cpu, typename EdgeValue>
struct louvain_kernel {
    vertex_partitioning_result<task::vertex_partitioning> operator()(
        const detail::descriptor_base<task::vertex_partitioning> &desc,
        const dal::preview::detail::topology<std::int32_t> &t,
        const std::int32_t *p,
        const bool use_p,
        const EdgeValue *vals,
        byte_alloc_iface *alloc_ptr) {
        {
            throw unimplemented(
                dal::detail::error_messages::louvain_algorithm_is_not_implemented());
        }
    }
};

} // namespace oneapi::dal::preview::louvain::backend
