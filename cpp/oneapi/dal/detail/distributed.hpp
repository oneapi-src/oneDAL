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

#include "oneapi/dal/detail/policy.hpp"

namespace oneapi::dal::detail {
namespace v1 {

template <typename Policy>
inline auto is_root_rank(const Policy& policy)
    -> std::enable_if_t<is_local_policy_v<Policy>, bool> {
    return true;
}

template <typename Policy>
inline auto is_root_rank(const Policy& policy)
    -> std::enable_if_t<is_distributed_policy_v<Policy>, bool> {
    const auto comm = policy.get_communicator();
    return comm.get_rank() == comm.get_root_rank();
}

} // namespace v1

using v1::is_root_rank;

} // namespace oneapi::dal::detail
