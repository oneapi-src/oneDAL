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

#include "oneapi/dal/policy/common.hpp"
#include "oneapi/dal/detail/policy.hpp"

namespace oneapi::dal::preview {

class host_policy : public base {
    friend dal::detail::internal_policy_accessor;

public:
    host_policy() = default;

private:
    const dal::detail::host_policy &get_internal_policy() const {
        return internal_policy_;
    }

    dal::detail::host_policy internal_policy_;
};

template <>
struct is_execution_policy<host_policy> : std::bool_constant<true> {};

template <>
struct is_host_policy<host_policy> : std::bool_constant<true> {};

template <>
struct is_local_policy<host_policy> : std::bool_constant<true> {};

} // namespace oneapi::dal::preview
