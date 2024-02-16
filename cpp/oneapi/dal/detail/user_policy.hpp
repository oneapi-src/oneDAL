/*******************************************************************************
* Copyright 2023 Intel Corporation
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
#include "oneapi/dal/detail/policy.hpp"

#pragma once

namespace oneapi::dal::detail {
class user_cpu_context_impl;

class user_cpu_context {
public:
    user_cpu_context();
    user_cpu_context(const host_policy& policy);
    host_policy get_host_policy();
    void set_host_policy(const host_policy& policy);

private:
    pimpl<user_cpu_context_impl> impl_;
};

} //namespace oneapi::dal::detail
