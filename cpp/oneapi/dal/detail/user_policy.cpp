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
#include "oneapi/dal/detail/user_policy.hpp"

namespace oneapi::dal::detail {
class user_cpu_context_impl {
public:
    user_cpu_context_impl() : policy_(host_policy::get_default()) {}
    user_cpu_context_impl(const host_policy& policy) : policy_(policy) {}
    void set_host_policy(const host_policy& policy) {
        policy_ = policy;
    }
    host_policy get_host_policy() {
        return policy_;
    }

private:
    detail::host_policy policy_;
};

user_cpu_context::user_cpu_context() : impl_(new user_cpu_context_impl()) {}

user_cpu_context::user_cpu_context(const host_policy& policy)
        : impl_(new user_cpu_context_impl(policy)) {}

void user_cpu_context::set_host_policy(const host_policy& policy) {
    impl_->set_host_policy(policy);
}

host_policy user_cpu_context::get_host_policy() {
    return impl_->get_host_policy();
}

} // namespace oneapi::dal::detail
