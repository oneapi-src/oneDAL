/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::detail {
namespace v1 {

class host_policy_impl : public base {
public:
    cpu_extension cpu_extensions_mask = backend::detect_top_cpu_extension();
    threading_policy threading_parameters;

    host_policy_impl() {
        threading_parameters = threading_policy();
    }
};

host_policy::host_policy() : impl_(new host_policy_impl()) {}

const auto default_impl = std::make_shared<host_policy_impl>();

auto host_policy::make_default_impl() -> std::shared_ptr<host_policy_impl> {
    return default_impl;
}

void host_policy::set_enabled_cpu_extensions_impl(const cpu_extension& extensions) noexcept {
    impl_->cpu_extensions_mask = extensions;
}

cpu_extension host_policy::get_enabled_cpu_extensions() const noexcept {
    return impl_->cpu_extensions_mask;
}

threading_policy host_policy::get_threading_policy() const noexcept {
    return impl_->threading_parameters;
}

void host_policy::set_threading_policy(const threading_policy& policy) noexcept {
    impl_->threading_parameters = policy;
}

#ifdef ONEDAL_DATA_PARALLEL
class data_parallel_policy_impl : public base {
public:
    threading_policy threading_parameters;

    data_parallel_policy_impl() {
        threading_parameters = threading_policy();
    }
};

threading_policy data_parallel_policy::get_threading_policy() const noexcept {
    return impl_->threading_parameters;
}

void data_parallel_policy::set_threading_policy(const threading_policy& policy) noexcept {
    impl_->threading_parameters = policy;
}

void data_parallel_policy::init_impl(const sycl::queue& queue) {
    this->impl_.reset(new data_parallel_policy_impl);
}
#endif

} // namespace v1
} // namespace oneapi::dal::detail
