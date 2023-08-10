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

    host_policy_impl(bool thread_pinning = false, int max_threads_per_core = 0) {
        threading_parameters = threading_policy(thread_pinning, max_threads_per_core);
    }
};

host_policy::host_policy(bool thread_pinning, int max_threads_per_core) : 
    impl_(new host_policy_impl(thread_pinning, max_threads_per_core)) {}

void host_policy::set_enabled_cpu_extensions_impl(const cpu_extension& extensions) noexcept {
    impl_->cpu_extensions_mask = extensions;
}

void host_policy::set_thread_pinning_impl(const bool& thread_pinning) noexcept {
    impl_->threading_parameters.thread_pinning = thread_pinning;
}

void host_policy::set_max_threads_per_core_impl(const int& max_threads_per_core) noexcept {
    impl_->threading_parameters.max_threads_per_core = max_threads_per_core;
}

cpu_extension host_policy::get_enabled_cpu_extensions() const noexcept {
    return impl_->cpu_extensions_mask;
}

bool host_policy::get_thread_pinning() const noexcept {
    return impl_->threading_parameters.thread_pinning;
}

int host_policy::get_max_threads_per_core() const noexcept {
    return impl_->threading_parameters.max_threads_per_core;
}

threading_policy host_policy::get_threading_policy() const noexcept {
    return impl_->threading_parameters;
}


#ifdef ONEDAL_DATA_PARALLEL
void data_parallel_policy::init_impl(const sycl::queue& queue) {
    this->impl_ = nullptr; // reserved for future use
}
#endif

} // namespace v1
} // namespace oneapi::dal::detail
