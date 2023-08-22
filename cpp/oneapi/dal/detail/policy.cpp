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

class host_policy_impl : public base {
public:
    cpu_extension cpu_extensions_mask = backend::detect_top_cpu_extension();
};

host_policy::host_policy() : impl_(new host_policy_impl()) {}

void host_policy::set_enabled_cpu_extensions_impl(const cpu_extension& extensions) noexcept {
    impl_->cpu_extensions_mask = extensions;
}

cpu_extension host_policy::get_enabled_cpu_extensions() const noexcept {
    return impl_->cpu_extensions_mask;
}

#ifdef ONEDAL_DATA_PARALLEL
void data_parallel_policy::init_impl(const sycl::queue& queue) {
    this->impl_ = nullptr; // reserved for future use
}
#endif

} // namespace oneapi::dal::detail
