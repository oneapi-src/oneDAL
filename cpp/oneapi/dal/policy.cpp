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

#include "oneapi/dal/policy.hpp"
#include "oneapi/dal/backend/interop/common.hpp"

namespace oneapi::dal {

class detail::host_policy_impl : public base {
public:
    cpu_extension cpu_extensions_mask = backend::interop::detect_top_cpu_extension();
};
using detail::host_policy_impl;

host_policy::host_policy() : impl_(new host_policy_impl()) {}

void host_policy::set_enabled_cpu_extensions_impl(const cpu_extension& extensions) noexcept {
    impl_->cpu_extensions_mask = extensions;
}

cpu_extension host_policy::get_enabled_cpu_extensions() const noexcept {
    return impl_->cpu_extensions_mask;
}

#ifdef ONEAPI_DAL_DATA_PARALLEL
class detail::data_parallel_policy_impl : public base {
public:
    explicit data_parallel_policy_impl(const sycl::queue& queue) : queue(queue) {}

    sycl::queue queue;
};
using detail::data_parallel_policy_impl;

data_parallel_policy::data_parallel_policy(const sycl::queue& queue)
        : impl_(new data_parallel_policy_impl(queue)) {}

sycl::queue& data_parallel_policy::get_queue() const noexcept {
    return impl_->queue;
}
#endif

} // namespace oneapi::dal
