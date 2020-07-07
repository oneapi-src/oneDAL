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

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/policy.hpp"

namespace oneapi::dal {

class detail::data_parallel_policy_impl : public base {
public:
    explicit data_parallel_policy_impl(const sycl::queue& queue)
            : queue(const_cast<sycl::queue&>(queue)) {}

    sycl::queue& queue;
};

using detail::data_parallel_policy_impl;

data_parallel_policy::data_parallel_policy(const sycl::queue& queue)
        : impl_(new data_parallel_policy_impl(queue)) {}

sycl::queue& data_parallel_policy::get_queue() const noexcept {
    return impl_->queue;
}

} // namespace oneapi::dal
