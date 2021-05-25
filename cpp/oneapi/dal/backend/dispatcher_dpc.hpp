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

#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::backend {

#ifdef ONEDAL_DATA_PARALLEL
class context_gpu : public base {
public:
    explicit context_gpu(const detail::data_parallel_policy& policy) : queue_(policy.get_queue()) {}

    context_gpu(const context_gpu&) = delete;
    context_gpu& operator=(const context_gpu&) = delete;

    sycl::queue& get_queue() const {
        return queue_;
    }

private:
    sycl::queue& queue_;
};
#endif

#ifdef ONEDAL_DATA_PARALLEL
template <typename CpuKernel, typename GpuKernel>
struct kernel_dispatcher<CpuKernel, GpuKernel> {
    template <typename... Args>
    auto operator()(const detail::data_parallel_policy& policy, Args&&... args) const {
        const auto device = policy.get_queue().get_device();
        if (device.is_host() || device.is_cpu()) {
            const auto cpu_policy = context_cpu{ detail::host_policy::get_default() };
            return CpuKernel()(cpu_policy, std::forward<Args>(args)...);
        }
        else if (device.is_gpu()) {
            const auto gpu_policy = context_gpu{ policy };
            return GpuKernel()(gpu_policy, std::forward<Args>(args)...);
        }
        else {
            throw unsupported_device(dal::detail::error_messages::unsupported_device_type());
        }
    }
};
#endif

} // namespace oneapi::dal::backend
