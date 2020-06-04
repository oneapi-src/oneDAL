/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

#include "oneapi/dal/execution_context.hpp"

namespace dal {
namespace backend {

template <typename ... Kernels>
struct kernel_dispatcher {};

class context_cpu {
  public:
    explicit context_cpu(const default_execution_context& ctx)
        : cpu_extensions_(ctx.get_enabled_cpu_extensions()) {}

    cpu_extension get_enabled_cpu_extensions() const {
        return cpu_extensions_;
    }

  private:
    cpu_extension cpu_extensions_;
};

template <typename CpuKernel>
struct kernel_dispatcher<CpuKernel> {
    template <typename ... Args>
    auto operator() (const default_execution_context& ctx, Args&& ...args) const {
        return CpuKernel()(context_cpu{ctx}, std::forward<Args>(args)...);
    }
};

struct cpu_dispatch_default {};
struct cpu_dispatch_avx {};
struct cpu_dispatch_avx2 {};
struct cpu_dispatch_avx512 {};

inline bool test_cpu_extension(cpu_extension mask, cpu_extension test) {
    return ((std::uint64_t)mask & (std::uint64_t)test) > 0;
}

template <typename Op>
constexpr auto dispatch_by_cpu(const context_cpu& ctx, Op&& op) {
    const cpu_extension cpu_ex = ctx.get_enabled_cpu_extensions();
    if (test_cpu_extension(cpu_ex, cpu_extension::avx512)) {
        return op(cpu_dispatch_avx512{});
    }
    else if (test_cpu_extension(cpu_ex, cpu_extension::avx2)) {
        return op(cpu_dispatch_avx2{});
    }
    else if (test_cpu_extension(cpu_ex, cpu_extension::avx)) {
        return op(cpu_dispatch_avx{});
    }
    // TODO: Add more extensions
    return op(cpu_dispatch_default{});
}

}  // namespace backend
}  // namespace dal
