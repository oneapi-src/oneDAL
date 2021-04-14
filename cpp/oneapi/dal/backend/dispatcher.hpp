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

#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/dispatcher_cpu.hpp"

namespace oneapi::dal::backend {

detail::cpu_extension detect_top_cpu_extension();

struct cpu_dispatch_sse2 {};
struct cpu_dispatch_ssse3 {};
struct cpu_dispatch_sse42 {};
struct cpu_dispatch_avx {};
struct cpu_dispatch_avx2 {};
struct cpu_dispatch_avx512 {};

using cpu_dispatch_default = cpu_dispatch_sse2;

template <typename... Kernels>
struct kernel_dispatcher {};

class context_cpu {
public:
    explicit context_cpu(const detail::host_policy& ctx = detail::host_policy::get_default())
            : cpu_extensions_(ctx.get_enabled_cpu_extensions()) {}

    detail::cpu_extension get_enabled_cpu_extensions() const {
        return cpu_extensions_;
    }

private:
    detail::cpu_extension cpu_extensions_;
};

template <typename CpuKernel>
struct kernel_dispatcher<CpuKernel> {
    template <typename... Args>
    auto operator()(const detail::host_policy& ctx, Args&&... args) const {
        return CpuKernel()(context_cpu{ ctx }, std::forward<Args>(args)...);
    }
};

inline bool test_cpu_extension(detail::cpu_extension mask, detail::cpu_extension test) {
    return ((std::uint64_t)mask & (std::uint64_t)test) > 0;
}

template <typename Op>
inline constexpr auto dispatch_by_cpu(const context_cpu& ctx, Op&& op) {
    using detail::cpu_extension;

    const cpu_extension cpu_ex = ctx.get_enabled_cpu_extensions();
    ONEDAL_IF_CPU_DISPATCH_AVX512(if (test_cpu_extension(cpu_ex, cpu_extension::avx512)) {
        return op(cpu_dispatch_avx512{});
    })
    ONEDAL_IF_CPU_DISPATCH_AVX2(
        if (test_cpu_extension(cpu_ex, cpu_extension::avx2)) { return op(cpu_dispatch_avx2{}); })
    ONEDAL_IF_CPU_DISPATCH_AVX(
        if (test_cpu_extension(cpu_ex, cpu_extension::avx)) { return op(cpu_dispatch_avx{}); })
    ONEDAL_IF_CPU_DISPATCH_SSE42(
        if (test_cpu_extension(cpu_ex, cpu_extension::sse42)) { return op(cpu_dispatch_sse42{}); })
    ONEDAL_IF_CPU_DISPATCH_SSSE3(
        if (test_cpu_extension(cpu_ex, cpu_extension::ssse3)) { return op(cpu_dispatch_ssse3{}); })
    return op(cpu_dispatch_default{});
}

template <typename Op>
inline constexpr auto dispatch_by_data_type(data_type dtype, Op&& op) {
    using msg = dal::detail::error_messages;

    switch (dtype) {
        case data_type::int8: return op(std::int8_t{});
        case data_type::uint8: return op(std::uint8_t{});
        case data_type::int16: return op(std::int16_t{});
        case data_type::uint16: return op(std::uint16_t{});
        case data_type::int32: return op(std::int32_t{});
        case data_type::uint32: return op(std::uint32_t{});
        case data_type::int64: return op(std::int64_t{});
        case data_type::uint64: return op(std::uint64_t{});
        case data_type::float32: return op(float{});
        case data_type::float64: return op(double{});
        default: throw unimplemented{ msg::unsupported_conversion_types() };
    }
}

} // namespace oneapi::dal::backend
