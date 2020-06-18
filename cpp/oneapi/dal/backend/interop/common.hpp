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

#pragma once

#include <daal/include/services/env_detect.h>
#include <daal/src/services/service_defines.h>

#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal {
namespace backend {
namespace interop {

template <typename DispatchId>
constexpr daal::CpuType to_daal_cpu_type(DispatchId);

template <>
constexpr daal::CpuType to_daal_cpu_type<cpu_dispatch_default>(cpu_dispatch_default) {
    return daal::CpuType::sse2;
}

template <>
constexpr daal::CpuType to_daal_cpu_type<cpu_dispatch_ssse3>(cpu_dispatch_ssse3) {
    return daal::CpuType::ssse3;
}

template <>
constexpr daal::CpuType to_daal_cpu_type<cpu_dispatch_sse42>(cpu_dispatch_sse42) {
    return daal::CpuType::sse42;
}

template <>
constexpr daal::CpuType to_daal_cpu_type<cpu_dispatch_avx>(cpu_dispatch_avx) {
    return daal::CpuType::avx;
}

template <>
constexpr daal::CpuType to_daal_cpu_type<cpu_dispatch_avx2>(cpu_dispatch_avx2) {
    return daal::CpuType::avx2;
}

template <>
constexpr daal::CpuType to_daal_cpu_type<cpu_dispatch_avx512>(cpu_dispatch_avx512) {
    return daal::CpuType::avx512;
}

inline constexpr cpu_extension from_daal_cpu_type(daal::CpuType cpu) {
    switch (cpu) {
        case daal::sse2:   return cpu_extension::sse2;
        case daal::ssse3:  return cpu_extension::ssse3;
        case daal::sse42:  return cpu_extension::sse42;
        case daal::avx:    return cpu_extension::avx;
        case daal::avx2:   return cpu_extension::avx2;
        case daal::avx512: return cpu_extension::avx512;
        case daal::avx512_mic:
        case daal::avx512_mic_e1:
            break;
    }
    return cpu_extension::none;
}

inline cpu_extension detect_top_cpu_extension() {
    const auto daal_cpu = (daal::CpuType)__daal_serv_cpu_detect(0);
    return from_daal_cpu_type(daal_cpu);
}

template <typename Float, template <typename, daal::CpuType> typename CpuKernel, typename... Args>
inline auto call_daal_kernel(const context_cpu& ctx, Args&&... args) {
    return dal::backend::dispatch_by_cpu(ctx, [&](const auto cpu) {
        constexpr daal::CpuType daal_cpu_type = to_daal_cpu_type(cpu);
        return CpuKernel<Float, daal_cpu_type>().compute(args...);
    });
}

} // namespace interop
} // namespace backend
} // namespace oneapi::dal
