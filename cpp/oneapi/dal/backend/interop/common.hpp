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

#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::backend::interop {

template <typename DispatchId>
struct to_daal_cpu_type;

template <daal::CpuType cpu>
struct daal_cpu_value {
    constexpr static daal::CpuType value = cpu;
};

template <>
struct to_daal_cpu_type<cpu_dispatch_default> : daal_cpu_value<daal::sse2> {};
template <>
struct to_daal_cpu_type<cpu_dispatch_sse42> : daal_cpu_value<daal::sse42> {};
template <>
struct to_daal_cpu_type<cpu_dispatch_avx2> : daal_cpu_value<daal::avx2> {};
template <>
struct to_daal_cpu_type<cpu_dispatch_avx512> : daal_cpu_value<daal::avx512> {};

template <typename Float, template <typename, daal::CpuType> typename CpuKernel, typename... Args>
inline auto call_daal_kernel(const context_cpu& ctx, Args&&... args) {
    return dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
        return CpuKernel<Float, to_daal_cpu_type<decltype(cpu)>::value>().compute(
            std::forward<Args>(args)...);
    });
}

template <typename Float, template <typename, daal::CpuType> typename CpuKernel, typename... Args>
inline auto call_daal_kernel_finalize_merge(const context_cpu& ctx, Args&&... args) {
    return dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
        return CpuKernel<Float, to_daal_cpu_type<decltype(cpu)>::value>().finalizeMerge(
            std::forward<Args>(args)...);
    });
}

template <typename Float, template <typename, daal::CpuType> typename CpuKernel, typename... Args>
inline auto call_daal_kernel_finalize_compute(const context_cpu& ctx, Args&&... args) {
    return dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
        return CpuKernel<Float, to_daal_cpu_type<decltype(cpu)>::value>().finalizeCompute(
            std::forward<Args>(args)...);
    });
}

template <typename Float, template <typename, daal::CpuType> typename CpuKernel, typename... Args>
inline auto call_daal_infer_kernel(const context_cpu& ctx, Args&&... args) {
    return dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
        return CpuKernel<Float, to_daal_cpu_type<decltype(cpu)>::value>().infer(
            std::forward<Args>(args)...);
    });
}

} // namespace oneapi::dal::backend::interop
