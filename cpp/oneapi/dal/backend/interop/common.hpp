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

#include "daal/include/services/env_detect.h"

#include "oneapi/dal/backend/dispatcher.hpp"

namespace dal {
namespace backend {
namespace interop {

template <typename DispatchId>
constexpr daal::CpuType get_daal_cpu_type(DispatchId);

template <>
constexpr daal::CpuType get_daal_cpu_type<cpu_dispatch_default>(cpu_dispatch_default) {
    return daal::CpuType::sse2;
}

template <>
constexpr daal::CpuType get_daal_cpu_type<cpu_dispatch_avx>(cpu_dispatch_avx) {
    return daal::CpuType::avx;
}

template <>
constexpr daal::CpuType get_daal_cpu_type<cpu_dispatch_avx2>(cpu_dispatch_avx2) {
    return daal::CpuType::avx2;
}

template <>
constexpr daal::CpuType get_daal_cpu_type<cpu_dispatch_avx512>(cpu_dispatch_avx512) {
    return daal::CpuType::avx512;
}

template <typename Float, template <typename, daal::CpuType> typename CpuKernel, typename... Args>
inline auto call_daal_kernel(const context_cpu& ctx, Args&&... args) {
    return dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
        constexpr daal::CpuType daal_cpu_type = get_daal_cpu_type(cpu);
        return CpuKernel<Float, daal_cpu_type>().compute(args...);
    });
}

} // namespace interop
} // namespace backend
} // namespace dal
