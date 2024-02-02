/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "oneapi/dal/cpu.hpp"
#include <daal/src/services/service_defines.h>

namespace oneapi::dal {
namespace v1 {

inline constexpr cpu_extension from_daal_cpu_type(daal::CpuType cpu) {
    switch (cpu) {
        case daal::sse2: return cpu_extension::sse2;
        case daal::sse42: return cpu_extension::sse42;
        case daal::avx2: return cpu_extension::avx2;
        case daal::avx512: return cpu_extension::avx512;
    }
    return cpu_extension::none;
}
cpu_extension detect_top_cpu_extension() {
    if (!__daal_serv_cpu_extensions_available()) {
        return cpu_extension::sse2;
    }
    const auto daal_cpu = (daal::CpuType)__daal_serv_cpu_detect(0);

    return from_daal_cpu_type(daal_cpu);
}

} // namespace v1
} // namespace oneapi::dal
