/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include <daal/src/services/service_defines.h>

#include <cstdint>

// TODO: Clean up this redefinition and import the defines globally.
#if defined(__x86_64__) || defined(__x86_64) || defined(__amd64) || defined(_M_AMD64)
#define TARGET_X86_64
#endif

#if defined(__ARM_ARCH) || defined(__aarch64__)
#define TARGET_ARM
#endif

namespace oneapi::dal::detail {
namespace v1 {

enum class cpu_vendor { unknown = 0, intel = 1, amd = 2, arm = 3 };

enum class cpu_extension : uint64_t {
    none = 0U,
#if defined(TARGET_X86_64)
    sse2 = 1U << 0,
    sse42 = 1U << 2,
    avx2 = 1U << 4,
    avx512 = 1U << 5
#elif defined(TARGET_ARM)
    sve = 1U << 0
#endif
};

inline constexpr cpu_extension from_daal_cpu_type(daal::CpuType cpu) {
    switch (cpu) {
#if defined(TARGET_X86_64)
        case daal::sse2: return cpu_extension::sse2;
        case daal::sse42: return cpu_extension::sse42;
        case daal::avx2: return cpu_extension::avx2;
        case daal::avx512: return cpu_extension::avx512;
#elif defined(TARGET_ARM)
        case daal::sve: return cpu_extension::sve;
#endif
    }
    return cpu_extension::none;
}

cpu_extension detect_top_cpu_extension();

} // namespace v1
using v1::cpu_vendor;
using v1::cpu_extension;
using v1::detect_top_cpu_extension;
} // namespace oneapi::dal::detail
