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

#pragma once

#include <cstdint>

namespace oneapi::dal::detail {
namespace v1 {

enum class cpu_extension : uint64_t {
    none = 0U,
    sse2 = 1U << 0,
    sse42 = 1U << 2,
    avx2 = 1U << 4,
    avx512 = 1U << 5
};

cpu_extension detect_top_cpu_extension();

} // namespace v1
using v1::cpu_extension;
using v1::detect_top_cpu_extension;
} // namespace oneapi::dal
