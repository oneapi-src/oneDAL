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

namespace oneapi::dal::detail {
namespace v1 {

struct cpu_dispatch_sve {};
struct cpu_dispatch_sse2 {};
struct cpu_dispatch_sse42 {};
struct cpu_dispatch_avx2 {};
struct cpu_dispatch_avx512 {};

using cpu_dispatch_default = cpu_dispatch_sse2;

} // namespace v1

using v1::cpu_dispatch_sve;
using v1::cpu_dispatch_sse2;
using v1::cpu_dispatch_sse42;
using v1::cpu_dispatch_avx2;
using v1::cpu_dispatch_avx512;
using v1::cpu_dispatch_default;

} // namespace oneapi::dal::detail
