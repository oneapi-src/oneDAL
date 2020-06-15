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

// This file shall not be included to any other file directly. It is used as a
// template to produce `_onedal_dispatcher_cpu.hpp` during the build.

#pragma once

#define ONEDAL_CPU_DISPATCH_SSSE3
#define ONEDAL_CPU_DISPATCH_SSE42
#define ONEDAL_CPU_DISPATCH_AVX
#define ONEDAL_CPU_DISPATCH_AVX2
#define ONEDAL_CPU_DISPATCH_AVX512

#ifdef ONEDAL_CPU_DISPATCH_SSSE3
    #define ONEDAL_IF_CPU_DISPATCH_SSSE3(x) x
#else
    #define ONEDAL_IF_CPU_DISPATCH_SSSE3(x)
#endif

#ifdef ONEDAL_CPU_DISPATCH_SSE42
    #define ONEDAL_IF_CPU_DISPATCH_SSE42(x) x
#else
    #define ONEDAL_IF_CPU_DISPATCH_SSE42(x)
#endif

#ifdef ONEDAL_CPU_DISPATCH_AVX
    #define ONEDAL_IF_CPU_DISPATCH_AVX(x) x
#else
    #define ONEDAL_IF_CPU_DISPATCH_AVX(x)
#endif

#ifdef ONEDAL_CPU_DISPATCH_AVX2
    #define ONEDAL_IF_CPU_DISPATCH_AVX2(x) x
#else
    #define ONEDAL_IF_CPU_DISPATCH_AVX2(x)
#endif

#ifdef ONEDAL_CPU_DISPATCH_AVX512
    #define ONEDAL_IF_CPU_DISPATCH_AVX512(x) x
#else
    #define ONEDAL_IF_CPU_DISPATCH_AVX512(x)
#endif
