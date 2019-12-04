/* file: service_dispatch.h */
/*******************************************************************************
* Copyright 2018-2019 Intel Corporation
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

/*
//++
//  Universal dispatcher of abstract function with cpu parameter.
//--
*/

#ifndef __SERVICE_DISPATCH_H__
#define __SERVICE_DISPATCH_H__

#include "daal_kernel_defines.h"

#define DAAL_DISPATCH_FUNCTION_BY_CPU(func, ...) \
    switch (static_cast<daal::CpuType>(daal::services::Environment::getInstance()->getCpuId())) { \
        DAAL_KERNEL_SSSE3_ONLY_CODE      ( case daal::CpuType::ssse3:      func(daal::CpuType::ssse3,      __VA_ARGS__); break; ) \
        DAAL_KERNEL_SSE42_ONLY_CODE      ( case daal::CpuType::sse42:      func(daal::CpuType::sse42,      __VA_ARGS__); break; ) \
        DAAL_KERNEL_AVX_ONLY_CODE        ( case daal::CpuType::avx:        func(daal::CpuType::avx,        __VA_ARGS__); break; ) \
        DAAL_KERNEL_AVX2_ONLY_CODE       ( case daal::CpuType::avx2:       func(daal::CpuType::avx2,       __VA_ARGS__); break; ) \
        DAAL_KERNEL_AVX512_ONLY_CODE     ( case daal::CpuType::avx512:     func(daal::CpuType::avx512,     __VA_ARGS__); break; ) \
        DAAL_KERNEL_AVX512_MIC_ONLY_CODE ( case daal::CpuType::avx512_mic: func(daal::CpuType::avx512_mic, __VA_ARGS__); break; ) \
        DAAL_EXPAND                      ( default: func(daal::CpuType::sse2, __VA_ARGS__); break; )                              \
    }

#endif
