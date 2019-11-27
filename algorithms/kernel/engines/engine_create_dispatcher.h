/* file: engine_create_dispatcher.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of dispatching of enginees.
//--
*/
#ifndef __ENGINE_CREATE_DISPATCHER_BATCH__
#define __ENGINE_CREATE_DISPATCHER_BATCH__

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace interface1
{

#define DISPATCH_RESET_ENGINE(engPtr, cpuId, algorithmFPType, method, ...)                                                                         \
    switch (cpuId) {                                                                                                                           \
    DAAL_KERNEL_SSSE3_ONLY_CODE(engPtr.reset(new BatchImpl<daal::CpuType::ssse3, algorithmFPType, method>(__VA_ARGS__)); break;)               \
    DAAL_KERNEL_SSE42_ONLY_CODE(engPtr.reset(new BatchImpl<daal::CpuType::sse42, algorithmFPType, method>(__VA_ARGS__)); break;)               \
    DAAL_KERNEL_AVX_ONLY_CODE(engPtr.reset(new BatchImpl<daal::CpuType::avx, algorithmFPType, method>(__VA_ARGS__)); break;)                   \
    DAAL_KERNEL_AVX2_ONLY_CODE(engPtr.reset(new BatchImpl<daal::CpuType::avx2, algorithmFPType, method>(__VA_ARGS__)); break;)                 \
    DAAL_KERNEL_AVX512_ONLY_CODE(engPtr.reset(new BatchImpl<daal::CpuType::avx512, algorithmFPType, method>(__VA_ARGS__)); break;)             \
    DAAL_KERNEL_AVX512_MIC_ONLY_CODE(engPtr.reset(new BatchImpl<daal::CpuType::avx512_mic_e1, algorithmFPType, method>(__VA_ARGS__)); break;)  \
    default: engPtr.reset(new BatchImpl<daal::CpuType::sse2, algorithmFPType, method>(__VA_ARGS__)); break;                                    \
    }

} // namespace interface1
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif
