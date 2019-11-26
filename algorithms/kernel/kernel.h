/* file: kernel.h */
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
//  Defines used for kernel allocation, deallocation and calling kernel methods
//--
*/

#ifndef __KERNEL_H__
#define __KERNEL_H__

#include "daal_defines.h"
#include "service_defines.h"
#include "services/daal_kernel_defines.h"
#include "services/internal/gpu_support_checker.h"

#undef __DAAL_INITIALIZE_KERNELS
#define __DAAL_INITIALIZE_KERNELS(KernelClass, ...)    \
    {                                                  \
        _kernel = (new KernelClass<__VA_ARGS__, cpu>); \
    }

#undef __DAAL_INITIALIZE_KERNELS_SYCL
#define __DAAL_INITIALIZE_KERNELS_SYCL(KernelClass, ...) \
    {                                                    \
        _kernel = (new KernelClass<__VA_ARGS__>);        \
    }

#undef __DAAL_DEINITIALIZE_KERNELS
#define __DAAL_DEINITIALIZE_KERNELS() \
    {                                 \
        if (_kernel) delete _kernel;  \
    }

#undef __DAAL_KERNEL_ARGUMENTS
#define __DAAL_KERNEL_ARGUMENTS(...) __VA_ARGS__

#undef __DAAL_CALL_KERNEL
#define __DAAL_CALL_KERNEL(env, KernelClass, templateArguments, method, ...)            \
    {                                                                                   \
        return ((KernelClass<templateArguments, cpu> *)(_kernel))->method(__VA_ARGS__); \
    }

#undef __DAAL_CALL_KERNEL_SYCL
#define __DAAL_CALL_KERNEL_SYCL(env, KernelClass, templateArguments, method, ...)  \
    {                                                                              \
        return ((KernelClass<templateArguments> *)(_kernel))->method(__VA_ARGS__); \
    }

#undef __DAAL_CALL_KERNEL_STATUS
#define __DAAL_CALL_KERNEL_STATUS(env, KernelClass, templateArguments, method, ...) \
    ((KernelClass<templateArguments, cpu> *)(_kernel))->method(__VA_ARGS__);

#define __DAAL_INSTANTIATE_DISPATCH_IMPL(ContainerTemplate, Mode, ClassName, BaseClassName, ...)                                                    \
    DAAL_KERNEL_SSE2_CONTAINER1(ContainerTemplate, __VA_ARGS__)                                                                                     \
    DAAL_KERNEL_SSSE3_CONTAINER1(ContainerTemplate, __VA_ARGS__)                                                                                    \
    DAAL_KERNEL_SSE42_CONTAINER1(ContainerTemplate, __VA_ARGS__)                                                                                    \
    DAAL_KERNEL_AVX_CONTAINER1(ContainerTemplate, __VA_ARGS__)                                                                                      \
    DAAL_KERNEL_AVX2_CONTAINER1(ContainerTemplate, __VA_ARGS__)                                                                                     \
    DAAL_KERNEL_AVX512_MIC_CONTAINER1(ContainerTemplate, __VA_ARGS__)                                                                               \
    DAAL_KERNEL_AVX512_CONTAINER1(ContainerTemplate, __VA_ARGS__)                                                                                   \
    namespace interface1                                                                                                                            \
    {                                                                                                                                               \
    template <>                                                                                                                                     \
    ClassName<Mode, ContainerTemplate<__VA_ARGS__, sse2> DAAL_KERNEL_SSSE3_CONTAINER(ContainerTemplate, __VA_ARGS__) DAAL_KERNEL_SSE42_CONTAINER(   \
                        ContainerTemplate, __VA_ARGS__) DAAL_KERNEL_AVX_CONTAINER(ContainerTemplate, __VA_ARGS__)                                   \
                        DAAL_KERNEL_AVX2_CONTAINER(ContainerTemplate, __VA_ARGS__) DAAL_KERNEL_AVX512_MIC_CONTAINER(ContainerTemplate, __VA_ARGS__) \
                            DAAL_KERNEL_AVX512_CONTAINER(ContainerTemplate, __VA_ARGS__)>::ClassName(daal::services::Environment::env * daalEnv)    \
        : BaseClassName(daalEnv), _cntr(nullptr)                                                                                                    \
    {                                                                                                                                               \
        switch (daalEnv->cpuid)                                                                                                                     \
        {                                                                                                                                           \
            DAAL_KERNEL_SSSE3_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                                                                        \
            DAAL_KERNEL_SSE42_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                                                                        \
            DAAL_KERNEL_AVX_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                                                                          \
            DAAL_KERNEL_AVX2_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                                                                         \
            DAAL_KERNEL_AVX512_MIC_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                                                                   \
            DAAL_KERNEL_AVX512_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                                                                       \
        default: _cntr = (new ContainerTemplate<__VA_ARGS__, sse2>(daalEnv)); break;                                                                \
        }                                                                                                                                           \
    }                                                                                                                                               \
                                                                                                                                                    \
    template class ClassName<Mode, ContainerTemplate<__VA_ARGS__, sse2> DAAL_KERNEL_SSSE3_CONTAINER(ContainerTemplate, __VA_ARGS__)                 \
                                       DAAL_KERNEL_SSE42_CONTAINER(ContainerTemplate, __VA_ARGS__) DAAL_KERNEL_AVX_CONTAINER(                       \
                                           ContainerTemplate, __VA_ARGS__) DAAL_KERNEL_AVX2_CONTAINER(ContainerTemplate, __VA_ARGS__)               \
                                           DAAL_KERNEL_AVX512_MIC_CONTAINER(ContainerTemplate, __VA_ARGS__)                                         \
                                               DAAL_KERNEL_AVX512_CONTAINER(ContainerTemplate, __VA_ARGS__)>;                                       \
    }

#define __DAAL_INSTANTIATE_DISPATCH_IMPL_OLD(ContainerTemplate, Mode, ClassName, BaseClassName, ...)                                                \
    namespace interface1                                                                                                                            \
    {                                                                                                                                               \
    template <>                                                                                                                                     \
    ClassName<Mode, ContainerTemplate<__VA_ARGS__, sse2> DAAL_KERNEL_SSSE3_CONTAINER(ContainerTemplate, __VA_ARGS__) DAAL_KERNEL_SSE42_CONTAINER(   \
                        ContainerTemplate, __VA_ARGS__) DAAL_KERNEL_AVX_CONTAINER(ContainerTemplate, __VA_ARGS__)                                   \
                        DAAL_KERNEL_AVX2_CONTAINER(ContainerTemplate, __VA_ARGS__) DAAL_KERNEL_AVX512_MIC_CONTAINER(ContainerTemplate, __VA_ARGS__) \
                            DAAL_KERNEL_AVX512_CONTAINER(ContainerTemplate, __VA_ARGS__)>::ClassName(daal::services::Environment::env * daalEnv)    \
        : BaseClassName(daalEnv), _cntr(nullptr)                                                                                                    \
    {                                                                                                                                               \
        switch (daalEnv->cpuid)                                                                                                                     \
        {                                                                                                                                           \
            DAAL_KERNEL_SSSE3_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                                                                        \
            DAAL_KERNEL_SSE42_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                                                                        \
            DAAL_KERNEL_AVX_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                                                                          \
            DAAL_KERNEL_AVX2_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                                                                         \
            DAAL_KERNEL_AVX512_MIC_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                                                                   \
            DAAL_KERNEL_AVX512_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                                                                       \
        default: _cntr = (new ContainerTemplate<__VA_ARGS__, sse2>(daalEnv)); break;                                                                \
        }                                                                                                                                           \
    }                                                                                                                                               \
                                                                                                                                                    \
    template class ClassName<Mode, ContainerTemplate<__VA_ARGS__, sse2> DAAL_KERNEL_SSSE3_CONTAINER(ContainerTemplate, __VA_ARGS__)                 \
                                       DAAL_KERNEL_SSE42_CONTAINER(ContainerTemplate, __VA_ARGS__) DAAL_KERNEL_AVX_CONTAINER(                       \
                                           ContainerTemplate, __VA_ARGS__) DAAL_KERNEL_AVX2_CONTAINER(ContainerTemplate, __VA_ARGS__)               \
                                           DAAL_KERNEL_AVX512_MIC_CONTAINER(ContainerTemplate, __VA_ARGS__)                                         \
                                               DAAL_KERNEL_AVX512_CONTAINER(ContainerTemplate, __VA_ARGS__)>;                                       \
    }

#define __DAAL_INSTANTIATE_DISPATCH_LAYER_CONTAINER(ContainerTemplate, ...) \
    __DAAL_INSTANTIATE_DISPATCH_IMPL(ContainerTemplate, batch, AlgorithmDispatchLayerContainer, LayerContainerIfaceImpl, __VA_ARGS__)

#define __DAAL_INSTANTIATE_DISPATCH_LAYER_CONTAINER_FORWARD(ContainerTemplate, ...) \
    __DAAL_INSTANTIATE_DISPATCH_IMPL_OLD(ContainerTemplate, batch, AlgorithmDispatchLayerContainer, LayerContainerIfaceImpl, __VA_ARGS__)

#define __DAAL_INSTANTIATE_DISPATCH_CONTAINER(ContainerTemplate, Mode, ...) \
    __DAAL_INSTANTIATE_DISPATCH_IMPL(ContainerTemplate, Mode, AlgorithmDispatchContainer, AlgorithmContainerImpl<Mode>, __VA_ARGS__)

#define __DAAL_INSTANTIATE_DISPATCH_CONTAINER_KM(ContainerTemplate, Mode, ...) \
    __DAAL_INSTANTIATE_DISPATCH_IMPL_OLD(ContainerTemplate, Mode, AlgorithmDispatchContainer, AlgorithmContainerImpl<Mode>, __VA_ARGS__)

#define __DAAL_INSTANTIATE_DISPATCH_SYCL_IMPL(ContainerTemplate, Mode, ClassName, BaseClassName, ...)                                               \
    DAAL_KERNEL_SSE2_CONTAINER1(ContainerTemplate, __VA_ARGS__)                                                                                     \
    DAAL_KERNEL_SSSE3_CONTAINER1(ContainerTemplate, __VA_ARGS__)                                                                                    \
    DAAL_KERNEL_SSE42_CONTAINER1(ContainerTemplate, __VA_ARGS__)                                                                                    \
    DAAL_KERNEL_AVX_CONTAINER1(ContainerTemplate, __VA_ARGS__)                                                                                      \
    DAAL_KERNEL_AVX2_CONTAINER1(ContainerTemplate, __VA_ARGS__)                                                                                     \
    DAAL_KERNEL_AVX512_MIC_CONTAINER1(ContainerTemplate, __VA_ARGS__)                                                                               \
    DAAL_KERNEL_AVX512_CONTAINER1(ContainerTemplate, __VA_ARGS__)                                                                                   \
    namespace interface1                                                                                                                            \
    {                                                                                                                                               \
    template <>                                                                                                                                     \
    ClassName<Mode, ContainerTemplate<__VA_ARGS__, sse2> DAAL_KERNEL_SSSE3_CONTAINER(ContainerTemplate, __VA_ARGS__) DAAL_KERNEL_SSE42_CONTAINER(   \
                        ContainerTemplate, __VA_ARGS__) DAAL_KERNEL_AVX_CONTAINER(ContainerTemplate, __VA_ARGS__)                                   \
                        DAAL_KERNEL_AVX2_CONTAINER(ContainerTemplate, __VA_ARGS__) DAAL_KERNEL_AVX512_MIC_CONTAINER(ContainerTemplate, __VA_ARGS__) \
                            DAAL_KERNEL_AVX512_CONTAINER(ContainerTemplate, __VA_ARGS__)>::ClassName(daal::services::Environment::env * daalEnv)    \
        : BaseClassName(daalEnv), _cntr(NULL)                                                                                                       \
    {                                                                                                                                               \
        switch (daalEnv->cpuid)                                                                                                                     \
        {                                                                                                                                           \
            DAAL_KERNEL_SSSE3_CONTAINER_CASE_SYCL(ContainerTemplate, __VA_ARGS__)                                                                   \
            DAAL_KERNEL_SSE42_CONTAINER_CASE_SYCL(ContainerTemplate, __VA_ARGS__)                                                                   \
            DAAL_KERNEL_AVX_CONTAINER_CASE_SYCL(ContainerTemplate, __VA_ARGS__)                                                                     \
            DAAL_KERNEL_AVX2_CONTAINER_CASE_SYCL(ContainerTemplate, __VA_ARGS__)                                                                    \
            DAAL_KERNEL_AVX512_MIC_CONTAINER_CASE_SYCL(ContainerTemplate, __VA_ARGS__)                                                              \
            DAAL_KERNEL_AVX512_CONTAINER_CASE_SYCL(ContainerTemplate, __VA_ARGS__)                                                                  \
        default:                                                                                                                                    \
        {                                                                                                                                           \
            using cntrTemplateInst = ContainerTemplate<__VA_ARGS__, sse2>;                                                                          \
            static volatile services::internal::GpuSupportRegistrar<cntrTemplateInst> registrar;                                                    \
            _cntr = (new cntrTemplateInst(daalEnv));                                                                                                \
            break;                                                                                                                                  \
        }                                                                                                                                           \
        }                                                                                                                                           \
    }                                                                                                                                               \
                                                                                                                                                    \
    template class ClassName<Mode, ContainerTemplate<__VA_ARGS__, sse2> DAAL_KERNEL_SSSE3_CONTAINER(ContainerTemplate, __VA_ARGS__)                 \
                                       DAAL_KERNEL_SSE42_CONTAINER(ContainerTemplate, __VA_ARGS__) DAAL_KERNEL_AVX_CONTAINER(                       \
                                           ContainerTemplate, __VA_ARGS__) DAAL_KERNEL_AVX2_CONTAINER(ContainerTemplate, __VA_ARGS__)               \
                                           DAAL_KERNEL_AVX512_MIC_CONTAINER(ContainerTemplate, __VA_ARGS__)                                         \
                                               DAAL_KERNEL_AVX512_CONTAINER(ContainerTemplate, __VA_ARGS__)>;                                       \
    }

#define __DAAL_INSTANTIATE_DISPATCH_CONTAINER_SYCL(ContainerTemplate, Mode, ...) \
    __DAAL_INSTANTIATE_DISPATCH_SYCL_IMPL(ContainerTemplate, Mode, AlgorithmDispatchContainer, AlgorithmContainerImpl<Mode>, __VA_ARGS__)

#endif
