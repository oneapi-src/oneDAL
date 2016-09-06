/* file: kernel.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

#undef __DAAL_INITIALIZE_KERNELS
#define __DAAL_INITIALIZE_KERNELS(KernelClass, ...)        \
    {                                                     \
        _kernel = (new KernelClass<__VA_ARGS__, cpu>);    \
    }

#undef __DAAL_DEINITIALIZE_KERNELS
#define __DAAL_DEINITIALIZE_KERNELS()    \
    {                                   \
        if(_kernel) delete _kernel;     \
    }

#undef __DAAL_KERNEL_ARGUMENTS
#define __DAAL_KERNEL_ARGUMENTS(...) __VA_ARGS__

#undef __DAAL_CALL_KERNEL
#define __DAAL_CALL_KERNEL(env, KernelClass, templateArguments, method, ...)            \
    {                                                                                   \
        ((KernelClass<templateArguments, cpu> *)(_kernel))->method(__VA_ARGS__);        \
    }

#define __DAAL_INSTANTIATE_DISPATCH_CONTAINER(ContainerTemplate, Mode, ...)                                     \
template<>                                                                                                      \
    AlgorithmDispatchContainer< Mode, \
    ContainerTemplate<__VA_ARGS__, sse2>                      \
    DAAL_KERNEL_SSSE3_CONTAINER(ContainerTemplate, __VA_ARGS__) \
    DAAL_KERNEL_SSE42_CONTAINER(ContainerTemplate, __VA_ARGS__) \
    DAAL_KERNEL_AVX_CONTAINER(ContainerTemplate, __VA_ARGS__) \
    DAAL_KERNEL_AVX2_CONTAINER(ContainerTemplate, __VA_ARGS__) \
    DAAL_KERNEL_AVX512_mic_CONTAINER(ContainerTemplate, __VA_ARGS__) \
    DAAL_KERNEL_AVX512_CONTAINER(ContainerTemplate, __VA_ARGS__) >::\
    AlgorithmDispatchContainer(daal::services::Environment::env *daalEnv) : AlgorithmContainerIface<Mode>(daalEnv), _cntr(NULL)  \
{                                                                                                               \
    switch (daalEnv->cpuid)                                                                                     \
    {                                                                                                           \
        DAAL_KERNEL_SSSE3_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__) \
        DAAL_KERNEL_SSE42_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__) \
        DAAL_KERNEL_AVX_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__) \
        DAAL_KERNEL_AVX2_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__) \
        DAAL_KERNEL_AVX512_mic_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__) \
        DAAL_KERNEL_AVX512_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__) \
        default: _cntr = (new ContainerTemplate<__VA_ARGS__, sse2>      (daalEnv)); break;              \
    }                                                                                                           \
}                                                                                                               \
                                                                                                                \
template                                                                                                        \
class     AlgorithmDispatchContainer< Mode, \
    ContainerTemplate<__VA_ARGS__, sse2>                      \
    DAAL_KERNEL_SSSE3_CONTAINER(ContainerTemplate, __VA_ARGS__) \
    DAAL_KERNEL_SSE42_CONTAINER(ContainerTemplate, __VA_ARGS__) \
    DAAL_KERNEL_AVX_CONTAINER(ContainerTemplate, __VA_ARGS__) \
    DAAL_KERNEL_AVX2_CONTAINER(ContainerTemplate, __VA_ARGS__) \
    DAAL_KERNEL_AVX512_mic_CONTAINER(ContainerTemplate, __VA_ARGS__) \
    DAAL_KERNEL_AVX512_CONTAINER(ContainerTemplate, __VA_ARGS__)>;

#endif
