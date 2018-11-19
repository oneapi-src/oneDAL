/* file: kernel.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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
    {                                                      \
        _kernel = (new KernelClass<__VA_ARGS__, cpu>);     \
    }

#undef __DAAL_DEINITIALIZE_KERNELS
#define __DAAL_DEINITIALIZE_KERNELS()    \
    {                                    \
        if(_kernel) delete _kernel;      \
    }

#undef __DAAL_KERNEL_ARGUMENTS
#define __DAAL_KERNEL_ARGUMENTS(...) __VA_ARGS__

#undef __DAAL_CALL_KERNEL
#define __DAAL_CALL_KERNEL(env, KernelClass, templateArguments, method, ...)            \
    {                                                                                   \
        return ((KernelClass<templateArguments, cpu> *)(_kernel))->method(__VA_ARGS__); \
    }

#undef __DAAL_CALL_KERNEL_STATUS
#define __DAAL_CALL_KERNEL_STATUS(env, KernelClass, templateArguments, method, ...) \
        ((KernelClass<templateArguments, cpu> *)(_kernel))->method(__VA_ARGS__);

#define __DAAL_INSTANTIATE_DISPATCH_IMPL(ContainerTemplate, Mode, ClassName, BaseClassName, ...) \
template<>                                                                                       \
    ClassName< Mode,                                                                             \
    ContainerTemplate<__VA_ARGS__, sse2>                                                         \
    DAAL_KERNEL_SSSE3_CONTAINER(ContainerTemplate, __VA_ARGS__)                                  \
    DAAL_KERNEL_SSE42_CONTAINER(ContainerTemplate, __VA_ARGS__)                                  \
    DAAL_KERNEL_AVX_CONTAINER(ContainerTemplate, __VA_ARGS__)                                    \
    DAAL_KERNEL_AVX2_CONTAINER(ContainerTemplate, __VA_ARGS__)                                   \
    DAAL_KERNEL_AVX512_mic_CONTAINER(ContainerTemplate, __VA_ARGS__)                             \
    DAAL_KERNEL_AVX512_CONTAINER(ContainerTemplate, __VA_ARGS__) >::                             \
    ClassName(daal::services::Environment::env *daalEnv) : BaseClassName(daalEnv), _cntr(NULL)   \
{                                                                                                \
    switch (daalEnv->cpuid)                                                                      \
    {                                                                                            \
        DAAL_KERNEL_SSSE3_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                         \
        DAAL_KERNEL_SSE42_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                         \
        DAAL_KERNEL_AVX_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                           \
        DAAL_KERNEL_AVX2_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                          \
        DAAL_KERNEL_AVX512_mic_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                    \
        DAAL_KERNEL_AVX512_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                        \
        default: _cntr = (new ContainerTemplate<__VA_ARGS__, sse2> (daalEnv)); break;            \
    }                                                                                            \
}                                                                                                \
                                                                                                 \
template                                                                                         \
class ClassName< Mode,                                                                           \
    ContainerTemplate<__VA_ARGS__, sse2>                                                         \
    DAAL_KERNEL_SSSE3_CONTAINER(ContainerTemplate, __VA_ARGS__)                                  \
    DAAL_KERNEL_SSE42_CONTAINER(ContainerTemplate, __VA_ARGS__)                                  \
    DAAL_KERNEL_AVX_CONTAINER(ContainerTemplate, __VA_ARGS__)                                    \
    DAAL_KERNEL_AVX2_CONTAINER(ContainerTemplate, __VA_ARGS__)                                   \
    DAAL_KERNEL_AVX512_mic_CONTAINER(ContainerTemplate, __VA_ARGS__)                             \
    DAAL_KERNEL_AVX512_CONTAINER(ContainerTemplate, __VA_ARGS__)>;

#define __DAAL_INSTANTIATE_DISPATCH_LAYER_CONTAINER(ContainerTemplate,  ...) \
        __DAAL_INSTANTIATE_DISPATCH_IMPL(ContainerTemplate, batch, AlgorithmDispatchLayerContainer, LayerContainerIfaceImpl, __VA_ARGS__)

#define __DAAL_INSTANTIATE_DISPATCH_CONTAINER(ContainerTemplate, Mode, ...) \
        __DAAL_INSTANTIATE_DISPATCH_IMPL(ContainerTemplate, Mode, AlgorithmDispatchContainer, AlgorithmContainerImpl<Mode>, __VA_ARGS__)

#endif
