/* file: kernel_inst_arm.h */
/*******************************************************************************
* Copyright 2023-24 FUJITSU LIMITED
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

#ifndef __KERNEL_INST_ARM_H__
#define __KERNEL_INST_ARM_H__

#include "services/daal_defines.h"
#include "src/services/service_defines.h"
#include "services/internal/daal_kernel_defines.h"
#include "services/internal/gpu_support_checker.h"

#define __DAAL_GET_CPUID int cpuid = daalEnv->cpuid;

#define __DAAL_GET_CPUID_SAFE  \
    int cpuid = DAAL_BASE_CPU; \
    DAAL_SAFE_CPU_CALL((cpuid = daalEnv->cpuid), (cpuid = DAAL_BASE_CPU))

#define __DAAL_KERNEL_MIN(a, b) ((a) < (b) ? (a) : (b))

#define __DAAL_INSTANTIATE_DISPATCH_IMPL(ContainerTemplate, Mode, ClassName, BaseClassName, GetCpuid, ...)                         \
    DAAL_KERNEL_SVE_CONTAINER1(ContainerTemplate, __VA_ARGS__)                                                                     \
    namespace interface1                                                                                                           \
    {                                                                                                                              \
    template <>                                                                                                                    \
    ClassName<Mode, ContainerTemplate<__VA_ARGS__, sve> DAAL_KERNEL_SVE_CONTAINER(ContainerTemplate, __VA_ARGS__)>::ClassName(     \
        daal::services::Environment::env * daalEnv)                                                                                \
        : BaseClassName(daalEnv), _cntr(nullptr)                                                                                   \
    {                                                                                                                              \
        GetCpuid switch (__DAAL_KERNEL_MIN(DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID, cpuid))                                       \
        {                                                                                                                          \
            DAAL_KERNEL_SVE_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                                                         \
        default: _cntr = (new ContainerTemplate<__VA_ARGS__, sve>(daalEnv)); break;                                                \
        }                                                                                                                          \
    }                                                                                                                              \
                                                                                                                                   \
    template class ClassName<Mode, ContainerTemplate<__VA_ARGS__, sve> DAAL_KERNEL_SVE_CONTAINER(ContainerTemplate, __VA_ARGS__)>; \
    }

#define __DAAL_INSTANTIATE_DISPATCH_CONTAINER_SAFE(ContainerTemplate, Mode, ...)                                                               \
    __DAAL_INSTANTIATE_DISPATCH_IMPL(ContainerTemplate, Mode, AlgorithmDispatchContainer, AlgorithmContainerImpl<Mode>, __DAAL_GET_CPUID_SAFE, \
                                     __VA_ARGS__)

#define __DAAL_INSTANTIATE_DISPATCH_CONTAINER(ContainerTemplate, Mode, ...) \
    __DAAL_INSTANTIATE_DISPATCH_IMPL(ContainerTemplate, Mode, AlgorithmDispatchContainer, AlgorithmContainerImpl<Mode>, __DAAL_GET_CPUID, __VA_ARGS__)

#define __DAAL_INSTANTIATE_DISPATCH_SYCL_IMPL(ContainerTemplate, Mode, ClassName, BaseClassName, GetCpuid, ...)                    \
    DAAL_KERNEL_SVE_CONTAINER1(ContainerTemplate, __VA_ARGS__)                                                                     \
    namespace interface1                                                                                                           \
    {                                                                                                                              \
    template <>                                                                                                                    \
    ClassName<Mode, ContainerTemplate<__VA_ARGS__, sve> DAAL_KERNEL_SVE_CONTAINER(ContainerTemplate, __VA_ARGS__)>::ClassName(     \
        daal::services::Environment::env * daalEnv)                                                                                \
        : BaseClassName(daalEnv), _cntr(NULL)                                                                                      \
    {                                                                                                                              \
        GetCpuid switch (__DAAL_KERNEL_MIN(DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID, cpuid))                                       \
        {                                                                                                                          \
            DAAL_KERNEL_SVE_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                                                         \
        default:                                                                                                                   \
        {                                                                                                                          \
            using cntrTemplateInst = ContainerTemplate<__VA_ARGS__, sve>;                                                          \
            static volatile services::internal::GpuSupportRegistrar<cntrTemplateInst> registrar;                                   \
            _cntr = (new cntrTemplateInst(daalEnv));                                                                               \
            break;                                                                                                                 \
        }                                                                                                                          \
        }                                                                                                                          \
    }                                                                                                                              \
                                                                                                                                   \
    template class ClassName<Mode, ContainerTemplate<__VA_ARGS__, sve> DAAL_KERNEL_SVE_CONTAINER(ContainerTemplate, __VA_ARGS__)>; \
    }

#define __DAAL_INSTANTIATE_DISPATCH_CONTAINER_SYCL(ContainerTemplate, Mode, ...)                                                               \
    __DAAL_INSTANTIATE_DISPATCH_SYCL_IMPL(ContainerTemplate, Mode, AlgorithmDispatchContainer, AlgorithmContainerImpl<Mode>, __DAAL_GET_CPUID, \
                                          __VA_ARGS__)

#define __DAAL_INSTANTIATE_DISPATCH_CONTAINER_SYCL_SAFE(ContainerTemplate, Mode, ...)                                                               \
    __DAAL_INSTANTIATE_DISPATCH_SYCL_IMPL(ContainerTemplate, Mode, AlgorithmDispatchContainer, AlgorithmContainerImpl<Mode>, __DAAL_GET_CPUID_SAFE, \
                                          __VA_ARGS__)

#endif
