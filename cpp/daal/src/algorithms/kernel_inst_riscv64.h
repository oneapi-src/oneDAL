/* file: kernel_inst_riscv64.h */
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

/*
//++
//  The defines used for kernel allocation, deallocation, and calling kernel methods
//--
*/

#ifndef __KERNEL_INST_RISCV64_H__
#define __KERNEL_INST_RISCV64_H__

#define __DAAL_INSTANTIATE_DISPATCH_IMPL(ContainerTemplate, Mode, ClassName, BaseClassName, GetCpuid, ...)                           \
    DAAL_KERNEL_RV64_CONTAINER1(ContainerTemplate, __VA_ARGS__)                                                                      \
    namespace interface1                                                                                                             \
    {                                                                                                                                \
    template <>                                                                                                                      \
    ClassName<Mode, ContainerTemplate<__VA_ARGS__, rv64> DAAL_KERNEL_RV64_CONTAINER(ContainerTemplate, __VA_ARGS__)>::ClassName(     \
        daal::services::Environment::env * daalEnv)                                                                                  \
        : BaseClassName(daalEnv), _cntr(nullptr)                                                                                     \
    {                                                                                                                                \
        GetCpuid switch (__DAAL_KERNEL_MIN(DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID, cpuid))                                         \
        {                                                                                                                            \
            DAAL_KERNEL_RV64_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                                                          \
        default: _cntr = (new ContainerTemplate<__VA_ARGS__, rv64>(daalEnv)); break;                                                 \
        }                                                                                                                            \
    }                                                                                                                                \
                                                                                                                                     \
    template class ClassName<Mode, ContainerTemplate<__VA_ARGS__, rv64> DAAL_KERNEL_RV64_CONTAINER(ContainerTemplate, __VA_ARGS__)>; \
    }

#define __DAAL_INSTANTIATE_DISPATCH_SYCL_IMPL(ContainerTemplate, Mode, ClassName, BaseClassName, GetCpuid, ...)                      \
    DAAL_KERNEL_RV64_CONTAINER1(ContainerTemplate, __VA_ARGS__)                                                                      \
    namespace interface1                                                                                                             \
    {                                                                                                                                \
    template <>                                                                                                                      \
    ClassName<Mode, ContainerTemplate<__VA_ARGS__, rv64> DAAL_KERNEL_RV64_CONTAINER(ContainerTemplate, __VA_ARGS__)>::ClassName(     \
        daal::services::Environment::env * daalEnv)                                                                                  \
        : BaseClassName(daalEnv), _cntr(NULL)                                                                                        \
    {                                                                                                                                \
        GetCpuid switch (__DAAL_KERNEL_MIN(DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID, cpuid))                                         \
        {                                                                                                                            \
            DAAL_KERNEL_RV64_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                                                          \
        default:                                                                                                                     \
        {                                                                                                                            \
            using cntrTemplateInst = ContainerTemplate<__VA_ARGS__, rv64>;                                                           \
            static volatile services::internal::GpuSupportRegistrar<cntrTemplateInst> registrar;                                     \
            _cntr = (new cntrTemplateInst(daalEnv));                                                                                 \
            break;                                                                                                                   \
        }                                                                                                                            \
        }                                                                                                                            \
    }                                                                                                                                \
                                                                                                                                     \
    template class ClassName<Mode, ContainerTemplate<__VA_ARGS__, rv64> DAAL_KERNEL_RV64_CONTAINER(ContainerTemplate, __VA_ARGS__)>; \
    }

#endif
