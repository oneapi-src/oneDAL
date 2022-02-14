/* file: level_zero_common.h */
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

#ifndef __DAAL_SERVICES_INTERNAL_SYCL_LEVEL_ZERO_COMMON_H__
#define __DAAL_SERVICES_INTERNAL_SYCL_LEVEL_ZERO_COMMON_H__

#ifdef DAAL_DISABLE_LEVEL_ZERO
    #error "DAAL_DISABLE_LEVEL_ZERO must be undefined to include this file"
#endif

#ifndef _ZE_API_H
    #include "services/internal/sycl/level_zero_types.h"
#endif

typedef ze_result_t (*zeModuleCreateFT)(ze_context_handle_t, ze_device_handle_t, const ze_module_desc_t *, ze_module_handle_t *,
                                        ze_module_build_log_handle_t *);

typedef ze_result_t (*zeModuleDestroyFT)(ze_module_handle_t hModule);

typedef ze_result_t (*zeKernelCreateFT)(ze_module_handle_t hModule, const ze_kernel_desc_t * desc, ze_kernel_handle_t * phKernel);

typedef ze_result_t (*zeKernelDestroyFT)(ze_kernel_handle_t hKernel);

#endif
