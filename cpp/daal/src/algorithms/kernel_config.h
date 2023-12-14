/* file: kernel_config.h */
/*******************************************************************************
* Copyright 2023 Intel Corporation
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
//  Wrapper for platform-specific kernels
//--
*/

#ifndef __KERNEL_CONFIG_H__
#define __KERNEL_CONFIG_H__

#ifdef __ARM_ARCH
    #include "src/algorithms/kernel_inst_arm.h"
#else
    #include "src/algorithms/kernel_inst_x86.h"
#endif
#endif
