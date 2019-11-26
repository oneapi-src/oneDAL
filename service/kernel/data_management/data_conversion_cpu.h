/** file data_conversion_cpu.h */
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

#ifndef __KERNEL_DATA_MANAGEMENT_DATA_CONVERSION_CPU_H__
#define __KERNEL_DATA_MANAGEMENT_DATA_CONVERSION_CPU_H__

#include "service_defines.h"

namespace daal
{
namespace data_management
{
namespace internal
{
template <typename T1, typename T2, CpuType cpu>
void vectorConvertFuncCpu(size_t n, const void * src, void * dst);

template <typename T1, typename T2, CpuType cpu>
void vectorStrideConvertFuncCpu(size_t n, const void * src, size_t srcByteStride, void * dst, size_t dstByteStride);

template <typename T, CpuType cpu>
void vectorAssignValueToArrayCpu(void * const ptr, const size_t n, const void * const value);

} // namespace internal
} // namespace data_management
} // namespace daal

#endif
