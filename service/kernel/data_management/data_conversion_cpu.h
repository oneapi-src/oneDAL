/** file data_conversion_cpu.h */
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

#ifndef __KERNEL_DATA_MANAGEMENT_DATA_CONVERSION_CPU_H__
#define __KERNEL_DATA_MANAGEMENT_DATA_CONVERSION_CPU_H__

#include "service_defines.h"

namespace daal
{
namespace data_management
{
namespace internal
{

template<typename T1, typename T2, CpuType cpu>
void vectorConvertFuncCpu(size_t n, const void *src, void *dst);

template<typename T1, typename T2, CpuType cpu>
void vectorStrideConvertFuncCpu(size_t n, const void *src, size_t srcByteStride, void *dst, size_t dstByteStride);

template<typename T, CpuType cpu>
void vectorAssignValueToArrayCpu(void* const ptr, const size_t n, const void* const value);

} // namespace internal
} // namespace data_management
} // namespace daal

#endif
