/* file: kernel_function_linear_base.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Declaration of template structs that calculate SVM Linear Kernel functions.
//--
*/

#ifndef __KERNEL_FUNCTION_LINEAR_BASE_H__
#define __KERNEL_FUNCTION_LINEAR_BASE_H__

#include "kernel_function_types_linear.h"
#include "kernel.h"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace linear
{
namespace internal
{

template <Method method, typename algorithmFPType, CpuType cpu>
struct KernelImplLinear {};

} // namespace internal

} // namespace linear

} // namespace kernel_function

} // namespace algorithms

} // namespace daal

#endif
