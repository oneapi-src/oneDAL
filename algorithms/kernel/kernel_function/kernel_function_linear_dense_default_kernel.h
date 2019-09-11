/* file: kernel_function_linear_dense_default_kernel.h */
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

#ifndef __KERNEL_FUNCTION_DENSE_KERNEL_H__
#define __KERNEL_FUNCTION_DENSE_KERNEL_H__

#include "kernel_function_dense_base.h"
#include "kernel_function_linear_base.h"

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

template <typename algorithmFPType, CpuType cpu>
struct KernelImplLinear<defaultDense, algorithmFPType, cpu> :
    public daal::algorithms::kernel_function::internal::KernelImplBase<algorithmFPType, cpu>
{
    virtual services::Status computeInternalVectorVector(const NumericTable *a1, const NumericTable *a2, NumericTable *r, const ParameterBase *par);
    virtual services::Status computeInternalMatrixVector(const NumericTable *a1, const NumericTable *a2, NumericTable *r, const ParameterBase *par);
    virtual services::Status computeInternalMatrixMatrix(const NumericTable *a1, const NumericTable *a2, NumericTable *r, const ParameterBase *par);
};

} //internal

} //linear

} // namespace kernel_function

} // namespace algorithms

} // namespace daal

#endif
