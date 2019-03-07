/* file: sgd_dense_kernel.h */
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

//++
//  Declaration of template function that calculate sgd.
//--


#ifndef __SGD_DENSE_KERNEL_H__
#define __SGD_DENSE_KERNEL_H__

#include "iterative_solver_kernel.h"
#include "sgd_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sgd
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
class SGDKernel : public iterative_solver::internal::IterativeSolverKernel<algorithmFPType, cpu>
{};


} // namespace daal::internal

} // namespace sgd

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
