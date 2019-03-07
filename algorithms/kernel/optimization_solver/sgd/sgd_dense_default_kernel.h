/* file: sgd_dense_default_kernel.h */
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


#ifndef __SGD_DENSE_DEFAULT_KERNEL_H__
#define __SGD_DENSE_DEFAULT_KERNEL_H__

#include "sgd_batch.h"
#include "kernel.h"
#include "numeric_table.h"
#include "iterative_solver_kernel.h"
#include "sgd_dense_kernel.h"

using namespace daal::data_management;

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

template<typename algorithmFPType, CpuType cpu>
class SGDKernel<algorithmFPType, defaultDense, cpu> : public iterative_solver::internal::IterativeSolverKernel<algorithmFPType, cpu>
{
public:
    services::Status compute(HostAppIface* pHost, NumericTable *inputArgument,
        NumericTable *minimum, NumericTable *nIterations,
        Parameter<defaultDense> *parameter, NumericTable *learningRateSequence,
        NumericTable *batchIndices, OptionalArgument *optionalArgument, OptionalArgument *optionalResult, engines::BatchBase &engine);
    using iterative_solver::internal::IterativeSolverKernel<algorithmFPType, cpu>::vectorNorm;
    using iterative_solver::internal::IterativeSolverKernel<algorithmFPType, cpu>::getRandom;
};


} // namespace daal::internal

} // namespace sgd

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
