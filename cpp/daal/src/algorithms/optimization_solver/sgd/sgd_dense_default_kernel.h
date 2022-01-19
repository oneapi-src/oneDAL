/* file: sgd_dense_default_kernel.h */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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

//++
//  Declaration of template function that calculate sgd.
//--

#ifndef __SGD_DENSE_DEFAULT_KERNEL_H__
#define __SGD_DENSE_DEFAULT_KERNEL_H__

#include "algorithms/optimization_solver/sgd/sgd_batch.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "src/algorithms/optimization_solver/iterative_solver_kernel.h"
#include "src/algorithms/optimization_solver/sgd/sgd_dense_kernel.h"

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
template <typename algorithmFPType, CpuType cpu>
class SGDKernel<algorithmFPType, defaultDense, cpu> : public iterative_solver::internal::IterativeSolverKernel<algorithmFPType, cpu>
{
public:
    services::Status compute(HostAppIface * pHost, NumericTable * inputArgument, NumericTable * minimum, NumericTable * nIterations,
                             Parameter<defaultDense> * parameter, NumericTable * learningRateSequence, NumericTable * batchIndices,
                             OptionalArgument * optionalArgument, OptionalArgument * optionalResult, engines::BatchBase & engine);

    using iterative_solver::internal::IterativeSolverKernel<algorithmFPType, cpu>::vectorNorm;
    using iterative_solver::internal::IterativeSolverKernel<algorithmFPType, cpu>::getRandom;
};

} // namespace internal

} // namespace sgd

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
