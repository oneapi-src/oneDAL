/* file: sgd_dense_momentum_kernel.h */
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

//++
//  Declaration of template function that calculate sgd.
//--

#ifndef __SGD_DENSE_MOMENTUM_KERNEL_H__
#define __SGD_DENSE_MOMENTUM_KERNEL_H__

#include "sgd_batch.h"
#include "kernel.h"
#include "numeric_table.h"
#include "iterative_solver_kernel.h"
#include "sgd_dense_kernel.h"
#include "sgd_dense_minibatch_kernel.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "service_math.h"
#include "service_utils.h"

using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;

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
/**
* Statuses of the indices of objective function terms that are used for gradient
*/

template <typename algorithmFPType, CpuType cpu>
class SGDKernel<algorithmFPType, momentum, cpu> : public iterative_solver::internal::IterativeSolverKernel<algorithmFPType, cpu>
{
public:
    services::Status compute(HostAppIface * pHost, NumericTable * inputArgument, NumericTable * minimum, NumericTable * nIterations,
                             Parameter<momentum> * parameter, NumericTable * learningRateSequence, NumericTable * batchIndices,
                             OptionalArgument * optionalArgument, OptionalArgument * optionalResult, engines::BatchBase & engine);
    using iterative_solver::internal::IterativeSolverKernel<algorithmFPType, cpu>::vectorNorm;
};

template <typename algorithmFPType, CpuType cpu>
struct SGDmomentumTask
{
    SGDmomentumTask(size_t batchSize_, size_t nTerms_, NumericTable * resultTable, NumericTable * batchIndicesTable, NumericTable * pastUpdateResult,
                    NumericTable * lastIterationResultNT, Parameter<momentum> * parameter);

    virtual ~SGDmomentumTask();

    Status init(NumericTable * batchIndicesTable, NumericTable * resultTable, Parameter<momentum> * parameter, NumericTable * pastUpdateInput,
                NumericTable * lastIterationInput);

    Status setStartValue(NumericTable * inputArgument, NumericTable * minimum);

    Status makeStep(NumericTable * gradient, NumericTable * minimum, NumericTable * pastUpdate, const algorithmFPType learningRate,
                    const algorithmFPType momentum);

    size_t batchSize;
    size_t nTerms;
    size_t startIteration;
    size_t nProceededIters;

    IndicesStatus indicesStatus;

    SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu> > ntBatchIndices;
    NumericTablePtr minimimWrapper;
    NumericTablePtr pastUpdate;
    NumericTablePtr lastIterationResult;
};

} // namespace internal

} // namespace sgd

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
