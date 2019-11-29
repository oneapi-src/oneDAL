/* file: sgd_dense_minibatch_kernel_v1.h */
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

#ifndef __SGD_DENSE_MINIBATCH_KERNEL_V1_H__
#define __SGD_DENSE_MINIBATCH_KERNEL_V1_H__

#include "sgd_batch.h"
#include "kernel.h"
#include "numeric_table.h"
#include "iterative_solver_kernel.h"
#include "sgd_dense_kernel_v1.h"
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
enum IndicesStatus
{
    random = 0, /*!< Indices of the terms are generated randomly */
    user   = 1, /*!< Indices of the terms are provided by user */
    all    = 2  /*!< All objective function terms are used for computations */
};

template <typename algorithmFPType, CpuType cpu>
class I1SGDKernel<algorithmFPType, miniBatch, cpu> : public iterative_solver::internal::IterativeSolverKernel<algorithmFPType, cpu>
{
public:
    services::Status compute(HostAppIface * pHost, NumericTable * inputArgument, NumericTable * minimum, NumericTable * nIterations,
                             interface1::Parameter<miniBatch> * parameter, NumericTable * learningRateSequence, NumericTable * batchIndices,
                             OptionalArgument * optionalArgument, OptionalArgument * optionalResult, engines::BatchBase & engine);
    using iterative_solver::internal::IterativeSolverKernel<algorithmFPType, cpu>::vectorNorm;
};

template <typename algorithmFPType, CpuType cpu>
struct I1SGDMiniBatchTask
{
    I1SGDMiniBatchTask(size_t nFeatures_, NumericTable * resultTable, NumericTable * startValueTable, NumericTable * nIterationsTable);

    I1SGDMiniBatchTask(size_t batchSize_, size_t nFeatures_, size_t maxIterations_, size_t nTerms_, NumericTable * resultTable,
                       NumericTable * startValueTable, NumericTable * learningRateSequenceTable, NumericTable * conservativeSequenceTable,
                       NumericTable * nIterationsTable, NumericTable * batchIndicesTable, NumericTable * pastWorkValueResultNT,
                       NumericTable * lastIterationResultNT);

    virtual ~I1SGDMiniBatchTask();

    services::Status init(NumericTable * startValueTable);

    services::Status init(NumericTable * startValueTable, NumericTable * learningRateSequenceTable, NumericTable * conservativeSequenceTable,
                          NumericTable * nIterationsTable, NumericTable * batchIndicesTable, OptionalArgument * optionalInput);

    services::Status setStartValue(NumericTable * startValueTable);

    void makeStep(const algorithmFPType * gradient, algorithmFPType learningRate, algorithmFPType consCoeff, size_t argumentSize);

    size_t batchSize;
    size_t argumentSize;
    size_t nIter;
    size_t nTerms;
    size_t startIteration;
    size_t nProceededIters;

    services::Status _status;

    int * nProceededIterations;
    const algorithmFPType * learningRateArray;
    const algorithmFPType * consCoeffsArray;
    size_t learningRateLength;
    size_t consCoeffsLength;
    TArray<algorithmFPType, cpu> prevWorkValue;
    IndicesStatus indicesStatus;

    WriteRows<algorithmFPType, cpu> mtWorkValue;
    SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu> > ntBatchIndices;
    SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > ntWorkValue;
    ReadRows<algorithmFPType, cpu> mtLearningRate;
    ReadRows<algorithmFPType, cpu> mtConsCoeffs;
    WriteRows<int, cpu> mtNIterations;
    ReadRows<int, cpu> mtPredefinedBatchIndices;

    NumericTablePtr lastIterationResult;
    NumericTablePtr pastWorkValueResult;
};

} // namespace internal

} // namespace sgd

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
