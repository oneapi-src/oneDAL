/* file: sgd_dense_minibatch_kernel.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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


#ifndef __SGD_DENSE_MINIBATCH_KERNEL_H__
#define __SGD_DENSE_MINIBATCH_KERNEL_H__

#include "sgd_batch.h"
#include "kernel.h"
#include "numeric_table.h"
#include "iterative_solver_kernel.h"
#include "sgd_dense_kernel.h"
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
enum IndicesStatus
{
    random = 0,     /*!< Indices of the terms are generated randomly */
    user   = 1,     /*!< Indices of the terms are provided by user */
    all    = 2      /*!< All objective function terms are used for computations */
};

template<typename algorithmFPType, CpuType cpu>
class SGDKernel<algorithmFPType, miniBatch, cpu> : public iterative_solver::internal::IterativeSolverKernel<algorithmFPType, cpu>
{
private:
    void makeStep(
        algorithmFPType *workValue,
        algorithmFPType *prevWorkValue,
        algorithmFPType *gradient,
        algorithmFPType learningRate,
        algorithmFPType consCoeff,
        size_t nFeatures);

public:
    void compute(NumericTable *inputArgument, NumericTable *minimum, NumericTable *nIterations,
                 Parameter<miniBatch> *parameter, NumericTable *learningRateSequence,
                 NumericTable *batchIndices);
    using iterative_solver::internal::IterativeSolverKernel<algorithmFPType, cpu>::vectorNorm;
    using iterative_solver::internal::IterativeSolverKernel<algorithmFPType, cpu>::getRandom;

};

template<typename algorithmFPType, CpuType cpu>
struct SGDMiniBatchTask
{
    services::SharedPtr<services::KernelErrorCollection> _errors;

    SGDMiniBatchTask(
        services::SharedPtr<services::KernelErrorCollection> errors_,
        size_t nFeatures_,
        NumericTable *resultTable,
        NumericTable *startValueTable,
        NumericTable *nIterationsTable
    );

    SGDMiniBatchTask(
        services::SharedPtr<services::KernelErrorCollection> errors_,
        size_t batchSize_,
        size_t nFeatures_,
        size_t maxIterations_,
        size_t nTerms_,
        NumericTable *resultTable,
        NumericTable *startValueTable,
        NumericTable *learningRateSequenceTable,
        NumericTable *conservativeSequenceTable,
        NumericTable *nIterationsTable,
        NumericTable *batchIndicesTable
    );

    virtual ~SGDMiniBatchTask();

    void setStartValue(NumericTable *startValueTable);
    void getNextPredefinedIndices(size_t iter);

    size_t batchSize;
    size_t nFeatures;
    size_t maxIterations;
    size_t nTerms;

    algorithmFPType *workValue;
    int             *nProceededIterations;
    algorithmFPType *learningRateArray;
    algorithmFPType *consCoeffsArray;
    size_t          learningRateLength;
    size_t          consCoeffsLength;
    algorithmFPType *prevWorkValue;
    int             *randomTerm;
    IndicesStatus   indicesStatus;

    BlockMicroTable<algorithmFPType, writeOnly, cpu> mtWorkValue;
    SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu>> ntBatchIndices;
    SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>> ntWorkValue;
    BlockMicroTable<algorithmFPType, readOnly, cpu> mtLearningRate;
    BlockMicroTable<algorithmFPType, readOnly, cpu> mtConsCoeffs;
    BlockMicroTable<int, writeOnly, cpu> mtNIterations;
    BlockMicroTable<int, readOnly, cpu> mtPredefinedBatchIndices;
};

} // namespace daal::internal

} // namespace sgd

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
