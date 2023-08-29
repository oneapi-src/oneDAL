/* file: sgd_dense_default_impl.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

/*
//++
//  Implementation of sgd algorithm
//--
*/

#ifndef __SGD_DENSE_DEFAULT_IMPL_I__
#define __SGD_DENSE_DEFAULT_IMPL_I__

#include "src/data_management/service_micro_table.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_math.h"
#include "src/services/service_utils.h"
#include "src/algorithms/optimization_solver/iterative_solver_kernel.h"
#include "src/threading/threading.h"
#include "src/services/service_data_utils.h"

using namespace daal::internal;
using namespace daal::services;
using namespace daal::algorithms::optimization_solver::iterative_solver::internal;

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
 *  \brief Kernel for SGD calculation
 */
template <typename algorithmFPType, CpuType cpu>
services::Status SGDKernel<algorithmFPType, defaultDense, cpu>::compute(HostAppIface * pHost, NumericTable * inputArgument, NumericTable * minimum,
                                                                        NumericTable * nIterations, Parameter<defaultDense> * parameter,
                                                                        NumericTable * learningRateSequence, NumericTable * batchIndices,
                                                                        OptionalArgument * optionalArgument, OptionalArgument * optionalResult,
                                                                        engines::BatchBase & engine)
{
    const size_t nRows = inputArgument->getNumberOfRows();
    SafeStatus safeStat;
    //init workValue
    {
        processByBlocks<cpu>(minimum->getNumberOfRows(), [=, &safeStat](size_t startOffset, size_t nRowsInBlock) {
            WriteRows<algorithmFPType, cpu, NumericTable> workValueBD(*minimum, startOffset, nRowsInBlock);
            DAAL_CHECK_BLOCK_STATUS_THR(workValueBD);
            algorithmFPType * minArray = workValueBD.get();
            ReadRows<algorithmFPType, cpu, NumericTable> startValueBD(*inputArgument, startOffset, nRowsInBlock);
            DAAL_CHECK_BLOCK_STATUS_THR(startValueBD);
            const algorithmFPType * startValueArray = startValueBD.get();
            if (minArray != startValueArray)
            {
                int result = daal::services::internal::daal_memcpy_s(minArray, nRowsInBlock * sizeof(algorithmFPType), startValueArray,
                                                                     nRowsInBlock * sizeof(algorithmFPType));
                if (result) safeStat.add(services::Status(services::ErrorMemoryCopyFailedInternal));
            }
        });
        DAAL_CHECK_SAFE_STATUS();
    }

    const size_t nIter = parameter->nIterations;
    /* if nIter == 0, set result as start point, the number of executed iters to 0 */
    if (nIter == 0)
    {
        WriteRows<int, cpu, NumericTable> nIterationsBD(*nIterations, 0, 1);
        int * nProceededIterations = nIterationsBD.get();
        nProceededIterations[0]    = 0;
        return Status();
    }

    NumericTable * lastIterationInput =
        (optionalArgument) ? NumericTable::cast(optionalArgument->get(iterative_solver::lastIteration)).get() : nullptr;
    NumericTable * lastIterationResult = (optionalResult) ? NumericTable::cast(optionalResult->get(iterative_solver::lastIteration)).get() : nullptr;

    sum_of_functions::BatchPtr function = parameter->function;

    const size_t nTerms = function->sumOfFunctionsParameter->numberOfTerms;
    ReadRows<int, cpu, NumericTable> predefinedBatchIndicesBD(batchIndices, 0, nIter);
    const bool bGenerateAllIndices = !(parameter->batchIndices || parameter->optionalResultRequired);
    TArray<int, cpu> aPredefinedBatchIndices(bGenerateAllIndices ? nIter : 0);
    if (bGenerateAllIndices)
    {
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nIter, sizeof(int));
        /*Get random indices for SGD from rng generator*/
        DAAL_CHECK_MALLOC(aPredefinedBatchIndices.get());
        getRandom(0, nTerms, aPredefinedBatchIndices.get(), nIter, engine);
    }
    using namespace iterative_solver::internal;
    const int * predefinedBatchIndices = predefinedBatchIndicesBD.get() ? predefinedBatchIndicesBD.get() : aPredefinedBatchIndices.get();
    RngTask<int, cpu> rngTask(predefinedBatchIndices, 1);
    DAAL_CHECK_MALLOC((predefinedBatchIndices) || rngTask.init(nTerms, engine));

    Status s;
    SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu> > ntBatchIndices(new daal::internal::HomogenNumericTableCPU<int, cpu>(NULL, 1, 1, s));

    NumericTablePtr minimimWrapper(minimum, EmptyDeleter());
    function->sumOfFunctionsParameter->batchIndices = ntBatchIndices;
    function->sumOfFunctionsInput->set(sum_of_functions::argument, minimimWrapper);

    ReadRows<algorithmFPType, cpu, NumericTable> learningRateBD(*learningRateSequence, 0, learningRateSequence->getNumberOfRows());
    const algorithmFPType * learningRateArray = learningRateBD.get();
    const size_t learningRateLength           = learningRateSequence->getNumberOfColumns();
    const double accuracyThreshold            = parameter->accuracyThreshold;

    WriteRows<int, cpu, NumericTable> nIterationsBD(*nIterations, 0, 1);
    int * nProceededIterations = nIterationsBD.get();
    DAAL_CHECK(nIter <= services::internal::MaxVal<int>::get(), ErrorIterativeSolverIncorrectMaxNumberOfIterations)
    *nProceededIterations = (int)nIter;

    size_t startIteration = 0, epoch = 0, nProceededIters = 0;
    if (lastIterationInput != nullptr)
    {
        ReadRows<int, cpu, NumericTable> lastIterationInputBD(lastIterationInput, 0, 1);
        const int * lastIterationInputArray = lastIterationInputBD.get();
        startIteration                      = lastIterationInputArray[0];
    }

    services::internal::HostAppHelper host(pHost, 10);
    for (epoch = startIteration; s.ok() && (epoch < (startIteration + nIter)); epoch++)
    {
        const int * pValues = nullptr;
        s                   = rngTask.get(pValues);
        if (s)
        {
            ntBatchIndices->setArray(const_cast<int *>(pValues), ntBatchIndices->getNumberOfRows());
            s = function->computeNoThrow();
        }
        if (!s || host.isCancelled(s, 1))
        {
            nProceededIterations[0] = nProceededIters;
            return s;
        }

        NumericTable * gradient = function->getResult()->get(objective_function::gradientIdx).get();
        if (nIter != 1)
        {
            algorithmFPType pointNorm, gradientNorm;
            s = vectorNorm(minimum, pointNorm);
            s |= vectorNorm(gradient, gradientNorm);
            DAAL_CHECK_BREAK(!s);

            const algorithmFPType one(1.0);
            const algorithmFPType gradientThreshold = accuracyThreshold * daal::internal::MathInst<algorithmFPType, cpu>::sMax(one, pointNorm);
            if (gradientNorm < gradientThreshold)
            {
                DAAL_ASSERT(nProceededIters <= services::internal::MaxVal<int>::get())
                nProceededIterations[0] = (int)nProceededIters;
                break;
            }
        }

        const algorithmFPType learningRate = learningRateArray[epoch % learningRateLength];

        processByBlocks<cpu>(
            nRows,
            [=, &safeStat](size_t startOffset, size_t nRowsInBlock) {
                WriteRows<algorithmFPType, cpu, NumericTable> workValueBD(*minimum, startOffset, nRowsInBlock);
                DAAL_CHECK_BLOCK_STATUS_THR(workValueBD);
                algorithmFPType * workLocal = workValueBD.get();
                ReadRows<algorithmFPType, cpu, NumericTable> ntGradientBD(*gradient, startOffset, nRowsInBlock);
                DAAL_CHECK_BLOCK_STATUS_THR(ntGradientBD);
                const algorithmFPType * gradientLocal = ntGradientBD.get();
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < nRowsInBlock; j++)
                {
                    workLocal[j] = workLocal[j] - learningRate * gradientLocal[j];
                }
            },
            256);
        if (!safeStat) s |= safeStat.detach();
        nProceededIters++;
    }
    if (lastIterationResult)
    {
        WriteRows<int, cpu, NumericTable> lastIterationResultBD(lastIterationResult, 0, 1);
        int * lastIterationResultArray = lastIterationResultBD.get();
        lastIterationResultArray[0]    = startIteration + nProceededIters;
    }
    return s;
}

} // namespace internal

} // namespace sgd

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
