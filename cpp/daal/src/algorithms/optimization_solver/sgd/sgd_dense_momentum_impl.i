/* file: sgd_dense_momentum_impl.i */
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
//  Implementation of sgd momentum algorithm
//--
*/

#ifndef __SGD_DENSE_MOMENTUM_IMPL_I__
#define __SGD_DENSE_MOMENTUM_IMPL_I__

#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_math.h"
#include "src/services/service_utils.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/optimization_solver/iterative_solver_kernel.h"
#include "src/threading/threading.h"
#include "src/services/service_data_utils.h"

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
 *  \brief Kernel for SGD momentum calculation
 */
template <typename algorithmFPType, CpuType cpu>
services::Status SGDKernel<algorithmFPType, momentum, cpu>::compute(HostAppIface * pHost, NumericTable * inputArgument, NumericTable * minimum,
                                                                    NumericTable * nProceededIterationsNT, Parameter<momentum> * parameter,
                                                                    NumericTable * learningRateSequence, NumericTable * batchIndices,
                                                                    OptionalArgument * optionalArgument, OptionalArgument * optionalResult,
                                                                    engines::BatchBase & engine)
{
    services::Status s;
    const size_t nIter             = parameter->nIterations;
    const size_t batchSize         = parameter->batchSize;
    const double accuracyThreshold = parameter->accuracyThreshold;
    const double momentum          = parameter->momentum;

    /* if nIter == 0, set result as start point, the number of executed iters to 0 */
    WriteRows<int, cpu, NumericTable> nProceededIterationsBD(*nProceededIterationsNT, 0, 1);
    int * nProceededIterations = nProceededIterationsBD.get();
    if (nIter == 0)
    {
        nProceededIterations[0] = 0;
        return s;
    }

    sum_of_functions::BatchPtr function = parameter->function;

    const size_t nTerms = function->sumOfFunctionsParameter->numberOfTerms;
    SGDmomentumTask<algorithmFPType, cpu> task(
        batchSize, nTerms, minimum, batchIndices, optionalResult ? NumericTable::cast(optionalResult->get(pastUpdateVector)).get() : nullptr,
        optionalResult ? NumericTable::cast(optionalResult->get(iterative_solver::lastIteration)).get() : nullptr, parameter);

    DAAL_CHECK_STATUS(
        s, task.init(batchIndices, minimum, parameter, optionalArgument ? NumericTable::cast(optionalArgument->get(pastUpdateVector)).get() : nullptr,
                     optionalArgument ? NumericTable::cast(optionalArgument->get(iterative_solver::lastIteration)).get() : nullptr));
    DAAL_CHECK_STATUS(s, task.setStartValue(inputArgument, minimum));
    function->sumOfFunctionsInput->set(sum_of_functions::argument, task.minimimWrapper);
    function->sumOfFunctionsParameter->batchIndices = task.ntBatchIndices;

    ReadRows<int, cpu> predefinedBatchIndicesBD(batchIndices, 0, nIter);
    using namespace iterative_solver::internal;
    RngTask<int, cpu> rngTask(predefinedBatchIndicesBD.get(), batchSize);
    DAAL_CHECK_MALLOC(batchIndices || rngTask.init(nTerms, engine));

    ReadRows<algorithmFPType, cpu, NumericTable> learningRateBD(*learningRateSequence, 0, learningRateSequence->getNumberOfRows());
    const algorithmFPType * learningRateArray = learningRateBD.get();
    const size_t learningRateLength           = learningRateSequence->getNumberOfRows();

    services::internal::HostAppHelper host(pHost, 10);
    for (size_t epoch = task.startIteration; epoch < (task.startIteration + nIter); epoch++)
    {
        if (task.indicesStatus == user || task.indicesStatus == random)
        {
            const int * pValues = nullptr;
            DAAL_CHECK_STATUS(s, rngTask.get(pValues));
            task.ntBatchIndices->setArray(const_cast<int *>(pValues), task.ntBatchIndices->getNumberOfRows());
        }
        DAAL_CHECK_STATUS(s, function->computeNoThrow());
        NumericTable * gradient = function->getResult()->get(objective_function::gradientIdx).get();
        if (nIter != 1)
        {
            algorithmFPType pointNorm, gradientNorm;
            s = vectorNorm(minimum, pointNorm);
            s |= vectorNorm(gradient, gradientNorm);

            DAAL_CHECK_BREAK(!s || host.isCancelled(s, 1));
            const algorithmFPType one(1.0);
            const algorithmFPType gradientThreshold = accuracyThreshold * daal::internal::MathInst<algorithmFPType, cpu>::sMax(one, pointNorm);
            DAAL_CHECK_BREAK(gradientNorm < gradientThreshold);
        }

        const algorithmFPType learningRate = learningRateArray[epoch % learningRateLength];
        DAAL_CHECK_STATUS(s, task.makeStep(gradient, minimum, task.pastUpdate.get(), learningRate, momentum));
    }
    DAAL_CHECK(task.nProceededIters <= services::internal::MaxVal<int>::get(), ErrorIterativeSolverIncorrectMaxNumberOfIterations)
    nProceededIterations[0] = (int)task.nProceededIters;

    return s;
}

template <typename algorithmFPType, CpuType cpu>
services::Status SGDmomentumTask<algorithmFPType, cpu>::makeStep(NumericTable * gradient, NumericTable * minimum, NumericTable * pastUpdate,
                                                                 const algorithmFPType learningRate, const algorithmFPType momentum)
{
    SafeStatus safeStat;
    processByBlocks<cpu>(minimum->getNumberOfRows(), [=, &safeStat](size_t startOffset, size_t nRowsInBlock) {
        WriteRows<algorithmFPType, cpu, NumericTable> workValueBD(*minimum, startOffset, nRowsInBlock);
        DAAL_CHECK_BLOCK_STATUS_THR(workValueBD);
        algorithmFPType * workArray = workValueBD.get();
        WriteRows<algorithmFPType, cpu, NumericTable> pastUpdateBD(*pastUpdate, startOffset, nRowsInBlock);
        DAAL_CHECK_BLOCK_STATUS_THR(pastUpdateBD);
        algorithmFPType * pastUpdateArray = pastUpdateBD.get();
        ReadRows<algorithmFPType, cpu, NumericTable> gradientBD(*gradient, startOffset, nRowsInBlock);
        DAAL_CHECK_BLOCK_STATUS_THR(gradientBD);
        const algorithmFPType * gradientArray = gradientBD.get();

        for (size_t j = 0; j < nRowsInBlock; j++)
        {
            pastUpdateArray[j] = -learningRate * gradientArray[j] + momentum * pastUpdateArray[j];
            workArray[j]       = workArray[j] + pastUpdateArray[j];
        }
    });
    nProceededIters++;
    return safeStat.detach();
}

template <typename algorithmFPType, CpuType cpu>
SGDmomentumTask<algorithmFPType, cpu>::~SGDmomentumTask()
{
    if (lastIterationResult)
    {
        WriteRows<int, cpu, NumericTable> lastIterationResultBD(lastIterationResult.get(), 0, 1);
        int * lastIterationResultArray = lastIterationResultBD.get();
        lastIterationResultArray[0]    = startIteration + nProceededIters;
    }
}

template <typename algorithmFPType, CpuType cpu>
Status SGDmomentumTask<algorithmFPType, cpu>::setStartValue(NumericTable * inputArgument, NumericTable * minimum)
{
    SafeStatus safeStat;
    processByBlocks<cpu>(minimum->getNumberOfRows(), [=, &safeStat](size_t startOffset, size_t nRowsInBlock) {
        WriteRows<algorithmFPType, cpu, NumericTable> workValueBD(*minimum, startOffset, nRowsInBlock);
        DAAL_CHECK_BLOCK_STATUS_THR(workValueBD);
        algorithmFPType * workArray = workValueBD.get();
        ReadRows<algorithmFPType, cpu, NumericTable> startValueBD(*inputArgument, startOffset, nRowsInBlock);
        DAAL_CHECK_BLOCK_STATUS_THR(startValueBD);
        const algorithmFPType * startValueArray = startValueBD.get();
        if (workArray != startValueArray)
        {
            int result = daal::services::internal::daal_memcpy_s(workArray, nRowsInBlock * sizeof(algorithmFPType), startValueArray,
                                                                 nRowsInBlock * sizeof(algorithmFPType));
            if (result) safeStat.add(services::Status(services::ErrorMemoryCopyFailedInternal));
        }
    });
    return safeStat.detach();
}

template <typename algorithmFPType, CpuType cpu>
SGDmomentumTask<algorithmFPType, cpu>::SGDmomentumTask(size_t batchSize_, size_t nTerms_, NumericTable * resultTable,
                                                       NumericTable * batchIndicesTable, NumericTable * pastUpdateResult,
                                                       NumericTable * lastIterationResultNT, Parameter<momentum> * parameter)
    : batchSize(batchSize_),
      nTerms(nTerms_),
      minimimWrapper(resultTable, EmptyDeleter()),
      pastUpdate(pastUpdateResult, EmptyDeleter()),
      lastIterationResult(lastIterationResultNT, EmptyDeleter()),
      startIteration(0),
      nProceededIters(0)
{}

template <typename algorithmFPType, CpuType cpu>
Status SGDmomentumTask<algorithmFPType, cpu>::init(NumericTable * batchIndicesTable, NumericTable * resultTable, Parameter<momentum> * parameter,
                                                   NumericTable * pastUpdateInput, NumericTable * lastIterationInput)
{
    indicesStatus = (batchIndicesTable ? user : (batchSize < nTerms ? random : all));
    Status s;
    if (indicesStatus == user || indicesStatus == random)
    {
        ntBatchIndices.reset(new HomogenNumericTableCPU<int, cpu>(NULL, batchSize, 1, s));
        DAAL_CHECK_MALLOC(ntBatchIndices.get());
    }

    size_t argumentSize = resultTable->getNumberOfRows();
    if (!parameter->optionalResultRequired)
    {
        SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > pastUpdateCpuNt(
            new HomogenNumericTableCPU<algorithmFPType, cpu>(1, argumentSize, s));
        pastUpdateCpuNt->assign(0.0);
        pastUpdate = pastUpdateCpuNt;
        return Status();
    }

    if (lastIterationInput != nullptr)
    {
        ReadRows<int, cpu, NumericTable> lastIterationInputBD(lastIterationInput, 0, 1);
        const int * lastIterationInputArray = lastIterationInputBD.get();
        startIteration                      = lastIterationInputArray[0];
    }

    if (pastUpdateInput != nullptr)
    {
        if (pastUpdateInput != pastUpdate.get())
        {
            SafeStatus safeStat;
            /* copy optional input ot optional result */
            processByBlocks<cpu>(argumentSize, [=, &safeStat](size_t startOffset, size_t nRowsInBlock) {
                WriteOnlyRows<algorithmFPType, cpu, NumericTable> pastUpdateBD(*pastUpdate, startOffset, nRowsInBlock);
                DAAL_CHECK_BLOCK_STATUS_THR(pastUpdateBD);
                algorithmFPType * pastUpdateArray = pastUpdateBD.get();
                ReadRows<algorithmFPType, cpu, NumericTable> pastUpdateInputBD(*pastUpdateInput, startOffset, nRowsInBlock);
                DAAL_CHECK_BLOCK_STATUS_THR(pastUpdateInputBD);
                const algorithmFPType * pastUpdateInputArray = pastUpdateInputBD.get();
                if (pastUpdateArray != pastUpdateInputArray)
                {
                    int result = daal::services::internal::daal_memcpy_s(pastUpdateArray, nRowsInBlock * sizeof(algorithmFPType),
                                                                         pastUpdateInputArray, nRowsInBlock * sizeof(algorithmFPType));
                    if (result) safeStat.add(services::Status(services::ErrorMemoryCopyFailedInternal));
                }
            });
            return safeStat.detach();
        }
    }
    else /* empty optional input, set optional result to zero */
    {
        SafeStatus safeStat;
        processByBlocks<cpu>(argumentSize, [=, &safeStat](size_t startOffset, size_t nRowsInBlock) {
            WriteOnlyRows<algorithmFPType, cpu, NumericTable> pastUpdateBD(*pastUpdate, startOffset, nRowsInBlock);
            DAAL_CHECK_BLOCK_STATUS_THR(pastUpdateBD);
            algorithmFPType * pastUpdateArray = pastUpdateBD.get();
            for (size_t i = 0; i < nRowsInBlock; i++)
            {
                pastUpdateArray[i] = 0.0;
            }
        });
        return safeStat.detach();
    }
    return Status();
}

} // namespace internal
} // namespace sgd
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

#endif
