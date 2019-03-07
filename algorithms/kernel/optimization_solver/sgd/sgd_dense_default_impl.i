/* file: sgd_dense_default_impl.i */
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
//  Implementation of sgd algorithm
//--
*/

#ifndef __SGD_DENSE_DEFAULT_IMPL_I__
#define __SGD_DENSE_DEFAULT_IMPL_I__

#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "service_math.h"
#include "service_utils.h"
#include "iterative_solver_kernel.h"
#include "threading.h"

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
template<typename algorithmFPType, CpuType cpu>
services::Status SGDKernel<algorithmFPType, defaultDense, cpu>::compute(HostAppIface* pHost,
    NumericTable *inputArgument, NumericTable *minimum, NumericTable *nIterations,
    Parameter<defaultDense> *parameter, NumericTable *learningRateSequence,
    NumericTable *batchIndices, OptionalArgument *optionalArgument, OptionalArgument *optionalResult, engines::BatchBase &engine)
{
    const size_t nRows = inputArgument->getNumberOfRows();
    SafeStatus safeStat;
    //init workValue
    {
        processByBlocks<cpu>(minimum->getNumberOfRows(), [=, &safeStat](size_t startOffset, size_t nRowsInBlock)
        {
            WriteRows<algorithmFPType, cpu, NumericTable> workValueBD(*minimum, startOffset, nRowsInBlock);
            DAAL_CHECK_BLOCK_STATUS_THR(workValueBD);
            algorithmFPType *minArray = workValueBD.get();
            ReadRows<algorithmFPType, cpu, NumericTable> startValueBD(*inputArgument, startOffset, nRowsInBlock);
            DAAL_CHECK_BLOCK_STATUS_THR(startValueBD);
            const algorithmFPType *startValueArray = startValueBD.get();
            if( minArray != startValueArray )
            {
                daal_memcpy_s(minArray, nRowsInBlock * sizeof(algorithmFPType), startValueArray, nRowsInBlock * sizeof(algorithmFPType));
            }
        });
        if(!safeStat)
            return safeStat.detach();
    }

    const size_t nIter = parameter->nIterations;
    /* if nIter == 0, set result as start point, the number of executed iters to 0 */
    if(nIter == 0)
    {
        WriteRows<int, cpu, NumericTable> nIterationsBD(*nIterations, 0, 1);
        int *nProceededIterations = nIterationsBD.get();
        nProceededIterations[0] = 0;
        return Status();
    }

    NumericTable *lastIterationInput = (optionalArgument) ? NumericTable::cast(optionalArgument->get(iterative_solver::lastIteration)).get() : nullptr;
    NumericTable *lastIterationResult = (optionalResult) ? NumericTable::cast(optionalResult->get(iterative_solver::lastIteration)).get() : nullptr;

    sum_of_functions::BatchPtr function = parameter->function;

    const size_t nTerms = function->sumOfFunctionsParameter->numberOfTerms;
    ReadRows<int, cpu, NumericTable> predefinedBatchIndicesBD(batchIndices, 0, nIter);
    const bool bGenerateAllIndices = !(parameter->batchIndices || parameter->optionalResultRequired);
    TArray<int, cpu> aPredefinedBatchIndices(bGenerateAllIndices ? nIter : 0);
    if(bGenerateAllIndices)
    {
        /*Get random indices for SGD from rng generator*/
        DAAL_CHECK_MALLOC(aPredefinedBatchIndices.get());
        getRandom(0, nTerms, aPredefinedBatchIndices.get(), nIter, engine);
    }
    using namespace iterative_solver::internal;
    const int *predefinedBatchIndices = predefinedBatchIndicesBD.get() ? predefinedBatchIndicesBD.get() : aPredefinedBatchIndices.get();
    RngTask<int, cpu> rngTask(predefinedBatchIndices, 1);
    DAAL_CHECK_MALLOC((predefinedBatchIndices) || rngTask.init(nTerms, engine));

    Status s;
    SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu>> ntBatchIndices(new daal::internal::HomogenNumericTableCPU<int, cpu>(NULL, 1, 1, s));

    NumericTablePtr minimimWrapper(minimum, EmptyDeleter());
    function->sumOfFunctionsParameter->batchIndices = ntBatchIndices;
    function->sumOfFunctionsInput->set(sum_of_functions::argument, minimimWrapper);

    ReadRows<algorithmFPType, cpu, NumericTable> learningRateBD(*learningRateSequence, 0, learningRateSequence->getNumberOfRows());
    const algorithmFPType *learningRateArray = learningRateBD.get();
    const size_t learningRateLength = learningRateSequence->getNumberOfColumns();
    const double accuracyThreshold = parameter->accuracyThreshold;

    WriteRows<int, cpu, NumericTable> nIterationsBD(*nIterations, 0, 1);
    int *nProceededIterations = nIterationsBD.get();
    *nProceededIterations = (int)nIter;

    size_t startIteration = 0, epoch = 0, nProceededIters = 0;
    if(lastIterationInput != nullptr)
    {
        ReadRows<int, cpu, NumericTable> lastIterationInputBD(lastIterationInput, 0, 1);
        const int *lastIterationInputArray = lastIterationInputBD.get();
        startIteration = lastIterationInputArray[0];
    }

    services::internal::HostAppHelper host(pHost, 10);
    for(epoch = startIteration; s.ok() && (epoch < (startIteration + nIter)); epoch++)
    {
        const int* pValues = nullptr;
        s = rngTask.get(pValues);
        if(s)
        {
            ntBatchIndices->setArray(const_cast<int *>(pValues), ntBatchIndices->getNumberOfRows());
            s = function->computeNoThrow();
        }
        if(!s || host.isCancelled(s, 1))
        {
            nProceededIterations[0] = nProceededIters;
            return s;
        }

        NumericTable *gradient = function->getResult()->get(objective_function::gradientIdx).get();
        if(nIter != 1)
        {
            algorithmFPType pointNorm, gradientNorm;
            s = vectorNorm(minimum, pointNorm);
            s |= vectorNorm(gradient, gradientNorm);
            if(!s)
                break;

            const algorithmFPType one(1.0);
            const algorithmFPType gradientThreshold = accuracyThreshold * daal::internal::Math<algorithmFPType, cpu>::sMax(one, pointNorm);
            if(gradientNorm < gradientThreshold)
            {
                nProceededIterations[0] = (int)nProceededIters;
                break;
            }
        }

        const algorithmFPType learningRate = learningRateArray[epoch % learningRateLength];

        processByBlocks<cpu>(nRows, [=, &safeStat](size_t startOffset, size_t nRowsInBlock)
        {
            WriteRows<algorithmFPType, cpu, NumericTable> workValueBD(*minimum, startOffset, nRowsInBlock);
            DAAL_CHECK_BLOCK_STATUS_THR(workValueBD);
            algorithmFPType *workLocal = workValueBD.get();
            ReadRows<algorithmFPType, cpu, NumericTable> ntGradientBD(*gradient, startOffset, nRowsInBlock);
            DAAL_CHECK_BLOCK_STATUS_THR(ntGradientBD);
            const algorithmFPType *gradientLocal = ntGradientBD.get();
            PRAGMA_VECTOR_ALWAYS
            for(int j = 0; j < nRowsInBlock; j++)
            {
                workLocal[j] = workLocal[j] - learningRate * gradientLocal[j];
            }
        },
        256);
        if(!safeStat)
            s |= safeStat.detach();
        nProceededIters++;
    }
    if(lastIterationResult)
    {
        WriteRows<int, cpu, NumericTable> lastIterationResultBD(lastIterationResult, 0, 1);
        int *lastIterationResultArray = lastIterationResultBD.get();
        lastIterationResultArray[0] = startIteration + nProceededIters;
    }
    return s;
}

} // namespace daal::internal

} // namespace sgd

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
