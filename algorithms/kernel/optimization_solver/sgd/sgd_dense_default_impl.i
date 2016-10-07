/* file: sgd_dense_default_impl.i */
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
#include "service_numeric_table.h"
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
void SGDKernel<algorithmFPType, defaultDense, cpu>::compute(NumericTable *inputArgument, NumericTable *minimum, NumericTable *nIterations,
        Parameter<defaultDense> *parameter, NumericTable *learningRateSequence,
        NumericTable *batchIndices, OptionalArgument *optionalArgument, OptionalArgument *optionalResult)
{
    const size_t nRows = inputArgument->getNumberOfRows();

    //init workValue
    {
        processByBlocks<cpu>(minimum->getNumberOfRows(), this->_errors.get(),  [ = ](size_t startOffset, size_t nRowsInBlock)
        {
            WriteRows<algorithmFPType, cpu, NumericTable> workValueBD(*minimum, startOffset, nRowsInBlock);
            algorithmFPType *minArray = workValueBD.get();
            ReadRows<algorithmFPType, cpu, NumericTable> startValueBD(*inputArgument, startOffset, nRowsInBlock);
            const algorithmFPType *startValueArray = startValueBD.get();
            if( minArray != startValueArray )
            {
                daal_memcpy_s(minArray, nRowsInBlock * sizeof(algorithmFPType), startValueArray, nRowsInBlock * sizeof(algorithmFPType));
            }
        });
    }

    const size_t maxIterations = parameter->nIterations;
    /* if maxIterations == 0, set result as start point, the number of executed iters to 0 */
    if(maxIterations == 0)
    {
        WriteRows<int, cpu, NumericTable> nIterationsBD(*nIterations, 0, 1);
        int *nProceededIterations = nIterationsBD.get();
        nProceededIterations[0] = 0;
        return;
    }

    sum_of_functions::BatchPtr function = parameter->function;

    const size_t nTerms = function->sumOfFunctionsParameter->numberOfTerms;
    ReadRows<int, cpu, NumericTable> predefinedBatchIndicesBD(batchIndices, 0, maxIterations);
    const bool bGenerateAllIndices = !(parameter->batchIndices || parameter->optionalResultRequired);
    TSmartPtr<int, cpu> aPredefinedBatchIndices(bGenerateAllIndices ? maxIterations : 0);
    if(bGenerateAllIndices)
    {
        /*Get random indices for SGD from rng generator*/
        DAAL_CHECK(aPredefinedBatchIndices.get(), ErrorMemoryAllocationFailed);
        getRandom(0, nTerms, aPredefinedBatchIndices.get(), maxIterations, parameter->seed);
    }
    using namespace iterative_solver::internal;
    const int *predefinedBatchIndices = predefinedBatchIndicesBD.get() ? predefinedBatchIndicesBD.get() : aPredefinedBatchIndices.get();
    RngTask<int, cpu> rngTask(predefinedBatchIndices, 1);
    if((!predefinedBatchIndices) && !rngTask.init(optionalArgument, nTerms, parameter->seed, sgd::rngState))
    {
        this->_errors->add(ErrorMemoryAllocationFailed);
        return;
    }

    SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu>> ntBatchIndices(new daal::internal::HomogenNumericTableCPU<int, cpu>(NULL, 1, 1));

    SharedPtr<NumericTable> minimimWrapper(minimum, EmptyDeleter<NumericTable>());
    function->sumOfFunctionsParameter->batchIndices = ntBatchIndices;
    function->sumOfFunctionsInput->set(sum_of_functions::argument, minimimWrapper);

    ReadRows<algorithmFPType, cpu, NumericTable> learningRateBD(*learningRateSequence, 0, 1);
    const algorithmFPType *learningRateArray = learningRateBD.get();
    const size_t learningRateLength = learningRateSequence->getNumberOfColumns();
    const double accuracyThreshold = parameter->accuracyThreshold;

    WriteRows<int, cpu, NumericTable> nIterationsBD(*nIterations, 0, 1);
    int *nProceededIterations = nIterationsBD.get();
    *nProceededIterations = (int)maxIterations;

    for(size_t epoch = 0; epoch < maxIterations; epoch++)
    {
        ntBatchIndices->setArray(const_cast<int *>(rngTask.get(*this->_errors)));
        function->computeNoThrow();
        if(function->getErrors()->size() != 0)
        {
            this->_errors->add(function->getErrors()->getErrors());
            *nProceededIterations = (int)epoch;
            return;
        }

        NumericTable *gradient = function->getResult()->get(objective_function::gradientIdx).get();
        if(maxIterations != 1)
        {
            const algorithmFPType pointNorm = vectorNorm(minimum);
            const algorithmFPType gradientNorm = vectorNorm(gradient);
            const algorithmFPType one = 1.0;
            const algorithmFPType gradientThreshold = accuracyThreshold * daal::internal::Math<algorithmFPType, cpu>::sMax(one, pointNorm);
            if(gradientNorm < gradientThreshold)
            {
                *nProceededIterations = (int)epoch;
                break;
            }
        }

        const algorithmFPType learningRate = (learningRateLength > 1 ? learningRateArray[epoch] : learningRateArray[0]);

        processByBlocks<cpu>(nRows, this->_errors.get(), [ = ](size_t startOffset, size_t nRowsInBlock)
        {
            WriteRows<algorithmFPType, cpu, NumericTable> workValueBD(*minimum, startOffset, nRowsInBlock);
            algorithmFPType *workLocal = workValueBD.get();
            ReadRows<algorithmFPType, cpu, NumericTable> ntGradientBD(*gradient, startOffset, nRowsInBlock);
            const algorithmFPType *gradientLocal = ntGradientBD.get();
            PRAGMA_SIMD_ASSERT
            for(int j = 0; j < nRowsInBlock; j++)
            {
                workLocal[j] = workLocal[j] - learningRate * gradientLocal[j];
            }
        },
        256);
    }
    if(parameter->optionalResultRequired && !rngTask.save(optionalResult, sgd::rngState, *this->_errors))
    {
        this->_errors->add(ErrorMemoryAllocationFailed);
        return;
    }
}

} // namespace daal::internal

} // namespace sgd

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
