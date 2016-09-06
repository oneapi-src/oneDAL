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
 *  \brief Kernel for SGD calculation
 */
template<typename algorithmFPType, CpuType cpu>
void SGDKernel<algorithmFPType, defaultDense, cpu>::compute(NumericTable *inputArgument, NumericTable *minimum, NumericTable *nIterations,
                                                            Parameter<defaultDense> *parameter, NumericTable *learningRateSequence,
                                                            NumericTable *batchIndices)
{
    const size_t nFeatures = inputArgument->getNumberOfColumns();

    WriteRows<algorithmFPType, cpu, NumericTable> workValueBD(*minimum, 0, 1);
    algorithmFPType* workValue = workValueBD.get();

    //init workValue
    {
        ReadRows<algorithmFPType, cpu, NumericTable> startValueBD(*inputArgument, 0, 1);
        const algorithmFPType* startValueArray = startValueBD.get();
        if( workValue != startValueArray )
        {
            daal_memcpy_s(workValue, nFeatures * sizeof(algorithmFPType), startValueArray, nFeatures * sizeof(algorithmFPType));
        }
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

    /*Get random indices for SGD from parameter or from rng generator*/
    const bool isPredefinedbatchIndices = parameter->batchIndices;
    SmartPtr<cpu> aPredefinedBatchIndices(isPredefinedbatchIndices ? 0 : maxIterations * sizeof(int));
    ReadRows<int, cpu, NumericTable> predefinedBatchIndicesBD(batchIndices, 0, maxIterations);
    int *randomTerm = nullptr;
    if(isPredefinedbatchIndices)
    {
        randomTerm = const_cast<int*>(predefinedBatchIndicesBD.get());
    }
    else
    {
        randomTerm = (int*)aPredefinedBatchIndices.get();
        if(!randomTerm) { this->_errors->add(ErrorMemoryAllocationFailed); return; }
        getRandom(0, function->sumOfFunctionsParameter->numberOfTerms, randomTerm, maxIterations, parameter->seed);
    }
    SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu>> ntBatchIndices(new daal::internal::HomogenNumericTableCPU<int, cpu>(NULL, 1, 1));

    function->sumOfFunctionsParameter->batchIndices = ntBatchIndices;
    function->sumOfFunctionsInput->set(sum_of_functions::argument,
        NumericTablePtr(new daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>(workValue, nFeatures, 1)));

    ReadRows<algorithmFPType, cpu, NumericTable> learningRateBD(*learningRateSequence, 0, 1);
    const algorithmFPType *learningRateArray = learningRateBD.get();
    const size_t learningRateLength = learningRateSequence->getNumberOfColumns();
    const double accuracyThreshold = parameter->accuracyThreshold;

    WriteRows<int, cpu, NumericTable> nIterationsBD(*nIterations, 0, 1);
    int *nProceededIterations = nIterationsBD.get();
    *nProceededIterations = (int)maxIterations;

    for(size_t epoch = 0; epoch < maxIterations; epoch++)
    {
        ntBatchIndices->setArray(&(randomTerm[epoch]));
        function->computeNoThrow();
        if(function->getErrors()->size() != 0)
        {
            *nProceededIterations = (int)epoch;
            return;
        }

        auto ntGradient = function->getResult()->get(objective_function::resultCollection, objective_function::gradientIdx);
        ReadRows<algorithmFPType, cpu, NumericTable> ntGradientBD(*ntGradient, 0, 1);
        const algorithmFPType* gradient = ntGradientBD.get();
        const algorithmFPType pointNorm = vectorNorm(workValue, nFeatures);
        const algorithmFPType gradientNorm = vectorNorm(gradient, nFeatures);
        const algorithmFPType one = 1.0;
        const algorithmFPType gradientThreshold = accuracyThreshold * daal::internal::Math<algorithmFPType,cpu>::sMax(one, pointNorm);
        if(gradientNorm <= gradientThreshold)
        {
            *nProceededIterations = (int)epoch;
            break;
        }

        const algorithmFPType learningRate = (learningRateLength > 1 ? learningRateArray[epoch] : learningRateArray[0]);
        for(size_t j = 0; j < nFeatures; j++)
            workValue[j] = workValue[j] - learningRate * gradient[j];
    }
}

} // namespace daal::internal

} // namespace sgd

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
