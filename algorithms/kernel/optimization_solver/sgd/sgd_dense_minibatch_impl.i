/* file: sgd_dense_minibatch_impl.i */
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
//  Implementation of sgd miniBatch algorithm
//
// Mu Li, Tong Zhang, Yuqiang Chen, Alexander J. Smola Efficient Mini-batch Training for Stochastic Optimization
//--
*/

#ifndef __SGD_DENSE_MINIBATCH_IMPL_I__
#define __SGD_DENSE_MINIBATCH_IMPL_I__

#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "service_math.h"
#include "service_utils.h"
#include "service_rng.h"

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
 *  \brief Kernel for SGD miniBatch calculation
 */
template<typename algorithmFPType, CpuType cpu>
void SGDKernel<algorithmFPType, miniBatch, cpu>::compute(NumericTable *inputArgument, NumericTable *minimum, NumericTable *nIterations,
    Parameter<miniBatch> *parameter, NumericTable *learningRateSequence,
    NumericTable *batchIndices, OptionalArgument *optionalArgument, OptionalArgument *optionalResult)
{
    const size_t argumentSize = inputArgument->getNumberOfRows();
    const size_t maxIterations = parameter->nIterations;
    const size_t L = parameter->innerNIterations;
    const size_t batchSize     = parameter->batchSize;

    /* if maxIterations == 0, set result as start point, the number of executed iters to 0 */
    if(maxIterations == 0 || L == 0)
    {
        SGDMiniBatchTask<algorithmFPType, cpu> task(
            this->_errors,
            argumentSize,
            minimum,
            inputArgument,
            nIterations
        );
        task.nProceededIterations[0] = 0;
        return;
    }

    SharedPtr<sum_of_functions::Batch> function = parameter->function;
    const size_t nTerms = function->sumOfFunctionsParameter->numberOfTerms;
    NumericTable *conservativeSequence = parameter->conservativeSequence.get();

    SGDMiniBatchTask<algorithmFPType, cpu> task(
        this->_errors,
        batchSize,
        argumentSize,
        maxIterations,
        nTerms,
        minimum,
        inputArgument,
        learningRateSequence,
        conservativeSequence,
        nIterations,
        batchIndices
    );
    if(this->_errors->size() != 0) { return; }

    NumericTablePtr previousArgument = function->sumOfFunctionsInput->get(sum_of_functions::argument);
    function->sumOfFunctionsInput->set(sum_of_functions::argument, task.ntWorkValue);

    double accuracyThreshold = parameter->accuracyThreshold;
    double pointNorm = 0, gradientNorm = 0, gradientThreshold = 0, one = 1.0;

    BlockDescriptor<algorithmFPType> gradientBlock;
    NumericTablePtr ntGradient;
    algorithmFPType learningRate, consCoeff;

    NumericTablePtr previousBatchIndices = function->sumOfFunctionsParameter->batchIndices;
    function->sumOfFunctionsParameter->batchIndices = task.ntBatchIndices;

    ReadRows<int, cpu> predefinedBatchIndicesBD(batchIndices, 0, maxIterations);
    using namespace iterative_solver::internal;
    RngTask<int, cpu> rngTask(predefinedBatchIndicesBD.get(), batchSize);
    if((!batchIndices) && !rngTask.init(optionalArgument, nTerms, parameter->seed, sgd::rngState))
    {
        this->_errors->add(ErrorMemoryAllocationFailed);
        return;
    }

    size_t epoch;
    for(epoch = 0; epoch < maxIterations; epoch++)
    {
        if(task.indicesStatus == user || task.indicesStatus == random)
        {
            task.ntBatchIndices->setArray(const_cast<int*>(rngTask.get(*this->_errors, RngTask<int, cpu>::eUniformWithoutReplacement)));
        }

        function->computeNoThrow();
        if(function->getErrors()->size() != 0) {this->_errors->add(function->getErrors()->getErrors()); break;}

        ntGradient = function->getResult()->get(objective_function::gradientIdx);
        ntGradient->getBlockOfRows(0, argumentSize, readOnly, gradientBlock);

        algorithmFPType *gradient = gradientBlock.getBlockPtr();
        pointNorm    = vectorNorm(task.workValue, argumentSize);
        gradientNorm = vectorNorm(gradient, argumentSize);

        gradientThreshold = accuracyThreshold * daal::internal::Math<algorithmFPType,cpu>::sMax(one, pointNorm);
        if(gradientNorm <= gradientThreshold) { ntGradient->releaseBlockOfRows(gradientBlock); break; }

        (task.learningRateLength > 1) ? learningRate = task.learningRateArray[epoch] : learningRate = task.learningRateArray[0];
        (task.consCoeffsLength > 1) ? consCoeff = task.consCoeffsArray[epoch] : consCoeff = task.consCoeffsArray[0];

        daal_memcpy_s(task.prevWorkValue, argumentSize * sizeof(algorithmFPType), task.workValue, argumentSize * sizeof(algorithmFPType));
        task.makeStep(gradient, learningRate, consCoeff, argumentSize);

        ntGradient->releaseBlockOfRows(gradientBlock);

        for(size_t inner = 1; inner < L; inner++)
        {
            function->computeNoThrow();
            if(function->getErrors()->size() != 0)
            {
                this->_errors->add(function->getErrors()->getErrors());
                task.nProceededIterations[0] = (int)epoch;
                function->sumOfFunctionsParameter->batchIndices = previousBatchIndices;
                function->sumOfFunctionsInput->set(sum_of_functions::argument, previousArgument);
                return;
            }
            ntGradient = function->getResult()->get(objective_function::gradientIdx);
            ntGradient->getBlockOfRows(0, argumentSize, readOnly, gradientBlock);

            task.makeStep(gradient, learningRate, consCoeff, argumentSize);

            ntGradient->releaseBlockOfRows(gradientBlock);
        }
    }
    task.nProceededIterations[0] = (int)epoch;
    function->sumOfFunctionsParameter->batchIndices = previousBatchIndices;
    function->sumOfFunctionsInput->set(sum_of_functions::argument, previousArgument);

    if(parameter->optionalResultRequired && !rngTask.save(optionalResult, sgd::rngState, *this->_errors))
    {
        this->_errors->add(ErrorMemoryAllocationFailed);
        return;
    }
}

template<typename algorithmFPType, CpuType cpu>
void SGDMiniBatchTask<algorithmFPType, cpu>::makeStep(
    const algorithmFPType *gradient,
    algorithmFPType learningRate,
    algorithmFPType consCoeff,
    size_t argumentSize)
{
    for(size_t j = 0; j < argumentSize; j++)
    {
        workValue[j] = workValue[j] - learningRate * (gradient[j] + consCoeff * (workValue[j] - prevWorkValue[j]));
    }
}

template<typename algorithmFPType, CpuType cpu>
SGDMiniBatchTask<algorithmFPType, cpu>::SGDMiniBatchTask(
    const services::KernelErrorCollectionPtr& errors_,
    size_t argumentSize_,
    NumericTable *resultTable,
    NumericTable *startValueTable,
    NumericTable *nIterationsTable
) :
    _errors(errors_),
    argumentSize(argumentSize_),
    mtWorkValue(resultTable),
    mtNIterations(nIterationsTable),
    workValue(NULL),
    nProceededIterations(NULL),
    learningRateArray(NULL),
    consCoeffsArray(NULL),
    prevWorkValue(NULL)
{
    mtWorkValue.getBlockOfRows(0, argumentSize, &workValue);
    mtNIterations.getBlockOfRows(0, 1, &nProceededIterations);
    setStartValue(startValueTable);
}

template<typename algorithmFPType, CpuType cpu>
SGDMiniBatchTask<algorithmFPType, cpu>::~SGDMiniBatchTask()
{
    mtNIterations.release();
    mtWorkValue.release();

    mtConsCoeffs.release();
    mtLearningRate.release();

    if(prevWorkValue)
        daal_free(prevWorkValue);

    if(indicesStatus == user)
    {
        mtPredefinedBatchIndices.release();
    }
}

template<typename algorithmFPType, CpuType cpu>
void SGDMiniBatchTask<algorithmFPType, cpu>::setStartValue(NumericTable *startValueTable)
{
    BlockMicroTable<algorithmFPType, readOnly, cpu> mtStartValue(startValueTable);
    algorithmFPType *startValueArray;
    mtStartValue.getBlockOfRows(0, argumentSize, &startValueArray);
    daal_memcpy_s(workValue, argumentSize * sizeof(algorithmFPType), startValueArray, argumentSize * sizeof(algorithmFPType));
    mtStartValue.release();
}

template<typename algorithmFPType, CpuType cpu>
SGDMiniBatchTask<algorithmFPType, cpu>::SGDMiniBatchTask(
    const services::KernelErrorCollectionPtr& errors_,
    size_t batchSize_,
    size_t argumentSize_,
    size_t maxIterations_,
    size_t nTerms_,
    NumericTable *resultTable,
    NumericTable *startValueTable,
    NumericTable *learningRateSequenceTable,
    NumericTable *conservativeSequenceTable,
    NumericTable *nIterationsTable,
    NumericTable *batchIndicesTable
) :
    _errors(errors_),
    batchSize(batchSize_),
    argumentSize(argumentSize_),
    maxIterations(maxIterations_),
    nTerms(nTerms_),
    mtWorkValue(resultTable),
    mtLearningRate(learningRateSequenceTable),
    mtConsCoeffs(conservativeSequenceTable),
    mtNIterations(nIterationsTable),
    mtPredefinedBatchIndices(batchIndicesTable),
    workValue(NULL),
    nProceededIterations(NULL),
    learningRateArray(NULL),
    consCoeffsArray(NULL),
    prevWorkValue(NULL)
{
    mtWorkValue.getBlockOfRows(0, argumentSize, &workValue);
    setStartValue(startValueTable);

    ntWorkValue = SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu>>(new HomogenNumericTableCPU<algorithmFPType, cpu>(workValue, 1, argumentSize));

    mtLearningRate.getBlockOfRows(0, 1, &learningRateArray);
    learningRateLength = mtLearningRate.getFullNumberOfColumns();

    mtConsCoeffs.getBlockOfRows(0, 1, &consCoeffsArray);
    consCoeffsLength = mtConsCoeffs.getFullNumberOfColumns();

    mtNIterations.getBlockOfRows(0, 1, &nProceededIterations);
    nProceededIterations[0] = 0;

    prevWorkValue = (algorithmFPType *) daal_malloc(argumentSize * sizeof(algorithmFPType));

    if(batchIndicesTable != NULL)
    {
        indicesStatus = user;
    }
    else
    {
        if(batchSize < nTerms)
        {
            indicesStatus = random;
        }
        else
        {
            indicesStatus = all;
        }
    }

    if(indicesStatus == user || indicesStatus == random)
    {
        ntBatchIndices = SharedPtr<HomogenNumericTableCPU<int, cpu>>(new HomogenNumericTableCPU<int, cpu>(NULL, batchSize, 1));
    }
    else if(indicesStatus == all)
    {
        ntBatchIndices = SharedPtr<HomogenNumericTableCPU<int, cpu>>();
    }
}

} // namespace daal::internal
} // namespace sgd
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

#endif
