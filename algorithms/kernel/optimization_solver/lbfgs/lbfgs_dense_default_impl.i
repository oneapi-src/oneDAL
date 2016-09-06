/* file: lbfgs_dense_default_impl.i */
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
//  Implementation of LBFGS algorithm
//--
*/

#ifndef __LBFGS_DENSE_DEFAULT_IMPL__
#define __LBFGS_DENSE_DEFAULT_IMPL__

#include "service_blas.h"
#include "service_rng.h"

using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace lbfgs
{
namespace internal
{

template<CpuType cpu>
bool getOptionalInputData(NumericTable* pCorrectionIndexInput, size_t& lastCorrectionIndex, size_t& lastIteration)
{
    if(!pCorrectionIndexInput)
        return false;
    ReadRows<int, cpu> bd(*pCorrectionIndexInput, 0, 1);
    const int* ptr = bd.get();
    lastCorrectionIndex = size_t(ptr[0]);
    lastIteration = size_t(ptr[1]);
    return true;
}

static size_t mod(size_t a, size_t m) { return (a - (a / m) * m); }

/**
 * \brief Kernel for LBFGS calculation
 */
template<typename algorithmFPType, CpuType cpu>
void LBFGSKernel<algorithmFPType, defaultDense, cpu>::compute(NumericTable* correctionPairsInput, NumericTable* correctionIndicesInput,
    NumericTable *inputArgument, NumericTable* averageArgLIterInput, OptionalArgument *optionalArgumentInput, NumericTable *correctionPairsResult,
    NumericTable *correctionIndicesResult, NumericTable *minimum, NumericTable *nIterationsNT, NumericTable *averageArgLIterResult,
    OptionalArgument *optionalArgumentResult, Parameter *parameter)
{
    size_t maxEpoch = parameter->nIterations;
    if(maxEpoch == 0)
    {
        /* Initialize the resulting argument of objective function with the input argument */
        LBFGSTaskBase<algorithmFPType, cpu> task(minimum, inputArgument);
        task.setToResult(correctionIndicesResult, nIterationsNT, optionalArgumentResult, 0, 0, 0);
        return;
    }
    const size_t L = parameter->L;
    const size_t m = parameter->m;
    const double accuracyThreshold = parameter->accuracyThreshold;

    sum_of_functions::BatchPtr gradientFunction = parameter->function;
    const size_t nTerms = gradientFunction->sumOfFunctionsParameter->numberOfTerms;

    const size_t batchSize = ((parameter->batchSize < nTerms) ? parameter->batchSize : nTerms);
    const size_t correctionPairBatchSize = ((parameter->correctionPairBatchSize < nTerms) ? parameter->correctionPairBatchSize : nTerms);

    LBFGSTask<algorithmFPType, cpu> task(inputArgument, correctionPairsInput, averageArgLIterInput, optionalArgumentInput, parameter, minimum,
        averageArgLIterResult, correctionPairsResult, nTerms, batchSize, correctionPairBatchSize, _errors);
    if (this->_errors->size() != 0) { return; }

    NumericTablePtr argumentTable(new HomogenNumericTableCPU<algorithmFPType, cpu>(task.argument, task.argumentSize, 1));
    gradientFunction->sumOfFunctionsParameter->batchIndices     = task.ntBatchIndices;
    gradientFunction->sumOfFunctionsParameter->resultsToCompute = objective_function::gradient;
    gradientFunction->sumOfFunctionsInput->set(sum_of_functions::argument, argumentTable);

    sum_of_functions::BatchPtr hessianFunction = gradientFunction->clone();
    hessianFunction->sumOfFunctionsParameter->batchIndices = task.ntCorrectionPairBatchIndices;
    hessianFunction->sumOfFunctionsParameter->resultsToCompute = objective_function::hessian;
    hessianFunction->sumOfFunctionsInput->set(sum_of_functions::argument, task.argumentLCurTable);

    BlockDescriptor<algorithmFPType> gradientBlock;
    BlockDescriptor<algorithmFPType> hessianBlock;
    NumericTablePtr ntGradient;
    NumericTablePtr ntHessian;

    algorithmFPType invL = (algorithmFPType)1.0 / (algorithmFPType)L;

    algorithmFPType *argument      = task.argument;
    algorithmFPType *argumentLCur  = task.argumentLCur;
    algorithmFPType *argumentLPrev = task.argumentLPrev;

    size_t lastCorrectionIndex = 0;
    size_t lastIteration = 0;
    size_t t = 0;
    size_t epoch = 0;
    size_t correctionIndex = m - 1;
    if(getOptionalInputData<cpu>(correctionIndicesInput, lastCorrectionIndex, lastIteration))
    {
        epoch = lastIteration + 1;
        t = epoch / L;
        maxEpoch += epoch;
        correctionIndex = lastCorrectionIndex;
    }
    size_t nLIterations = maxEpoch / L;
    size_t curIteration = 0;
    size_t iPredefinedIndicesCorrectionRow = 0;

    for (; t < nLIterations;)
    {
        for(; epoch < (t + 1) * L; ++epoch, ++curIteration)
        {
            for (size_t j = 0; j < task.argumentSize; j++)
            {
                argumentLCur[j] += argument[j];
            }

            if(!task.updateArgument(curIteration, t, epoch, m, correctionIndex, nTerms, batchSize,
                                accuracyThreshold, gradientFunction, ntGradient, gradientBlock,
                                argument))
            {
                task.setToResult(correctionIndicesResult, nIterationsNT, optionalArgumentResult, curIteration, epoch, correctionIndex);
                return;
            }
        }

        for (size_t j = 0; j < task.argumentSize; j++)
        {
            argumentLCur[j] *= invL;
        }

        t++;
        if (t >= 2)
        {
            /* Compute new correction pair */
            correctionIndex = mod(correctionIndex + 1, m);

            task.updateCorrectionPairBatchIndices(iPredefinedIndicesCorrectionRow, nTerms, correctionPairBatchSize);
            iPredefinedIndicesCorrectionRow++;

            hessianFunction->computeNoThrow();
            if(hessianFunction->getErrors()->size() != 0) { task.setToResult(correctionIndicesResult, nIterationsNT, optionalArgumentResult,
                                                                             curIteration, epoch, correctionIndex); return; }
            if (task.correctionPairBatchIndicesStatus == user) { task.mtCorrectionPairBatchIndices.release(); }

            ntHessian = hessianFunction->getResult()->get(objective_function::resultCollection, objective_function::hessianIdx);
            ntHessian->getBlockOfRows(0, task.argumentSize, readOnly, hessianBlock);
            task.computeCorrectionPair(correctionIndex, hessianBlock.getBlockPtr());
            ntHessian->releaseBlockOfRows(hessianBlock);
        }
        for (size_t j = 0; j < task.argumentSize; j++)
        {
            argumentLPrev[j] = argumentLCur[j];
            argumentLCur[j] = 0.0;
        }
    }

    for(; epoch < maxEpoch; ++epoch, ++curIteration)
    {
        if(!task.updateArgument(curIteration, t, epoch, m, correctionIndex, nTerms, batchSize,
            accuracyThreshold, gradientFunction, ntGradient, gradientBlock, argument))
        {
            task.setToResult(correctionIndicesResult, nIterationsNT, optionalArgumentResult, curIteration, epoch, correctionIndex);
            return;
        }
    }

    task.setToResult(correctionIndicesResult, nIterationsNT, optionalArgumentResult, curIteration, epoch - 1, correctionIndex);
}

/**
 * Computes the dot product of two vectors
 *
 * \param[in] n     Number of elements in each input vector
 * \param[in] x     Array that contains elements of the first input vector
 * \param[in] y     Array that contains elements of the second input vector
 * \return Resulting dot product
 */
template<typename algorithmFPType, CpuType cpu>
algorithmFPType dotProduct(size_t n, const algorithmFPType *x, const algorithmFPType *y)
{
    algorithmFPType dot = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        dot += x[i] * y[i];
    }
    return dot;
}

/**
 * Updates argument of the objective function
 *
 * \param[in] t                     Index of the outer loop of LBFGS algorithm
 * \param[in] epoch                 Index of the inner loop of LBFGS algorithm
 * \param[in] m                     Memory parameter of LBFGS
 * \param[in] correctionIndex       Index of starting correction pair in a cyclic buffer
 * \param[in] batchSize             Number of terms of objective function to be used in stochastic gradient
 *                                  computation
 * \param[in] accuracyThreshold     Accuracy of the LBFGS algorithm
 * \param[in] gradientFunction      Objective function for stochastic gradient computations
 * \param[in] ntGradient            Numeric table that stores the stochastic gradient
 * \param[in] gradientBlock         Block descriptor related to the stochastic gradient
 * \param[in] task                  Structure for storing data used in itermediate computations in LBFGS algorithm
 * \param[in] rng                   Random number generator
 * \param[in,out] argument          Array that contains the argument of objective function
 *
 * \return Flag. True if the argument was updated successfully. False, otherwise.
 */
template<typename algorithmFPType, CpuType cpu>
bool LBFGSTask<algorithmFPType, cpu>::updateArgument(size_t iIteration,
            size_t t, size_t epoch, size_t m, size_t correctionIndex, size_t nTerms,
            size_t batchSize, algorithmFPType accuracyThreshold,
            sum_of_functions::BatchPtr &gradientFunction,
            NumericTablePtr &ntGradient, BlockDescriptor<algorithmFPType> &gradientBlock,
            algorithmFPType *argument)
{
    const algorithmFPType one = 1.0;

    updateBatchIndices(iIteration, nTerms, batchSize, &(batchIndices), batchIndicesStatus,
                       mtBatchIndices, ntBatchIndices);

    gradientFunction->compute();
    if (gradientFunction->getErrors()->size() != 0) { return false; }
    if (batchIndicesStatus == user) { mtBatchIndices.release(); }

    ntGradient = gradientFunction->getResult()->get(objective_function::resultCollection, objective_function::gradientIdx);
    ntGradient->getBlockOfRows(0, 1, readOnly, gradientBlock);

    algorithmFPType *gradient = gradientBlock.getBlockPtr();

    /* Check accuracy */
    if(dotProduct<algorithmFPType, cpu>(this->argumentSize, gradient, gradient) <
        accuracyThreshold * daal::internal::Math<algorithmFPType,cpu>::sMax(one, dotProduct<algorithmFPType, cpu>(this->argumentSize, argument, argument)))
    {
        ntGradient->releaseBlockOfRows(gradientBlock);
        return false;
    }

    /* Get step length on this iteration */
    const algorithmFPType stepLengthVal = ((nStepLength > 1) ? stepLength[epoch] : stepLength[0]);

    if (t >= 2)
    {
        /* Compute H * gradient */
        twoLoopRecursion(m, correctionIndex, gradient);
    }

    /* Update argument */
    for (size_t j = 0; j < this->argumentSize; j++)
    {
        argument[j] -= stepLengthVal * gradient[j];
    }
    ntGradient->releaseBlockOfRows(gradientBlock);

    return true;
}

/**
 * Updates the array of objective function terms indices that are used in stochastic gradient
 * or Hessian matrix computations
 *
 * \param[in]  iPredefinedIndicesRow    Index of row of predefined indices to use
 * \param[in]  nTerms                   Full number of summands (terms) in objective function
 * \param[in]  batchSize                Number of terms of objective function to be used in stochastic gradient
 *                                      or Hessian matrix computations
 * \param[out] batchIndices             Array of indices of objective function terms that are used
 *                                      in stochastic gradient or Hessian matrix computations
 * \param[in]  batchIndicesStatus       Status of the indices of objective function terms
 * \param[in]  ntBatchIndices           Numeric table that stores the indices of objective function terms
 */
template<typename algorithmFPType, CpuType cpu>
void LBFGSTask<algorithmFPType, cpu>::updateBatchIndices(size_t iPredefinedIndicesRow,
            size_t nTerms, size_t batchSize, int **batchIndices, IndicesStatus batchIndicesStatus,
            daal::internal::BlockMicroTable<int, readOnly, cpu> &mtBatchIndices,
            services::SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu> > &ntBatchIndices)
{
    if (batchIndicesStatus == all)
    {
        return;
    }
    else if (batchIndicesStatus == user)
    {
        mtBatchIndices.getBlockOfRows(iPredefinedIndicesRow, 1, batchIndices);
        ntBatchIndices->setArray(*batchIndices);
    }
    else // if (batchIndicesStatus == random)
    {
        _rng.uniformWithoutReplacement(batchSize, 0, (int)nTerms, *batchIndices);
        _rngStateChanged = true;
    }
}

/**
 * Two-loop recursion algorithm that computes approximation of inverse Hessian matrix
 * multiplied by input gradient vector from a set of correction pairs (s(j), y(j)), j = 1,...,m.
 *
 * See Algorithm 7.4 in [2].
 *
 * \param[in]  m               Number of correction pairs
 * \param[in]  correctionIndex Index of starting correction pair in a cyclic buffer
 * \param[in,out] gradient     On input:  Gradient vector.
 *                             On output: iterative_solver::Result of two-loop recursion.
 */
template<typename algorithmFPType, CpuType cpu>
void LBFGSTask<algorithmFPType, cpu>::twoLoopRecursion(size_t m, size_t correctionIndex, algorithmFPType *gradient)
{
    size_t index = 0;

    for (size_t k = 0; k < m; k++)
    {
        index = mod(correctionIndex + m - 1 - k, m);
        const algorithmFPType *correctionSPtr = correctionS + index * this->argumentSize;
        const algorithmFPType *correctionYPtr = correctionY + index * this->argumentSize;

        alpha[index] = rho[index] * dotProduct<algorithmFPType, cpu>(this->argumentSize, correctionSPtr, gradient);

        for(size_t j = 0; j < this->argumentSize; j++)
        {
            gradient[j] -= alpha[index] * correctionYPtr[j];
        }
    }

    for (size_t k = 0; k < m; k++)
    {
        index = mod(correctionIndex + k, m);
        const algorithmFPType *correctionSPtr = correctionS + index * this->argumentSize;
        const algorithmFPType *correctionYPtr = correctionY + index * this->argumentSize;

        algorithmFPType beta = rho[index] * dotProduct<algorithmFPType, cpu>(this->argumentSize, correctionYPtr, gradient);

        for(size_t j = 0; j < this->argumentSize; j++)
        {
            gradient[j] += correctionSPtr[j] * (alpha[index] - beta);
        }
    }
}

/**
 * Computes the correction pair (s, y) and the corresponding value rho
 *
 * \param[out] correctionIndex  Index of correction pair to be changed
 * \param[in]  hessian          Approximation of Hessian matrix of the objective function on the current iteration
 *                              See formula (7.17) in [2]
 */
template<typename algorithmFPType, CpuType cpu>
void LBFGSTask<algorithmFPType, cpu>::computeCorrectionPair(size_t correctionIndex, algorithmFPType *hessian)
{
    algorithmFPType* s = correctionS + correctionIndex * this->argumentSize;
    for(size_t j = 0; j < this->argumentSize; j++)
    {
        s[j] = argumentLCur[j] - argumentLPrev[j];
    }

    algorithmFPType* y = correctionY + correctionIndex * this->argumentSize;
    char trans = 'N';
    algorithmFPType zero = 0.0;
    algorithmFPType one = 1.0;
    MKL_INT n = (MKL_INT)(this->argumentSize);
    MKL_INT ione = 1;
    Blas<algorithmFPType, cpu>::xgemv(&trans, &n, &n, &one, hessian, &n, s, &ione, &zero, y, &ione);

    rho[correctionIndex] = dotProduct<algorithmFPType, cpu>(this->argumentSize, s, y);
    if(rho[correctionIndex] != 0.0) // threshold
    {
        rho[correctionIndex] = 1.0 / rho[correctionIndex];
    }
}

/**
 * Creates structure for storing data used in itermediate computations in LBFGS algorithm
 * if number of iterations is 0
 *
 * \param[in] argumentTable     Numeric table that stores the argument of objective function
 * \param[in] startValueTable   Numeric table that stores the starting point, the initial argument of objective function
 */
template<typename algorithmFPType, CpuType cpu>
LBFGSTaskBase<algorithmFPType, cpu>::LBFGSTaskBase(NumericTable *argumentTable, NumericTable *startValueTable) :
mtArgument(argumentTable), argument(nullptr), argumentSize(argumentTable->getNumberOfColumns())
{
    /* Initialize work value with a start value provided by user */
    setStartArgument(startValueTable);
}

template<typename algorithmFPType, CpuType cpu>
LBFGSTaskBase<algorithmFPType, cpu>::~LBFGSTaskBase()
{
    if(argument) { mtArgument.release(); }
}

/**
* Sets the initial argument of objective function
*
* \param[in] startValueTable   Numeric table that stores the starting point, the initial argument of objective function
*/
template<typename algorithmFPType, CpuType cpu>
void LBFGSTaskBase<algorithmFPType, cpu>::setStartArgument(NumericTable *startValueTable)
{
    /* Initialize work value with a start value provided by user */
    mtArgument.getBlockOfRows(0, 1, &argument);

    BlockMicroTable<algorithmFPType, readOnly, cpu> mtStartValue(startValueTable);
    algorithmFPType *startValue;
    mtStartValue.getBlockOfRows(0, 1, &startValue);
    daal_memcpy_s(argument, argumentSize * sizeof(algorithmFPType), startValue, argumentSize * sizeof(algorithmFPType));
    mtStartValue.release();
}

template<typename algorithmFPType, CpuType cpu>
void LBFGSTaskBase<algorithmFPType, cpu>::setToResult(NumericTable *correctionIndicesResult, NumericTable *nIterationsNT, OptionalArgument *optionalArgumentResult,
                                                      size_t nIterations, size_t epoch, size_t corrIndex)
{
    BlockMicroTable<int, writeOnly, cpu> mtNIterations(nIterationsNT);
    int *nProceededIterations = NULL;
    mtNIterations.getBlockOfRows(0, 1, &nProceededIterations);
    *nProceededIterations = (int)nIterations;
    mtNIterations.release();

    if(correctionIndicesResult)
    {
        WriteRows<int, cpu> bd(*correctionIndicesResult, 0, 1);
        int* ptr = bd.get();
        ptr[0] = corrIndex;
        ptr[1] = epoch;
    }
}

/**
* Creates structure for storing data used in itermediate computations in LBFGS algorithm
*
* \param[in] input             Algorithm's input
* \param[in] parameter         Algorithm's parameter
* \param[in] result            Algorithm's result
* \param[in] nTerms            Full number of summands (terms) in objective function
* \param[in] batchSize         Number of terms to compute the stochastic gradient
* \param[in] correctionPairBatchSize           Number of terms to compute the sub-sampled Hessian
*                                              for correction pairs computation
* \param[in] _errors           Error collection of LBFGS algorithm
*/
template<typename algorithmFPType, CpuType cpu>
LBFGSTask<algorithmFPType, cpu>::LBFGSTask(NumericTable *inputArgument, NumericTable* correctionPairsInput, NumericTable* averageArgLIterInput,
    OptionalArgument *optionalArgumentInput, const Parameter *parameter, NumericTable *minimum, NumericTable *averageArgLIterResult,
    NumericTable *correctionPairsResult, size_t nTerms, size_t batchSize, size_t correctionPairBatchSize, services::KernelErrorCollectionPtr &_errors) :
    super(minimum, inputArgument),
    mtBatchIndices(parameter->batchIndices.get()), mtCorrectionPairBatchIndices(parameter->correctionPairBatchIndices.get()),
    mtStepLength(parameter->stepLengthSequence.get()), correctionPairs(nullptr), _errors(_errors),
    correctionS(NULL), correctionY(NULL),
    batchIndices(NULL), batchIndicesStatus(all),
    correctionPairBatchIndices(NULL), correctionPairBatchIndicesStatus(all),
    nStepLength(parameter->stepLengthSequence->getNumberOfColumns()),
    _rng(parameter->seed), _rngStateChanged(false), _rngStateRequired(false)
{
    initArgumentL(averageArgLIterInput, averageArgLIterResult, parameter);
    alpha = (algorithmFPType *)daal_malloc(parameter->m * sizeof(algorithmFPType));
    if(!argumentLCur || !argumentLPrev || !alpha)
    {
        this->_errors->add(ErrorMemoryAllocationFailed);
        return;
    }

    if(!initCorrectionPairs(correctionPairsInput, parameter, correctionPairsResult))
        return;

    /* Get step-length sequence */
    mtStepLength.getBlockOfRows(0, 1, &stepLength);

    /* Create numeric table for storing the average of work values for the last L iterations */
    argumentLCurTable = SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> >(
        new HomogenNumericTableCPU<algorithmFPType, cpu>(argumentLCur, this->argumentSize, 1));
    if(!argumentLCurTable)
    {
        this->_errors->add(ErrorMemoryAllocationFailed);
        return;
    }

    /* Initialize indices for objective function gradient computations */
    NumericTable *batchIndicesTable = parameter->batchIndices.get();
    if(batchSize < nTerms)
    {
        getBatchIndices(batchSize, batchIndicesTable, &batchIndices, mtBatchIndices, &batchIndicesStatus,
            ntBatchIndices);
        if(this->_errors->size() != 0) { return; }
    }
    /* Initialize indices for objective function Hessian computations */
    if(correctionPairBatchSize < nTerms)
    {
        getBatchIndices(correctionPairBatchSize, parameter->correctionPairBatchIndices.get(), &correctionPairBatchIndices,
            mtCorrectionPairBatchIndices, &correctionPairBatchIndicesStatus, ntCorrectionPairBatchIndices);
        if(this->_errors->size() != 0) { return; }
    }

    initRngState(optionalArgumentInput, parameter);
}


template<typename algorithmFPType, CpuType cpu>
LBFGSTask<algorithmFPType, cpu>::~LBFGSTask()
{
    if(argumentLCur && !argumentLCurRows.get())  { daal_free(argumentLCur); }
    if(argumentLPrev && !argumentLPrevRows.get()) { daal_free(argumentLPrev); }
    if (rho)           { daal_free(rho);           }
    if (alpha)         { daal_free(alpha);         }
    if(correctionPairs)
    {
        correctionPairs->releaseBlockOfRows(correctionPairsBD);
    }
    else
    {
        if(correctionS)   { daal_free(correctionS); }
        if(correctionY)   { daal_free(correctionY); }
    }
    if (stepLength)    { mtStepLength.release();   }
    releaseBatchIndices(batchIndices, mtBatchIndices, batchIndicesStatus);
    releaseBatchIndices(correctionPairBatchIndices, mtCorrectionPairBatchIndices, correctionPairBatchIndicesStatus);
}

template<typename algorithmFPType, CpuType cpu>
void LBFGSTask<algorithmFPType, cpu>::setToResult(NumericTable *correctionIndicesResult, NumericTable *nIterationsNT, OptionalArgument *optionalArgumentResult,
                                                  size_t nIterations, size_t epoch, size_t corrIndex)
{
    super::setToResult(correctionIndicesResult, nIterationsNT, optionalArgumentResult, nIterations, epoch, corrIndex);
    if(!_rngStateRequired || !_rngStateChanged)
        return;

    //pOpt should exist by now
    data_management::MemoryBlock* pState = dynamic_cast<data_management::MemoryBlock*>(optionalArgumentResult->get(lbfgs::rngState).get());
    //pState should exist by now
    auto stateSize = _rng.getStreamSize();
    if(!stateSize)
        return;
    pState->reserve(stateSize);
    if(!pState->get())
    {
        this->_errors->add(ErrorMemoryAllocationFailed);
        return;
    }
    _rng.saveStream(pState->get());
}

template<typename algorithmFPType, CpuType cpu>
void LBFGSTask<algorithmFPType, cpu>::initArgumentL(NumericTable* averageArgLIterInput, NumericTable *averageArgLIterResult, const Parameter* parameter)
{
    NumericTable* pOptRes = parameter->optionalResultRequired ? averageArgLIterResult : nullptr;
    if(pOptRes)
    {
        argumentLPrevRows.set(pOptRes, 0, 1);
        argumentLPrev = argumentLPrevRows.get();
        argumentLCurRows.set(pOptRes, 1, 1);
        argumentLCur = argumentLCurRows.get();
    }
    else
    {
        argumentLCur = (algorithmFPType *)daal::services::internal::service_calloc<algorithmFPType, cpu>(this->argumentSize);
        argumentLPrev = (algorithmFPType *)daal::services::internal::service_calloc<algorithmFPType, cpu>(this->argumentSize);
    }

    if(averageArgLIterInput && (averageArgLIterInput != pOptRes))
    {
        ReadRows<algorithmFPType, cpu> rr(averageArgLIterInput, 0, 1);
        const auto cMemSize = sizeof(algorithmFPType)*averageArgLIterInput->getNumberOfColumns();
        daal_memcpy_s(argumentLPrev, cMemSize, rr.get(), cMemSize);
        rr.next(1, 1);
        daal_memcpy_s(argumentLCur, cMemSize, rr.get(), cMemSize);
    }
    else if(pOptRes)
    {
        daal::services::internal::service_memset<algorithmFPType, cpu>(argumentLPrev, 0, this->argumentSize);
        daal::services::internal::service_memset<algorithmFPType, cpu>(argumentLCur, 0, this->argumentSize);
    }
}

template<typename algorithmFPType, CpuType cpu>
bool LBFGSTask<algorithmFPType, cpu>::initCorrectionPairs(NumericTable* correctionPairsInput, const Parameter* parameter, NumericTable *correctionPairsResult)
{
    const auto cCorrectionPairSize = parameter->m * this->argumentSize;
    NumericTable* correctionPairsOutput = parameter->optionalResultRequired ? correctionPairsResult : nullptr;
    if(correctionPairsOutput)
    {
        correctionPairs = correctionPairsOutput;
        correctionPairs->getBlockOfRows(0, 2 * parameter->m, readWrite, correctionPairsBD);
        correctionS = correctionPairsBD.getBlockPtr();
        if(correctionS)
            correctionY = correctionS + cCorrectionPairSize; /* second half */
    }
    else
    {
        correctionS = (algorithmFPType *)daal::services::internal::service_calloc<algorithmFPType, cpu>(cCorrectionPairSize);
        correctionY = (algorithmFPType *)daal::services::internal::service_calloc<algorithmFPType, cpu>(cCorrectionPairSize);
    }
    rho = (algorithmFPType *)daal::services::internal::service_calloc<algorithmFPType, cpu>(parameter->m);
    if(!correctionS || !correctionY || !rho)
    {
        this->_errors->add(ErrorMemoryAllocationFailed);
        return false;
    }

    /* If input correction pairs are given ... */
    if(correctionPairsInput)
    {
        /* and if it is not the same as output correction pairs then copy them to correctionS, correctionY */
        if(correctionPairsOutput != correctionPairsInput)
        {
            ReadRows<algorithmFPType, cpu> correctionPairsInputBD(*correctionPairsInput, 0, correctionPairsInput->getNumberOfRows());
            const auto cMemSize = sizeof(algorithmFPType)*cCorrectionPairSize;
            daal_memcpy_s(correctionS, cMemSize, correctionPairsInputBD.get(), cMemSize);
            daal_memcpy_s(correctionY, cMemSize, correctionPairsInputBD.get() + cMemSize, cMemSize);
        }
        /* initialize rho form S and Y */
        for(auto i = 0; i < parameter->m; ++i)
        {
            auto s = correctionS + i*this->argumentSize;
            auto y = correctionY + i*this->argumentSize;
            rho[i] = dotProduct<algorithmFPType, cpu>(this->argumentSize, s, y);
            if(rho[i] != 0.0) // threshold
            {
                rho[i] = 1.0 / rho[i];
            }
        }
    }
    else if(correctionPairsOutput)
    {
        daal::services::internal::service_memset<algorithmFPType, cpu>(correctionS, 0, cCorrectionPairSize);
        daal::services::internal::service_memset<algorithmFPType, cpu>(correctionY, 0, cCorrectionPairSize);
    }
    return true;
}

template<typename algorithmFPType, CpuType cpu>
void LBFGSTask<algorithmFPType, cpu>::initRngState(OptionalArgument *optionalArgumentInput, const Parameter* parameter)
{
    _rngStateRequired = parameter->optionalResultRequired;

    if(!optionalArgumentInput)
        return;
    data_management::MemoryBlock* pState = dynamic_cast<data_management::MemoryBlock*>(optionalArgumentInput->get(lbfgs::rngState).get());
    if(!pState || !pState->size())
        return;
    _rng.loadStream(pState->get());
}

/**
 * Returns the array that contains the indices of objective function terms used for
 * stochastic gradient or sub-sampled Hessian matrix computation
 *
 * \param[in]  size             Number of indices
 * \param[in]  indicesTable     Numeric table that represent indices that will be used
 *                              instead of random values for the stochastic gradient
 *                              or sub-sampled Hessian matrix computations
 * \param[out] indices          Resulting array that contains the indices provided by user
 *                              or memory for storing randomly generated indices
 * \param[out] mtIndices        Micro-table that stores the indices
 * \param[out] indicesStatusPtr Status of the indices array
 * \param[out] ntIndices        Numeric table that stores the indices
 */
template<typename algorithmFPType, CpuType cpu>
void LBFGSTask<algorithmFPType, cpu>::getBatchIndices(
            size_t size, NumericTable *indicesTable, int **indices,
            daal::internal::BlockMicroTable<int, readOnly, cpu> &mtIndices, IndicesStatus *indicesStatusPtr,
            services::SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu> > &ntIndices)
{
    if (indicesTable)
    {
        *indicesStatusPtr = user;
    }
    else
    {
        *indicesStatusPtr = random;
        *indices = (int *)daal_malloc(size * sizeof(int));
        if (!(*indices)) { this->_errors->add(ErrorMemoryAllocationFailed); return; }
    }

    ntIndices = SharedPtr<HomogenNumericTableCPU<int, cpu> >(new HomogenNumericTableCPU<int, cpu>(*indices, size, 1));
    if (!ntIndices)
    { this->_errors->add(ErrorMemoryAllocationFailed); return; }
}

/**
 * Releases the array that contains the indices of objective function terms used for
 * stochastic gradient or sub-sampled Hessian matrix computation
 *
 * \param[in]  indices       Array that contains the indices provided by user
 *                           or memory for storing randomly generated indices
 * \param[in]  mtIndices     Micro-table that stores the indices
 * \param[in]  indicesStatus Status of the array of indices
 */
template<typename algorithmFPType, CpuType cpu>
void LBFGSTask<algorithmFPType, cpu>::releaseBatchIndices(int *indices,
            daal::internal::BlockMicroTable<int, readOnly, cpu> &mtIndices, IndicesStatus indicesStatus)
{
    if (indicesStatus == random)
    {
        daal_free(indices);
    }
}

} // namespace daal::internal

} // namespace lbfgs

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
