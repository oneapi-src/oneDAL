/* file: lbfgs_base.h */
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

//++
//  Declaration of template function that computes LBFGS.
//--

#ifndef __LBFGS_BASE_H__
#define __LBFGS_BASE_H__

#include "algorithms/optimization_solver/lbfgs/lbfgs_batch.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "src/externals/service_math.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_rng.h"
#include "src/algorithms/engines/engine_batch_impl.h"

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
using namespace daal::data_management;
using namespace daal::internal;

/**
 * Statuses of the indices of objective function terms that are used for gradient and Hessian computation
 */
enum IndicesStatus
{
    random = 0, /*!< Indices of the terms are generated randomly */
    user   = 1, /*!< Indices of the terms are provided by user */
    all    = 2  /*!< All objective function terms are used for computations */
};

/**
 *  \brief Structure for storing data used in itermediate computations in LBFGS algorithm
 */
template <typename algorithmFPType, CpuType cpu>
struct LBFGSTaskBase
{
    LBFGSTaskBase(NumericTable * argumentTable);

    /*
    * Sets the initial argument of objective function
    */
    Status setStartArgument(NumericTable * startValueTable);

    /*
    * Set nIterations result data and optional result as follows: correctionIndex, lastIteration index (epoch)
    */
    Status setToResult(NumericTable * correctionIndicesResult, NumericTable * nIterationsNT, OptionalArgument * optionalArgumentResult,
                       size_t nIterations, size_t epoch, size_t corrIndex);

public:
    const size_t argumentSize;  /*!< Number of elements in the argument of objective function */
    algorithmFPType * argument; /*!< Argument of the objective function. The optimized value */

protected:
    /** Work values access */
    WriteRows<algorithmFPType, cpu> mtArgument;
};

/**
 *  \brief Structure for storing data used in itermediate computations in LBFGS algorithm
 */
template <typename algorithmFPType, CpuType cpu>
class LBFGSTask : public LBFGSTaskBase<algorithmFPType, cpu>
{
public:
    typedef LBFGSTaskBase<algorithmFPType, cpu> super;
    typedef daal::internal::RNGsInst<int, cpu> RNGs;

    LBFGSTask(const Parameter * parameter, NumericTable * minimum);

    ~LBFGSTask();

    Status init(NumericTable * inputArgument, NumericTable * correctionPairsInput, NumericTable * averageArgLIterInput,
                OptionalArgument * optionalArgumentInput, const Parameter * parameter, NumericTable * minimum, NumericTable * averageArgLIterResult,
                NumericTable * correctionPairsResult, size_t nTerms, size_t batchSize, size_t correctionPairBatchSize);
    /*
     * Returns array of batch indices provided by user or the memory allcated for sampled batch indices
     */
    Status getBatchIndices(size_t size, NumericTable * indicesTable, int *& indices, IndicesStatus & indicesStatus,
                           services::SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu> > & ntIndices);

    /*
    * Computes the correction pair (s, y) and the corresponding value rho
    */
    Status computeCorrectionPair(size_t correctionIndex, NumericTable * hessian, bool useWolfeConditions);

    void computeCorrectionPairImpl(size_t correctionIndex, const algorithmFPType * hessian, bool useWolfeConditions);

    algorithmFPType lineSearch(algorithmFPType * x, NumericTablePtr & ntValue, NumericTablePtr & ntGradient, algorithmFPType * dx,
                               sum_of_functions::BatchPtr & gradientFunction, bool & continueSearch);
    /*
    * Updates argument of the objective function
    */
    Status updateArgument(size_t iIteration, size_t t, size_t epoch, size_t m, size_t correctionIndex, size_t nTerms, size_t batchSize,
                          algorithmFPType accuracyThreshold, sum_of_functions::BatchPtr & gradientFunction, NumericTablePtr & ntGradient,
                          NumericTablePtr & ntValue, algorithmFPType * argument, bool & bContinue,
                          daal::algorithms::engines::internal::BatchBaseImpl * engine, bool useWolfeConditions);

    /*
    * Updates the array of objective function terms indices that are used in stochastic gradient
    * or Hessian matrix computations
    */
    Status updateCorrectionPairBatchIndices(size_t iPredefinedIndicesRow, size_t nTerms, size_t batchSize,
                                            daal::algorithms::engines::internal::BatchBaseImpl * engine)
    {
        return updateBatchIndices(iPredefinedIndicesRow, nTerms, batchSize, correctionPairBatchIndices, correctionPairBatchIndicesStatus,
                                  mtCorrectionPairBatchIndices, ntCorrectionPairBatchIndices, engine);
    }

    /*
    * Set nIterations result data and optional result as follows: correctionIndex, lastIteration index (epoch)
    */
    Status setToResult(NumericTable * correctionIndicesResult, NumericTable * nIterationsNT, OptionalArgument * optionalArgumentResult,
                       size_t nIterations, size_t epoch, size_t corrIndex);

private:
    /*
     * Releases the memory allocated to store the batch indices
     */
    void releaseBatchIndices(int * indices, IndicesStatus indicesStatus);

    /*
    * Initializes argumentLCur, argumentLPrev
    */
    Status initArgumentL(NumericTable * averageArgLIterInput, NumericTable * averageArgLIterResult, const Parameter * parameter);

    /*
    * Initializes correction pairs data if passed in an optional input data
    */
    Status initCorrectionPairs(NumericTable * correctionPairsInput, const Parameter * parameter, NumericTable * correctionPairsResult);

    /*
    * Updates the array of objective function terms indices that are used in stochastic gradient
    * or Hessian matrix computations
    */
    Status updateBatchIndices(size_t iPredefinedIndicesRow, size_t nTerms, size_t batchSize, int *& batchIndices, IndicesStatus batchIndicesStatus,
                              daal::internal::ReadRows<int, cpu> & mtBatchIndices,
                              services::SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu> > & ntBatchIndices,
                              daal::algorithms::engines::internal::BatchBaseImpl * engine);

    /*
    * Two-loop recursion algorithm that computes approximation of inverse Hessian matrix
    * multiplied by input gradient vector from a set of correction pairs (s(j), y(j)), j = 1,...,m.
    */
    void twoLoopRecursion(size_t m, size_t correctionIndex, algorithmFPType * gradient);

public:
    bool continueLineSearch;
    IndicesStatus batchIndicesStatus;               /*!< Status of the objective function indices for gradient computation */
    IndicesStatus correctionPairBatchIndicesStatus; /*!< Status of the objective function indices for Hessian computation */
    int * batchIndices;                             /*!< Array that contains the batch indices */
    int * correctionPairBatchIndices;               /*!< Array that contains the correction pair batch indices */
    algorithmFPType * argumentLCur;                 /*!< Average of objective function arguments for last L iterations. See formula (2.1) in [1] */
    algorithmFPType * argumentLPrev; /*!< Average of objective function arguments for previous L iterations. See formula (2.1) in [1] */
    TArray<algorithmFPType, cpu> _gradientPrevPtr;
    TArray<algorithmFPType, cpu> _gradientCurrPtr;
    /** Numeric table that stores the batch indices */
    services::SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu> > ntBatchIndices;
    /** Numeric table that stores the correction pair batch indices */
    services::SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu> > ntCorrectionPairBatchIndices;
    /** Numeric table that stores the average of work values for last L iterations */
    services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > argumentLCurTable;
    /** Correction pair batch indices block descriptor */
    ReadRows<int, cpu> mtCorrectionPairBatchIndices;

private:
    /** Step-length sequence block descriptor*/
    ReadRows<algorithmFPType, cpu> mtStepLength;
    /** Hessian block descriptor */
    ReadRows<algorithmFPType, cpu> mtHessian;
    /** Gradient block descriptor */
    WriteRows<algorithmFPType, cpu> mtGradient;
    /** Batch indices block descriptor */
    ReadRows<int, cpu> mtBatchIndices;
    /** Numeric table that stores the correction pairs if they should be returned as an optional result */
    NumericTable * correctionPairs;
    /** BlockDescriptor for the correctionPairs table */
    BlockDescriptor<algorithmFPType> correctionPairsBD;

    /** Holder of argumentLCur passed as optional result */
    daal::internal::WriteRows<algorithmFPType, cpu, NumericTable> argumentLCurRows;
    /** Holder of argumentLPrev passed as optional result */
    daal::internal::WriteRows<algorithmFPType, cpu, NumericTable> argumentLPrevRows;

    algorithmFPType * correctionS;      /*!< Array of correction pairs parts s(1), ..., s(m). See formula (2.1) in [1] */
    algorithmFPType * correctionY;      /*!< Array of correction pairs parts y(1), ..., y(m). See formula (2.2) in [1] */
    algorithmFPType * rho;              /*!< Array of parameters rho of BFGS update. See formula (7.17) in [2] */
    algorithmFPType * alpha;            /*!< Intermediate values used in two-loop recursion. See algorithm 7.4 in [2] */
    const size_t nStepLength;           /*!< Number of values in the provided step-length sequence */
    const algorithmFPType * stepLength; /*!< Array that stores step-length sequence */
    RNGs _rng;                          /*!< Random number generator */
};

/**
 *  \brief Kernel for LBFGS computation
 *  for different floating point types of intermediate calculations and methods
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class LBFGSKernel
{};

} // namespace internal

} // namespace lbfgs

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
