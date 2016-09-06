/* file: lbfgs_base.h */
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
//  Declaration of template function that computes LBFGS.
//--


#ifndef __LBFGS_BASE_H__
#define __LBFGS_BASE_H__

#include "lbfgs_batch.h"
#include "kernel.h"
#include "numeric_table.h"
#include "service_math.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "service_rng.h"

using namespace daal::data_management;

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
/**
 * Statuses of the indices of objective function terms that are used for gradient and Hessian computation
 */
enum IndicesStatus
{
    random = 0,     /*!< Indices of the terms are generated randomly */
    user   = 1,     /*!< Indices of the terms are provided by user */
    all    = 2      /*!< All objective function terms are used for computations */
};

/**
 *  \brief Structure for storing data used in itermediate computations in LBFGS algorithm
 */
template<typename algorithmFPType, CpuType cpu>
struct LBFGSTaskBase
{
    LBFGSTaskBase(NumericTable *argumentTable, NumericTable *startValueTable);

    ~LBFGSTaskBase();

    /*
    * Sets the initial argument of objective function
    */
    void setStartArgument(NumericTable *startValueTable);

    /*
    * Set nIterations result data and optional result as follows: correctionIndex, lastIteration index (epoch)
    */
    void setToResult(NumericTable *correctionIndicesResult, NumericTable *nIterationsNT, OptionalArgument *optionalArgumentResult,
                     size_t nIterations, size_t epoch, size_t corrIndex);

public:
    const size_t argumentSize;                       /*!< Number of elements in the argument of objective function */
    algorithmFPType *argument;               /*!< Argument of the objective function. The optimized value */

protected:
    /** Micro-table that stores the work value */
    daal::internal::BlockMicroTable<algorithmFPType, writeOnly, cpu> mtArgument;
};

/**
 *  \brief Structure for storing data used in itermediate computations in LBFGS algorithm
 */
template<typename algorithmFPType, CpuType cpu>
class LBFGSTask : public LBFGSTaskBase<algorithmFPType, cpu>
{
public:
    typedef LBFGSTaskBase<algorithmFPType, cpu> super;
    typedef daal::internal::IntRng<int, cpu> IntRngType;

    LBFGSTask(NumericTable *inputArgument, NumericTable* correctionPairsInput, NumericTable* averageArgLIterInput,
              OptionalArgument *optionalArgumentInput, const Parameter *parameter, NumericTable *minimum, NumericTable *averageArgLIterResult,
              NumericTable *correctionPairsResult, size_t nTerms, size_t batchSize, size_t correctionPairBatchSize, services::KernelErrorCollectionPtr &_errors);

    ~LBFGSTask();

    /*
     * Returns array of batch indices provided by user or the memory allcated for sampled batch indices
     */
    void getBatchIndices(size_t size, NumericTable *indicesTable, int **indices,
            daal::internal::BlockMicroTable<int, readOnly, cpu> &mtIndices, IndicesStatus *indicesStatusPtr,
            services::SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu> > &ntIndices);

    /*
    * Computes the correction pair (s, y) and the corresponding value rho
    */
    void computeCorrectionPair(size_t correctionIndex, algorithmFPType *hessian);

    /*
    * Updates argument of the objective function
    */
    bool updateArgument(size_t iIteration, size_t t, size_t epoch, size_t m, size_t correctionIndex, size_t nTerms,
        size_t batchSize, algorithmFPType accuracyThreshold,
        sum_of_functions::BatchPtr &gradientFunction,
        NumericTablePtr &ntGradient, BlockDescriptor<algorithmFPType> &gradientBlock,
        algorithmFPType *argument);

    /*
    * Updates the array of objective function terms indices that are used in stochastic gradient
    * or Hessian matrix computations
    */
    void updateCorrectionPairBatchIndices(size_t iPredefinedIndicesRow, size_t nTerms, size_t batchSize)
    {
        updateBatchIndices(iPredefinedIndicesRow, nTerms, batchSize, &correctionPairBatchIndices,
            correctionPairBatchIndicesStatus, mtCorrectionPairBatchIndices, ntCorrectionPairBatchIndices);
    }

    /*
    * Set nIterations result data and optional result as follows: correctionIndex, lastIteration index (epoch)
    */
    void setToResult(NumericTable *correctionIndicesResult, NumericTable *nIterationsNT, OptionalArgument *optionalArgumentResult,
                     size_t nIterations, size_t epoch, size_t corrIndex);

private:
    /*
     * Releases the memory allocated to store the batch indices
     */
    void releaseBatchIndices(int *indices, daal::internal::BlockMicroTable<int, readOnly, cpu> &mtIndices, IndicesStatus indicesStatus);

    /*
    * Initializes argumentLCur, argumentLPrev
    */
    void initArgumentL(NumericTable* averageArgLIterInput, NumericTable *averageArgLIterResult, const Parameter* parameter);

    /*
    * Initializes correction pairs data if passed in an optional input data
    */
    bool initCorrectionPairs(NumericTable* correctionPairsInput, const Parameter* parameter, NumericTable *correctionPairsResult);

    /*
    * Initializes rng state if passed in an optional input data
    */
    void initRngState(OptionalArgument *optionalArgumentInput, const Parameter* parameter);

    /*
    * Updates the array of objective function terms indices that are used in stochastic gradient
    * or Hessian matrix computations
    */
    void updateBatchIndices(size_t iPredefinedIndicesRow, size_t nTerms, size_t batchSize, int **batchIndices,
        IndicesStatus batchIndicesStatus, daal::internal::BlockMicroTable<int, readOnly, cpu> &mtBatchIndices,
        services::SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu> > &ntBatchIndices);

    /*
    * Two-loop recursion algorithm that computes approximation of inverse Hessian matrix
    * multiplied by input gradient vector from a set of correction pairs (s(j), y(j)), j = 1,...,m.
    */
    void twoLoopRecursion(size_t m, size_t correctionIndex, algorithmFPType *gradient);

public:
    IndicesStatus batchIndicesStatus;                /*!< Status of the objective function indices for gradient computation */
    IndicesStatus correctionPairBatchIndicesStatus;  /*!< Status of the objective function indices for Hessian computation */
    int *batchIndices;                       /*!< Array that contains the batch indices */
    int *correctionPairBatchIndices;         /*!< Array that contains the correction pair batch indices */
    algorithmFPType *argumentLCur;           /*!< Average of objective function arguments for last L iterations. See formula (2.1) in [1] */
    algorithmFPType *argumentLPrev;          /*!< Average of objective function arguments for previous L iterations. See formula (2.1) in [1] */
    /** Numeric table that stores the batch indices */
    services::SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu> > ntBatchIndices;
    /** Numeric table that stores the correction pair batch indices */
    services::SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu> > ntCorrectionPairBatchIndices;
    /** Numeric table that stores the average of work values for last L iterations */
    services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > argumentLCurTable;
    /** Micro-table that stores the correction pair batch indices */
    daal::internal::BlockMicroTable<int, readOnly, cpu> mtCorrectionPairBatchIndices;

private:
    /** Micro-table that stores the step-length sequence */
    daal::internal::BlockMicroTable<algorithmFPType, readOnly, cpu>  mtStepLength;

    /** Micro-table that stores the batch indices */
    daal::internal::BlockMicroTable<int, readOnly, cpu> mtBatchIndices;
    /** Numeric table that stores the correction pairs if they should be returned as an optional result */
    NumericTable *correctionPairs;
    /** BlockDescriptor for the correctionPairs table */
    BlockDescriptor<algorithmFPType> correctionPairsBD;

    /** Holder of argumentLCur passed as optional result */
    daal::internal::WriteRows<algorithmFPType, cpu, NumericTable> argumentLCurRows;
    /** Holder of argumentLPrev passed as optional result */
    daal::internal::WriteRows<algorithmFPType, cpu, NumericTable> argumentLPrevRows;

    algorithmFPType *correctionS;            /*!< Array of correction pairs parts s(1), ..., s(m). See formula (2.1) in [1] */
    algorithmFPType *correctionY;            /*!< Array of correction pairs parts y(1), ..., y(m). See formula (2.2) in [1] */
    algorithmFPType *rho;                    /*!< Array of parameters rho of BFGS update. See formula (7.17) in [2] */
    algorithmFPType *alpha;                  /*!< Intermediate values used in two-loop recursion. See algorithm 7.4 in [2] */
    const size_t nStepLength;                /*!< Number of values in the provided step-length sequence */
    algorithmFPType *stepLength;             /*!< Array that stores step-length sequence */
    IntRngType _rng;                         /*!< Random number generator */
    bool _rngStateChanged;                   /*!< True if rng state was changed */
    bool _rngStateRequired;                  /*!< True if saving of rng state is required */
    services::KernelErrorCollectionPtr _errors; /*!< Error collection of LBFGS algorithm */
};

/**
 *  \brief Kernel for LBFGS computation
 *  for different floating point types of intermediate calculations and methods
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class LBFGSKernel {};

} // namespace daal::internal

} // namespace lbfgs

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
