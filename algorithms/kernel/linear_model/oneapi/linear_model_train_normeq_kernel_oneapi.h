/* file: linear_model_train_normeq_kernel_oneapi.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Declaration of common base classes for normal equations model training.
//--
*/

#ifndef __LINEAR_MODEL_TRAIN_NORMEQ_KERNEL_ONEAPI_H__
#define __LINEAR_MODEL_TRAIN_NORMEQ_KERNEL_ONEAPI_H__

#include "env_detect.h"
#include "numeric_table.h"
#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace normal_equations
{
namespace training
{
namespace internal
{
/**
 * Abstract class that defines interface for the helper function that computes the regression coefficients.
 */
template <typename algorithmFPType>
class KernelHelperOneAPIIface
{
public:
    /**
     * Computes regression coefficients by solving the system of linear equations
     * \param[in] p         Size of the system of linear equations
     * \param[in] a         Matrix of size P x P with semifinished left hand side of the system
     * \param[in] ny        Number of right hand sides of the system
     * \param[in,out] b     Matrix of size Ny x P.
     *                      On input, the right hand sides of the system of linear equations
     *                      On output, the regression coefficients
     * \param[in] interceptFlag Flag. If true, then it is required to compute an intercept term
     * \return Status of the computations
     */
    virtual services::Status computeBetasImpl(size_t p, services::Buffer<algorithmFPType> & a, size_t ny, services::Buffer<algorithmFPType> & b,
                                              bool inteceptFlag) const = 0;

    virtual services::Status copyBetaToResult(const services::Buffer<algorithmFPType> & betaTmp, services::Buffer<algorithmFPType> & betaRes,
                                              const size_t nBetas, const size_t nResponses, const bool interceptFlag) const = 0;
};

/**
 * Implements the common part of the regression coefficients computation from partial result
 */
template <typename algorithmFPType>
class FinalizeKernelOneAPI
{
public:
    /**
     * Computes regression coefficients by solving the symmetric system of linear equations
     *      - X' - matrix of size N x P' that contains input data set of size N x P
     *             and optionally a column of 1's.
     *             Column of 1's is added when it is required to compute an intercept term
     *      - P' - number of columns in X'.
     *             P' = P + 1, when it is required to compute an intercept term;
     *             P' = P, otherwise
     * \param[in]  xtx      Input matrix \f$X'^T \times X'\f$ of size P' x P'
     * \param[in]  xty      Input matrix \f$X'^T \times Y\f$ of size Ny x P'
     * \param[out] xtxFinal Resulting matrix \f$X'^T \times X'\f$ of size P' x P'
     * \param[out] xtyFinal Resulting matrix \f$X'^T \times Y\f$ of size Ny x P'
     * \param[out] beta     Matrix with regression coefficients of size Ny x (P + 1)
     * \param[in]  interceptFlag    Flag. True if intercept term is not zero, false otherwise
     * \param[in]  helper   Object that implements the differences in the regression
     *                      coefficients computation
     * \return Status of the computations
     */
    static services::Status compute(NumericTable & xtx, NumericTable & xty, NumericTable & xtxFinal, NumericTable & xtyFinal, NumericTable & beta,
                                    bool interceptFlag, const KernelHelperOneAPIIface<algorithmFPType> & helper);

    static services::Status copyDataToFinalTable(NumericTable & srcTable, NumericTable & dstTable);

    /**
     * Solves the symmetric system of linear equations
     * \param[in] p             Size of the system of linear equations
     * \param[in] a             Matrix of size P x P with the left hand side of the system
     * \param[in] ny            Number of right hand sides of the system
     * \param[in,out] b         Matrix of size Ny x P.
     *                          On input, the right hand sides of the system of linear equations
     *                          On output, the regression coefficients
     * \param[in] internalError Error code that have to be returned in case incorrect parameters
     *                          are passed into lapack routines
     * \return Status of the computations
     */
    static services::Status solveSystem(const size_t p, services::Buffer<algorithmFPType> & a, const size_t ny,
                                        services::Buffer<algorithmFPType> & b);
};

/**
 * Implements the common part of the partial results update with new block of input data
 */
template <typename algorithmFPType>
class UpdateKernelOneAPI
{
public:
    /**
     * Updates normal equations model with the new block of data
     * \param[in]  x        Input data set of size N x P
     * \param[in]  y        Input responses of size N x Ny
     * \param[out] xtx      Matrix \f$X'^T \times X'\f$ of size P' x P'
     * \param[out] xty      Matrix \f$X'^T \times Y\f$ of size Ny x P'
     * \param[in]  interceptFlag    Flag.
     *                              - True if it is required to compute an intercept term and P' = P + 1
     *                              - False otherwis, P' = P
     * \return Status of the computations
     */
    static services::Status compute(NumericTable & x, NumericTable & y, NumericTable & xtx, NumericTable & xty, bool interceptFlag);

private:
    static services::Status copyReduceResultsY(const services::Buffer<algorithmFPType> & src, const size_t srcSize,
                                               services::Buffer<algorithmFPType> & dst, const size_t nColsDst);
};

} // namespace internal
} // namespace training
} // namespace normal_equations
} // namespace linear_model
} // namespace algorithms
} // namespace daal

#endif
