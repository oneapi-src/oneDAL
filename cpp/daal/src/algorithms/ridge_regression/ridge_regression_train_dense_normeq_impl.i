/* file: ridge_regression_train_dense_normeq_impl.i */
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
//  Implementation of auxiliary functions for ridge regression Normal Equations (normEqDense) method.
//--
*/

#ifndef __RIDGE_REGRESSION_TRAIN_DENSE_NORMEQ_IMPL_I__
#define __RIDGE_REGRESSION_TRAIN_DENSE_NORMEQ_IMPL_I__

#include "src/algorithms/ridge_regression/ridge_regression_train_kernel.h"
#include "src/algorithms/service_kernel_math.h"
#include "src/externals/service_blas.h"
#include "src/externals/service_stat.h"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace training
{
namespace internal
{
using namespace daal::algorithms::linear_model::normal_equations::training::internal;

/**
 * Solves ridge regression through the dual system: (X * X' + ridge * I) * alpha = y.
 *
 * The resulting coefficients beta = X' * alpha.
 *
 * The intercept case is solved for centered X and y so that the intercept is not penalized.
 *
 * @param[in]   xTable          Input data table with features of size nRows x nFeatures.
 * @param[in]   yTable          Input data table with responses of size nRows x nResponses.
 * @param[out]  betaTable       Output data table for coefficients of size nResponses x (nFeatures + 1).
 *                              The first column corresponds to the intercept (if interceptFlag is true) or zero otherwise.
 * @param[in]  interceptFlag    Flag indicating whether to include an intercept term.
 * @param[in]  ridgeHelper      Helper object containing ridge parameter and other kernel-related information.
 *
 * @return Status indicating success or failure of the operation.
 */
template <typename algorithmFPType, CpuType cpu>
Status solveDualRidgeProblem(const NumericTable & xTable, const NumericTable & yTable, NumericTable & betaTable, bool interceptFlag,
                             const KernelHelper<algorithmFPType, cpu> & ridgeHelper)
{
    const DAAL_INT nRows(xTable.getNumberOfRows());
    const DAAL_INT nFeatures(xTable.getNumberOfColumns());
    const DAAL_INT nResponses(yTable.getNumberOfColumns());
    const DAAL_INT nBetas(betaTable.getNumberOfColumns());

    DAAL_ASSERT(nBetas == nFeatures + 1);
    DAAL_CHECK(nRows > 0, services::ErrorIncorrectNumberOfObservations);

    ReadRows<algorithmFPType, cpu> xBlock(const_cast<NumericTable &>(xTable), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(xBlock);
    const algorithmFPType * x = xBlock.get();

    ReadRows<algorithmFPType, cpu> yBlock(const_cast<NumericTable &>(yTable), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(yBlock);
    const algorithmFPType * y = yBlock.get();

    TArrayCalloc<algorithmFPType, cpu> xMeanArray(interceptFlag ? nFeatures : 0);
    TArrayCalloc<algorithmFPType, cpu> yMeanArray(interceptFlag ? nResponses : 0);
    algorithmFPType * xMean = xMeanArray.get();
    algorithmFPType * yMean = yMeanArray.get();
    if (interceptFlag)
    {
        DAAL_CHECK_MALLOC(xMean);
        DAAL_CHECK_MALLOC(yMean);
        if (daal::internal::StatisticsInst<algorithmFPType, cpu>::xmeansOnePass(x, nFeatures, nRows, xMean))
            return services::Status(services::ErrorRidgeRegressionInternal);
        if (daal::internal::StatisticsInst<algorithmFPType, cpu>::xmeansOnePass(y, nResponses, nRows, yMean))
            return services::Status(services::ErrorRidgeRegressionInternal);
    }

    /* Compute X * X' */
    const algorithmFPType one(1);
    const algorithmFPType zero(0);
    TArray<algorithmFPType, cpu> xxtArray(nRows * nRows);
    algorithmFPType * xxt = xxtArray.get();
    DAAL_CHECK_MALLOC(xxt);
    BlasInst<algorithmFPType, cpu>::xsyrk("U", "T", &nRows, &nFeatures, &one, x, &nFeatures, &zero, xxt, &nRows);

    if (interceptFlag)
    {
        /* Compute centered left hand side:
           Xc * Xc' = X * X' - u * 1' - 1 * u' + c * 1 * 1', where u = X * xMean and c = xMean' * xMean */
        TArray<algorithmFPType, cpu> uArray(nRows);
        algorithmFPType * u = uArray.get();
        DAAL_CHECK_MALLOC(u);
        const DAAL_INT incOne(1);
        BlasInst<algorithmFPType, cpu>::xgemv("T", &nFeatures, &nRows, &one, x, &nFeatures, xMean, &incOne, &zero, u, &incOne);
        algorithmFPType c = zero;
        for (DAAL_INT f = 0; f < nFeatures; ++f) c += xMean[f] * xMean[f];

        /* Only the lower triangle is used by the solver */
        for (DAAL_INT j = 0; j < nRows; ++j)
        {
            PRAGMA_OMP_SIMD
            for (DAAL_INT i = 0; i <= j; ++i) xxt[j * nRows + i] += c - u[i] - u[j];
        }
    }

    /* Form right hand side, the solver works on dualCoef of size nResponses x nRows,
       where i-th row represents the dual coefficients for the i-th observation */
    TArray<algorithmFPType, cpu> dualCoefArray(nRows * nResponses);
    algorithmFPType * dualCoef = dualCoefArray.get();
    DAAL_CHECK_MALLOC(dualCoef);
    for (DAAL_INT i = 0; i < nRows; ++i)
    {
        if (interceptFlag)
        {
            for (DAAL_INT j = 0; j < nResponses; ++j)
            {
                dualCoef[j * nRows + i] = y[i * nResponses + j] - yMean[j];
            }
        }
        else
        {
            for (DAAL_INT j = 0; j < nResponses; ++j)
            {
                dualCoef[j * nRows + i] = y[i * nResponses + j];
            }
        }
    }

    /* The solver overwrites input matrix, so we create a copy */
    TArray<algorithmFPType, cpu> xxtCopyArray(nRows * nRows);
    algorithmFPType * xxtCopy = xxtCopyArray.get();
    DAAL_CHECK_MALLOC(xxtCopy);
    const size_t xxtSizeInBytes = nRows * nRows * sizeof(algorithmFPType);
    DAAL_CHECK(!daal::services::internal::daal_memcpy_s(xxtCopy, xxtSizeInBytes, xxt, xxtSizeInBytes), services::ErrorMemoryCopyFailedInternal);

    Status st;
    DAAL_CHECK_STATUS(st, ridgeHelper.computeBetasImpl(nRows, xxt, xxtCopy, nResponses, dualCoef, false));

    /* betaBuffer = Xc' * dualCoef */
    TArray<algorithmFPType, cpu> betaBufferArray(nFeatures * nResponses);
    algorithmFPType * betaBuffer = betaBufferArray.get();
    DAAL_CHECK_MALLOC(betaBuffer);
    BlasInst<algorithmFPType, cpu>::xgemm("N", "N", &nFeatures, &nResponses, &nRows, &one, x, &nFeatures, dualCoef, &nRows, &zero, betaBuffer,
                                          &nFeatures);

    /* Each beta row is [intercept, coefficients] */
    WriteOnlyRows<algorithmFPType, cpu> betaBlock(betaTable, 0, nResponses);
    DAAL_CHECK_BLOCK_STATUS(betaBlock);
    algorithmFPType * beta = betaBlock.get();
    for (DAAL_INT j = 0; j < nResponses; ++j)
    {
        algorithmFPType * coef    = betaBuffer + j * nFeatures;
        algorithmFPType intercept = zero;
        if (interceptFlag)
        {
            /* Adjust coefficients for the mean of X */
            algorithmFPType dualSum = zero;
            PRAGMA_OMP_SIMD_ARGS(reduction(+ : dualSum))
            for (DAAL_INT i = 0; i < nRows; ++i) dualSum += dualCoef[j * nRows + i];

            PRAGMA_OMP_SIMD
            for (DAAL_INT i = 0; i < nFeatures; ++i) coef[i] -= xMean[i] * dualSum;

            intercept = yMean[j];
            PRAGMA_OMP_SIMD_ARGS(reduction(- : intercept))
            for (DAAL_INT i = 0; i < nFeatures; ++i) intercept -= xMean[i] * coef[i];
        }
        beta[j * nBetas] = intercept;
        DAAL_CHECK(!daal::services::internal::daal_memcpy_s(beta + j * nBetas + 1, nFeatures * sizeof(algorithmFPType), coef,
                                                            nFeatures * sizeof(algorithmFPType)),
                   services::ErrorMemoryCopyFailedInternal);
    }

    return st;
}

template <typename algorithmFPType, CpuType cpu>
Status BatchKernel<algorithmFPType, training::normEqDense, cpu>::compute(const NumericTable & x, const NumericTable & y, NumericTable & xtx,
                                                                         NumericTable & xty, NumericTable & beta, bool interceptFlag,
                                                                         const NumericTable & ridge) const
{
    const size_t nFeatures = x.getNumberOfColumns();
    const size_t nRows     = x.getNumberOfRows();
    Status st;
    KernelHelper<algorithmFPType, cpu> ridgeHelper(ridge);
    if (nRows >= nFeatures)
    {
        // Use normal equations method
        st = UpdateKernelType::compute(x, y, xtx, xty, true, interceptFlag);
        if (st) st = FinalizeKernelType::compute(xtx, xty, xtx, xty, beta, interceptFlag, ridgeHelper);
    }
    else
    {
        // Use dual ridge problem solver
        st = solveDualRidgeProblem<algorithmFPType, cpu>(x, y, beta, interceptFlag, ridgeHelper);
    }
    return st;
}

template <typename algorithmFPType, CpuType cpu>
Status OnlineKernel<algorithmFPType, training::normEqDense, cpu>::compute(const NumericTable & x, const NumericTable & y, NumericTable & xtx,
                                                                          NumericTable & xty, bool interceptFlag) const
{
    return UpdateKernelType::compute(x, y, xtx, xty, false, interceptFlag);
}

template <typename algorithmFPType, CpuType cpu>
Status OnlineKernel<algorithmFPType, training::normEqDense, cpu>::finalizeCompute(const NumericTable & xtx, const NumericTable & xty,
                                                                                  NumericTable & xtxFinal, NumericTable & xtyFinal,
                                                                                  NumericTable & beta, bool interceptFlag,
                                                                                  const NumericTable & ridge) const
{
    return FinalizeKernelType::compute(xtx, xty, xtxFinal, xtyFinal, beta, interceptFlag, KernelHelper<algorithmFPType, cpu>(ridge));
}

} // namespace internal
} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif
