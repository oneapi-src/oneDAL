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

#include <iostream>

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

template <typename algorithmFPType, CpuType cpu>
Status computeDualCoefficients(DAAL_INT p, const algorithmFPType * a, algorithmFPType * aCopy, DAAL_INT ny,
                               algorithmFPType * b, bool interceptFlag, DAAL_INT nRidge, const algorithmFPType * ridge)
{
    const DAAL_INT pToFix = (interceptFlag ? p - 1 : p);

    Status st;
    int result = 0;
    if (nRidge == 1)
    {
        for (DAAL_INT i = 0, idx = 0; i < pToFix; i++, idx += (p + 1))
        {
            aCopy[idx] += *ridge;
        }

        if (!daal::algorithms::internal::solveDualSymmetricEquationsSystem<algorithmFPType, cpu>(aCopy, b, p, ny, false, interceptFlag, *ridge))
            return services::Status(services::ErrorRidgeRegressionInternal);
    }
    else
    {
        algorithmFPType * bPtr    = b;
        const size_t aSizeInBytes = p * p * sizeof(algorithmFPType);
        for (DAAL_INT j = 0; j < ny; j++, bPtr += (pToFix + 1))
        {
            result |= daal::services::internal::daal_memcpy_s(aCopy, aSizeInBytes, a, aSizeInBytes);
            for (DAAL_INT i = 0, idx = 0; i < pToFix; i++, idx += (p + 1))
            {
                aCopy[idx] += ridge[j];
            }

            DAAL_INT one(1);

            if (!daal::algorithms::internal::solveDualSymmetricEquationsSystem<algorithmFPType, cpu>(aCopy, bPtr, p, one, false, interceptFlag, ridge[j]))
                return services::Status(services::ErrorRidgeRegressionInternal);
        }
    }
    return (!result) ? st : services::Status(services::ErrorMemoryCopyFailedInternal);
}


template <typename algorithmFPType, CpuType cpu>
Status solveDualRidgeProblem(const NumericTable & xTable, const NumericTable & yTable, NumericTable & xxtTable,
                 NumericTable & xytTable, NumericTable & betaTable, bool interceptFlag,
                 const NumericTable & ridgeTable)
{
    Status st;
    std::cout << "Solving dual ridge problem..." << std::endl;
    DAAL_INT nRows(xTable.getNumberOfRows());         /* observations */
    DAAL_INT nFeatures(xTable.getNumberOfColumns());  /* features */
    DAAL_INT nResponses(yTable.getNumberOfColumns()); /* responses */
    DAAL_INT nBetas(xTable.getNumberOfColumns() + interceptFlag); /* coefficients */
    DAAL_INT nDualCoeffs(interceptFlag ? nRows + 1 : nRows); /* dual coefficients */

    ReadRows<algorithmFPType, cpu> xBlock(const_cast<NumericTable &>(xTable), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(xBlock);
    const algorithmFPType * x = xBlock.get();

    std::cout << "X matrix:" << std::endl;
    for (DAAL_INT i = 0; i < nRows; ++i) {
        for (DAAL_INT j = 0; j < nFeatures; ++j) {
            std::cout << x[i * nFeatures + j] << " ";
        }
        std::cout << std::endl;
    }

    ReadRows<algorithmFPType, cpu> yBlock(const_cast<NumericTable &>(yTable), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(yBlock);
    const algorithmFPType * y = yBlock.get();

    WriteRows<algorithmFPType, cpu> xxtBlock(xxtTable, 0, nDualCoeffs);
    DAAL_CHECK_BLOCK_STATUS(xxtBlock);
    algorithmFPType * xxt = xxtBlock.get();

    algorithmFPType one(1.0);
    const algorithmFPType zero(0.0);
    BlasInst<algorithmFPType, cpu>::xsyrk("U", "T", &nRows, &nFeatures, &one, const_cast<algorithmFPType *>(x), &nFeatures, &one, xxt,
                                           (DAAL_INT *)(&nDualCoeffs));
    if (interceptFlag)
    {
        for (DAAL_INT i = 0; i < nRows; ++i) {
            xxt[nRows * nDualCoeffs + i] = one;
        }
        xxt[nDualCoeffs * nDualCoeffs - 1] = zero;
    }
    std::cout << "XX^T matrix:" << std::endl;
    for (DAAL_INT i = 0; i < nDualCoeffs; ++i) {
        for (DAAL_INT j = 0; j < nDualCoeffs; ++j) {
            std::cout << xxt[i * nDualCoeffs + j] << " ";
        }
        std::cout << std::endl;
    }

    TArray<algorithmFPType, cpu> xxtCopyArray(nDualCoeffs * nDualCoeffs);
    algorithmFPType * xxtCopy = xxtCopyArray.get();
    DAAL_CHECK_MALLOC(xxtCopy);

    const DAAL_INT xxtSize = nDualCoeffs * nDualCoeffs * sizeof(algorithmFPType);
    int result = daal::services::internal::daal_memcpy_s(xxtCopy, xxtSize, xxt, xxtSize);
    DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal)

    TArray<algorithmFPType, cpu> dualCoefArray(nResponses * nDualCoeffs);
    algorithmFPType * dualCoef = dualCoefArray.get();
    DAAL_CHECK_MALLOC(dualCoef);

    const DAAL_INT dualCoefSize = nResponses * nDualCoeffs * sizeof(algorithmFPType);
    if (!interceptFlag)
    {
        result = daal::services::internal::daal_memcpy_s(dualCoef, dualCoefSize, y, dualCoefSize);
        DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal);
    }
    else
    {
        for (DAAL_INT i = 0; i < nDualCoeffs - 1; ++i)
        {
            for (DAAL_INT j = 0; j < nResponses; ++j)
            {
                dualCoef[i * nResponses + j] = y[i * nResponses + j];
            }
        }
        for (DAAL_INT j = 0; j < nResponses; ++j)
        {
            dualCoef[(nDualCoeffs - 1) * nResponses + j] = 0.0;
        }
    }

    size_t nRidge = ridgeTable.getNumberOfRows();
    ReadRows<algorithmFPType, cpu> ridgeBlock(const_cast<NumericTable &>(ridgeTable), 0, nRidge);
    const algorithmFPType * ridge = ridgeBlock.get();
    DAAL_CHECK_MALLOC(ridge);
    DAAL_CHECK_STATUS(st, (computeDualCoefficients<algorithmFPType, cpu>(nDualCoeffs, xxt, xxtCopy, nResponses, dualCoef, interceptFlag, nRidge, ridge)));

    std::cout << "dual coefficients matrix:" << std::endl;
    for (DAAL_INT i = 0; i < nDualCoeffs; ++i) {
        for (DAAL_INT j = 0; j < nResponses; ++j) {
            std::cout << dualCoef[i * nResponses + j] << " ";
        }
        std::cout << std::endl;
    }

    /* Compute beta coefficients from dual coefficients as b = X^T * dualCoef */
    TArray<algorithmFPType, cpu> betaBufferArray(nResponses * nFeatures);
    algorithmFPType * betaBuffer = betaBufferArray.get();
    DAAL_CHECK_MALLOC(betaBuffer);
    BlasInst<algorithmFPType, cpu>::xgemm("N", "N", &nFeatures, &nResponses, &nRows, &one, x, &nFeatures, dualCoef, &nDualCoeffs, &zero, betaBuffer,
                                          &nFeatures);

    std::cout << "beta buffer matrix:" << std::endl;
    for (DAAL_INT i = 0; i < nResponses; ++i) {
        for (DAAL_INT j = 0; j < nFeatures; ++j) {
            std::cout << betaBuffer[i * nFeatures + j] << " ";
        }
        std::cout << std::endl;
    }

    /* Copy beta coefficients from buffer to result table */
    DAAL_ASSERT(betaTable.getNumberOfColumns() == nBetas);
    WriteOnlyRows<algorithmFPType, cpu> betaBlock(betaTable, 0, nResponses);
    DAAL_CHECK_BLOCK_STATUS(betaBlock);
    algorithmFPType * beta = betaBlock.get();

    for (size_t i = 0; i < nResponses; i++)
    {
        PRAGMA_OMP_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 1; j < nFeatures; j++)
        {
            beta[i * nBetas + j] = betaBuffer[i * nFeatures + j - 1];
        }
        beta[i * nBetas] = betaBuffer[i * nFeatures + nFeatures - 1];
    }
    if (interceptFlag)
    {
        for (size_t i = 0; i < nResponses; i++)
        {
            beta[i * nBetas] = dualCoef[nRows * nResponses + i];
        }
    }

    return st;
}

template <typename algorithmFPType, CpuType cpu>
Status BatchKernel<algorithmFPType, training::normEqDense, cpu>::compute(const NumericTable & x, const NumericTable & y, NumericTable & xtx,
                                                                         NumericTable & xty, NumericTable & beta, bool interceptFlag,
                                                                         const NumericTable & ridge) const
{
    const size_t nFeatures = x.getNumberOfColumns();
    const size_t nRows = x.getNumberOfRows();
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
        st = solveDualRidgeProblem<algorithmFPType, cpu>(x, y, xtx, xty, beta, interceptFlag, ridge); // Initialize st to a default success status if the condition is not met
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
