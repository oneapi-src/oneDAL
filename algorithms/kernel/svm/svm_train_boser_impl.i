/* file: svm_train_boser_impl.i */
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
//  SVM training algorithm implementation
//--
*/
/*
//  DESCRIPTION
//
//  Definition of the functions for training with SVM 2-class classifier.
//
//  REFERENCES
//
//  1. Rong-En Fan, Pai-Hsuen Chen, Chih-Jen Lin,
//     Working Set Selection Using Second Order Information
//     for Training Support Vector Machines,
//     Journal of Machine Learning Research 6 (2005), pp. 1889___1918
//  2. Bernard E. boser, Isabelle M. Guyon, Vladimir N. Vapnik,
//     A Training Algorithm for Optimal Margin Classifiers.
//  3. Thorsten Joachims, Making Large-Scale SVM Learning Practical,
//     Advances in Kernel Methods - Support Vector Learning
*/

#ifndef __SVM_TRAIN_BOSER_IMPL_I__
#define __SVM_TRAIN_BOSER_IMPL_I__

#include "service_memory.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "service_utils.h"
#include "service_data_utils.h"

using namespace daal::internal;
using namespace daal::services::internal;

#if defined(__INTEL_COMPILER)
    #if defined(_M_AMD64) || defined(__amd64) || defined(__x86_64) || defined(__x86_64__)
        #if (__CPUID__(DAAL_CPU) == __avx512__)

            #include <immintrin.h>

using namespace daal;
using namespace daal::algorithms::svm::training::internal;

            #include "svm_train_boser_avx512_impl.i"
            #include "inner/svm_train_boser_avx512_impl_v1.i"

        #endif // __CPUID__(DAAL_CPU) == __avx512__
    #endif     // defined (_M_AMD64) || defined (__amd64) || defined (__x86_64) || defined (__x86_64__)
#endif         // __INTEL_COMPILER

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, typename ParameterType, CpuType cpu>
services::Status SVMTrainImpl<boser, algorithmFPType, ParameterType, cpu>::compute(const NumericTablePtr & xTable, NumericTable & yTable,
                                                                                   daal::algorithms::Model * r, const ParameterType * svmPar)
{
    SVMTrainTask<algorithmFPType, ParameterType, cpu> task(xTable->getNumberOfRows());
    Status s = task.setup(*svmPar, xTable, yTable);
    if (!s) return s;
    s = task.compute(*svmPar);
    return s.ok() ? task.setResultsToModel(*xTable, *static_cast<Model *>(r), svmPar->C) : s;
}

template <typename algorithmFPType, typename ParameterType, CpuType cpu>
Status SVMTrainTask<algorithmFPType, ParameterType, cpu>::compute(const ParameterType & svmPar)
{
    const algorithmFPType C(svmPar.C);
    const algorithmFPType eps(svmPar.accuracyThreshold);
    const algorithmFPType tau(svmPar.tau);
    Status s = init(C);
    Status status;
    if (!s) return s;

    size_t nActiveVectors(_nVectors);
    algorithmFPType curEps = MaxVal<algorithmFPType>::get();
    if (!svmPar.doShrinking)
    {
        for (size_t iter = 0; s.ok() && (iter < svmPar.maxIterations) && (eps < curEps); ++iter)
        {
            int Bi, Bj;
            algorithmFPType delta, ma, Ma;
            if (!findMaximumViolatingPair(nActiveVectors, tau, Bi, Bj, delta, ma, Ma, curEps, s)) break;
            s = update(nActiveVectors, C, Bi, Bj, delta);
        }
        return s;
    }

    bool unshrink = false;
    for (size_t iter = 0, shrinkingIter = 1; (iter < svmPar.maxIterations) && (eps < curEps); ++iter, ++shrinkingIter)
    {
        int Bi, Bj;
        algorithmFPType delta, ma, Ma;
        if (!findMaximumViolatingPair(nActiveVectors, tau, Bi, Bj, delta, ma, Ma, curEps, s)) return s;

        if (curEps < eps)
        {
            /* Check the optimality condition for the task with excluded variables */
            if (unshrink && nActiveVectors < _nVectors) s = reconstructGradient(nActiveVectors);
            if (!s || !findMaximumViolatingPair(nActiveVectors, tau, Bi, Bj, delta, ma, Ma, curEps, s)) return s;
            if (curEps < eps) return s; /* Here if the optimality condition holds for the excluded variables */
            shrinkingIter = 0;
        }
        s = update(nActiveVectors, C, Bi, Bj, delta);
        if ((shrinkingIter % svmPar.shrinkingStep) == 0)
        {
            if ((!unshrink) && (curEps < 10.0 * eps))
            {
                unshrink = true;
                if (nActiveVectors < _nVectors)
                {
                    s = reconstructGradient(nActiveVectors);
                    if (!s) return s;
                }
            }
            /* Update shrinking flags and do shrinking if needed*/
            if (updateShrinkingFlags(nActiveVectors, C, ma, Ma) > 0)
            {
                status |= _cache->updateShrinkingRowIndices(nActiveVectors, _I.get());
                nActiveVectors = doShrink(nActiveVectors);
            }
        }
    }
    if (status) return status;
    if (nActiveVectors < _nVectors) s = reconstructGradient(nActiveVectors);
    return s;
}

template <typename algorithmFPType, typename ParameterType, CpuType cpu>
Status SVMTrainTask<algorithmFPType, ParameterType, cpu>::setResultsToModel(const NumericTable & xTable, Model & model, algorithmFPType C) const
{
    const algorithmFPType * alpha = _alpha.get();
    const algorithmFPType zero(0.0);
    size_t nSV = 0;
    for (size_t i = 0; i < _nVectors; i++)
    {
        if (alpha[i] > zero) nSV++;
    }

    model.setNFeatures(xTable.getNumberOfColumns());
    Status s;
    DAAL_CHECK_STATUS(s, setSVCoefficients(nSV, model));
    DAAL_CHECK_STATUS(s, setSVIndices(nSV, model));
    if (xTable.getDataLayout() == NumericTableIface::csrArray)
    {
        DAAL_CHECK_STATUS(s, setSV_CSR(model, xTable, nSV));
    }
    else
    {
        DAAL_CHECK_STATUS(s, setSV_Dense(model, xTable, nSV));
    }
    /* Calculate bias and write it into model */
    model.setBias(double(calculateBias(C)));
    return s;
}

/**
 * \brief Write classification coefficients into resulting model
 *template <typename algorithmFPType, typename ParameterType, CpuType cpu>
 * \param[in]  nSV          Number of support vectors
 * \param[out] model        Resulting model
 */
template <typename algorithmFPType, typename ParameterType, CpuType cpu>
Status SVMTrainTask<algorithmFPType, ParameterType, cpu>::setSVCoefficients(size_t nSV, Model & model) const
{
    const algorithmFPType zero(0.0);
    NumericTablePtr svCoeffTable = model.getClassificationCoefficients();
    Status s;
    DAAL_CHECK_STATUS(s, svCoeffTable->resize(nSV));

    WriteOnlyRows<algorithmFPType, cpu> mtSvCoeff(*svCoeffTable, 0, nSV);
    DAAL_CHECK_BLOCK_STATUS(mtSvCoeff);
    algorithmFPType * svCoeff     = mtSvCoeff.get();
    const algorithmFPType * y     = _y.get();
    const algorithmFPType * alpha = _alpha.get();

    for (size_t i = 0, iSV = 0; i < _nVectors; i++)
    {
        if (alpha[i] != zero)
        {
            svCoeff[iSV] = y[i] * alpha[i];
            iSV++;
        }
    }
    return s;
}

/**
 * \brief Write indices of the support vectors into resulting model
 *
 * \param[in]  nSV          Number of support vectors
 * \param[out] model        Resulting model
 */
template <typename algorithmFPType, typename ParameterType, CpuType cpu>
Status SVMTrainTask<algorithmFPType, ParameterType, cpu>::setSVIndices(size_t nSV, Model & model) const
{
    NumericTablePtr svIndicesTable = model.getSupportIndices();
    Status s;
    DAAL_CHECK_STATUS(s, svIndicesTable->resize(nSV));

    WriteOnlyRows<int, cpu> mtSvIndices(*svIndicesTable, 0, nSV);
    DAAL_CHECK_BLOCK_STATUS(mtSvIndices);
    int * svIndices = mtSvIndices.get();

    const algorithmFPType zero(0.0);
    for (size_t i = 0, iSV = 0; i < _nVectors; i++)
    {
        if (_alpha[i] != zero)
        {
            DAAL_ASSERT(_cache->getDataRowIndex(i) <= services::internal::MaxVal<int>::get())
            svIndices[iSV++] = (int)_cache->getDataRowIndex(i);
        }
    }
    return s;
}

/**
 * \brief Write support vectors in dense format into resulting model
 *
 * \param[out] model        Resulting model
 * \param[in]  xTable       Input data set in dense layout
 * \param[in]  nSV          Number of support vectors
 */
template <typename algorithmFPType, typename ParameterType, CpuType cpu>
Status SVMTrainTask<algorithmFPType, ParameterType, cpu>::setSV_Dense(Model & model, const NumericTable & xTable, size_t nSV) const
{
    const size_t nFeatures = xTable.getNumberOfColumns();
    /* Allocate memory for support vectors and coefficients */
    NumericTablePtr svTable = model.getSupportVectors();
    Status s;
    DAAL_CHECK_STATUS(s, svTable->resize(nSV));
    if (nSV == 0) return s;

    WriteOnlyRows<algorithmFPType, cpu> mtSv(*svTable, 0, nSV);
    DAAL_CHECK_BLOCK_STATUS(mtSv);
    algorithmFPType * sv = mtSv.get();

    const algorithmFPType zero(0.0);
    ReadRows<algorithmFPType, cpu> mtX;
    for (size_t i = 0, iSV = 0; i < _nVectors; i++)
    {
        if (_alpha[i] == zero) continue;
        const size_t rowIndex = _cache->getDataRowIndex(i);
        mtX.set(const_cast<NumericTable *>(&xTable), rowIndex, 1);
        DAAL_CHECK_BLOCK_STATUS(mtX);
        const algorithmFPType * xi = mtX.get();
        for (size_t j = 0; j < nFeatures; j++)
        {
            sv[iSV * nFeatures + j] = xi[j];
        }
        iSV++;
    }
    return s;
}

/**
 * \brief Write support vectors in CSR format into resulting model
 *
 * \param[out] model        Resulting model
 * \param[in]  xTable       Input data set in CSR layout
 * \param[in]  nSV          Number of support vectors
 */
template <typename algorithmFPType, typename ParameterType, CpuType cpu>
Status SVMTrainTask<algorithmFPType, ParameterType, cpu>::setSV_CSR(Model & model, const NumericTable & xTable, size_t nSV) const
{
    TArray<size_t, cpu> aSvRowOffsets(nSV + 1);
    DAAL_CHECK_MALLOC(aSvRowOffsets.get());
    size_t * svRowOffsetsBuffer = aSvRowOffsets.get();

    CSRNumericTableIface * csrIface = dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(&xTable));
    ReadRowsCSR<algorithmFPType, cpu> mtX;

    const algorithmFPType zero(0.0);
    /* Calculate row offsets for the table that stores support vectors */
    svRowOffsetsBuffer[0] = 1;
    for (size_t i = 0, iSV = 0; i < _nVectors; i++)
    {
        if (_alpha[i] > zero)
        {
            const size_t rowIndex = _cache->getDataRowIndex(i);
            mtX.set(csrIface, rowIndex, 1);
            DAAL_CHECK_BLOCK_STATUS(mtX);
            svRowOffsetsBuffer[iSV + 1] = svRowOffsetsBuffer[iSV] + (mtX.rows()[1] - mtX.rows()[0]);
            iSV++;
        }
    }

    Status s;
    /* Allocate memory for storing support vectors and coefficients */
    CSRNumericTablePtr svTable = services::staticPointerCast<CSRNumericTable, NumericTable>(model.getSupportVectors());
    DAAL_CHECK_STATUS(s, svTable->resize(nSV));
    if (nSV == 0) return s;

    const size_t svDataSize = svRowOffsetsBuffer[nSV] - svRowOffsetsBuffer[0];
    DAAL_CHECK_STATUS(s, svTable->allocateDataMemory(svDataSize));

    /* Copy row offsets into the table */
    size_t * svRowOffsets = nullptr;
    svTable->getArrays<algorithmFPType>(NULL, NULL, &svRowOffsets);
    for (size_t i = 0; i < nSV + 1; i++)
    {
        svRowOffsets[i] = svRowOffsetsBuffer[i];
    }

    WriteOnlyRowsCSR<algorithmFPType, cpu> mtSv(*svTable, 0, nSV);
    DAAL_CHECK_BLOCK_STATUS(mtSv);
    algorithmFPType * sv  = mtSv.values();
    size_t * svColIndices = mtSv.cols();

    for (size_t i = 0, iSV = 0, svOffset = 0; i < _nVectors; i++)
    {
        if (_alpha[i] == zero) continue;
        const size_t rowIndex = _cache->getDataRowIndex(i);
        mtX.set(csrIface, rowIndex, 1);
        DAAL_CHECK_BLOCK_STATUS(mtX);
        const algorithmFPType * xi       = mtX.values();
        const size_t * xiColIndices      = mtX.cols();
        const size_t nNonZeroValuesInRow = mtX.rows()[1] - mtX.rows()[0];
        for (size_t j = 0; j < nNonZeroValuesInRow; j++, svOffset++)
        {
            sv[svOffset]           = xi[j];
            svColIndices[svOffset] = xiColIndices[j];
        }
        iSV++;
    }
    return s;
}

/**
 * \brief Calculate SVM model bias
 *
 * \param[in]  C        Upper bound in constraints of the quadratic optimization problem
 * \return Bias for the SVM model
 */
template <typename algorithmFPType, typename ParameterType, CpuType cpu>
algorithmFPType SVMTrainTask<algorithmFPType, ParameterType, cpu>::calculateBias(algorithmFPType C) const
{
    algorithmFPType bias;
    const algorithmFPType zero(0.0);
    const algorithmFPType one(1.0);
    size_t num_yg          = 0;
    algorithmFPType sum_yg = 0.0;

    const algorithmFPType fpMax   = MaxVal<algorithmFPType>::get();
    algorithmFPType ub            = -(fpMax);
    algorithmFPType lb            = fpMax;
    const algorithmFPType * alpha = _alpha.get();
    const algorithmFPType * y     = _y.get();
    const algorithmFPType * grad  = _grad.get();
    for (size_t i = 0; i < _nVectors; i++)
    {
        const algorithmFPType yg = -y[i] * grad[i];
        if (y[i] == -one && alpha[i] == C)
        {
            ub = ((ub > yg) ? ub : yg);
        } /// SVM_MAX(ub, yg);
        else if (y[i] == one && alpha[i] == C)
        {
            lb = ((lb < yg) ? lb : yg);
        } /// SVM_MIN(lb, yg);
        else if (y[i] == one && alpha[i] == zero)
        {
            ub = ((ub > yg) ? ub : yg);
        } /// SVM_MAX(ub, yg);
        else if (y[i] == -one && alpha[i] == zero)
        {
            lb = ((lb < yg) ? lb : yg);
        } /// SVM_MIN(lb, yg);
        else
        {
            sum_yg += yg;
            num_yg++;
        }
    }

    if (num_yg == 0)
    {
        bias = 0.5 * (ub + lb);
    }
    else
    {
        bias = sum_yg / (algorithmFPType)num_yg;
    }

    return bias;
}

/**
 * \brief Working set selection (WSS3) function.
 *        Select an index i from a pair of indices B = {i, j} using WSS 3 algorithm from [1].
 *
 * \param[in] nActiveVectors    number of observations in a training data set that are used
 *                              in sequential minimum optimization at the current iteration
 * \param[out] Bi            resulting index i
 *
 * \return The function returns m(alpha) = max(-y[i]*grad[i]): i belongs to I_UP (alpha)
 */
template <typename algorithmFPType, typename ParameterType, CpuType cpu>
algorithmFPType SVMTrainTask<algorithmFPType, ParameterType, cpu>::WSSi(size_t nActiveVectors, int & Bi) const
{
    Bi                   = -1;
    algorithmFPType GMax = -(MaxVal<algorithmFPType>::get()); // some big negative number

    const char * I               = _I.get();
    const algorithmFPType * grad = _grad.get();
    const algorithmFPType * y    = _y.get();
    /* Find i index of the working set (Bi) */
    for (size_t i = 0; i < nActiveVectors; i++)
    {
        if ((I[i] & up) != up)
        {
            continue;
        }
        algorithmFPType objFunc = -y[i] * grad[i];
        if (objFunc >= GMax)
        {
            GMax = objFunc;
            Bi   = i;
        }
    }
    return GMax;
}

template <typename algorithmFPType, typename ParameterType, CpuType cpu>
void SVMTrainTask<algorithmFPType, ParameterType, cpu>::WSSjLocalBaseline(const size_t jStart, const size_t jEnd, const algorithmFPType * KiBlock,
                                                                          const algorithmFPType GMax, const algorithmFPType Kii,
                                                                          const algorithmFPType tau, int & Bj, algorithmFPType & GMin,
                                                                          algorithmFPType & GMin2, algorithmFPType & delta) const
{
    algorithmFPType fpMax = MaxVal<algorithmFPType>::get();
    GMin                  = fpMax; // some big positive number
    GMin2                 = fpMax;

    const algorithmFPType zero(0.0);
    const algorithmFPType two(2.0);

    for (size_t j = jStart; j < jEnd; j++)
    {
        algorithmFPType ygrad = -_y[j] * _grad[j];
        if ((_I[j] & low) != low)
        {
            continue;
        }
        if (ygrad <= GMin2)
        {
            GMin2 = ygrad;
        }
        if (ygrad >= GMax)
        {
            continue;
        }

        algorithmFPType b = GMax - ygrad;
        algorithmFPType a = Kii + _kernelDiag[j] - two * KiBlock[j - jStart];
        if (a <= zero)
        {
            a = tau;
        }
        algorithmFPType dt      = b / a;
        algorithmFPType objFunc = -b * dt;
        if (objFunc <= GMin)
        {
            GMin  = objFunc;
            Bj    = j;
            delta = dt;
        }
    }
}

template <typename algorithmFPType, typename ParameterType, CpuType cpu>
void SVMTrainTask<algorithmFPType, ParameterType, cpu>::WSSjLocal(const size_t jStart, const size_t jEnd, const algorithmFPType * KiBlock,
                                                                  const algorithmFPType GMax, const algorithmFPType Kii, const algorithmFPType tau,
                                                                  int & Bj, algorithmFPType & GMin, algorithmFPType & GMin2,
                                                                  algorithmFPType & delta) const
{
    WSSjLocalBaseline(jStart, jEnd, KiBlock, GMax, Kii, tau, Bj, GMin, GMin2, delta);
}

/**
 * \brief Working set selection (WSS3) function for the case when the matrix Q is cached.
 *        Select an index j from a pair of indices B = {i, j} using WSS 3 algorithm from [1].
 *
 * \param[in] nActiveVectors    number of observations in a training data set that are used
 *                              in sequential minimum optimization at the current iteration
 * \param[in] tau               parameter of the working set selection algorithm
 * \param[in] Bi                index i from a pair of working set indices B = {i, j}
 * \param[in] GMax              value of m(alpha) = max(-y[i]*grad[i]): i belongs to I_UP (alpha) (see p.1891, eqn.6 in [1])
 * \param[out] Bj               resulting index j
 * \param[out] delta            optimal solution of the sub-problem of size 2:
 *                                  delta =  alpha[i]* - alpha[i]
 *                                  delta = -alpha[j]* + alpha[j]
 *
 * \return The function returns M(alpha) where
 *              M(alpha) = min(-y[i]*grad[i]): i belongs to I_low(alpha)
 */
template <typename algorithmFPType, typename ParameterType, CpuType cpu>
Status SVMTrainTask<algorithmFPType, ParameterType, cpu>::WSSj(size_t nActiveVectors, algorithmFPType tau, int Bi, algorithmFPType GMax, int & Bj,
                                                               algorithmFPType & delta, algorithmFPType & res) const
{
    Bj                    = -1;
    algorithmFPType fpMax = MaxVal<algorithmFPType>::get();
    algorithmFPType GMin  = fpMax; // some big positive number
    algorithmFPType GMin2 = fpMax;

    const algorithmFPType zero(0.0);
    const algorithmFPType two(2.0);
    algorithmFPType Kii = _kernelDiag[Bi];

    Status s;
    size_t nBlocks = nActiveVectors / kernelFunctionBlockSize;
    if (nBlocks * kernelFunctionBlockSize < nActiveVectors)
    {
        nBlocks++;
    }
    for (size_t iBlock = 0; iBlock < nBlocks; iBlock++)
    {
        size_t jStart = iBlock * kernelFunctionBlockSize;
        size_t jEnd   = jStart + kernelFunctionBlockSize;
        if (jEnd > nActiveVectors)
        {
            jEnd = nActiveVectors;
        }

        const algorithmFPType * KiBlock = nullptr;
        s                               = _cache->getRowBlock(Bi, jStart, (jEnd - jStart), KiBlock);
        if (!s) break;

        int Bj_local = -1;
        algorithmFPType GMin_local, GMin2_local, delta_local;
        WSSjLocal(jStart, jEnd, KiBlock, GMax, Kii, tau, Bj_local, GMin_local, GMin2_local, delta_local);

        if (GMin_local <= GMin)
        {
            GMin  = GMin_local;
            Bj    = Bj_local;
            delta = delta_local;
        }
        if (GMin2_local <= GMin2)
        {
            GMin2 = GMin2_local;
        }
    }
    res = GMin2;
    return s;
}

template <typename algorithmFPType, typename ParameterType, CpuType cpu>
bool SVMTrainTask<algorithmFPType, ParameterType, cpu>::findMaximumViolatingPair(size_t nActiveVectors, algorithmFPType tau, int & Bi, int & Bj,
                                                                                 algorithmFPType & delta, algorithmFPType & ma, algorithmFPType & Ma,
                                                                                 algorithmFPType & curEps, Status & s) const
{
    Bi = -1;
    ma = WSSi(nActiveVectors, Bi);
    if (Bi == -1) return false;

    Bj = -1;
    s |= WSSj(nActiveVectors, tau, Bi, ma, Bj, delta, Ma);
    curEps = ma - Ma;
    return (Bj != -1);
}

template <typename algorithmFPType, typename ParameterType, CpuType cpu>
Status SVMTrainTask<algorithmFPType, ParameterType, cpu>::update(size_t nActiveVectors, algorithmFPType C, int Bi, int Bj, algorithmFPType delta)
{
    /* Update alpha */
    algorithmFPType newDeltai, newDeltaj;
    updateAlpha(C, Bi, Bj, delta, newDeltai, newDeltaj);
    updateI(C, Bj);
    updateI(C, Bi);

    const algorithmFPType dyj = _y[Bj] * newDeltaj;
    const algorithmFPType dyi = _y[Bi] * newDeltai;

    /* Update gradient */
    const size_t blockSize = (kernelFunctionBlockSize >> 1); // 2 rows from kernel function matrix are used
    size_t nBlocks         = nActiveVectors / blockSize;
    if (nBlocks * blockSize < nActiveVectors) ++nBlocks;

    algorithmFPType * grad    = _grad.get();
    const algorithmFPType * y = _y.get();
    Status s;
    for (size_t iBlock = 0; s.ok() && (iBlock < nBlocks); iBlock++)
    {
        size_t tStart = iBlock * blockSize;
        size_t tEnd   = tStart + blockSize;
        if (tEnd > nActiveVectors)
        {
            tEnd = nActiveVectors;
        }

        const algorithmFPType * KiBlock = nullptr;
        const algorithmFPType * KjBlock = nullptr;
        s                               = _cache->getTwoRowsBlock(Bi, Bj, tStart, (tEnd - tStart), KiBlock, KjBlock);

        for (size_t t = tStart; t < tEnd; t++)
        {
            grad[t] += dyi * y[t] * KiBlock[t - tStart];
            grad[t] += dyj * y[t] * KjBlock[t - tStart];
        }
    }
    return s;
}

/**
 * \brief Update the array of classification coefficients. Step 4 of the Algorithm 1 in [1].
 *
 * \param[in] C          Upper bound in constraints of the quadratic optimization problem
 * \param[in] Bi         Index i from a pair of working set indices B = {i, j}
 * \param[in] Bj         Index j from a pair of working set indices B = {i, j}
 * \param[in] delta      Estimated difference between old and new value of the classification coefficients
 * \param[out] newDeltai Resulting difference between old and new value of the Bi-th classification coefficient
 * \param[out] newDeltaj Resulting difference between old and new value of the Bj-th classification coefficient
 */
template <typename algorithmFPType, typename ParameterType, CpuType cpu>
inline void SVMTrainTask<algorithmFPType, ParameterType, cpu>::updateAlpha(algorithmFPType C, int Bi, int Bj, algorithmFPType delta,
                                                                           algorithmFPType & newDeltai, algorithmFPType & newDeltaj)
{
    const algorithmFPType zero(0.0);
    const algorithmFPType oldAlphai = _alpha[Bi];
    const algorithmFPType oldAlphaj = _alpha[Bj];
    const algorithmFPType yi        = _y[Bi];
    const algorithmFPType yj        = _y[Bj];

    algorithmFPType newAlphai = oldAlphai + yi * delta;
    algorithmFPType newAlphaj = oldAlphaj - yj * delta;

    /* Project alpha back to the feasible region */
    algorithmFPType sum = yi * oldAlphai + yj * oldAlphaj;

    if (newAlphai > C)
    {
        newAlphai = C;
    }
    if (newAlphai < zero)
    {
        newAlphai = zero;
    }
    newAlphaj = yj * (sum - yi * newAlphai);

    if (newAlphaj > C)
    {
        newAlphaj = C;
    }
    if (newAlphaj < zero)
    {
        newAlphaj = zero;
    }
    newAlphai = yi * (sum - yj * newAlphaj);

    _alpha[Bj] = newAlphaj;
    _alpha[Bi] = newAlphai;
    newDeltai  = newAlphai - oldAlphai;
    newDeltaj  = newAlphaj - oldAlphaj;
}

/**
 * \brief Performs shrinking by moving the variables that corresponf to the shrunk feature vectors
 *        at the end of the respective arrays.
 *
 * \param[in] nActiveVectors Number of observations in a training data set that are used
 *                           in sequential minimum optimization at the current iteration
 * \return Number of observations that remain active (not shrunk) after shrinking
 */
template <typename algorithmFPType, typename ParameterType, CpuType cpu>
size_t SVMTrainTask<algorithmFPType, ParameterType, cpu>::doShrink(size_t nActiveVectors)
{
    size_t i                     = 0;
    size_t j                     = nActiveVectors - 1;
    algorithmFPType * y          = _y.get();
    algorithmFPType * grad       = _grad.get();
    algorithmFPType * alpha      = _alpha.get();
    algorithmFPType * kernelDiag = _kernelDiag.get();
    char * I                     = _I.get();
    while (i < j)
    {
        while (!(I[i] & shrink) && i < nActiveVectors - 1) i++;
        while ((I[j] & shrink) && j > 0) j--;
        if (i >= j) break;
        daal::services::internal::swap<cpu, char>(I[i], I[j]);
        daal::services::internal::swap<cpu, algorithmFPType>(y[i], y[j]);
        daal::services::internal::swap<cpu, algorithmFPType>(alpha[i], alpha[j]);
        daal::services::internal::swap<cpu, algorithmFPType>(grad[i], grad[j]);
        daal::services::internal::swap<cpu, algorithmFPType>(kernelDiag[i], kernelDiag[j]);
    }
    return i;
}

/**
 * \brief Update the array of flags that specify if the feature vector will be shrunk or not.
 *
 * \param[in] nActiveVectors Number of observations in a training data set that are used
 *                           in sequential minimum optimization at the current iteration
 * \param[in] C              Upper bound in constraints of the quadratic optimization problem
 * \param[in] ma             Value of m(alpha) = max(-y[i]*grad[i]): i belongs to I_UP (alpha)  (see p.1891, eqn.6 in [1])
 * \param[in] Ma             Value of M(alpha) = min(-y[i]*grad[i]): i belongs to I_LOW (alpha)
 * \return Number of observations to be shrunk
 */
template <typename algorithmFPType, typename ParameterType, CpuType cpu>
size_t SVMTrainTask<algorithmFPType, ParameterType, cpu>::updateShrinkingFlags(size_t nActiveVectors, algorithmFPType C, algorithmFPType ma,
                                                                               algorithmFPType Ma)
{
    const algorithmFPType * y     = _y.get();
    const algorithmFPType * grad  = _grad.get();
    const algorithmFPType * alpha = _alpha.get();
    char * I                      = _I.get();
    size_t nShrink                = 0;
    for (size_t i = 0; i < nActiveVectors; i++)
    {
        I[i] &= (~shrink);
        const algorithmFPType yg = -y[i] * grad[i];
        if (yg > ma)
        {
            if (alpha[i] <= 0.0 && y[i] == -1.0)
            {
                I[i] |= shrink;
                nShrink++;
            }
            if (alpha[i] >= C && y[i] == 1.0)
            {
                I[i] |= shrink;
                nShrink++;
            }
        }
        if (yg < Ma)
        {
            if (alpha[i] <= 0.0 && y[i] == 1.0)
            {
                I[i] |= shrink;
                nShrink++;
            }
            if (alpha[i] >= C && y[i] == -1.0)
            {
                I[i] |= shrink;
                nShrink++;
            }
        }
    }
    return nShrink;
}

/**
 * \brief Recompute the values of optimization function gradient for the vectors that were shrunk
 *
 * \param[in] nVectors       Number of observations in a training data set
 * \param[in] nActiveVectors Number of observations in a training data set that are used
 *                           in sequential minimum optimization at the current iteration
 */
template <typename algorithmFPType, typename ParameterType, CpuType cpu>
Status SVMTrainTask<algorithmFPType, ParameterType, cpu>::reconstructGradient(size_t & nActiveVectors)
{
    size_t nBlocks = _nVectors / kernelFunctionBlockSize;
    if (nBlocks * kernelFunctionBlockSize < _nVectors)
    {
        nBlocks++;
    }
    algorithmFPType * grad        = _grad.get();
    const algorithmFPType * y     = _y.get();
    const algorithmFPType * alpha = _alpha.get();
    Status s;
    for (size_t i = nActiveVectors; i < _nVectors; i++)
    {
        algorithmFPType yi = y[i];
        grad[i]            = algorithmFPType(-1.0);

        for (size_t jBlock = 0; s.ok() && (jBlock < nBlocks); jBlock++)
        {
            size_t jStart = jBlock * kernelFunctionBlockSize;
            size_t jEnd   = jStart + kernelFunctionBlockSize;
            if (jEnd > _nVectors)
            {
                jEnd = _nVectors;
            }

            const algorithmFPType * cacheRow = nullptr;
            s                                = _cache->getRowBlock(i, jStart, (jEnd - jStart), cacheRow);
            for (size_t j = jStart; j < jEnd; j++)
            {
                grad[i] += yi * y[j] * cacheRow[j - jStart] * alpha[j];
            }
        }
    }
    nActiveVectors = _nVectors;
    return s;
}

/**
 * \brief Construct the structure that stores the intermediate data used in SVM training
 *
 * \param[in] svmPar        Parameters of the algorithm
 * \param[in] xTable        Pointer to numeric table that contains input data set
 * \param[in] yTable        Pointer to numeric table that contains class labels
 */
template <typename algorithmFPType, typename ParameterType, CpuType cpu>
Status SVMTrainTask<algorithmFPType, ParameterType, cpu>::setup(const ParameterType & svmPar, const NumericTablePtr & xTable, NumericTable & yTable)
{
    _alpha.reset(_nVectors);
    daal::services::internal::service_memset<algorithmFPType, cpu>(_alpha.get(), algorithmFPType(0.0), _nVectors);
    _I.reset(_nVectors);
    daal::services::internal::service_memset<char, cpu>(_I.get(), char(0), _nVectors);
    _y.reset(_nVectors);
    _grad.reset(_nVectors);
    _kernelDiag.reset(_nVectors);
    DAAL_CHECK_MALLOC(_alpha.get() && _I.get() && _y.get() && _grad.get() && _kernelDiag.get());

    kernel_function::KernelIfacePtr kernel = svmPar.kernel->clone();
    size_t cacheSize                       = svmPar.cacheSize;
    Status s;
    if (cacheSize >= _nVectors * _nVectors * sizeof(algorithmFPType))
    {
        _cache = SVMCache<simpleCache, algorithmFPType, cpu>::create(_nVectors, svmPar.doShrinking, xTable, kernel, s);
    }
    else
    {
        cacheSize = kernelFunctionBlockSize;
        _cache    = SVMCache<noCache, algorithmFPType, cpu>::create(cacheSize, _nVectors, svmPar.doShrinking, xTable, kernel, s);
    }
    if (!s) return s;
    DAAL_ASSERT(_cache);
    ReadColumns<algorithmFPType, cpu> mtY(yTable, 0, 0, _nVectors);
    DAAL_CHECK_BLOCK_STATUS(mtY);
    int result =
        daal::services::internal::daal_memcpy_s(_y.get(), _nVectors * sizeof(algorithmFPType), mtY.get(), _nVectors * sizeof(algorithmFPType));
    return (!result) ? Status() : Status(ErrorMemoryCopyFailedInternal);
}

template <typename algorithmFPType, typename ParameterType, CpuType cpu>
SVMTrainTask<algorithmFPType, ParameterType, cpu>::~SVMTrainTask()
{
    delete _cache;
}

/**
 * \brief Initialize the intermediate data used in SVM training
 *
 * \param[in] C     Upper bound in constraints of the quadratic optimization problem
 */
template <typename algorithmFPType, typename ParameterType, CpuType cpu>
Status SVMTrainTask<algorithmFPType, ParameterType, cpu>::init(algorithmFPType C)
{
    const algorithmFPType negOne(-1.0);
    algorithmFPType * grad = _grad.get();
    for (size_t i = 0; i < _nVectors; i++)
    {
        grad[i] = negOne;
        updateI(C, i);
    }

    Status s;
    algorithmFPType * kernelDiag = _kernelDiag.get();
    for (size_t i = 0; s.ok() && (i < _nVectors); i++)
    {
        const algorithmFPType * KiiPtr = nullptr;
        s                              = _cache->getRowBlock(i, i, 1, KiiPtr);
        kernelDiag[i]                  = *KiiPtr;
    }
    return s;
}

/**
 * \brief Update the flag that specify the status of the feature vector in the input data set
 *
 * \param[in] C     Upper bound in constraints of the quadratic optimization problem
 * \param[in] index Index of the feature vector
 */
template <typename algorithmFPType, typename ParameterType, CpuType cpu>
inline void SVMTrainTask<algorithmFPType, ParameterType, cpu>::updateI(algorithmFPType C, size_t index)
{
    const algorithmFPType zero(0.0);
    const algorithmFPType one(1.0);
    char Ii                = _I[index];
    algorithmFPType alphai = _alpha[index];
    algorithmFPType yi     = _y[index];
    Ii &= (char)shrink;
    if (alphai < C && yi == +one)
    {
        Ii |= up;
    }
    if (alphai > zero && yi == -one)
    {
        Ii |= up;
    }
    if (alphai < C && yi == -one)
    {
        Ii |= low;
    }
    if (alphai > zero && yi == +one)
    {
        Ii |= low;
    }
    _I[index] = Ii;
}

/**
 * \brief Move the indices of the shrunk feature vector to the end of the array
 *
 * \param[in] nActiveVectors Number of observations in a training data set that are used
 *                           in sequential minimum optimization at the current iteration
 * \param[in] I              Array of flags that describe the status of feature vectors
 * \return                   Status of the call
 */
template <typename algorithmFPType, CpuType cpu>
Status SVMCache<noCache, algorithmFPType, cpu>::updateShrinkingRowIndices(size_t nActiveVectors, const char * I)
{
    size_t i = 0;
    size_t j = nActiveVectors - 1;
    while (i < j)
    {
        while (!(I[i] & shrink) && i < nActiveVectors - 1) i++;
        while ((I[j] & shrink) && j > 0) j--;
        if (i >= j) break;
        daal::services::internal::swap<cpu, size_t>(_shrinkingRowIndices[i], _shrinkingRowIndices[j]);
        i++;
        j--;
    }
    return Status();
}

/**
 * \brief Move the indices of the shrunk feature vector to the end of the array and
 *        re-order rows and columns in the cache accordingly
 *
 * \param[in] nActiveVectors Number of observations in a training data set that are used
 *                           in sequential minimum optimization at the current iteration
 * \param[in] I              Array of flags that describe the status of feature vectors
 * \return                   Status of the call
 */
template <typename algorithmFPType, CpuType cpu>
Status SVMCache<simpleCache, algorithmFPType, cpu>::updateShrinkingRowIndices(size_t nActiveVectors, const char * I)
{
    size_t i   = 0;
    size_t j   = nActiveVectors - 1;
    int result = 0;
    while (i < j)
    {
        while (!(I[i] & shrink) && i < nActiveVectors - 1) i++;
        while ((I[j] & shrink) && j > 0) j--;
        if (i >= j) break;
        daal::services::internal::swap<cpu, size_t>(_shrinkingRowIndices[i], _shrinkingRowIndices[j]);

        algorithmFPType cacheii = _cache[i * _lineSize + i];
        algorithmFPType cacheij = _cache[i * _lineSize + j];
        algorithmFPType cachejj = _cache[j * _lineSize + j];

        /* Swap i-th and j-th row in cache */
        size_t lineSizeInBytes = _lineSize * sizeof(algorithmFPType);
        result |= daal::services::internal::daal_memcpy_s(_tmp.get(), lineSizeInBytes, _cache.get() + i * _lineSize, lineSizeInBytes);
        result |=
            daal::services::internal::daal_memcpy_s(_cache.get() + i * _lineSize, lineSizeInBytes, _cache.get() + j * _lineSize, lineSizeInBytes);
        result |= daal::services::internal::daal_memcpy_s(_cache.get() + j * _lineSize, lineSizeInBytes, _tmp.get(), lineSizeInBytes);

        /* Swap i-th and j-th column in cache */
        for (size_t k = 0; k < _nLines; k++)
        {
            daal::services::internal::swap<cpu, algorithmFPType>(_cache[i + k * _lineSize], _cache[j + k * _lineSize]);
        }

        // maybe not needed
        _cache[i * _lineSize + i] = cachejj;
        _cache[j * _lineSize + j] = cacheii;
        _cache[j * _lineSize + i] = cacheij;
        _cache[i * _lineSize + j] = cacheij;
        i++;
        j--;
    }
    return (!result) ? Status() : Status(ErrorMemoryCopyFailedInternal);
}
} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
