/* file: cross_entropy_loss_dense_default_batch_v1_impl.i */
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
//  Implementation of cross-entropy loss algorithm
//--
*/
#include "service_math.h"
#include "service_utils.h"
#include "service_environment.h"

#include "service_ittnotify.h"

DAAL_ITTNOTIFY_DOMAIN(cross_entropy_loss.dense.default.batch);

#include "common/objective_function_utils.i"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace cross_entropy_loss
{
namespace internal
{
//////////////////////////////////////////////////////////////////////////////////////////
// Cross entropy loss function, L(x,y,b)=(1/n)*sum l(xi, yi, b),
// where l(x, y, b) =-sum(I(y=k)*ln(pk)), pk = exp(fk)/sum(exp(f)), fk = x*bk
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
static void applyBetaImpl(const algorithmFPType * x, const algorithmFPType * beta, algorithmFPType * xb, size_t nRows, size_t nClasses, size_t nCols,
                          bool bIntercept, bool bThreaded)
{
    char trans           = 'T';
    char notrans         = 'N';
    algorithmFPType one  = 1.0;
    algorithmFPType zero = 0.0;
    DAAL_INT m           = (DAAL_INT)nClasses;
    DAAL_INT n           = (DAAL_INT)nRows;
    DAAL_INT k           = (DAAL_INT)nCols;
    size_t nBetaPerClass = nCols + 1;
    DAAL_INT ldb         = (DAAL_INT)nBetaPerClass;
    if (bThreaded)
        Blas<algorithmFPType, cpu>::xgemm(&trans, &notrans, &m, &n, &k, &one, beta + 1, &ldb, x, &k, &zero, xb, &m);
    else
        Blas<algorithmFPType, cpu>::xxgemm(&trans, &notrans, &m, &n, &k, &one, beta + 1, &ldb, x, &k, &zero, xb, &m);
    if (bIntercept)
    {
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < nRows; ++i)
        {
            for (size_t j = 0; j < nClasses; ++j) xb[i * nClasses + j] += beta[j * nBetaPerClass + 0];
        }
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
void I1CrossEntropyLossKernel<algorithmFPType, method, cpu>::applyBeta(const algorithmFPType * x, const algorithmFPType * beta, algorithmFPType * xb,
                                                                       size_t nRows, size_t nClasses, size_t nCols, bool bIntercept)
{
    applyBetaImpl<algorithmFPType, cpu>(x, beta, xb, nRows, nClasses, nCols, bIntercept, false);
}

template <typename algorithmFPType, Method method, CpuType cpu>
void I1CrossEntropyLossKernel<algorithmFPType, method, cpu>::applyBetaThreaded(const algorithmFPType * x, const algorithmFPType * beta,
                                                                               algorithmFPType * xb, size_t nRows, size_t nClasses, size_t nCols,
                                                                               bool bIntercept)
{
    applyBetaImpl<algorithmFPType, cpu>(x, beta, xb, nRows, nClasses, nCols, bIntercept, true);
}

template <typename algorithmFPType, Method method, CpuType cpu>
void I1CrossEntropyLossKernel<algorithmFPType, method, cpu>::softmax(const algorithmFPType * arg, algorithmFPType * res, size_t nRows, size_t nCols)
{
    const algorithmFPType expThreshold = daal::internal::Math<algorithmFPType, cpu>::vExpThreshold();
    for (size_t iRow = 0; iRow < nRows; ++iRow)
    {
        const algorithmFPType * pArg = arg + iRow * nCols;
        algorithmFPType * pRes       = res + iRow * nCols;
        algorithmFPType maxArg       = pArg[0];
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 1; i < nCols; ++i)
        {
            if (maxArg < pArg[i]) maxArg = pArg[i];
        }
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < nCols; ++i)
        {
            pRes[i] = pArg[i] - maxArg;
            /* make all values less than threshold as threshold value
            to fix slow work on vExp on large negative inputs */
            if (pRes[i] < expThreshold) pRes[i] = expThreshold;
        }
    }
    daal::internal::Math<algorithmFPType, cpu>::vExp(nRows * nCols, res, res);
    for (size_t iRow = 0; iRow < nRows; ++iRow)
    {
        algorithmFPType * pRes = res + iRow * nCols;
        algorithmFPType sum(0.);
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < nCols; ++i) sum += pRes[i];
        sum = algorithmFPType(1.) / sum;
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < nCols; ++i) pRes[i] *= sum;
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
void I1CrossEntropyLossKernel<algorithmFPType, method, cpu>::softmaxThreaded(const algorithmFPType * arg, algorithmFPType * res, size_t nRows,
                                                                             size_t nCols)
{
    const size_t nRowsInBlockDefault = 500;
    const size_t nRowsInBlock        = services::internal::getNumElementsFitInMemory(services::internal::getL1CacheSize() * 0.8,
                                                                              nCols * sizeof(algorithmFPType), nRowsInBlockDefault);
    const size_t nDataBlocks         = nRows / nRowsInBlock + !!(nRows % nRowsInBlock);
    daal::threader_for(nDataBlocks, nDataBlocks, [&](size_t iBlock) {
        const size_t iStartRow       = iBlock * nRowsInBlock;
        const size_t nRowsToProcess  = (iBlock == nDataBlocks - 1) ? nRows - iBlock * nRowsInBlock : nRowsInBlock;
        const algorithmFPType * pArg = arg + iStartRow * nCols;
        algorithmFPType * pRes       = res + iStartRow * nCols;
        softmax(pArg, pRes, nRowsToProcess, nCols);
    });
}

template <typename algorithmFPType, CpuType cpu>
void addGradInPt(algorithmFPType * g, const algorithmFPType * xi, const algorithmFPType * pi, size_t yi, algorithmFPType interceptFactor,
                 size_t nClasses, size_t nBetaPerClass)
{
    for (size_t k = 0; k < nClasses; ++k)
    {
        const algorithmFPType pk = pi[k];
        algorithmFPType gk       = ((yi == k) ? (pk - algorithmFPType(1.)) : pk);
        g[nBetaPerClass * k] += interceptFactor * gk;
        for (size_t iBeta = 1; iBeta < nBetaPerClass; ++iBeta) g[nBetaPerClass * k + iBeta] += gk * xi[iBeta - 1];
    }
}

template <typename algorithmFPType, CpuType cpu>
void addHessInPt(algorithmFPType * h, const algorithmFPType * xi, const algorithmFPType * pi, const algorithmFPType interceptFactor, size_t nClasses,
                 size_t nBetaPerClass, size_t nBetaTotal)
{
    for (size_t k = 0; k < nBetaTotal; k++)
    {
        const algorithmFPType pk  = pi[k / nBetaPerClass];
        const algorithmFPType xij = k % nBetaPerClass ? xi[k % nBetaPerClass - 1] : interceptFactor;
        for (size_t m = k; m < nBetaTotal; m++)
        {
            const algorithmFPType pm  = pi[m / nBetaPerClass];
            const algorithmFPType xit = m % nBetaPerClass ? xi[m % nBetaPerClass - 1] : interceptFactor;
            const algorithmFPType pxx = pk * xij * xit;
            h[k * nBetaTotal + m] -= pm * pxx;
            h[k * nBetaTotal + m] += (k / nBetaPerClass == m / nBetaPerClass) ? pxx : 0;
        }
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status I1CrossEntropyLossKernel<algorithmFPType, method, cpu>::doCompute(
    const algorithmFPType * x, const algorithmFPType * y, size_t nRows, size_t n, size_t p, NumericTable * betaNT, NumericTable * valueNT,
    NumericTable * hessianNT, NumericTable * gradientNT, NumericTable * nonSmoothTermValue, NumericTable * proximalProjection,
    NumericTable * lipschitzConstant, interface1::Parameter * parameter)
{
    const size_t nClasses = parameter->nClasses;

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, nClasses);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * nClasses, sizeof(algorithmFPType));

    TArrayScalable<algorithmFPType, cpu> f(n * nClasses);
    const size_t nBetaPerClass = p + 1;
    DAAL_ASSERT(betaNT->getNumberOfColumns() == 1);
    DAAL_ASSERT(betaNT->getNumberOfRows() == nClasses * nBetaPerClass);
    const size_t nBeta = betaNT->getNumberOfColumns() * betaNT->getNumberOfRows();
    ReadRows<algorithmFPType, cpu> betar(betaNT, 0, nBeta);
    DAAL_CHECK_BLOCK_STATUS(betar);
    const algorithmFPType * b = betar.get();

    if (proximalProjection)
    {
        WriteRows<algorithmFPType, cpu> proxPtr(proximalProjection, 0, nBeta);
        DAAL_CHECK_BLOCK_STATUS(proxPtr);
        algorithmFPType * prox = proxPtr.get();

        for (size_t i = 0; i < nClasses; i++) prox[i * nBetaPerClass] = b[i * nBetaPerClass];
        for (size_t i = 0; i < nClasses; i++)
        {
            for (size_t j = 1; j < nBetaPerClass; j++)
            {
                if (b[i * nBetaPerClass + j] > parameter->penaltyL1)
                {
                    prox[i * nBetaPerClass + j] = b[i * nBetaPerClass + j] - parameter->penaltyL1;
                }
                if (b[i * nBetaPerClass + j] < -parameter->penaltyL1)
                {
                    prox[i * nBetaPerClass + j] = b[i * nBetaPerClass + j] + parameter->penaltyL1;
                }
                if (daal::internal::Math<algorithmFPType, cpu>::sFabs(b[i * nBetaPerClass + j]) <= parameter->penaltyL1)
                {
                    prox[i * nBetaPerClass + j] = 0;
                }
            }
        }
    }

    algorithmFPType notSmoothTerm = 0;
    if (nonSmoothTermValue)
    {
        WriteRows<algorithmFPType, cpu> vr(nonSmoothTermValue, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(vr);
        algorithmFPType & value = *vr.get();
        for (size_t i = 0; i < nClasses; i++)
        {
            for (size_t j = 1; j < nBetaPerClass; j++)
                notSmoothTerm += (b[i * nBetaPerClass + j] < 0 ? -b[i * nBetaPerClass + j] : b[i * nBetaPerClass + j]) * parameter->penaltyL1;
        }
        value = notSmoothTerm;
    }

    if (lipschitzConstant)
    {
        DAAL_ASSERT(lipschitzConstant->getNumberOfRows() == 1);
        WriteRows<algorithmFPType, cpu> lipschitzConstantPtr(lipschitzConstant, 0, 1);
        algorithmFPType & c = *lipschitzConstantPtr.get();

        const size_t blockSize = 256;
        size_t nBlocks         = n / blockSize;
        nBlocks += (nBlocks * blockSize != n);
        algorithmFPType globalMaxNorm = 0;

        TlsMem<algorithmFPType, cpu, services::internal::ScalableCalloc<algorithmFPType, cpu> > tlsData(lipschitzConstant->getNumberOfRows());
        daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
            algorithmFPType & _maxNorm = *tlsData.local();
            const size_t startRow      = iBlock * blockSize;
            const size_t finishRow     = (iBlock + 1 == nBlocks ? n : (iBlock + 1) * blockSize);
            algorithmFPType curentNorm = 0;
            for (size_t i = startRow; i < finishRow; i++)
            {
                curentNorm = 0;

                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (int j = 0; j < p; j++)
                {
                    curentNorm += x[i * p + j] * x[i * p + j];
                }
                if (curentNorm > _maxNorm)
                {
                    _maxNorm = curentNorm;
                }
            }
        });
        tlsData.reduce([&](algorithmFPType * maxNorm) {
            if (globalMaxNorm < *maxNorm)
            {
                globalMaxNorm = *maxNorm;
            }
        });

        algorithmFPType alpha_scaled = algorithmFPType(parameter->penaltyL2) / algorithmFPType(n);
        algorithmFPType lipschitz    = 0.25 * (globalMaxNorm + algorithmFPType(parameter->interceptFlag)) + alpha_scaled;
        algorithmFPType displacement = daal::internal::Math<algorithmFPType, cpu>::sMin(2 * parameter->penaltyL2, lipschitz);
        c                            = 2 * lipschitz + displacement;
    }

    if (valueNT || gradientNT || hessianNT)
    {
        //f = X*b + b0
        applyBetaThreaded(x, b, f.get(), n, nClasses, p, parameter->interceptFlag);

        //f = softmax(f)
        softmaxThreaded(f.get(), f.get(), n, nClasses);

        const bool bL1          = (parameter->penaltyL1 > 0);
        const bool bL2          = (parameter->penaltyL2 > 0);
        const size_t iFirstBeta = parameter->interceptFlag ? 0 : nClasses;

        const algorithmFPType div = algorithmFPType(1) / algorithmFPType(n);
        if (valueNT)
        {
            TArrayScalable<algorithmFPType, cpu> logP(f.size());
            daal::internal::Math<algorithmFPType, cpu>::vLog(n * nClasses, f.get(), logP.get());

            WriteRows<algorithmFPType, cpu> vr(valueNT, 0, 1);
            DAAL_CHECK_BLOCK_STATUS(vr);
            algorithmFPType & value               = *vr.get();
            const algorithmFPType interceptFactor = (parameter->interceptFlag ? 1 : 0);
            value                                 = 0.0;
            const algorithmFPType * lp            = logP.get();

            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j < nClasses; ++j) value += (size_t(y[i]) == j) * lp[i * nClasses + j];
            }

            value *= -div;

            if (bL2)
            {
                for (size_t i = 0; i < nClasses; i++)
                {
                    for (size_t j = 1; j < nBetaPerClass; j++) value += b[i * nBetaPerClass + j] * b[i * nBetaPerClass + j] * parameter->penaltyL2;
                }
            }

            if (bL1)
            {
                if (nonSmoothTermValue)
                {
                    value += notSmoothTerm;
                }
                else
                {
                    for (size_t i = 0; i < nClasses; i++)
                    {
                        for (size_t j = 1; j < nBetaPerClass; j++)
                            value += (b[i * nBetaPerClass + j] < 0 ? -b[i * nBetaPerClass + j] : b[i * nBetaPerClass + j]) * parameter->penaltyL1;
                    }
                }
            }
        }

        if (gradientNT)
        {
            const algorithmFPType * pp = f.get();
            DAAL_ASSERT(gradientNT->getNumberOfRows() == nBeta);
            WriteRows<algorithmFPType, cpu> gr(gradientNT, 0, nBeta);
            DAAL_CHECK_BLOCK_STATUS(gr);
            algorithmFPType * g                   = gr.get();
            const algorithmFPType interceptFactor = (parameter->interceptFlag ? 1 : 0);
            if (n > 10 * daal::threader_get_threads_number())
            {
                TlsSum<algorithmFPType, cpu> tlsData(nBeta);
                daal::threader_for(n, n, [&](size_t i) {
                    addGradInPt<algorithmFPType, cpu>(tlsData.local(), x + i * p, pp + i * nClasses, size_t(y[i]), interceptFactor, nClasses,
                                                      nBetaPerClass);
                });
                tlsData.reduceTo(g, nBeta);
            }
            else
            {
                for (size_t i = 0; i < nBeta; ++i) g[i] = 0;
                for (size_t i = 0; i < n; ++i)
                    addGradInPt<algorithmFPType, cpu>(g, x + i * p, pp + i * nClasses, size_t(y[i]), interceptFactor, nClasses, nBetaPerClass);
            }

            for (size_t i = 0; i < nBeta; ++i) g[i] *= div;

            if (bL2)
            {
                for (size_t i = 0; i < nClasses; i++)
                {
                    for (size_t j = 1; j < nBetaPerClass; j++) g[i * nBetaPerClass + j] += 2 * b[i * nBetaPerClass + j] * parameter->penaltyL2;
                }
            }
        }
        if (hessianNT)
        {
            const algorithmFPType * pp = f.get();
            WriteRows<algorithmFPType, cpu> hr(hessianNT, 0, n);
            DAAL_CHECK_BLOCK_STATUS(hr);
            DAAL_ASSERT(hessianNT->getNumberOfColumns() == nBeta);
            DAAL_ASSERT(hessianNT->getNumberOfRows() == nBeta);
            algorithmFPType * h                   = hr.get();
            const algorithmFPType interceptFactor = (parameter->interceptFlag ? 1 : 0);
            const auto hSize                      = nBeta * nBeta;
            TlsSum<algorithmFPType, cpu> tlsData(hSize);

            daal::threader_for(n, n, [&](size_t i) {
                addHessInPt<algorithmFPType, cpu>(tlsData.local(), x + i * p, pp + i * nClasses, interceptFactor, nClasses, nBetaPerClass, nBeta);
            });
            tlsData.reduceTo(h, hSize);

            //hessian is a symmetrical matrix
            for (size_t i = 0; i < nBeta; ++i)
            {
                h[i * nBeta + i] *= div;
                for (size_t j = i + 1; j < nBeta; ++j)
                {
                    h[i * nBeta + j] *= div;
                    h[j * nBeta + i] = h[i * nBeta + j];
                }
            }

            if (bL2)
            {
                for (size_t i = 0; i < nBeta; i++)
                {
                    const algorithmFPType regularValue = 2 * parameter->penaltyL2;
                    h[i * nBeta + i] += (i % nBetaPerClass) ? regularValue : 0;
                }
            }
        }
    }
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status I1CrossEntropyLossKernel<algorithmFPType, method, cpu>::compute(NumericTable * dataNT, NumericTable * dependentVariablesNT,
                                                                                 NumericTable * betaNT, NumericTable * valueNT,
                                                                                 NumericTable * hessianNT, NumericTable * gradientNT,
                                                                                 NumericTable * nonSmoothTermValue, NumericTable * proximalProjection,
                                                                                 NumericTable * lipschitzConstant, interface1::Parameter * parameter)
{
    const size_t nRows                                = dataNT->getNumberOfRows();
    const daal::data_management::NumericTable * ntInd = parameter->batchIndices.get();
    if (ntInd && (ntInd->getNumberOfColumns() == nRows)) ntInd = nullptr;
    services::Status s;
    const size_t p = dataNT->getNumberOfColumns();
    if (ntInd)
    {
        const size_t n = ntInd->getNumberOfColumns();

        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, sizeof(algorithmFPType));
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, p);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * p, sizeof(algorithmFPType));

        TArrayScalable<algorithmFPType, cpu> aX(n * p);
        TArrayScalable<algorithmFPType, cpu> aY(n);
        s |= objective_function::internal::getXY<algorithmFPType, cpu>(dataNT, dependentVariablesNT, ntInd, aX.get(), aY.get(), nRows, n, p);
        s |= doCompute(aX.get(), aY.get(), nRows, n, p, betaNT, valueNT, hessianNT, gradientNT, nonSmoothTermValue, proximalProjection,
                       lipschitzConstant, parameter);
        return s;
    }
    ReadRows<algorithmFPType, cpu> xr(dataNT, 0, nRows);
    ReadRows<algorithmFPType, cpu> yr(dependentVariablesNT, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(xr);
    DAAL_CHECK_BLOCK_STATUS(yr);

    return doCompute(xr.get(), yr.get(), nRows, nRows, p, betaNT, valueNT, hessianNT, gradientNT, nonSmoothTermValue, proximalProjection,
                     lipschitzConstant, parameter);
}

} // namespace internal

} // namespace cross_entropy_loss

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal
