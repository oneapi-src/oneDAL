/* file: logistic_loss_dense_default_batch_v1_impl.i */
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
//  Implementation of logloss algorithm
//--
*/
#include "service_math.h"
#include "service_ittnotify.h"

DAAL_ITTNOTIFY_DOMAIN(logistic_loss.dense.default.batch);

#include "common/objective_function_utils.i"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace logistic_loss
{
namespace internal
{
//////////////////////////////////////////////////////////////////////////////////////////
// Logistic loss function, L(x,y,b) = -[y*ln(sigmoid(f)) + (1 - y)*ln(1-sigmoid(f))]
// where sigmoid(f) = 1/(1 + exp(-f), f = x*b
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
static void applyBetaImpl(const algorithmFPType * x, const algorithmFPType * beta, algorithmFPType * xb, size_t nRows, size_t nCols, bool bIntercept,
                          bool bThreaded)
{
    char trans           = 'T';
    algorithmFPType one  = 1.0;
    algorithmFPType zero = 0.0;
    DAAL_INT n           = (DAAL_INT)nRows;
    DAAL_INT dim         = (DAAL_INT)nCols;
    DAAL_INT ione        = 1;
    if (bThreaded)
    {
        Blas<algorithmFPType, cpu>::xgemv(&trans, &dim, &n, &one, x, &dim, beta + 1, &ione, &zero, xb, &ione);
    }
    else
    {
        Blas<algorithmFPType, cpu>::xxgemv(&trans, &dim, &n, &one, x, &dim, beta + 1, &ione, &zero, xb, &ione);
    }
    if (bIntercept)
    {
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < n; ++i) xb[i] += beta[0];
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
void I1LogLossKernel<algorithmFPType, method, cpu>::applyBeta(const algorithmFPType * x, const algorithmFPType * beta, algorithmFPType * xb,
                                                              size_t nRows, size_t nCols, bool bIntercept)
{
    applyBetaImpl<algorithmFPType, cpu>(x, beta, xb, nRows, nCols, bIntercept, false);
}

template <typename algorithmFPType, Method method, CpuType cpu>
void I1LogLossKernel<algorithmFPType, method, cpu>::applyBetaThreaded(const algorithmFPType * x, const algorithmFPType * beta, algorithmFPType * xb,
                                                                      size_t nRows, size_t nCols, bool bIntercept)
{
    applyBetaImpl<algorithmFPType, cpu>(x, beta, xb, nRows, nCols, bIntercept, true);
}

template <typename algorithmFPType, CpuType cpu>
static void vexp(const algorithmFPType * f, algorithmFPType * exp, size_t n)
{
    const algorithmFPType expThreshold = daal::internal::Math<algorithmFPType, cpu>::vExpThreshold();
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < n; ++i)
    {
        exp[i] = -f[i];
        /* make all values less than threshold as threshold value
        to fix slow work on vExp on large negative inputs */
        if (exp[i] < expThreshold) exp[i] = expThreshold;
    }
    daal::internal::Math<algorithmFPType, cpu>::vExp(n, exp, exp);
}

template <typename algorithmFPType, CpuType cpu>
static void sigmoids(algorithmFPType * exp, size_t n)
{
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < n; ++i)
    {
        const auto sigm = algorithmFPType(1.0) / (algorithmFPType(1.0) + exp[i]);
        exp[i]          = sigm;
        exp[i + n]      = 1 - sigm;
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
void I1LogLossKernel<algorithmFPType, method, cpu>::sigmoid(const algorithmFPType * f, algorithmFPType * s, size_t n)
{
    //s = exp(-f)
    vexp<algorithmFPType, cpu>(f, s, n);
    //s = sigm(f)
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < n; ++i) s[i] = algorithmFPType(1.0) / (algorithmFPType(1.0) + s[i]);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status I1LogLossKernel<algorithmFPType, method, cpu>::doCompute(const algorithmFPType * x, const algorithmFPType * y, size_t n, size_t p,
                                                                          NumericTable * betaNT, NumericTable * valueNT, NumericTable * hessianNT,
                                                                          NumericTable * gradientNT, NumericTable * nonSmoothTermValue,
                                                                          NumericTable * proximalProjection, NumericTable * lipschitzConstant,
                                                                          interface1::Parameter * parameter)
{
    const size_t nBeta = p + 1;
    DAAL_ASSERT(betaNT->getNumberOfColumns() == 1);
    DAAL_ASSERT(betaNT->getNumberOfRows() == nBeta);

    const algorithmFPType * b;
    HomogenNumericTable<algorithmFPType> * hmgBeta = dynamic_cast<HomogenNumericTable<algorithmFPType> *>(betaNT);
    ReadRows<algorithmFPType, cpu> betar;

    if (hmgBeta)
    {
        b = (*hmgBeta).getArray();
    }
    else
    {
        betar.set(betaNT, 0, nBeta);
        DAAL_CHECK_BLOCK_STATUS(betar);
        b = betar.get();
    }

    if (proximalProjection)
    {
        DAAL_ASSERT(proximalProjection->getNumberOfRows() == nBeta);
        algorithmFPType * prox;

        HomogenNumericTable<algorithmFPType> * hmgProx = dynamic_cast<HomogenNumericTable<algorithmFPType> *>(proximalProjection);
        WriteRows<algorithmFPType, cpu> pr;
        if (hmgProx)
        {
            prox = hmgProx->getArray();
        }
        else
        {
            pr.set(proximalProjection, 0, nBeta);
            DAAL_CHECK_BLOCK_STATUS(pr);
            prox = pr.get();
        }

        prox[0] = b[0];
        for (int i = 1; i < nBeta; i++)
        {
            if (b[i] > parameter->penaltyL1)
            {
                prox[i] = b[i] - parameter->penaltyL1;
            }
            if (b[i] < -parameter->penaltyL1)
            {
                prox[i] = b[i] + parameter->penaltyL1;
            }
            if (daal::internal::Math<algorithmFPType, cpu>::sFabs(b[i]) <= parameter->penaltyL1)
            {
                prox[i] = 0;
            }
        }
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

    algorithmFPType nonSmoothTerm = 0;
    if (nonSmoothTermValue)
    {
        WriteRows<algorithmFPType, cpu> vr(nonSmoothTermValue, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(vr);
        algorithmFPType & v = *vr.get();

        if ((parameter->penaltyL1 > 0))
        {
            for (size_t i = 1; i < nBeta; ++i)
            {
                nonSmoothTerm += (b[i] < 0 ? -b[i] : b[i]) * parameter->penaltyL1;
            }
        }
        v = nonSmoothTerm;
    }

    if (valueNT || gradientNT || hessianNT)
    {
        TNArray<algorithmFPType, 16, cpu> f;
        TNArray<algorithmFPType, 32, cpu> sg;

        TArrayScalable<algorithmFPType, cpu> fScalable;
        TArrayScalable<algorithmFPType, cpu> sgScalable;

        algorithmFPType * fPtr;
        algorithmFPType * sgPtr;
        if (n < 16)
        {
            f.reset(n);
            sg.reset(2 * n);
            fPtr  = f.get();
            sgPtr = sg.get();
        }
        else
        {
            fScalable.reset(n);
            sgScalable.reset(2 * n);
            fPtr  = fScalable.get();
            sgPtr = sgScalable.get();
        }
        //f = X*b + b0
        applyBetaThreaded(x, b, fPtr, n, p, parameter->interceptFlag);
        //s = exp(-f)
        vexp<algorithmFPType, cpu>(fPtr, sgPtr, n);

        //s = sigm(f), s1 = 1 - s
        sigmoids<algorithmFPType, cpu>(sgPtr, n);

        const bool bL1 = (parameter->penaltyL1 > 0);
        const bool bL2 = (parameter->penaltyL2 > 0);

        const size_t iFirstBeta   = parameter->interceptFlag ? 0 : 1;
        const algorithmFPType div = algorithmFPType(1) / algorithmFPType(n);

        if (valueNT)
        {
            TArrayScalable<algorithmFPType, cpu> logS(2 * n);
            daal::internal::Math<algorithmFPType, cpu>::vLog(2 * n, sgPtr, logS.get());

            const algorithmFPType * ls  = logS.get();
            const algorithmFPType * ls1 = ls + n;

            WriteRows<algorithmFPType, cpu> vr(valueNT, 0, 1);
            DAAL_CHECK_BLOCK_STATUS(vr);
            algorithmFPType & value = *vr.get();
            value                   = 0.0;
            for (size_t i = 0; i < n; ++i) value += y[i] * ls[i] + (algorithmFPType(1) - y[i]) * ls1[i];

            value *= -div;
            if (bL2)
            {
                for (size_t i = 1; i < nBeta; ++i) value += b[i] * b[i] * parameter->penaltyL2;
            }

            if (bL1)
            {
                if (nonSmoothTermValue)
                {
                    value += nonSmoothTerm;
                }
                else
                {
                    for (size_t i = 1; i < nBeta; ++i) value += (b[i] < 0 ? -b[i] : b[i]) * parameter->penaltyL1;
                }
            }
        }

        if (gradientNT)
        {
            DAAL_ASSERT(gradientNT->getNumberOfRows() == nBeta);
            const algorithmFPType * s = sgPtr;

            algorithmFPType * g;
            HomogenNumericTable<algorithmFPType> * hmgGrad = dynamic_cast<HomogenNumericTable<algorithmFPType> *>(gradientNT);
            WriteRows<algorithmFPType, cpu> gr;
            if (hmgGrad)
            {
                g = hmgGrad->getArray();
            }
            else
            {
                gr.set(gradientNT, 0, nBeta);
                DAAL_CHECK_BLOCK_STATUS(gr);
                g = gr.get();
            }
            const size_t nRowsInBlock = (p < 5000 ? 5000 / p : 1);
            const size_t nDataBlocks  = n / nRowsInBlock + !!(n % nRowsInBlock);
            const auto nThreads       = daal::threader_get_threads_number();

            if ((nThreads > 1) && (nDataBlocks > 1))
            {
                TlsSum<algorithmFPType, cpu> tlsData(nBeta);
                daal::threader_for(nDataBlocks, nDataBlocks, [&](size_t iBlock) {
                    const size_t iStartRow      = iBlock * nRowsInBlock;
                    const size_t nRowsToProcess = (iBlock == nDataBlocks - 1) ? n - iBlock * nRowsInBlock : nRowsInBlock;
                    const auto px               = x + iStartRow * p;
                    auto pg                     = tlsData.local();
                    for (size_t i = 0; i < nRowsToProcess; ++i)
                    {
                        const algorithmFPType d = (s[iStartRow + i] - y[iStartRow + i]);
                        for (size_t j = 0; j < p; ++j) pg[j + 1] += d * px[i * p + j];
                    }
                });
                tlsData.reduceTo(g, nBeta);
            }
            else
            {
                for (size_t i = 0; i < nBeta; ++i) g[i] = 0.0;
                for (size_t i = 0; i < n; ++i)
                {
                    const algorithmFPType d = (s[i] - y[i]);
                    PRAGMA_IVDEP
                    PRAGMA_VECTOR_ALWAYS
                    for (size_t j = 0; j < p; ++j) g[j + 1] += d * x[i * p + j];
                }
            }

            if (parameter->interceptFlag)
            {
                for (size_t i = 0; i < n; ++i) g[0] += (s[i] - y[i]);
            }
            for (size_t i = iFirstBeta; i < nBeta; ++i) g[i] *= div;

            if (bL2)
            {
                for (size_t i = 1; i < nBeta; ++i) g[i] += 2. * b[i] * parameter->penaltyL2;
            }
        }
        if (hessianNT)
        {
            DAAL_ASSERT(hessianNT->getNumberOfRows() == nBeta);
            WriteRows<algorithmFPType, cpu> hr(hessianNT, 0, nBeta * nBeta);
            DAAL_CHECK_BLOCK_STATUS(hr);
            algorithmFPType * h = hr.get();

            algorithmFPType * s = sgPtr;
            for (size_t i = 0; i < n; ++i) s[i] *= s[i + n]; //sigmoid derivative at x[i]

            h[0] = 0;
            if (parameter->interceptFlag)
            {
                for (size_t i = 0; i < n; ++i) h[0] += s[i];
                h[0] *= div; //average of sigmoid derivatives

                //first row and column
                for (size_t k = 1; k < nBeta; ++k)
                {
                    algorithmFPType val = 0;
                    for (size_t i = 0; i < n; ++i) val += s[i] * x[i * p + k - 1];
                    h[k]         = val * div;
                    h[k * nBeta] = val * div;
                }
            }
            else
            {
                //first row and column
                for (size_t k = 1; k < nBeta; ++k)
                {
                    h[k]         = 0;
                    h[k * nBeta] = 0;
                }
            }
            //rows 1,..
            for (size_t j = 1; j < nBeta; ++j)
            {
                for (size_t k = j; k < nBeta; ++k)
                {
                    algorithmFPType val = 0;
                    for (size_t i = 0; i < n; ++i) val += x[i * p + j - 1] * x[i * p + k - 1] * s[i];
                    h[j * nBeta + k] = val * div;
                    h[k * nBeta + j] = val * div;
                }
                h[j * nBeta + j] += 2. * parameter->penaltyL2;
            }
        }
    }
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status I1LogLossKernel<algorithmFPType, method, cpu>::compute(NumericTable * dataNT, NumericTable * dependentVariablesNT,
                                                                        NumericTable * betaNT, NumericTable * valueNT, NumericTable * hessianNT,
                                                                        NumericTable * gradientNT, NumericTable * nonSmoothTermValue,
                                                                        NumericTable * proximalProjection, NumericTable * lipschitzConstant,
                                                                        interface1::Parameter * parameter)
{
    const size_t nRows                                = dataNT->getNumberOfRows();
    const daal::data_management::NumericTable * ntInd = parameter->batchIndices.get();

    if (ntInd && (ntInd->getNumberOfColumns() == nRows)) ntInd = nullptr;
    services::Status s;
    const size_t p = dataNT->getNumberOfColumns();
    if (ntInd)
    {
        const size_t n                                               = ntInd->getNumberOfColumns();
        HomogenNumericTable<algorithmFPType> * hmgData               = dynamic_cast<HomogenNumericTable<algorithmFPType> *>(dataNT);
        HomogenNumericTable<algorithmFPType> * hmgDependentVariables = dynamic_cast<HomogenNumericTable<algorithmFPType> *>(dependentVariablesNT);

        if (n == 1 && hmgData && hmgDependentVariables)
        {
            int ind                    = ntInd->getValue<int>(0, 0);
            const algorithmFPType * aX = (*hmgData)[ind];
            const algorithmFPType * aY = (*hmgDependentVariables)[ind];
            s |=
                doCompute(aX, aY, n, p, betaNT, valueNT, hessianNT, gradientNT, nonSmoothTermValue, proximalProjection, lipschitzConstant, parameter);
            return s;
        }
        else
        {
            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, sizeof(algorithmFPType));
            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, p);
            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * p, sizeof(algorithmFPType));

            TArrayScalable<algorithmFPType, cpu> aX(n * p);
            TArrayScalable<algorithmFPType, cpu> aY(n);
            s |= objective_function::internal::getXY<algorithmFPType, cpu>(dataNT, dependentVariablesNT, ntInd, aX.get(), aY.get(), nRows, n, p);
            s |= doCompute(aX.get(), aY.get(), n, p, betaNT, valueNT, hessianNT, gradientNT, nonSmoothTermValue, proximalProjection,
                           lipschitzConstant, parameter);
        }
        return s;
    }

    ReadRows<algorithmFPType, cpu> xr(dataNT, 0, nRows);
    ReadRows<algorithmFPType, cpu> yr(dependentVariablesNT, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(xr);
    DAAL_CHECK_BLOCK_STATUS(yr);

    s |= doCompute(xr.get(), yr.get(), nRows, p, betaNT, valueNT, hessianNT, gradientNT, nonSmoothTermValue, proximalProjection, lipschitzConstant,
                   parameter);
    return s;
}

} // namespace internal

} // namespace logistic_loss

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal
