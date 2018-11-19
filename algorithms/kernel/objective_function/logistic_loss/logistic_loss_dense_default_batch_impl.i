/* file: logistic_loss_dense_default_batch_impl.i */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of logloss algorithm
//--
*/
#include "service_math.h"
#include "objective_function_utils.i"

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
template<typename algorithmFPType, CpuType cpu>
static void applyBetaImpl(const algorithmFPType* x, const algorithmFPType* beta,
    algorithmFPType* xb, size_t nRows, size_t nCols, bool bIntercept, bool bThreaded)
{
    char trans = 'T';
    algorithmFPType one = 1.0;
    algorithmFPType zero = 0.0;
    DAAL_INT n = (DAAL_INT)nRows;
    DAAL_INT dim = (DAAL_INT)nCols;
    DAAL_INT ione = 1;
    if(bThreaded)
        Blas<algorithmFPType, cpu>::xgemv(&trans, &dim, &n, &one, x, &dim, beta + 1, &ione, &zero, xb, &ione);
    else
        Blas<algorithmFPType, cpu>::xxgemv(&trans, &dim, &n, &one, x, &dim, beta + 1, &ione, &zero, xb, &ione);
    if(bIntercept)
    {
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < n; ++i)
            xb[i] += beta[0];
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
void LogLossKernel<algorithmFPType, method, cpu>::applyBeta(const algorithmFPType* x, const algorithmFPType* beta,
    algorithmFPType* xb, size_t nRows, size_t nCols, bool bIntercept)
{
    applyBetaImpl<algorithmFPType, cpu>(x, beta, xb, nRows, nCols, bIntercept, false);
}

template<typename algorithmFPType, Method method, CpuType cpu>
void LogLossKernel<algorithmFPType, method, cpu>::applyBetaThreaded(const algorithmFPType* x, const algorithmFPType* beta,
    algorithmFPType* xb, size_t nRows, size_t nCols, bool bIntercept)
{
    applyBetaImpl<algorithmFPType, cpu>(x, beta, xb, nRows, nCols, bIntercept, true);
}

template<typename algorithmFPType, CpuType cpu>
static void vexp(const algorithmFPType* f, algorithmFPType* exp, size_t n)
{
    const algorithmFPType expThreshold = daal::internal::Math<algorithmFPType, cpu>::vExpThreshold();
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for(size_t i = 0; i < n; ++i)
    {
        exp[i] = -f[i];
        /* make all values less than threshold as threshold value
        to fix slow work on vExp on large negative inputs */
        if(exp[i] < expThreshold)
            exp[i] = expThreshold;
    }
    daal::internal::Math<algorithmFPType, cpu>::vExp(n, exp, exp);
}

template<typename algorithmFPType, CpuType cpu>
static void sigmoids(algorithmFPType* exp, size_t n)
{
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for(size_t i = 0; i < n; ++i)
    {
        const auto sigm = algorithmFPType(1.0) / (algorithmFPType(1.0) + exp[i]);
        exp[i] = sigm;
        exp[i + n] = 1 - sigm;
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
void LogLossKernel<algorithmFPType, method, cpu>::sigmoid(const algorithmFPType* f, algorithmFPType* s, size_t n)
{
    //s = exp(-f)
    vexp<algorithmFPType, cpu>(f, s, n);
    //s = sigm(f)
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for(size_t i = 0; i < n; ++i)
        s[i] = algorithmFPType(1.0) / (algorithmFPType(1.0) + s[i]);
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status LogLossKernel<algorithmFPType, method, cpu>::doCompute(const algorithmFPType* x, const algorithmFPType* y,
    size_t n, size_t p, NumericTable *betaNT,
    NumericTable *valueNT, NumericTable *hessianNT, NumericTable *gradientNT, Parameter *parameter)
{
    TArrayScalable<algorithmFPType, cpu> f(n);
    TArrayScalable<algorithmFPType, cpu> sg(2*n);

    const size_t nBeta = p + 1;
    DAAL_ASSERT(betaNT->getNumberOfColumns() == 1);
    DAAL_ASSERT(betaNT->getNumberOfRows() == nBeta);
    ReadRows<algorithmFPType, cpu> betar(betaNT, 0, nBeta);
    DAAL_CHECK_BLOCK_STATUS(betar);
    const algorithmFPType* b = betar.get();

    //f = X*b + b0
    applyBetaThreaded(x, b, f.get(), n, p, parameter->interceptFlag);

    //s = exp(-f)
    vexp<algorithmFPType, cpu>(f.get(), sg.get(), n);

    //s = sigm(f), s1 = 1 - s
    sigmoids<algorithmFPType, cpu>(sg.get(), n);

    const bool bL1 = (parameter->penaltyL1 > 0);
    const bool bL2 = (parameter->penaltyL2 > 0);

    const size_t iFirstBeta = parameter->interceptFlag ? 0 : 1;
    const algorithmFPType div = algorithmFPType(1) / algorithmFPType(n);
    if(valueNT)
    {
        TArrayScalable<algorithmFPType, cpu> logS(2*n);
        daal::internal::Math<algorithmFPType, cpu>::vLog(2*n, sg.get(), logS.get());

        const algorithmFPType* ls = logS.get();
        const algorithmFPType* ls1 = ls + n;

        WriteRows<algorithmFPType, cpu> vr(valueNT, 0, n);
        DAAL_CHECK_BLOCK_STATUS(vr);
        algorithmFPType& value = *vr.get();
        value = 0.0;
        for(size_t i = 0; i < n; ++i)
            value += y[i] * ls[i] + (algorithmFPType(1) - y[i]) * ls1[i];
        value *= -div;
        if(bL1)
        {
            for(size_t i = 1; i < nBeta; ++i)
                value += (b[i] < 0 ? -b[i] : b[i])*parameter->penaltyL1;
        }
        if(bL2)
        {
            for(size_t i = 1; i < nBeta; ++i)
                value += b[i] * b[i] * parameter->penaltyL2;
        }
    }

    if(gradientNT)
    {
        DAAL_ASSERT(gradientNT->getNumberOfRows() == nBeta);
        const algorithmFPType* s = sg.get();
        WriteRows<algorithmFPType, cpu> gr(gradientNT, 0, nBeta);
        DAAL_CHECK_BLOCK_STATUS(gr);

        algorithmFPType* g = gr.get();
        const size_t nRowsInBlock = (p < 5000 ? 5000/p : 1);
        const size_t nDataBlocks = n / nRowsInBlock + !!(n%nRowsInBlock);
        const auto nThreads = daal::threader_get_threads_number();
        if((nThreads > 1) && (nDataBlocks > 1))
        {
            TlsSum<algorithmFPType, cpu> tlsData(nBeta);
            daal::threader_for(nDataBlocks, nDataBlocks, [&](size_t iBlock)
            {
                const size_t iStartRow = iBlock*nRowsInBlock;
                const size_t nRowsToProcess = (iBlock == nDataBlocks - 1) ? n - iBlock * nRowsInBlock : nRowsInBlock;
                const auto px = x + iStartRow * p;
                auto pg = tlsData.local();
                for(size_t i = 0; i < nRowsToProcess; ++i)
                {
                    const algorithmFPType d = (s[iStartRow + i] - y[iStartRow + i]);
                    for(size_t j = 0; j < p; ++j)
                        pg[j + 1] += d * px[i * p + j];
                }
            });
            tlsData.reduceTo(g, nBeta);
        }
        else
        {
            for(size_t i = 0; i < nBeta; ++i)
                g[i] = 0.0;
            for(size_t i = 0; i < n; ++i)
            {
                const algorithmFPType d = (s[i] - y[i]);
                for(size_t j = 0; j < p; ++j)
                    g[j + 1] += d * x[i * p + j];
            }
        }

        if(parameter->interceptFlag)
        {
            for(size_t i = 0; i < n; ++i)
                g[0] += (s[i] - y[i]);
        }

        for(size_t i = iFirstBeta; i < nBeta; ++i)
            g[i] *= div;

        if(bL1)
        {
            for(size_t i = 1; i < nBeta; ++i)
                g[i] += (b[i] < 0 ? -1 : (b[i] > 0))*parameter->penaltyL1;
        }
        if(bL2)
        {
            for(size_t i = 1; i < nBeta; ++i)
                g[i] += 2.* b[i] * parameter->penaltyL2;
        }
    }
    if(hessianNT)
    {
        DAAL_ASSERT(hessianNT->getNumberOfRows() == nBeta*nBeta);
        WriteRows<algorithmFPType, cpu> hr(hessianNT, 0, nBeta*nBeta);
        DAAL_CHECK_BLOCK_STATUS(hr);
        algorithmFPType* h = hr.get();

        algorithmFPType* s = sg.get();
        for(size_t i = 0; i < n; ++i)
            s[i] *= s[i + n]; //sigmoid derivative at x[i]

        h[0] = 0;
        if(parameter->interceptFlag)
        {
            for(size_t i = 0; i < n; ++i)
                h[0] += s[i];
            h[0] *= div; //average of sigmoid derivatives

            //first row and column
            for(size_t k = 1; k < nBeta; ++k)
            {
                algorithmFPType val = 0;
                for(size_t i = 0; i < n; ++i)
                    val += s[i] * x[i*p + k - 1];
                h[k] = val*div;
                h[k * nBeta] = val*div;
            }
        }
        else
        {
            //first row and column
            for(size_t k = 1; k < nBeta; ++k)
            {
                h[k] = 0;
                h[k * nBeta] = 0;
            }
        }
        //rows 1,..
        for(size_t j = 1; j < nBeta; ++j)
        {
            for(size_t k = j; k < nBeta; ++k)
            {
                algorithmFPType val = 0;
                for(size_t i = 0; i < n; ++i)
                    val += x[i * p + j-1] * x[i * p + k-1] * s[i];
                h[j*nBeta + k] = val*div;
                h[k*nBeta + j] = val*div;
            }
            h[j*nBeta + j] += 2.*parameter->penaltyL2;
        }
    }
    return services::Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status LogLossKernel<algorithmFPType, method, cpu>::compute(NumericTable *dataNT,
    NumericTable *dependentVariablesNT, NumericTable *betaNT,
    NumericTable *valueNT, NumericTable *hessianNT, NumericTable *gradientNT, Parameter *parameter)
{
    const size_t nRows = dataNT->getNumberOfRows();
    const daal::data_management::NumericTable* ntInd = parameter->batchIndices.get();

    if(ntInd && (ntInd->getNumberOfColumns() == nRows))
        ntInd = nullptr;

    const size_t p = dataNT->getNumberOfColumns();
    if(ntInd)
    {
        const size_t n = ntInd->getNumberOfColumns();
        TArrayScalable<algorithmFPType, cpu> aX(n*p);
        TArrayScalable<algorithmFPType, cpu> aY(n);
        services::Status s = objective_function::internal::getXY<algorithmFPType, cpu>(dataNT, dependentVariablesNT, ntInd, aX.get(), aY.get(), nRows, n, p);
        if(s)
            s |= doCompute(aX.get(), aY.get(), n, p, betaNT, valueNT, hessianNT, gradientNT, parameter);
        return s;
    }
    ReadRows<algorithmFPType, cpu> xr(dataNT, 0, nRows);
    ReadRows<algorithmFPType, cpu> yr(dependentVariablesNT, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(xr);
    DAAL_CHECK_BLOCK_STATUS(yr);
    return doCompute(xr.get(), yr.get(), nRows, p, betaNT, valueNT, hessianNT, gradientNT, parameter);
}

} // namespace daal::internal

} // namespace logistic_loss

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal
