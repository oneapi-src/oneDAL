/* file: linear_regression_single_beta_dense_default_batch_impl.i */
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
//  Declaration of template class that computes linear regression quality metrics.
//--
*/

#ifndef __LINEAR_REGRESSION_SINGLE_BETA_DEFAULT_IMPL_I__
#define __LINEAR_REGRESSION_SINGLE_BETA_DEFAULT_IMPL_I__

#include "service_memory.h"
#include "service_math.h"
#include "service_lapack.h"
#include "service_numeric_table.h"
#include "service_data_utils.h"
#include "threading.h"

using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace quality_metric
{
namespace single_beta
{
namespace internal
{

template<Method method, typename algorithmFPType, CpuType cpu>
void SingleBetaKernel<method, algorithmFPType, cpu>::computeTestStatistics(
    const NumericTable* betas,
    const algorithmFPType* v,
    algorithmFPType alpha, algorithmFPType accuracyThreshold, SingleBetaOutput& out)
{
    const size_t nBeta = betas->getNumberOfColumns();
    const size_t nResponse = betas->getNumberOfRows();

    ReadRows<algorithmFPType, cpu> betasBD(*const_cast<NumericTable*>(betas), 0, nResponse);
    const algorithmFPType* beta = betasBD.get();

    WriteRows<algorithmFPType, cpu> zScoreBD(*out.zScore, 0, nResponse);
    algorithmFPType* zScore = zScoreBD.get();

    WriteRows<algorithmFPType, cpu> confidenceIntervalsBD(*out.confidenceIntervals, 0, nResponse);
    algorithmFPType* confInt = confidenceIntervalsBD.get();

    ReadRows<algorithmFPType, cpu> varianceBD(*out.variance, 0, 1);
    const algorithmFPType* variance = varianceBD.get();

    const algorithmFPType z_1_alpha = daal::internal::Math<algorithmFPType,cpu>::sCdfNormInv(1 - alpha);
    for(size_t i = 0; i < nResponse; ++i)
    {
        const algorithmFPType sigma = z_1_alpha*daal::internal::Math<algorithmFPType,cpu>::sSqrt(variance[i]);
        for(size_t j = 0; j < nBeta; ++j)
        {
            algorithmFPType vsigma = sigma*v[j];
            if(vsigma < accuracyThreshold)
                vsigma = accuracyThreshold;
            const algorithmFPType betaVal = beta[i*nBeta + j];
            zScore[i*nBeta + j] = betaVal/vsigma;
            confInt[i*2*nBeta + 2*j] = betaVal - vsigma;
            confInt[i*2*nBeta + 2*j + 1] = betaVal + vsigma;
        }
    }
}

template<typename algorithmFPType, CpuType cpu>
static bool regularizeTriangularMatrix(algorithmFPType* pMat, size_t dim)
{
    const algorithmFPType DIAGVALUE_THRESHOLD = 10 * daal::data_feature_utils::internal::MinVal<algorithmFPType, cpu>::get();
    const algorithmFPType minShift = 2 * daal::data_feature_utils::internal::MinVal<algorithmFPType, cpu>::get();
    algorithmFPType shift = 0;
    //find minimal absolute value of diagonal elements assuming they are > -DIAGVALUE_THRESHOLD
    for(auto i = 0; i < dim; ++i)
    {
        if(pMat[dim*i + i] < -DIAGVALUE_THRESHOLD)
            return false;
        const algorithmFPType val = pMat[dim*i + i] < 0 ? -pMat[dim*i + i] : pMat[dim*i + i];
        if((val < DIAGVALUE_THRESHOLD) && (shift > val))
            shift = val;
    }
    if(shift < minShift)
        shift = minShift;
    for(auto i = 0; i < dim; ++i)
    {
        algorithmFPType val = pMat[dim*i + i] < 0 ? -pMat[dim*i + i] : pMat[dim*i + i];
        if(val < DIAGVALUE_THRESHOLD)
            pMat[dim*i + i] = DIAGVALUE_THRESHOLD + shift;
        else
            pMat[dim*i + i] += shift;
    }
    return true;
}

template<Method method, typename algorithmFPType, CpuType cpu>
bool SingleBetaKernel<method, algorithmFPType, cpu>::computeInverseXtX(const NumericTable* xtx, bool bModelNe, NumericTable* xtxInv)
{
    const auto nBetas = xtx->getNumberOfColumns();

    ReadRows<algorithmFPType, cpu> xtxBD(*const_cast<NumericTable*>(xtx), 0, nBetas);
    const algorithmFPType *pXtX = xtxBD.get();

    {
        WriteRows<algorithmFPType, cpu> xtxInvBD(*xtxInv, 0, nBetas);
        algorithmFPType *pXtXInv = xtxInvBD.get();

        //if bModelNe == true then xtx contains Cholesky decompositon matrix L (xtx is assumed L*Lt)
        //else xtx contains Rt matrix from QR (xtx is assumed Rt*R)

        //calculate inverse of triangular matrix
        char uplo = 'U';

        //Copy xtx to xtxInv
        const auto dataSize = nBetas * nBetas * sizeof(algorithmFPType);
        services::daal_memcpy_s(pXtXInv, dataSize, pXtX, dataSize);

        MKL_INT nBeta(nBetas);
        MKL_INT info = 0;
        Lapack<algorithmFPType, cpu>::xpotri(&uplo, &nBeta, pXtXInv, &nBeta, &info);
        if(info == 0)
            return true;
        if(info < 0)
            return false;
        services::daal_memcpy_s(pXtXInv, dataSize, pXtX, dataSize);
        if(!regularizeTriangularMatrix<algorithmFPType, cpu>(pXtXInv, nBetas))
            return false;
        Lapack<algorithmFPType, cpu>::xpotri(&uplo, &nBeta, pXtXInv, &nBeta, &info);
        return (info == 0);
    }
}

template<Method method, typename algorithmFPType, CpuType cpu>
void SingleBetaKernel<method, algorithmFPType, cpu>::compute(
    const NumericTable* y, const NumericTable* z, size_t p,
    const NumericTable* betas, const NumericTable* xtx, bool bModelNe,
    algorithmFPType accuracyThreshold, algorithmFPType alpha, SingleBetaOutput& out)
{
    if(!computeRmsVariance(y, z, p, out.rms, out.variance))
        return;

    // Calculate inverse(Xt*X)
    if(!computeInverseXtX(xtx, bModelNe, out.inverseOfXtX))
    {
        this->_errors->add(services::ErrorLinRegXtXInvFailed);
        return;
    }

    const auto nBetas = xtx->getNumberOfColumns();
    //Compute vector V (sqrt of inverse (Xt*X) diagonal elements)
    SmartPtr<cpu> aDiagElem(nBetas * sizeof(algorithmFPType));
    algorithmFPType* v = (algorithmFPType *)aDiagElem.get();
    if(!v)
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed);
        return;
    }

    {
        ReadRows<algorithmFPType, cpu> xtxInvBD(*out.inverseOfXtX, 0, nBetas);
        const algorithmFPType *xtxInv = xtxInvBD.get();

        const algorithmFPType* pXtxInv = xtxInv;
        algorithmFPType* pV = v;
        for(auto i = 0; i < nBetas; ++i, pXtxInv += nBetas + 1, ++pV)
            *pV = (*pXtxInv < 0 ? daal::internal::Math<algorithmFPType,cpu>::sSqrt(-*pXtxInv) : daal::internal::Math<algorithmFPType,cpu>::sSqrt(*pXtxInv));

        //Compute beta variance-covariance matrices
        ReadRows<algorithmFPType, cpu> varianceBD(*out.variance, 0, 1);
        const algorithmFPType* variance = varianceBD.get();

        const auto k = y->getNumberOfColumns();
        for(auto i = 0; i < k; ++i)
        {
            WriteRows<algorithmFPType, cpu> betaCovBD(*out.betaCovariances[i], 0, nBetas);
            algorithmFPType *betaCov = betaCovBD.get();
            const algorithmFPType sigmaSq = variance[i];
            for(auto j = 0; j < nBetas*nBetas; ++j)
                betaCov[j] = xtxInv[j]*sigmaSq;
        }
    }
    computeTestStatistics(betas, v, alpha, accuracyThreshold, out);
}

template<Method method, typename algorithmFPType, CpuType cpu>
bool SingleBetaKernel<method, algorithmFPType, cpu>::computeRmsVariance(const NumericTable* y,
    const NumericTable* z, size_t p, NumericTable* rms, NumericTable* variance)
{
    const auto nInputRows = y->getNumberOfRows();
    const auto k = y->getNumberOfColumns();

    WriteRows<algorithmFPType, cpu> rmsBD(*rms, 0, 1);
    algorithmFPType *pRms = rmsBD.get();

    WriteRows<algorithmFPType, cpu> varBD(*variance, 0, 1);
    algorithmFPType *pVar = varBD.get();

    for(size_t j = 0; j < k; pRms[j] = 0, pVar[j] = 0, ++j);

    daal::tls<algorithmFPType *> rmsPartial([=]()-> algorithmFPType*
    {
        const size_t nCols = k;
        algorithmFPType* ptr = (algorithmFPType *)daal_malloc(nCols * sizeof(algorithmFPType));
        if(ptr)
            for(size_t j = 0; j < nCols; ptr[j] = 0, ++j);
        return ptr;
    });

    size_t nBlocks = nInputRows / _nRowsInBlock;
    nBlocks += (nBlocks * _nRowsInBlock != nInputRows);

    bool bMemoryAllocationFailed = false;
    daal::threader_for(nBlocks, nBlocks, [=, &rmsPartial, &bMemoryAllocationFailed](size_t block)
    {
        algorithmFPType* pRmsPartial = rmsPartial.local();
        if(!pRmsPartial)
        {
            bMemoryAllocationFailed = true;
            return;
        }
        size_t nRowsToProcess = _nRowsInBlock;
        if(block == nBlocks - 1)
            nRowsToProcess = nInputRows - block * _nRowsInBlock;
        const size_t nRowsProcessed = block * _nRowsInBlock;
        const size_t nCols = k;

        ReadRows<algorithmFPType, cpu> yBD(*const_cast<NumericTable*>(y), nRowsProcessed, nRowsToProcess);
        const algorithmFPType* py = yBD.get();

        ReadRows<algorithmFPType, cpu> zBD(*const_cast<NumericTable*>(z), nRowsProcessed, nRowsToProcess);
        const algorithmFPType* pz = zBD.get();

        for(size_t i = 0; i < nRowsToProcess; ++i, py += nCols, pz += nCols)
        {
            for(size_t j = 0; j < nCols; ++j)
                pRmsPartial[j] += (py[j] - pz[j])*(py[j] - pz[j]);
        }
    });
    rmsPartial.reduce([=](algorithmFPType * pRmsPartial)-> void
    {
        if(!pRmsPartial)
            return;
        const size_t nCols = k;
        for(size_t j = 0; j < nCols; ++j)
            pRms[j] += pRmsPartial[j];
        daal_free(pRmsPartial);
    });

    if(bMemoryAllocationFailed)
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed);
        return false;
    }

    const algorithmFPType div1 = 1. / nInputRows;
    const algorithmFPType div2 = 1. / (nInputRows - p - 1);
    for(size_t j = 0; j < k; ++j)
    {
        pVar[j] = div2*pRms[j];
        pRms[j] = div1*daal::internal::Math<algorithmFPType,cpu>::sSqrt(pRms[j]);
    }
    return true;
}

}
}
}
}
}
}

#endif
