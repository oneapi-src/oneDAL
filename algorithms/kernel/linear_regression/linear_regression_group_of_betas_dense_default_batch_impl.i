/* file: linear_regression_group_of_betas_dense_default_batch_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Declaration of template class that computes linear regression quality metrics.
//--
*/

#ifndef __LINEAR_REGRESSION_GROUP_OF_BETAS_DEFAULT_IMPL_I__
#define __LINEAR_REGRESSION_GROUP_OF_BETAS_DEFAULT_IMPL_I__

#include "service_memory.h"
#include "service_lapack.h"
#include "threading.h"
#include "service_numeric_table.h"
#include "service_error_handling.h"
#include "linear_regression_group_of_betas_dense_default_batch_kernel.h"

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
namespace group_of_betas
{
namespace internal
{

template<Method method, typename algorithmFPType, CpuType cpu>
services::Status GroupOfBetasKernel<method, algorithmFPType, cpu>::compute(
    const NumericTable* y, const NumericTable* z, const NumericTable* zReduced,
    size_t numBeta, size_t numBetaReduced, algorithmFPType accuracyThreshold,
    NumericTable* out[])
{
    const auto nInputRows = y->getNumberOfRows();
    const auto k = y->getNumberOfColumns();

    TArray<algorithmFPType, cpu> aResSS0(k);
    DAAL_CHECK_MALLOC(aResSS0.get());

    WriteRows<algorithmFPType, cpu> ermBD(*out[expectedMeans], 0, 1);
    DAAL_CHECK_BLOCK_STATUS(ermBD);
    WriteRows<algorithmFPType, cpu> resSSBD(*out[resSS], 0, 1);
    DAAL_CHECK_BLOCK_STATUS(resSSBD);

    algorithmFPType *pErm = ermBD.get();
    algorithmFPType *pResSS = resSSBD.get();
    algorithmFPType *pResSS0 = aResSS0.get();

    for(size_t j = 0; j < k; pErm[j] = 0, pResSS[j] = 0, pResSS0[j] = 0, ++j);

    const algorithmFPType divN = 1. / algorithmFPType(nInputRows);

    size_t nBlocks = nInputRows / _nRowsInBlock;
    nBlocks += (nBlocks * _nRowsInBlock != nInputRows);

    SafeStatus safeStat;
    //Compute ERM, resSS, resSS0
    {
        daal::tls<algorithmFPType *> resPartial([=]()-> algorithmFPType*
        {
            const size_t size = 3 * k;
            algorithmFPType* ptr = (algorithmFPType *)daal_malloc(size * sizeof(algorithmFPType));
            if(ptr)
                for(size_t j = 0; j < size; ptr[j] = 0, ++j);
            return ptr;
        });

        daal::threader_for(nBlocks, nBlocks, [=, &resPartial, &safeStat](size_t block)
        {
            algorithmFPType* pResPartial = resPartial.local();
            DAAL_CHECK_THR(pResPartial, ErrorMemoryAllocationFailed);

            size_t nRowsToProcess = _nRowsInBlock;
            if(block == nBlocks - 1)
                nRowsToProcess = nInputRows - block * _nRowsInBlock;
            const size_t nRowsProcessed = block * _nRowsInBlock;
            const size_t nCols = k;

            ReadRows<algorithmFPType, cpu> yBD(*const_cast<NumericTable*>(y), nRowsProcessed, nRowsToProcess);
            DAAL_CHECK_BLOCK_STATUS_THR(yBD);
            const algorithmFPType* py = yBD.get();

            ReadRows<algorithmFPType, cpu> zBD(*const_cast<NumericTable*>(z), nRowsProcessed, nRowsToProcess);
            DAAL_CHECK_BLOCK_STATUS_THR(zBD);
            const algorithmFPType* pz = zBD.get();

            ReadRows<algorithmFPType, cpu> zReducedBD(*const_cast<NumericTable*>(zReduced), nRowsProcessed, nRowsToProcess);
            DAAL_CHECK_BLOCK_STATUS_THR(zReducedBD);
            const algorithmFPType* pz0 = zReducedBD.get();

            for(size_t i = 0; i < nRowsToProcess; ++i, py += nCols, pz += nCols, pz0 += nCols)
            {
                for(size_t j = 0; j < nCols; ++j)
                {
                    //erm
                    pResPartial[j] += py[j];
                    //resSS
                    pResPartial[nCols + j] += (py[j] - pz[j])*(py[j] - pz[j]);
                    //resSS0
                    pResPartial[2 * nCols + j] += (py[j] - pz0[j])*(py[j] - pz0[j]);
                }
            }
        });
        resPartial.reduce([=](algorithmFPType * pResPartial)-> void
        {
            if(!pResPartial)
                return;
            const size_t nCols = k;
            for(size_t j = 0; j < nCols; ++j)
            {
                pErm[j] += pResPartial[j];
                pResSS[j] += pResPartial[nCols + j];
                pResSS0[j] += pResPartial[2 * nCols + j];
            }
            daal_free(pResPartial);
        });
        DAAL_CHECK_SAFE_STATUS();
        for(size_t j = 0; j < k; pErm[j] *= divN, ++j);
    }

    //Compute ERV, regSS, tSS
    {
        WriteRows<algorithmFPType, cpu> tSSBD(*out[tSS], 0, 1);
        DAAL_CHECK_BLOCK_STATUS(tSSBD);
        algorithmFPType *pTSS = tSSBD.get();

        WriteRows<algorithmFPType, cpu> regSSBD(*out[regSS], 0, 1);
        DAAL_CHECK_BLOCK_STATUS(regSSBD);
        algorithmFPType *pRegSS = regSSBD.get();
        for(size_t j = 0; j < k; pRegSS[j] = 0, pTSS[j] = 0, ++j);

        const algorithmFPType *pErm = ermBD.get();
        daal::tls<algorithmFPType *> resPartial([=]()-> algorithmFPType*
        {
            const size_t size = 2 * k;
            algorithmFPType* ptr = (algorithmFPType *)daal_malloc(size * sizeof(algorithmFPType));
            if(ptr)
                for(size_t j = 0; j < size; ptr[j] = 0, ++j);
            return ptr;
        });

        daal::threader_for(nBlocks, nBlocks, [=, &resPartial, &safeStat](size_t block)
        {
            algorithmFPType* pResPartial = resPartial.local();
            DAAL_CHECK_THR(pResPartial, ErrorMemoryAllocationFailed);
            size_t nRowsToProcess = _nRowsInBlock;
            if(block == nBlocks - 1)
                nRowsToProcess = nInputRows - block * _nRowsInBlock;
            const size_t nRowsProcessed = block * _nRowsInBlock;
            const size_t nCols = k;

            ReadRows<algorithmFPType, cpu> yBD(*const_cast<NumericTable*>(y), nRowsProcessed, nRowsToProcess);
            DAAL_CHECK_BLOCK_STATUS_THR(yBD);
            const algorithmFPType* py = yBD.get();

            ReadRows<algorithmFPType, cpu> zBD(*const_cast<NumericTable*>(z), nRowsProcessed, nRowsToProcess);
            DAAL_CHECK_BLOCK_STATUS_THR(zBD);
            const algorithmFPType* pz = zBD.get();

            for(size_t i = 0; i < nRowsToProcess; ++i, py += nCols, pz += nCols)
            {
                for(size_t j = 0; j < nCols; ++j)
                {
                    //TSS
                    pResPartial[j] += (py[j] - pErm[j])*(py[j] - pErm[j]);
                    //RegSS
                    pResPartial[nCols + j] += (pz[j] - pErm[j])*(pz[j] - pErm[j]);
                }
            }
        });

        resPartial.reduce([=](algorithmFPType * pResPartial)-> void
        {
            if(!pResPartial)
                return;
            const size_t nCols = k;
            for(size_t j = 0; j < nCols; ++j)
            {
                pTSS[j] += pResPartial[j];
                pRegSS[j] += pResPartial[nCols + j];
            }
            daal_free(pResPartial);
        });

        DAAL_CHECK_SAFE_STATUS();

        WriteRows<algorithmFPType, cpu> ervBD(*out[expectedVariance], 0, 1);
        DAAL_CHECK_BLOCK_STATUS(ervBD);
        algorithmFPType *pErv = ervBD.get();
        WriteRows<algorithmFPType, cpu> detBD(*out[determinationCoeff], 0, 1);
        DAAL_CHECK_BLOCK_STATUS(detBD);
        algorithmFPType *pDet = detBD.get();
        WriteRows<algorithmFPType, cpu> fBD(*out[fStatistics], 0, 1);
        DAAL_CHECK_BLOCK_STATUS(fBD);
        algorithmFPType *pF = fBD.get();

        const algorithmFPType *pResSS = resSSBD.get();
        const algorithmFPType *pResSS0 = aResSS0.get();
        const algorithmFPType divN_1 = 1. / algorithmFPType(nInputRows - 1);
        const algorithmFPType multF = algorithmFPType(nInputRows - numBeta) / algorithmFPType(numBeta - numBetaReduced);
        for(size_t j = 0; j < k; ++j)
        {
            pErv[j] = pTSS[j]*divN_1;
            pRegSS[j] *= divN;
            pDet[j] = pRegSS[j]/pTSS[j];
            const algorithmFPType div = (pResSS[j] < accuracyThreshold ? accuracyThreshold : pResSS[j]);
            pF[j] = multF*(pResSS0[j] - pResSS[j])/div;
        }
    }
    return Status();
}

}
}
}
}
}
}

#endif
