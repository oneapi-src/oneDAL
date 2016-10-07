/* file: linear_regression_group_of_betas_dense_default_batch_impl.i */
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

#ifndef __LINEAR_REGRESSION_GROUP_OF_BETAS_DEFAULT_IMPL_I__
#define __LINEAR_REGRESSION_GROUP_OF_BETAS_DEFAULT_IMPL_I__

#include "service_memory.h"
#include "service_micro_table.h"
#include "service_lapack.h"
#include "threading.h"
#include "service_numeric_table.h"
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
void GroupOfBetasKernel<method, algorithmFPType, cpu>::compute(
    const NumericTable* y, const NumericTable* z, const NumericTable* zReduced,
    size_t numBeta, size_t numBetaReduced, algorithmFPType accuracyThreshold,
    NumericTable* out[])
{
    const auto nInputRows = y->getNumberOfRows();
    const auto k = y->getNumberOfColumns();

    SmartPtr<cpu> aResSS0(k * sizeof(algorithmFPType));
    if(!aResSS0.get())
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed);
        return;
    }

    WriteRows<algorithmFPType, cpu> ermBD(*out[expectedMeans], 0, 1);
    WriteRows<algorithmFPType, cpu> resSSBD(*out[resSS], 0, 1);

    algorithmFPType *pErm = ermBD.get();
    algorithmFPType *pResSS = resSSBD.get();
    algorithmFPType *pResSS0 = (algorithmFPType *)aResSS0.get();

    for(size_t j = 0; j < k; pErm[j] = 0, pResSS[j] = 0, pResSS0[j] = 0, ++j);

    const algorithmFPType divN = 1. / algorithmFPType(nInputRows);

    size_t nBlocks = nInputRows / _nRowsInBlock;
    nBlocks += (nBlocks * _nRowsInBlock != nInputRows);

    bool bMemoryAllocationFailed = false;
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

        daal::threader_for(nBlocks, nBlocks, [=, &resPartial, &bMemoryAllocationFailed](size_t block)
        {
            algorithmFPType* pResPartial = resPartial.local();
            if(!pResPartial)
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

            ReadRows<algorithmFPType, cpu> zReducedBD(*const_cast<NumericTable*>(zReduced), nRowsProcessed, nRowsToProcess);
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
        if(bMemoryAllocationFailed)
        {
            this->_errors->add(services::ErrorMemoryAllocationFailed);
            return;
        }
        for(size_t j = 0; j < k; pErm[j] *= divN, ++j);
    }

    //Compute ERV, regSS, tSS
    {
        WriteRows<algorithmFPType, cpu> tSSBD(*out[tSS], 0, 1);
        algorithmFPType *pTSS = tSSBD.get();

        WriteRows<algorithmFPType, cpu> regSSBD(*out[regSS], 0, 1);
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

        daal::threader_for(nBlocks, nBlocks, [=, &resPartial, &bMemoryAllocationFailed](size_t block)
        {
            algorithmFPType* pResPartial = resPartial.local();
            if(!pResPartial)
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
        if(bMemoryAllocationFailed)
        {
            this->_errors->add(services::ErrorMemoryAllocationFailed);
            return;
        }
        WriteRows<algorithmFPType, cpu> ervBD(*out[expectedVariance], 0, 1);
        algorithmFPType *pErv = ervBD.get();
        WriteRows<algorithmFPType, cpu> detBD(*out[determinationCoeff], 0, 1);
        algorithmFPType *pDet = detBD.get();
        WriteRows<algorithmFPType, cpu> fBD(*out[fStatistics], 0, 1);
        algorithmFPType *pF = fBD.get();

        const algorithmFPType *pResSS = resSSBD.get();
        const algorithmFPType *pResSS0 = (algorithmFPType *)aResSS0.get();
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
}

}
}
}
}
}
}

#endif
