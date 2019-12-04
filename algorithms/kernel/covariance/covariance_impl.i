/* file: covariance_impl.i */
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
//  Covariance matrix computation algorithm implementation
//--
*/

#ifndef __COVARIANCE_IMPL_I__
#define __COVARIANCE_IMPL_I__

#include "numeric_table.h"
#include "csr_numeric_table.h"
#include "service_memory.h"
#include "service_math.h"
#include "service_blas.h"
#include "service_spblas.h"
#include "service_stat.h"
#include "service_numeric_table.h"
#include "service_error_handling.h"
#include "threading.h"
#include "service_ittnotify.h"

using namespace daal::internal;
using namespace daal::services::internal;

DAAL_ITTNOTIFY_DOMAIN(covariance.dense.batch);

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status prepareSums(NumericTable * dataTable, algorithmFPType * sums)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.prepareSums);

    const size_t nFeatures = dataTable->getNumberOfColumns();
    int result             = 0;

    if (method == sumDense || method == sumCSR)
    {
        NumericTable * dataSumsTable = dataTable->basicStatistics.get(NumericTable::sum).get();
        DEFINE_TABLE_BLOCK(ReadRows, userSumsBlock, dataSumsTable);

        const size_t nFeaturesSize = nFeatures * sizeof(algorithmFPType);
        result                     = daal::services::internal::daal_memcpy_s(sums, nFeaturesSize, userSumsBlock.get(), nFeaturesSize);
    }
    else
    {
        const algorithmFPType zero = 0.0;
        services::internal::service_memset<algorithmFPType, cpu>(sums, zero, nFeatures);
    }

    return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
}

template <typename algorithmFPType, CpuType cpu>
services::Status prepareCrossProduct(size_t nFeatures, algorithmFPType * crossProduct)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.prepareCrossProduct);

    const algorithmFPType zero = 0.0;
    services::internal::service_memset<algorithmFPType, cpu>(crossProduct, zero, nFeatures * nFeatures);
    return services::Status();
}

/********************* tls_data_t class *******************************************************/
template <typename algorithmFPType, CpuType cpu>
struct tls_data_t
{
    algorithmFPType * sums;
    algorithmFPType * crossProduct;

    static tls_data_t<algorithmFPType, cpu> * create(bool isNormalized, size_t nFeatures)
    {
        auto object = new tls_data_t<algorithmFPType, cpu>(isNormalized, nFeatures);
        if (!object)
        {
            return nullptr;
        }

        if (!(object->crossProduct))
        {
            return nullptr;
        }
        if (!(object->sums) && !isNormalized)
        {
            return nullptr;
        }

        return object;
    }

    tls_data_t(bool isNormalized, size_t nFeatures)
    {
        crossProductArray.reset(nFeatures * nFeatures);
        if (!isNormalized)
        {
            sumsArray.reset(nFeatures);
        }

        sums         = sumsArray.get();
        crossProduct = crossProductArray.get();
    }

private:
    TArrayScalableCalloc<algorithmFPType, cpu> sumsArray;
    TArrayScalableCalloc<algorithmFPType, cpu> crossProductArray;
};

/* Optimal block size for AVX512 low dimensions case (1024) and other CPU's and cases (140) */
template <CpuType cpu>
static inline size_t getBlockSize(size_t nrows)
{
    return 140;
}

template <>
inline size_t getBlockSize<avx512>(size_t nrows)
{
    return (nrows > 5000 && nrows <= 50000) ? 1024 : 140;
}

/********************* updateDenseCrossProductAndSums ********************************************/
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status updateDenseCrossProductAndSums(bool isNormalized, size_t nFeatures, size_t nVectors, algorithmFPType * dataBlock,
                                                algorithmFPType * crossProduct, algorithmFPType * sums, algorithmFPType * nObservations)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.updateDenseCrossProductAndSums);
    if (((isNormalized) || ((!isNormalized) && ((method == defaultDense) || (method == sumDense)))))
    {
        /* Inverse number of rows (for normalization) */
        algorithmFPType nVectorsInv = 1.0 / (double)(nVectors);

        /* Split rows by blocks */
        size_t numRowsInBlock = getBlockSize<cpu>(nVectors);
        size_t numBlocks      = nVectors / numRowsInBlock;
        if (numBlocks * numRowsInBlock < nVectors)
        {
            numBlocks++;
        }

        /* TLS data initialization */
        SafeStatus safeStat;
        daal::tls<tls_data_t<algorithmFPType, cpu> *> tls_data([=, &safeStat]() {
            auto tlsData = tls_data_t<algorithmFPType, cpu>::create(isNormalized, nFeatures);
            if (!tlsData)
            {
                safeStat.add(services::ErrorMemoryAllocationFailed);
            }
            return tlsData;
        });

        /* Threaded loop with syrk seq calls */
        daal::threader_for(numBlocks, numBlocks, [&](int iBlock) {
            struct tls_data_t<algorithmFPType, cpu> * tls_data_local = tls_data.local();
            if (!tls_data_local)
            {
                return;
            }

            char uplo             = 'U';
            char trans            = 'N';
            algorithmFPType alpha = 1.0;
            algorithmFPType beta  = 1.0;

            size_t startRow = iBlock * numRowsInBlock;
            size_t endRow   = startRow + numRowsInBlock;
            if (endRow > nVectors)
            {
                endRow = nVectors;
            }
            DAAL_INT nFeatures_local             = nFeatures;
            DAAL_INT nVectors_local              = endRow - startRow;
            algorithmFPType * dataBlock_local    = dataBlock + startRow * nFeatures;
            algorithmFPType * crossProduct_local = tls_data_local->crossProduct;
            algorithmFPType * sums_local         = tls_data_local->sums;

            {
                DAAL_ITTNOTIFY_SCOPED_TASK(gemmData);
                Blas<algorithmFPType, cpu>::xxsyrk(&uplo, &trans, (DAAL_INT *)&nFeatures_local, (DAAL_INT *)&nVectors_local, &alpha, dataBlock_local,
                                                   (DAAL_INT *)&nFeatures_local, &beta, crossProduct_local, (DAAL_INT *)&nFeatures_local);
            }

            if (!isNormalized && (method == defaultDense))
            {
                DAAL_ITTNOTIFY_SCOPED_TASK(cumputeSums.local);
                /* Sum input array elements in case of non-normalized data */
                for (int i = 0; i < nVectors_local; i++)
                {
                    PRAGMA_IVDEP
                    PRAGMA_VECTOR_ALWAYS
                    for (int j = 0; j < nFeatures_local; j++)
                    {
                        sums_local[j] += dataBlock_local[i * nFeatures_local + j];
                    }
                }
            }
        });
        DAAL_CHECK_SAFE_STATUS();

        /* TLS reduction: sum all partial cross products and sums */
        tls_data.reduce([=](tls_data_t<algorithmFPType, cpu> * tls_data_local) {
            DAAL_ITTNOTIFY_SCOPED_TASK(computeSums.reduce);
            /* Sum all cross products */
            if (tls_data_local->crossProduct)
            {
                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = 0; i < (nFeatures * nFeatures); i++)
                {
                    crossProduct[i] += tls_data_local->crossProduct[i];
                }
            }

            /* Update sums vector in case of non-normalized data */
            if (!isNormalized && (method == defaultDense))
            {
                if (tls_data_local->sums)
                {
                    PRAGMA_IVDEP
                    PRAGMA_VECTOR_ALWAYS
                    for (int i = 0; i < nFeatures; i++)
                    {
                        sums[i] += tls_data_local->sums[i];
                    }
                }
            }

            delete tls_data_local;
        });

        /* If data is not normalized, perform subtractions of(sums[i]*sums[j])/n */
        if (!isNormalized)
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(gemmSums);
            for (int i = 0; i < nFeatures; i++)
            {
                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (int j = 0; j < nFeatures; j++)
                {
                    crossProduct[i * nFeatures + j] -= (nVectorsInv * sums[i] * sums[j]);
                }
            }
        }
    }
    else
    {
        __int64 mklMethod = __DAAL_VSL_SS_METHOD_FAST;
        switch (method)
        {
        case defaultDense: mklMethod = __DAAL_VSL_SS_METHOD_FAST; break;
        case singlePassDense: mklMethod = __DAAL_VSL_SS_METHOD_1PASS; break;
        case sumDense: mklMethod = __DAAL_VSL_SS_METHOD_FAST_USER_MEAN; break;
        default: break;
        }

        int errcode =
            Statistics<algorithmFPType, cpu>::xcp(dataBlock, (__int64)nFeatures, (__int64)nVectors, nObservations, sums, crossProduct, mklMethod);
        DAAL_CHECK(errcode == 0, services::ErrorCovarianceInternal);
    }

    *nObservations += (algorithmFPType)nVectors;
    return services::Status();
}

/********************** updateCSRCrossProductAndSums *********************************************/
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status updateCSRCrossProductAndSums(size_t nFeatures, size_t nVectors, algorithmFPType * dataBlock, size_t * colIndices,
                                              size_t * rowOffsets, algorithmFPType * crossProduct, algorithmFPType * sums,
                                              algorithmFPType * nObservations)
{
    char transa = 'T';
    SpBlas<algorithmFPType, cpu>::xcsrmultd(&transa, (DAAL_INT *)&nVectors, (DAAL_INT *)&nFeatures, (DAAL_INT *)&nFeatures, dataBlock,
                                            (DAAL_INT *)colIndices, (DAAL_INT *)rowOffsets, dataBlock, (DAAL_INT *)colIndices, (DAAL_INT *)rowOffsets,
                                            crossProduct, (DAAL_INT *)&nFeatures);

    if (method != sumCSR)
    {
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nVectors, sizeof(algorithmFPType));

        TArray<algorithmFPType, cpu> onesArray(nVectors);
        DAAL_CHECK_MALLOC(onesArray.get());

        algorithmFPType one    = 1.0;
        algorithmFPType * ones = onesArray.get();
        daal::services::internal::service_memset<algorithmFPType, cpu>(ones, one, nVectors);

        char matdescra[6];
        matdescra[0] = 'G'; // general matrix
        matdescra[3] = 'F'; // 1-based indexing

        matdescra[1] = (char)0;
        matdescra[2] = (char)0;
        matdescra[4] = (char)0;
        matdescra[5] = (char)0;
        SpBlas<algorithmFPType, cpu>::xcsrmv(&transa, (DAAL_INT *)&nVectors, (DAAL_INT *)&nFeatures, &one, matdescra, dataBlock,
                                             (DAAL_INT *)colIndices, (DAAL_INT *)rowOffsets, (DAAL_INT *)rowOffsets + 1, ones, &one, sums);
    }

    nObservations[0] += (algorithmFPType)nVectors;
    return services::Status();
}

/*********************** mergeCrossProductAndSums ************************************************/
template <typename algorithmFPType, CpuType cpu>
void mergeCrossProductAndSums(size_t nFeatures, const algorithmFPType * partialCrossProduct, const algorithmFPType * partialSums,
                              const algorithmFPType * partialNObservations, algorithmFPType * crossProduct, algorithmFPType * sums,
                              algorithmFPType * nObservations)
{
    /* Merge cross-products */
    algorithmFPType partialNObsValue = partialNObservations[0];

    if (partialNObsValue != 0)
    {
        algorithmFPType nObsValue = nObservations[0];

        if (nObsValue == 0)
        {
            daal::threader_for(nFeatures, nFeatures, [=](size_t i) {
                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j <= i; j++)
                {
                    crossProduct[i * nFeatures + j] += partialCrossProduct[i * nFeatures + j];
                    crossProduct[j * nFeatures + i] = crossProduct[i * nFeatures + j];
                }
            });
        }
        else
        {
            algorithmFPType invPartialNObs = 1.0 / partialNObsValue;
            algorithmFPType invNObs        = 1.0 / nObsValue;
            algorithmFPType invNewNObs     = 1.0 / (nObsValue + partialNObsValue);

            daal::threader_for(nFeatures, nFeatures, [=](size_t i) {
                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j <= i; j++)
                {
                    crossProduct[i * nFeatures + j] += partialCrossProduct[i * nFeatures + j];
                    crossProduct[i * nFeatures + j] += partialSums[i] * partialSums[j] * invPartialNObs;
                    crossProduct[i * nFeatures + j] += sums[i] * sums[j] * invNObs;
                    crossProduct[i * nFeatures + j] -= (partialSums[i] + sums[i]) * (partialSums[j] + sums[j]) * invNewNObs;
                    crossProduct[j * nFeatures + i] = crossProduct[i * nFeatures + j];
                }
            });
        }

        /* Merge number of observations */
        nObservations[0] += partialNObservations[0];

        /* Merge sums */
        for (size_t i = 0; i < nFeatures; i++)
        {
            sums[i] += partialSums[i];
        }
    }
}

/*********************** finalizeCovariance ******************************************************/
template <typename algorithmFPType, CpuType cpu>
services::Status finalizeCovariance(size_t nFeatures, algorithmFPType nObservations, algorithmFPType * crossProduct, algorithmFPType * sums,
                                    algorithmFPType * cov, algorithmFPType * mean, const Parameter * parameter)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.finalizeCovariance);

    algorithmFPType invNObservations   = 1.0 / nObservations;
    algorithmFPType invNObservationsM1 = 1.0;
    if (nObservations > 1.0)
    {
        invNObservationsM1 = 1.0 / (nObservations - 1.0);
    }

    /* Calculate resulting mean vector */
    for (size_t i = 0; i < nFeatures; i++)
    {
        mean[i] = sums[i] * invNObservations;
    }

    if (parameter->outputMatrixType == covariance::correlationMatrix)
    {
        /* Calculate resulting correlation matrix */
        TArray<algorithmFPType, cpu> diagInvSqrtsArray(nFeatures);
        DAAL_CHECK_MALLOC(diagInvSqrtsArray.get());

        algorithmFPType * diagInvSqrts = diagInvSqrtsArray.get();
        for (size_t i = 0; i < nFeatures; i++)
        {
            diagInvSqrts[i] = 1.0 / daal::internal::Math<algorithmFPType, cpu>::sSqrt(crossProduct[i * nFeatures + i]);
        }

        for (size_t i = 0; i < nFeatures; i++)
        {
            for (size_t j = 0; j < i; j++)
            {
                cov[i * nFeatures + j] = crossProduct[i * nFeatures + j] * diagInvSqrts[i] * diagInvSqrts[j];
            }
            cov[i * nFeatures + i] = 1.0; //diagonal element
        }
    }
    else
    {
        /* Calculate resulting covariance matrix */
        for (size_t i = 0; i < nFeatures; i++)
        {
            for (size_t j = 0; j <= i; j++)
            {
                cov[i * nFeatures + j] = crossProduct[i * nFeatures + j] * invNObservationsM1;
            }
        }
    }

    /* Copy results into symmetric upper triangle */
    for (size_t i = 0; i < nFeatures; i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            cov[j * nFeatures + i] = cov[i * nFeatures + j];
        }
    }

    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status finalizeCovariance(NumericTable * nObservationsTable, NumericTable * crossProductTable, NumericTable * sumTable,
                                    NumericTable * covTable, NumericTable * meanTable, const Parameter * parameter)
{
    const size_t nFeatures = covTable->getNumberOfColumns();

    DEFINE_TABLE_BLOCK(ReadRows, sumBlock, sumTable);
    DEFINE_TABLE_BLOCK(ReadRows, crossProductBlock, crossProductTable);
    DEFINE_TABLE_BLOCK(ReadRows, nObservationsBlock, nObservationsTable);
    DEFINE_TABLE_BLOCK(WriteOnlyRows, covBlock, covTable);
    DEFINE_TABLE_BLOCK(WriteOnlyRows, meanBlock, meanTable);

    algorithmFPType * cov           = covBlock.get();
    algorithmFPType * mean          = meanBlock.get();
    algorithmFPType * sums          = const_cast<algorithmFPType *>(sumBlock.get());
    algorithmFPType * crossProduct  = const_cast<algorithmFPType *>(crossProductBlock.get());
    algorithmFPType * nObservations = const_cast<algorithmFPType *>(nObservationsBlock.get());

    return finalizeCovariance<algorithmFPType, cpu>(nFeatures, *nObservations, crossProduct, sums, cov, mean, parameter);
}

} // namespace internal
} // namespace covariance
} // namespace algorithms
} // namespace daal

#endif
