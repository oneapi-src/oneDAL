/* file: covariance_impl.i */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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

#include "data_management/data/numeric_table.h"
#include "data_management/data/csr_numeric_table.h"
#include "src/externals/service_memory.h"
#include "src/externals/service_math.h"
#include "src/externals/service_blas.h"
#include "src/externals/service_spblas.h"
#include "src/externals/service_stat.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"
#include "src/threading/threading.h"
#include "src/externals/service_ittnotify.h"
#include "src/services/service_environment.h"
#include <iostream>
#include <chrono>

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

template <typename algorithmFPType, CpuType cpu>
inline DAAL_INT64 getBlockSizeTry(size_t nrows, size_t ncols)
{
    // size_t dataBytes = nrows * ncols * sizeof(algorithmFPType);
    // size_t resultBytes = ncols * ncols * sizeof(algorithmFPType);
    DAAL_INT64 l2_size = getL2CacheSize();
    DAAL_INT64 rows_in_block = (0.8 * l2_size - ncols * 128 * sizeof(algorithmFPType)) / (ncols * sizeof(algorithmFPType));
    return rows_in_block;
}

template <typename algorithmFPType, CpuType cpu>
inline DAAL_INT64 getFeaturesBlockSize(size_t numRowsInBlock) {
    DAAL_INT64 l2_size = getL2CacheSize();
    algorithmFPType alpha = 2., beta = 1.05, k = 0.8, f = sizeof(algorithmFPType);

    // algorithmFPType discr = alpha * alpha * f * f * numRowsInBlock * numRowsInBlock + 4 * k * l2_size * beta * f;
    // DAAL_INT64 numFeaturesInBlock = (-alpha * f * numRowsInBlock + daal::internal::Math<algorithmFPType, cpu>::sSqrt(discr)) / (2 * beta * f);
    return (0.8 * l2_size) / (numRowsInBlock * f + 128 * f);
}

template <typename algorithmFPType, Method method, CpuType cpu>
size_t calculateBlockIndex(size_t numFeatureBlocks, size_t firstFeatureBlockIndex, size_t secondFeatureBlockIndex)
{
    return firstFeatureBlockIndex * (2 * numFeatureBlocks - firstFeatureBlockIndex + 1) / 2 + firstFeatureBlockIndex + secondFeatureBlockIndex;
}

template <typename algorithmFPType, Method method, CpuType cpu>
void checkBlockIndices(size_t iBlockPair, size_t numFeatureBlocks, size_t& firstFeatureBlockIndex, size_t& secondFeatureBlockIndex)
{
    if (calculateBlockIndex<algorithmFPType, method, cpu>(numFeatureBlocks, firstFeatureBlockIndex, secondFeatureBlockIndex) == iBlockPair) 
    {
        // std::cout << "OK BLOCK" << std::endl;
        return;    
    } else {
        // std::cout << "(!) NO OK BLOCK" << std::endl;
        for (int firstInc = -1; firstInc <= 1; ++firstInc) {
            for (int secondInc = -1; secondInc <= 1; ++secondInc) {
                if (firstInc == -1 && firstFeatureBlockIndex == 0 || secondInc == -1 && secondFeatureBlockIndex == 0 || 
                    firstInc == 1 && firstFeatureBlockIndex == static_cast<size_t>(-1) || secondInc == 1 && secondFeatureBlockIndex == static_cast<size_t>(-1))
                {
                    continue;
                } 
                else
                {
                    if (calculateBlockIndex<algorithmFPType, method, cpu>(numFeatureBlocks, firstFeatureBlockIndex + firstInc, secondFeatureBlockIndex + secondInc) == iBlockPair) {
                        firstFeatureBlockIndex += firstInc;
                        secondFeatureBlockIndex += secondInc;
                    }
                }
            }
        }
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
void findFeatureBlockIndices(size_t iBlockPair, size_t numFeatureBlocks, size_t& firstFeatureBlockIndex, size_t& secondFeatureBlockIndex)
{
    // firstFeatureBlockIndex = 0, secondFeatureBlockIndex = 0;
    // size_t diff = numFeatureBlocks;
    // while (iBlockPair >= diff) {
    //     iBlockPair -= diff;
    //     firstFeatureBlockIndex += 1;
    //     secondFeatureBlockIndex += 1;
    //     diff -= 1;
    // }
    // secondFeatureBlockIndex += iBlockPair;
    double discr = (numFeatureBlocks + 0.5) * (numFeatureBlocks + 0.5) - 2 * iBlockPair;
    firstFeatureBlockIndex = numFeatureBlocks + 0.5 - daal::internal::Math<algorithmFPType, cpu>::sSqrt(discr);
    secondFeatureBlockIndex = iBlockPair - firstFeatureBlockIndex * 0.5 * (2 * numFeatureBlocks - firstFeatureBlockIndex + 1) + firstFeatureBlockIndex;
    // std::cout << "PreInd: " << firstFeatureBlockIndex << ' ' << secondFeatureBlockIndex << std::endl;
    checkBlockIndices<algorithmFPType, method, cpu>(iBlockPair, numFeatureBlocks, firstFeatureBlockIndex, secondFeatureBlockIndex);
}

/********************* updateDenseCrossProductAndSums ********************************************/
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status updateDenseCrossProductAndSums(bool isNormalized, size_t nFeatures, size_t nVectors, NumericTable * dataTable,
                                                algorithmFPType * crossProduct, algorithmFPType * sums, algorithmFPType * nObservations)
{
    //std::cout << "************************************************" << std::endl;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.updateDenseCrossProductAndSums);
    if (((isNormalized) || ((!isNormalized) && ((method == defaultDense) || (method == sumDense)))))
    {
        /* Inverse number of rows (for normalization) */
        algorithmFPType nVectorsInv = 1.0 / (double)(nVectors);

        /* Split rows by blocks */
        DAAL_INT64 numRowsInBlockDefault = getBlockSizeTry<algorithmFPType, cpu>(nVectors, nFeatures);
        size_t numRowsInBlock = 0;
        size_t numFeaturesInBlock = 0;
        const size_t numRowsLimit = 128;
        if (numRowsInBlockDefault <= numRowsLimit) {
            numRowsInBlock = numRowsLimit;
            numFeaturesInBlock = getFeaturesBlockSize<algorithmFPType, cpu>(numRowsInBlock);
        } else {
            numRowsInBlock = numRowsInBlockDefault;
            numFeaturesInBlock = nFeatures;
        }
        if (numRowsInBlock > nVectors) {
            numRowsInBlock = nVectors;
        }
        if (numFeaturesInBlock > nFeatures) {
            numFeaturesInBlock = nFeatures;
        }
        size_t numBlocks      = nVectors / numRowsInBlock;
        if (numBlocks * numRowsInBlock < nVectors)
        {
            numBlocks++;
        }
        size_t numFeatureBlocks      = nFeatures / numFeaturesInBlock;
        if (numFeatureBlocks * numFeaturesInBlock < nFeatures)
        {
            numFeatureBlocks++;
        }

        /* TLS data initialization */
        auto start = std::chrono::steady_clock::now();
        SafeStatus safeStat;
        daal::tls<tls_data_t<algorithmFPType, cpu> *> tls_data([=, &safeStat]() {
            auto tlsData = tls_data_t<algorithmFPType, cpu>::create(isNormalized, nFeatures);
            if (!tlsData)
            {
                safeStat.add(services::ErrorMemoryAllocationFailed);
            }
            return tlsData;
        });
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "CPP COVARIANCE FIT updateDenseCrossProductAndSums_init: " << elapsed << std::endl;
        
        /* Threaded loop with syrk seq calls */
        start = std::chrono::steady_clock::now();

        DataBlocker<algorithmFPType, cpu, NumericTable> dataBlocker(dataTable, numBlocks, numRowsInBlock, numFeatureBlocks, numFeaturesInBlock);
        DAAL_INT lda = nFeatures, ldb = nFeatures, ldc = nFeatures;
        DAAL_INT nFeatures_local             = nFeatures;
        algorithmFPType alpha = 1.0;
        algorithmFPType beta  = 1.0;
        size_t pairsToProcess = (numFeatureBlocks * numFeatureBlocks - numFeatureBlocks) / 2 + numFeatureBlocks;
        size_t pairsForThread = pairsToProcess / threader_get_threads_number();
        std::cout << "Threads num: " << threader_get_threads_number() << std::endl;
        const size_t pairsForThreadLimit = 4;
        if (pairsForThread < pairsForThreadLimit) {
            pairsForThread = pairsForThreadLimit;
        }
        if (pairsForThread > pairsToProcess) {
            pairsForThread = pairsToProcess;
        }
        size_t pairBlocks = pairsToProcess / pairsForThread;
        if (pairBlocks * pairsForThread != pairsToProcess) {
            ++pairBlocks;
        }
        daal::conditional_threader_for((numBlocks > 2), numBlocks, [&](int iBlock) {
            size_t nRows    = (iBlock < numBlocks - 1) ? numRowsInBlock : nVectors - iBlock * numRowsInBlock;
            daal::conditional_threader_for((numFeatureBlocks > 3), pairBlocks, [&](const size_t iBlockPair) {
                size_t firstFeatureBlockIndex = 0, secondFeatureBlockIndex = 0;
                size_t indexPair = iBlockPair * pairsForThread;
                findFeatureBlockIndices<algorithmFPType, method, cpu>(indexPair, numFeatureBlocks, firstFeatureBlockIndex, secondFeatureBlockIndex);
                size_t localPairsToProcess = iBlockPair + 1 == pairBlocks ? pairsToProcess - iBlockPair * pairsForThread : pairsForThread;
                for (size_t pairIndex = 0; pairIndex < localPairsToProcess; ++pairIndex) {
                    size_t featuresFirstPart = firstFeatureBlockIndex * numFeaturesInBlock;
                    size_t firstFeaturesToProcess = nFeatures - featuresFirstPart < numFeaturesInBlock ? nFeatures - featuresFirstPart : numFeaturesInBlock;
                    size_t featuresSecondPart = secondFeatureBlockIndex * numFeaturesInBlock;
                    size_t secondFeaturesToProcess = nFeatures - featuresSecondPart < numFeaturesInBlock ? nFeatures - featuresSecondPart : numFeaturesInBlock;
                    struct tls_data_t<algorithmFPType, cpu> * tls_data_local = tls_data.local();
                    if (!tls_data_local)
                    {
                        return;
                    }
                    algorithmFPType * crossProduct_local = tls_data_local->crossProduct;
                    DataView<algorithmFPType, cpu, NumericTable> dataViewFirst = dataBlocker.getView(iBlock, firstFeatureBlockIndex);
                    DataView<algorithmFPType, cpu, NumericTable> dataViewSecond = dataBlocker.getView(iBlock, secondFeatureBlockIndex);
                    algorithmFPType * firstBlock_local = const_cast<algorithmFPType *>(dataViewFirst.get());
                    algorithmFPType * secondBlock_local = const_cast<algorithmFPType *>(dataViewSecond.get());
                    algorithmFPType * sums_local         = tls_data_local->sums;
                    {
                        DAAL_ITTNOTIFY_SCOPED_TASK(gemmData);
                        if (dataBlocker.isRows()) {
                            char transa = 'N', transb = 'T';
                            Blas<algorithmFPType, cpu>::xxgemm(&transa, &transb, (DAAL_INT *)&firstFeaturesToProcess, (DAAL_INT *)&secondFeaturesToProcess, 
                                (DAAL_INT *)&nRows, &alpha, firstBlock_local, &lda, secondBlock_local, &ldb, 
                                &beta, crossProduct_local + nFeatures * featuresSecondPart + featuresFirstPart, &ldc);
                        } else {
                            char transa = 'T', transb = 'N';
                            DAAL_INT lda = dataViewFirst.getLD(), ldb = dataViewSecond.getLD(), ldc = nFeatures;
                            Blas<algorithmFPType, cpu>::xxgemm(&transa, &transb, (DAAL_INT *)&firstFeaturesToProcess, (DAAL_INT *)&secondFeaturesToProcess, 
                                (DAAL_INT *)&nRows, &alpha, firstBlock_local, &lda, secondBlock_local, &ldb, 
                                &beta, crossProduct_local + nFeatures * featuresSecondPart + featuresFirstPart, &ldc);
                        }
                    }
                    if (!isNormalized && (method == defaultDense) && (featuresFirstPart == featuresSecondPart))
                    {
                        DAAL_ITTNOTIFY_SCOPED_TASK(cumputeSums.local);
                        /* Sum input array elements in case of non-normalized data */
                        if (dataBlocker.isRows()) {
                            DAAL_INT ld = dataViewFirst.getLD();
                            for (DAAL_INT i = 0; i < nRows; i++)
                            {
                                PRAGMA_IVDEP
                                PRAGMA_VECTOR_ALWAYS
                                for (DAAL_INT j = 0; j < firstFeaturesToProcess; j++)
                                {
                                    sums_local[featuresFirstPart + j] += firstBlock_local[i * ld + j];
                                }
                            }
                        } else {
                            DAAL_INT ld = dataViewFirst.getLD();
                            for (DAAL_INT j = 0; j < firstFeaturesToProcess; j++)
                            {
                                PRAGMA_IVDEP
                                PRAGMA_VECTOR_ALWAYS
                                for (DAAL_INT i = 0; i < nRows; i++)
                                {
                                    sums_local[featuresFirstPart + j] += firstBlock_local[ld * j + i];
                                }
                            }
                        }
                    }
                    if (secondFeatureBlockIndex + 1 >= numFeatureBlocks) {
                        firstFeatureBlockIndex += 1;
                        secondFeatureBlockIndex = firstFeatureBlockIndex;
                    } else {
                        secondFeatureBlockIndex += 1;
                    }
                }
            });
        });
        DAAL_CHECK_SAFE_STATUS();
        end = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "CPP COVARIANCE FIT updateDenseCrossProductAndSums_threaderfor: " << elapsed << std::endl;
        DAAL_CHECK_SAFE_STATUS();

        start = std::chrono::steady_clock::now();
        /* TLS reduction: sum all partial cross products and sums */
        tls_data.reduce([=](tls_data_t<algorithmFPType, cpu> * tls_data_local) {
            DAAL_ITTNOTIFY_SCOPED_TASK(computeSums.reduce);
            /* Sum all cross products */
            if (tls_data_local->crossProduct)
            {
                for (size_t i = 0; i < nFeatures; i++)
                {
                    PRAGMA_IVDEP
                    PRAGMA_VECTOR_ALWAYS
                    for (size_t j = 0; j <= i; j++) {
                        crossProduct[i * nFeatures + j] += tls_data_local->crossProduct[i * nFeatures + j];
                    }
                }
            }

            /* Update sums vector in case of non-normalized data */
            if (!isNormalized && (method == defaultDense))
            {
                if (tls_data_local->sums)
                {
                    PRAGMA_IVDEP
                    PRAGMA_VECTOR_ALWAYS
                    for (size_t i = 0; i < nFeatures; i++)
                    {
                        sums[i] += tls_data_local->sums[i];
                    }
                }
            }

            delete tls_data_local;
        });
        end = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "CPP COVARIANCE FIT updateDenseCrossProductAndSums_reduce: " << elapsed << std::endl;
        start = std::chrono::steady_clock::now();
        /* If data is not normalized, perform subtractions of(sums[i]*sums[j])/n */
        if (!isNormalized)
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(gemmSums);
            for (size_t i = 0; i < nFeatures; i++)
            {
                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j <= i; j++)
                {
                    crossProduct[i * nFeatures + j] -= (nVectorsInv * sums[i] * sums[j]);
                }
            }
        }
        end = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "CPP COVARIANCE FIT updateDenseCrossProductAndSums_isnormalized: " << elapsed << std::endl;
    }
    else
    {
        auto start = std::chrono::steady_clock::now();
        __int64 mklMethod = __DAAL_VSL_SS_METHOD_FAST;
        switch (method)
        {
        case defaultDense: mklMethod = __DAAL_VSL_SS_METHOD_FAST; break;
        case singlePassDense: mklMethod = __DAAL_VSL_SS_METHOD_1PASS; break;
        case sumDense: mklMethod = __DAAL_VSL_SS_METHOD_FAST_USER_MEAN; break;
        default: break;
        }
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        start = std::chrono::steady_clock::now();
        DEFINE_TABLE_BLOCK(ReadRows, dataBlock, dataTable);
        algorithmFPType * dataBlockPtr = const_cast<algorithmFPType *>(dataBlock.get());

        int errcode =
            Statistics<algorithmFPType, cpu>::xcp(dataBlockPtr, (__int64)nFeatures, (__int64)nVectors, nObservations, sums, crossProduct, mklMethod);
        end = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
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

    // auto start = std::chrono::steady_clock::now();
    /* Calculate resulting mean vector */
    for (size_t i = 0; i < nFeatures; i++)
    {
        mean[i] = sums[i] * invNObservations;
    }
    // auto end = std::chrono::steady_clock::now();
    // auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // //std::cout << "CPP Covariance finalizeCovariance loop_on_features: " << elapsed << std::endl;

    // start = std::chrono::steady_clock::now();
    if (parameter->outputMatrixType == covariance::correlationMatrix)
    {
        // //std::cout << '#' << std::endl;
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
        // //std::cout << '@' << std::endl;
        /* Calculate resulting covariance matrix */
        for (size_t i = 0; i < nFeatures; i++)
        {
            for (size_t j = 0; j <= i; j++)
            {
                cov[i * nFeatures + j] = crossProduct[i * nFeatures + j] * invNObservationsM1;
            }
        }
    }
    // end = std::chrono::steady_clock::now();
    // elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // //std::cout << "CPP Covariance finalizeCovariance compute: " << elapsed << std::endl;

    // start = std::chrono::steady_clock::now();
    /* Copy results into symmetric upper triangle */
    for (size_t i = 0; i < nFeatures; i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            cov[j * nFeatures + i] = cov[i * nFeatures + j];
        }
    }
    // end = std::chrono::steady_clock::now();
    // elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // //std::cout << "CPP Covariance finalizeCovariance loop_on_features2: " << elapsed << std::endl;

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
