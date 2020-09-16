/* file: bf_knn_impl.i */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef __BF_KNN_IMPL_I__
#define __BF_KNN_IMPL_I__

#include "algorithms/engines/engine.h"
#include "services/daal_defines.h"
#include "algorithms/classifier/classifier_model.h"
#include "algorithms/k_nearest_neighbors/bf_knn_classification_model.h"
#include "algorithms/kernel/k_nearest_neighbors/bf_knn_classification_train_kernel.h"
#include "algorithms/kernel/k_nearest_neighbors/bf_knn_classification_predict_kernel.h"
#include "algorithms/kernel/k_nearest_neighbors/bf_knn_classification_model_impl.h"
#include "data_management/data/numeric_table.h"
#include "services/env_detect.h"
#include "service/kernel/service_data_utils.h"
#include "service/kernel/service_utils.h"
#include "service/kernel/service_defines.h"
#include "algorithms/threading/threading.h"
#include "service/kernel/data_management/service_numeric_table.h"
#include "algorithms/kernel/service_kernel_math.h"
#include "algorithms/kernel/service_sort.h"
#include "externals/service_math.h"

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace internal
{

template <typename FPType, CpuType cpu>
class BruteForceNearestNeighbors
{
public:
    BruteForceNearestNeighbors() {}

    ~BruteForceNearestNeighbors() {}

    services::Status kNeighbors(const size_t k, const size_t nClasses, VoteWeights voteWeights, DAAL_UINT64 resultsToCompute,
                                DAAL_UINT64 resultsToEvaluate, const NumericTable * trainTable, const NumericTable * testTable,
                                const NumericTable * trainLabelTable, NumericTable * testLabelTable, NumericTable * indicesTable,
                                NumericTable * distancesTable)
    {
        const size_t nDims  = trainTable->getNumberOfColumns();
        const size_t nTrain = trainTable->getNumberOfRows();
        const size_t nTest  = testTable->getNumberOfRows();

        int * trainLabel   = nullptr;
        BlockDescriptor<int> trainLabelBlock;

        NumericTable * newTrainLabelTable = const_cast<NumericTable *>(trainLabelTable);
        if (resultsToEvaluate & daal::algorithms::classifier::computeClassLabels)
        {
            newTrainLabelTable->getBlockOfRows(0, nTrain, readWrite, trainLabelBlock);
            trainLabel = trainLabelBlock.getBlockPtr();
            DAAL_CHECK_MALLOC(trainLabel);
        }

        daal::algorithms::internal::EuclideanDistances<FPType, cpu> euclDist(*testTable, *trainTable, true);
        euclDist.init();

        const size_t outBlockSize = 128;
        const size_t inBlockSize  = 128;
        const size_t nOuterBlocks = nTest / outBlockSize + !!(nTest % outBlockSize);

        TlsMem<FPType, cpu> tlsDistances(inBlockSize * outBlockSize);
        TlsMem<int,    cpu> tlsIdx(outBlockSize);
        TlsMem<FPType, cpu> tlsMaxs(inBlockSize);
        TlsMem<FPType, cpu> tlsKDistances(inBlockSize * k);
        TlsMem<int,    cpu> tlsKIndexes(inBlockSize * k);
        TlsMem<int,    cpu> tlsVoting(nClasses);

        SafeStatus safeStat;

        daal::threader_for(nOuterBlocks, nOuterBlocks, [&](size_t outerBlock) {
            const size_t outerStart = outerBlock * outBlockSize;
            const size_t outerEnd   = outerBlock + 1 == nOuterBlocks ? nTest : outerStart + outBlockSize;
            const size_t outerSize  = outerEnd - outerStart;

            DAAL_CHECK_STATUS_THR(computeKNearestBlock(&euclDist,
                                                       outerSize,
                                                       inBlockSize,
                                                       outerStart,
                                                       nTrain,
                                                       resultsToEvaluate,
                                                       resultsToCompute,
                                                       nClasses,
                                                       k,
                                                       voteWeights,
                                                       trainLabel,
                                                       trainTable,
                                                       testTable,
                                                       testLabelTable,
                                                       indicesTable,
                                                       distancesTable,
                                                       tlsDistances,
                                                       tlsIdx,
                                                       tlsMaxs,
                                                       tlsKDistances,
                                                       tlsKIndexes,
                                                       tlsVoting));
        });

        if (resultsToEvaluate & daal::algorithms::classifier::computeClassLabels)
        {
            newTrainLabelTable->releaseBlockOfRows(trainLabelBlock);
        }

        return safeStat.detach();
    }

protected:
    services::Status computeKNearestBlock(daal::algorithms::internal::EuclideanDistances<FPType, cpu> * distancesInstance,
                                          const size_t blockSize,
                                          const size_t trainBlockSize,
                                          const size_t startTestIdx,
                                          const size_t nTrain,
                                          DAAL_UINT64 resultsToEvaluate,
                                          DAAL_UINT64 resultsToCompute,
                                          const size_t nClasses,
                                          const size_t k,
                                          VoteWeights voteWeights,
                                          int * trainLabel,
                                          const NumericTable * trainTable,
                                          const NumericTable * testTable,
                                          NumericTable * testLabelTable,
                                          NumericTable * indicesTable,
                                          NumericTable * distancesTable,
                                          TlsMem<FPType, cpu>& tlsDistances,
                                          TlsMem<int, cpu>& tlsIdx,
                                          TlsMem<FPType, cpu>& tlsMaxs,
                                          TlsMem<FPType, cpu>& tlsKDistances,
                                          TlsMem<int, cpu>&    tlsKIndexes,
                                          TlsMem<int, cpu>&    tlsVoting)
    {
        FPType* distancesBuff =  tlsDistances.local();
        DAAL_CHECK_MALLOC(distancesBuff);

        int* idx = tlsIdx.local();
        DAAL_CHECK_MALLOC(idx);

        FPType* maxs = tlsMaxs.local();
        DAAL_CHECK_MALLOC(maxs);

        FPType* kDistances = tlsKDistances.local();
        DAAL_CHECK_MALLOC(kDistances);

        int* kIndexes = tlsKIndexes.local();
        DAAL_CHECK_MALLOC(kIndexes);

        int* voting = tlsVoting.local();
        DAAL_CHECK_MALLOC(voting);

        service_memset_seq<FPType, cpu>(maxs, MaxVal<FPType>::get(), trainBlockSize);
        service_memset_seq<FPType, cpu>(kDistances, MaxVal<FPType>::get(), blockSize * k);

        const size_t i1    = startTestIdx;
        const size_t i2    = startTestIdx + blockSize;
        const size_t iSize = blockSize;

        ReadRows<FPType, cpu> inDataRows(const_cast<NumericTable *>(testTable), i1, i2 - i1);
        DAAL_CHECK_BLOCK_STATUS(inDataRows);
        const FPType * const testData = inDataRows.get();

        const size_t outBlockSize = trainBlockSize;
        const size_t outRows = nTrain;
        const size_t nOutBlocks   = outRows / outBlockSize + (outRows % outBlockSize > 0);

        for (size_t outBlock = 0; outBlock < nOutBlocks; outBlock++)
        {
            size_t j1    = outBlock * outBlockSize;
            size_t j2    = (outBlock + 1 == nOutBlocks ? outRows : j1 + outBlockSize);
            size_t jSize = j2 - j1;

            ReadRows<FPType, cpu> outDataRows(const_cast<NumericTable *>(trainTable), j1, j2 - j1);
            DAAL_CHECK_BLOCK_STATUS(outDataRows);
            const FPType * const trainData = outDataRows.get();

            DAAL_CHECK_STATUS_VAR(distancesInstance->computeBatch(testData, trainData, i1, iSize, j1, jSize, distancesBuff));

            for (size_t i = 0; i < iSize; i++)
            {
                const size_t indexes = getIndexesWithLessDistances(idx, distancesBuff + i * jSize, jSize, maxs[i]);

                if (indexes)
                {
                    updateLocalNeighbours(indexes, idx, jSize, i, k, kDistances, kIndexes,
                                     maxs, distancesBuff, j1);
                }
            }
        }

        // Euclidean Distances are computed without Sqrt, fixing it here
        Math<FPType, cpu>::vSqrt(blockSize * k, kDistances, kDistances);

        // sort by distances
        for (size_t i = 0; i < blockSize; ++i)
        {
            daal::algorithms::internal::qSort<FPType, int, cpu>(k, kDistances + i * k, kIndexes + i * k);
        }

        if (resultsToCompute & computeIndicesOfNeightbors)
        {
            daal::internal::WriteRows<int, cpu> indexesBlock(indicesTable, startTestIdx, blockSize);
            DAAL_CHECK_BLOCK_STATUS(indexesBlock);
            int * indices = indexesBlock.get();

            const size_t size = blockSize * k * sizeof(*indices);
            daal::services::internal::daal_memcpy_s(indices, size, kIndexes, size);
        }

        if (resultsToCompute & computeDistances)
        {
            daal::internal::WriteRows<FPType, cpu> distancesBlock(distancesTable, startTestIdx, blockSize);
            DAAL_CHECK_BLOCK_STATUS(distancesBlock);
            FPType * distances = distancesBlock.get();

            const size_t size = blockSize * k * sizeof(FPType);
            daal::services::internal::daal_memcpy_s(distances, size, kDistances, size);
        }

        if (resultsToEvaluate & daal::algorithms::classifier::computeClassLabels)
        {
            daal::internal::WriteRows<int, cpu> testLabelRows(testLabelTable, startTestIdx, blockSize);
            DAAL_CHECK_BLOCK_STATUS(testLabelRows);
            int * testLabel = testLabelRows.get();

            if (voteWeights == VoteWeights::voteUniform)
            {
                DAAL_CHECK_STATUS_VAR(uniformWeightedVoting(nClasses, k, blockSize, nTrain, kIndexes, trainLabel, testLabel, voting));
            }
            else
            {
                DAAL_CHECK_STATUS_VAR(distanceWeightedVoting(nClasses, k, blockSize, nTrain, kDistances, kIndexes, trainLabel, testLabel, voting));
            }
        }

        return services::Status();
    }

    size_t getIndexesWithLessDistances(int* idx, FPType* array, size_t size, FPType cmp)
    {
        size_t count = 0;

        for(size_t i = 0; i < size; ++i)
        {
            if (array[i] < cmp)
            {
                idx[count++] = i;
            }
        }
        return count;
    }

    void updateLocalNeighbours(size_t indexes, int* idx, size_t jSize, size_t i, size_t k,
        FPType* kDistances, int* kIndexes, FPType* maxs, FPType* distances, size_t j1)
    {
        for(size_t j = 0; j < indexes; j++)
        {
            FPType d = distances[i * jSize + idx[j]];

            int min_idx = 0;
            for(size_t kk = 0; kk < k; kk++)
            {
                if (d < kDistances[i * k + kk] && kDistances[i * k + kk] > kDistances[i * k + min_idx])
                {
                    min_idx = kk;
                }
            }
            if (kDistances[i * k + min_idx] > d)
            {
                kDistances[i * k + min_idx] = d;
                kIndexes[i * k + min_idx] = idx[j] + j1;
            }
        }

        FPType max = kDistances[i * k + 0];
        for(size_t kk = 1; kk < k; kk++)
        {
            if (kDistances[i * k + kk] > max)
            {
                max = kDistances[i * k + kk];
            }
        }
        maxs[i] = max;

    }

    services::Status uniformWeightedVoting(const size_t nClasses, const size_t k, const size_t n, const size_t nTrain, int * indices,
                                           const int * trainLabel, int * testLabel, int* classWeights)
    {
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < nClasses; ++j)
            {
                classWeights[j] = 0;
            }
            for (size_t j = 0; j < k; ++j)
            {
                classWeights[trainLabel[indices[i * k + j]]] += 1;
            }
            size_t maxWeightClass = 0;
            size_t maxWeight      = 0;
            for (size_t j = 0; j < nClasses; ++j)
            {
                if (classWeights[j] > maxWeight)
                {
                    maxWeight      = classWeights[j];
                    maxWeightClass = j;
                }
            }
            testLabel[i] = maxWeightClass;
        }
        return services::Status();
    }

    services::Status distanceWeightedVoting(const size_t nClasses, const size_t k, const size_t n, const size_t nTrain, FPType * distances,
                                            int * indices, const int * trainLabel, int * testLabel, int* classWeights)
    {
        const FPType epsilon = daal::services::internal::EpsilonVal<FPType>::get();
        bool isContainZero   = false;
        for (size_t i = 0; i < k * n; ++i)
        {
            if (distances[i] < epsilon)
            {
                isContainZero = true;
                break;
            }
        }

        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < nClasses; ++j)
            {
                classWeights[j] = 0;
            }
            for (size_t j = 0; j < k; ++j)
            {
                if (isContainZero)
                {
                    if (distances[i] < epsilon)
                    {
                        classWeights[trainLabel[indices[i * k + j]]] += 1;
                    }
                }
                else
                {
                    classWeights[trainLabel[indices[i * k + j]]] += 1 / distances[i * k + j];
                }
            }
            size_t maxWeightClass = 0;
            FPType maxWeight      = 0;
            for (size_t j = 0; j < nClasses; ++j)
            {
                if (classWeights[j] > maxWeight)
                {
                    maxWeight      = classWeights[j];
                    maxWeightClass = j;
                }
            }
            testLabel[i] = maxWeightClass;
        }
        return services::Status();
    }
};

} // namespace internal
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

#endif
