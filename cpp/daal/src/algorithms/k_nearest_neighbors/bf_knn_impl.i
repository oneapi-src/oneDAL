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
#include "src/algorithms/k_nearest_neighbors/bf_knn_classification_train_kernel.h"
#include "src/algorithms/k_nearest_neighbors/bf_knn_classification_predict_kernel.h"
#include "src/algorithms/k_nearest_neighbors/oneapi/bf_knn_classification_model_ucapi_impl.h"
#include "data_management/data/numeric_table.h"
#include "services/env_detect.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_utils.h"
#include "src/services/service_defines.h"
#include "src/threading/threading.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_kernel_math.h"
#include "src/algorithms/service_sort.h"
#include "src/externals/service_math.h"

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace internal
{
#define __BF_KNN_CLASS_BUFFER_SIZE 10

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
        daal::SafeStatus s;

        const size_t nDims  = trainTable->getNumberOfColumns();
        const size_t nTrain = trainTable->getNumberOfRows();
        const size_t nTest  = testTable->getNumberOfRows();

        int * trainLabel   = nullptr;
        int * indices      = nullptr;
        FPType * distances = nullptr;
        BlockDescriptor<int> trainLabelBlock;
        BlockDescriptor<int> indicesBlock;
        BlockDescriptor<FPType> distancesBlock;

        if (resultsToCompute & computeIndicesOfNeightbors)
        {
            indicesTable->getBlockOfRows(0, nTest, readWrite, indicesBlock);
            indices = indicesBlock.getBlockPtr();
            DAAL_CHECK_MALLOC(indices);
        }

        if (resultsToCompute & computeDistances)
        {
            distancesTable->getBlockOfRows(0, nTest, readWrite, distancesBlock);
            distances = distancesBlock.getBlockPtr();
            DAAL_CHECK_MALLOC(distances);
        }

        NumericTable * newTrainLabelTable = const_cast<NumericTable *>(trainLabelTable);
        if (resultsToEvaluate & daal::algorithms::classifier::computeClassLabels)
        {
            newTrainLabelTable->getBlockOfRows(0, nTrain, readWrite, trainLabelBlock);
            trainLabel = trainLabelBlock.getBlockPtr();
            DAAL_CHECK_MALLOC(trainLabel);
        }

        daal::algorithms::internal::EuclideanDistances<FPType, cpu> euclDist(
            *testTable, *trainTable, !(resultsToCompute & computeDistances) && !(voteWeights & VoteWeights::voteDistance));
        euclDist.init();

        const size_t blockSize    = 128;
        const size_t nOuterBlocks = nTest / blockSize + !!(nTest % blockSize);
        daal::threader_for(nOuterBlocks, nOuterBlocks, [&](size_t outerBlock) {
            const size_t outerStart = outerBlock * blockSize;
            const size_t outerEnd   = outerBlock + 1 == nOuterBlocks ? nTest : outerStart + blockSize;
            const size_t outerSize  = outerEnd - outerStart;

            s |= computeKNearestBlock(&euclDist, outerSize, outerStart, nTrain, resultsToEvaluate, resultsToCompute, indices, distances, nClasses, k,
                                      voteWeights, trainLabel, testLabelTable);
        });

        if (resultsToEvaluate & daal::algorithms::classifier::computeClassLabels)
        {
            newTrainLabelTable->releaseBlockOfRows(trainLabelBlock);
        }
        if (resultsToCompute & computeIndicesOfNeightbors)
        {
            indicesTable->releaseBlockOfRows(indicesBlock);
        }
        if (resultsToCompute & computeDistances)
        {
            distancesTable->releaseBlockOfRows(distancesBlock);
        }

        return s.detach();
    }

protected:
    services::Status computeKNearestBlock(daal::algorithms::internal::EuclideanDistances<FPType, cpu> * distancesInstance, const size_t blockSize,
                                          const size_t startTestIdx, const size_t nTrain, DAAL_UINT64 resultsToEvaluate, DAAL_UINT64 resultsToCompute,
                                          int * indices, FPType * distances, const size_t nClasses, const size_t k, VoteWeights voteWeights,
                                          int * trainLabel, NumericTable * testLabelTable)
    {
        services::Status s;

        daal::services::internal::TArray<FPType, cpu> tmpDistancesArr(blockSize * nTrain);
        daal::services::internal::TArray<int, cpu> tmpIndicesArr(blockSize * nTrain);
        FPType * tmpDistances = tmpDistancesArr.get();
        int * tmpIndices      = tmpIndicesArr.get();
        DAAL_CHECK_MALLOC(tmpDistances);
        DAAL_CHECK_MALLOC(tmpIndices);
        s |= distancesInstance->computeBatch(startTestIdx, blockSize, 0, nTrain, tmpDistances);

        for (size_t i = 0; i < blockSize; ++i)
        {
            for (size_t j = 0; j < nTrain; ++j)
            {
                tmpIndices[i * nTrain + j] = j;
            }

            daal::algorithms::internal::qSort<FPType, int, cpu>(nTrain, tmpDistances + i * nTrain, tmpIndices + i * nTrain);
        }

        if (resultsToCompute & computeIndicesOfNeightbors)
        {
            for (size_t i = 0; i < blockSize; ++i)
            {
                for (size_t j = 0; j < k; ++j)
                {
                    indices[(i + startTestIdx) * k + j] = tmpIndices[i * nTrain + j];
                }
            }
        }
        if (resultsToCompute & computeDistances)
        {
            for (size_t i = 0; i < blockSize; ++i)
            {
                for (size_t j = 0; j < k; ++j)
                {
                    distances[(i + startTestIdx) * k + j] = tmpDistances[i * nTrain + j];
                }
            }
        }

        if (resultsToEvaluate & daal::algorithms::classifier::computeClassLabels)
        {
            daal::internal::WriteRows<int, cpu> testLabelRows(testLabelTable, startTestIdx, blockSize);
            int * testLabel = testLabelRows.get();
            if (voteWeights == VoteWeights::voteUniform)
            {
                s |= uniformWeightedVoting(nClasses, k, blockSize, nTrain, tmpIndices, trainLabel, testLabel);
            }
            else
            {
                s |= distanceWeightedVoting(nClasses, k, blockSize, nTrain, tmpDistances, tmpIndices, trainLabel, testLabel);
            }
        }
        return s;
    }

    services::Status uniformWeightedVoting(const size_t nClasses, const size_t k, const size_t n, const size_t nTrain, int * indices,
                                           const int * trainLabel, int * testLabel)
    {
        daal::services::internal::TNArray<int, __BF_KNN_CLASS_BUFFER_SIZE, cpu> classWeightsArr(nClasses);
        int * classWeights = classWeightsArr.get();
        DAAL_CHECK_MALLOC(classWeights);

        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < nClasses; ++j)
            {
                classWeights[j] = 0;
            }
            for (size_t j = 0; j < k; ++j)
            {
                classWeights[trainLabel[indices[i * nTrain + j]]] += 1;
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
                                            int * indices, const int * trainLabel, int * testLabel)
    {
        daal::services::internal::TNArray<FPType, __BF_KNN_CLASS_BUFFER_SIZE, cpu> classWeightsArr(nClasses);
        FPType * classWeights = classWeightsArr.get();
        DAAL_CHECK_MALLOC(classWeights);

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
                        classWeights[trainLabel[indices[i * nTrain + j]]] += 1;
                    }
                }
                else
                {
                    classWeights[trainLabel[indices[i * nTrain + j]]] += 1 / distances[i * nTrain + j];
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
