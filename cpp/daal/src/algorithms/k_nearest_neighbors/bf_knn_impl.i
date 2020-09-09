/* file: bf_knn_impl.i */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
const size_t classBufSize = 10;

template <typename FPType, CpuType cpu>
class BruteForceNearestNeighbors
{
public:
    BruteForceNearestNeighbors() {}

    ~BruteForceNearestNeighbors() {}

    services::Status kNeighbors(const size_t k, const size_t nClasses, VoteWeights voteWeights, DAAL_UINT64 resultsToCompute,
                                DAAL_UINT64 resultsToEvaluate, const NumericTable * trainTable, const NumericTable * testTable,
                                const NumericTable * trainLabelTable, NumericTable * testLabelTable, int * indices, FPType * distances)
    {
        daal::SafeStatus s;

        const size_t nDims  = trainTable->getNumberOfColumns();
        const size_t nTrain = trainTable->getNumberOfRows();
        const size_t nTest  = testTable->getNumberOfRows();

        int * trainLabel = nullptr;
        BlockDescriptor<int> trainLabelBlock;
        NumericTable * newTrainLabelTable = const_cast<NumericTable *>(trainLabelTable);
        if (resultsToEvaluate & daal::algorithms::classifier::computeClassLabels)
        {
            newTrainLabelTable->getBlockOfRows(0, nTrain, readWrite, trainLabelBlock);
            trainLabel = trainLabelBlock.getBlockPtr();
        }

        daal::algorithms::internal::EuclideanDistances<FPType, cpu> euclDist(*testTable, *trainTable, !(resultsToCompute & computeDistances));
        euclDist.init();

        const size_t blockSize    = 128;
        const size_t nOuterBlocks = nTest / blockSize + !!(nTest % blockSize);
        daal::threader_for(nOuterBlocks, nOuterBlocks, [&](size_t outerBlock) {
            const size_t outerStart = outerBlock * blockSize;
            const size_t outerEnd   = outerBlock + 1 == nOuterBlocks ? nTest : outerStart + blockSize;
            const size_t outerSize  = outerEnd - outerStart;

            daal::services::internal::TArray<FPType, cpu> tmpDistancesArr(outerSize * nTrain);
            FPType * tmpDistances = tmpDistancesArr.get();
            s |= euclDist.computeBatch(outerStart, outerSize, 0, nTrain, tmpDistances);

            for (size_t i = outerStart; i < outerEnd; ++i)
            {
                daal::services::internal::TArray<int, cpu> indicesArr(nTrain);
                int * tmpIndices = indicesArr.get();
                for (size_t j = 0; j < nTrain; ++j)
                {
                    tmpIndices[j] = j;
                }

                daal::algorithms::internal::qSort<FPType, int, cpu>(nTrain, tmpDistances + (i - outerStart) * nTrain, tmpIndices);

                for (size_t j = 0; j < k; ++j)
                {
                    indices[i * k + j]   = tmpIndices[j];
                    distances[i * k + j] = tmpDistances[(i - outerStart) * nTrain + j];
                }
            }

            if (resultsToEvaluate & daal::algorithms::classifier::computeClassLabels)
            {
                daal::internal::WriteRows<int, cpu> testLabelRows(testLabelTable, outerStart, outerSize);
                int * testLabel = testLabelRows.get();
                if (voteWeights == VoteWeights::voteUniform)
                {
                    s |= uniformWeightedVoting(nClasses, k, outerSize, indices + outerStart * k, trainLabel, testLabel);
                }
                else
                {
                    s |= distanceWeightedVoting(nClasses, k, outerSize, distances + outerStart * k, indices + outerStart * k, trainLabel, testLabel);
                }
            }
        });

        if (resultsToEvaluate & daal::algorithms::classifier::computeClassLabels)
        {
            newTrainLabelTable->releaseBlockOfRows(trainLabelBlock);
        }

        return s.detach();
    }

protected:
    services::Status uniformWeightedVoting(const size_t nClasses, const size_t k, const size_t n, int * indices, const int * trainLabel,
                                           int * testLabel)
    {
        daal::services::internal::TNArray<int, classBufSize, cpu> classWeightsArr(nClasses);
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

    services::Status distanceWeightedVoting(const size_t nClasses, const size_t k, const size_t n, FPType * distances, int * indices,
                                            const int * trainLabel, int * testLabel)
    {
        daal::services::internal::TNArray<FPType, classBufSize, cpu> classWeightsArr(nClasses);
        FPType * classWeights = classWeightsArr.get();
        DAAL_CHECK_MALLOC(classWeights);

        const FPType epsilon = daal::services::internal::EpsilonVal<FPType>::get();
        bool isContainZero   = false;
        for (size_t i = 0; i < k * n; ++i)
        {
            if (distances[i] < epsilon && distances[i] > -epsilon)
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
                    if (distances[i] < epsilon && distances[i] > -epsilon)
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
