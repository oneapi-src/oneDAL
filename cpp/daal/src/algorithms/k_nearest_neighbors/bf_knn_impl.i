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
template <typename FPType, CpuType cpu>
class BruteForceNearestNeighbors
{
public:
    BruteForceNearestNeighbors(char metric) : _metric(metric) {}

    ~BruteForceNearestNeighbors()
    {
        if (_distances)
        {
            services::daal_free(_distances);
        }
    }

    services::Status kNearest(const size_t k, int * neighborsIndices, FPType * neighborsDistances, const NumericTable * trainTable,
                              const NumericTable * testTable)
    {
        computeDistances(trainTable, testTable);

        const size_t nTrain = trainTable->getNumberOfRows();
        const size_t nTest  = testTable->getNumberOfRows();

        const size_t outerBlockSize = 1024;
        const size_t nOuterBlocks   = nTest / outerBlockSize + !!(nTest % outerBlockSize);

        daal::threader_for(nOuterBlocks, nOuterBlocks, [&](size_t outerBlock) {
            const size_t outerStart = outerBlock * outerBlockSize;
            const size_t outerEnd   = outerBlock + 1 == nOuterBlocks ? nTest : outerStart + outerBlockSize;

            for (size_t i = outerStart; i < outerEnd; ++i)
            {
                daal::services::internal::TArray<int, cpu> indicesArr(nTrain);
                int * indices = indicesArr.get();
                for (size_t j = 0; j < nTrain; ++j)
                {
                    indices[j] = j;
                }

                daal::algorithms::internal::qSort<FPType, int, cpu>(nTrain, _distances + i * nTrain, indices);

                for (size_t j = 0; j < k; ++j)
                {
                    neighborsIndices[i * k + j]   = indices[j];
                    neighborsDistances[i * k + j] = _distances[i * nTrain + j];
                }
            }
        });

        return services::Status();
    }

    services::Status kClassification(const size_t k, const size_t nClasses, VoteWeights voteWeights, const NumericTable * trainTable,
                                     const NumericTable * testTable, const NumericTable * trainLabelTable, NumericTable * testLabelTable,
                                     NumericTable * indicesTable, NumericTable * distancesTable)
    {
        daal::SafeStatus s;

        const size_t nTrain = trainTable->getNumberOfRows();
        const size_t nTest  = testTable->getNumberOfRows();

        daal::internal::WriteRows<FPType, cpu> distancesRows(distancesTable, 0, nTest);
        daal::internal::WriteRows<int, cpu> indicesRows(indicesTable, 0, nTest);
        FPType * neighborsDistances = distancesRows.get();
        int * neighborsIndices      = indicesRows.get();
        DAAL_CHECK_MALLOC(neighborsDistances);
        DAAL_CHECK_MALLOC(neighborsIndices);

        kNearest(k, neighborsIndices, neighborsDistances, trainTable, testTable);

        daal::internal::ReadRows<int, cpu> trainLabelRows(const_cast<NumericTable *>(trainLabelTable), 0, nTrain);
        daal::internal::WriteRows<int, cpu> testLabelRows(testLabelTable, 0, nTest);

        const int * trainLabel = trainLabelRows.get();
        int * testLabel        = testLabelRows.get();
        DAAL_CHECK_MALLOC(trainLabel);
        DAAL_CHECK_MALLOC(testLabel);

        const size_t blockSize = 1024;
        const size_t nBlocks   = nTest / blockSize + !!(nTest % blockSize);

        daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
            const size_t start = iBlock * blockSize;
            const size_t end   = iBlock + 1 == nBlocks ? nTest : start + blockSize;

            if (voteWeights == VoteWeights::voteUniform)
            {
                s |= uniformWeightedVoting(nClasses, k, end - start, neighborsIndices + start * k, trainLabel, testLabel + start * k);
            }
            else
            {
                s |= distanceWeightedVoting(nClasses, k, end - start, neighborsDistances + start * k, neighborsIndices + start * k, trainLabel,
                                            testLabel + start);
            }
        });

        return s.detach();
    }

protected:
    services::Status uniformWeightedVoting(size_t nClasses, size_t k, size_t n, int * indices, const int * trainLabel, int * testLabel)
    {
        daal::services::internal::TArray<int, cpu> classWeightsArr(nClasses);
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

    services::Status distanceWeightedVoting(size_t nClasses, size_t k, size_t n, FPType * distances, int * indices, const int * trainLabel,
                                            int * testLabel)
    {
        daal::services::internal::TArray<FPType, cpu> classWeightsArr(nClasses);
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

    services::Status computeDistances(const NumericTable * trainTable, const NumericTable * testTable)
    {
        daal::SafeStatus s;

        const size_t nDims  = trainTable->getNumberOfColumns();
        const size_t nTrain = trainTable->getNumberOfRows();
        const size_t nTest  = testTable->getNumberOfRows();

        _distances = static_cast<FPType *>(services::internal::service_malloc<FPType, cpu>(nTrain * nTest * sizeof(FPType)));

        daal::algorithms::internal::EuclideanDistances<FPType, cpu> euclDist(*testTable, *trainTable);
        euclDist.init();

        const size_t blockSize    = 128;
        const size_t nOuterBlocks = nTest / blockSize + !!(nTest % blockSize);
        const size_t nInnerBlocks = nTrain / blockSize + !!(nTrain % blockSize);

        daal::threader_for(nOuterBlocks, nOuterBlocks, [&](size_t outerBlock) {
            const size_t outerStart = outerBlock * blockSize;
            const size_t outerEnd   = outerBlock + 1 == nOuterBlocks ? nTest : outerStart + blockSize;
            const size_t outerSize  = outerEnd - outerStart;

            daal::threader_for(nInnerBlocks, nInnerBlocks, [&](size_t innerBlock) {
                const size_t innerStart = innerBlock * blockSize;
                const size_t innerEnd   = innerBlock + 1 == nInnerBlocks ? nTrain : innerStart + blockSize;
                const size_t innerSize  = innerEnd - innerStart;

                daal::services::internal::TArray<FPType, cpu> tmpArr(outerSize * innerSize);
                FPType * tmp = tmpArr.get();

                s |= euclDist.computeBatch(outerStart, outerSize, innerStart, innerSize, tmp);

                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = outerStart; i < outerEnd; ++i)
                {
                    for (size_t j = innerStart; j < innerEnd; ++j)
                    {
                        _distances[i * nTrain + j] = tmp[(i - outerStart) * innerSize + j - innerStart];
                    }
                }
            });
        });

        daal::internal::Math<FPType, cpu> math;
        daal::threader_for(nOuterBlocks, nOuterBlocks, [&](size_t outerBlock) {
            const size_t outerStart = outerBlock * blockSize;
            const size_t outerEnd   = outerBlock + 1 == nOuterBlocks ? nTest : outerStart + blockSize;
            const size_t outerSize  = outerEnd - outerStart;

            daal::services::internal::TArray<FPType, cpu> tmpArr(outerSize * nTrain);
            FPType * tmp = tmpArr.get();
            math.vSqrt(outerSize * nTrain, _distances + outerStart * nTrain, tmp);

            services::internal::daal_memcpy_s(_distances + outerStart * nTrain, outerSize * nTrain * sizeof(FPType), tmp,
                                              outerSize * nTrain * sizeof(FPType));
        });

        return s.detach();
    }

private:
    char _metric;
    FPType * _distances;
};

} // namespace internal
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

#endif
