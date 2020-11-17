/** file roc_auc_score.cpp */
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

#include "data_management/data/internal/roc_auc_score.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "src/algorithms/service_kernel_math.h"
#include "src/algorithms/service_sort.h"
#include "src/externals/service_dispatch.h"
#include "src/externals/service_memory.h"
#include "src/services/service_data_utils.h"

namespace daal
{
namespace data_management
{
namespace internal
{
template <typename FPType, daal::CpuType cpu>
services::Status calculateRankDataImpl(FPType * predictedRank, NumericTablePtr & testPrediction, const int & size)
{
    services::Status s;
    ReadRows<FPType, cpu> testPredictionBlock(testPrediction.get(), 0, 1);
    const FPType * const testPredictionPtr = testPredictionBlock.get();
    DAAL_CHECK_MALLOC(testPredictionPtr);

    TArray<FPType, cpu> values(size);
    TArray<size_t, cpu> indeces(size);
    DAAL_CHECK_MALLOC(values.get() && indeces.get());

    size_t blockSizeDefault = 256;
    size_t nBlocks          = size / blockSizeDefault + !!(size % blockSizeDefault);

    daal::threader_for(nBlocks, nBlocks, [&](int iBlock) {
        const size_t blockSize  = (iBlock == nBlocks - 1) ? size % blockSizeDefault : blockSizeDefault;
        const size_t blockBegin = iBlock * blockSizeDefault;
        for (size_t i = 0;i < blockSize;++i)
        {
            values[blockBegin + i]  = testPredictionPtr[blockBegin + i];
            indeces[blockBegin + i] = blockBegin + i;
        }
    });

    daal::algorithms::internal::qSort<FPType, size_t, cpu>(size, values.get(), indeces.get());

    size_t r    = 1;
    size_t n    = 1;
    size_t i    = 0;
    while (i < size)
    {
        size_t j = i;
        while ((j < (size - 1)) && (values[j] == values[j + 1]))
        {
            j++;
        }
        n = j - i + 1;

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < n; ++j) // parallel this
        {
            int idx            = indeces[i + j];
            predictedRank[idx] = r + ((n - 1) * 0.5);
        }
        r += n;
        i += n;
    }
    return s;
}

template <typename FPType>
DAAL_EXPORT void calculateRankData(FPType * predictedRank, NumericTablePtr & testPrediction, const int & size)
{
#define DAAL_CALC_RANK_DATA(cpuId, ...) calculateRankDataImpl<FPType, cpuId>(__VA_ARGS__);

    DAAL_DISPATCH_FUNCTION_BY_CPU_SAFE(DAAL_CALC_RANK_DATA, predictedRank, testPrediction, size);

#undef DAAL_CALC_RANK_DATA
}

template DAAL_EXPORT void calculateRankData<float>(float * predictedRank, NumericTablePtr & testPrediction, const int & size);
template DAAL_EXPORT void calculateRankData<double>(double * predictedRank, NumericTablePtr & testPrediction, const int & size);

template <typename FPType, daal::CpuType cpu>
services::Status rocAucScoreImpl(const FPType * predictedRank, NumericTablePtr & truePrediction, const int & size, FPType * score)
{
    services::Status s;
    ReadRows<FPType, cpu> truePredictionBlock(truePrediction.get(), 0, 1);
    const FPType * const truePredictionPtr = truePredictionBlock.get();
    DAAL_CHECK_MALLOC(truePredictionPtr);
    FPType sum = 0;

    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < size; ++i) sum += truePredictionPtr[i];

    const FPType nPos = sum;
    const FPType nNeg = size - nPos;

    FPType filteredRankSum = 0;

    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < size; ++i) // parallel this
    {
        if (truePredictionPtr[i] == 1)
        {
            filteredRankSum += predictedRank[i];
        }
    }

    *score = (filteredRankSum - (nPos * (nPos + 1) / 2)) / (nPos * nNeg);
    return s;
}

template <typename FPType>
DAAL_EXPORT FPType rocAucScore(const FPType * predictedRank, NumericTablePtr & truePrediction, const int & size)
{
    FPType score = 0;
#define DAAL_ROC_AUC_SCORE(cpuId, ...) rocAucScoreImpl<FPType, cpuId>(__VA_ARGS__);

    DAAL_DISPATCH_FUNCTION_BY_CPU_SAFE(DAAL_ROC_AUC_SCORE, predictedRank, truePrediction, size, &score);

#undef DAAL_ROC_AUC_SCORE
    return score;
}

template DAAL_EXPORT float rocAucScore<float>(const float * predictedRank, NumericTablePtr & truePrediction, const int & size);
template DAAL_EXPORT double rocAucScore<double>(const double * predictedRank, NumericTablePtr & truePrediction, const int & size);

} // namespace internal
} // namespace data_management
} // namespace daal
