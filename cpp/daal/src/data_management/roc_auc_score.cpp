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
#include "src/threading/threading.h"

template <typename FPType>
void sort(IdxValType<FPType> * beginPtr, IdxValType<FPType> * endPtr) {}

template <>
void sort<float>(IdxValType<float> * beginPtr, IdxValType<float> * endPtr)
{
    daal::parallel_sort_pair_fp32_uint64(beginPtr, endPtr);
}

template <>
void sort<double>(IdxValType<double> * beginPtr, IdxValType<double> * endPtr)
{
    daal::parallel_sort_pair_fp64_uint64(beginPtr, endPtr);
}

namespace daal
{
namespace data_management
{
namespace internal
{
template <typename FPType, daal::CpuType cpu>
services::Status calculateRankDataImpl(FPType * predictedRank, const NumericTablePtr & testPrediction, const size_t & nElements)
{
    services::Status s;
    ReadRows<FPType, cpu> testPredictionBlock(testPrediction.get(), 0, 1);
    const FPType * const testPredictionPtr = testPredictionBlock.get();
    DAAL_CHECK_BLOCK_STATUS(testPredictionBlock);

    TArray<IdxValType<FPType>, cpu> predict(nElements);
    DAAL_CHECK_MALLOC(predict.get());

    const size_t blockSizeDefault = 256;
    const size_t nBlocks          = nElements / blockSizeDefault + !!(nElements % blockSizeDefault);

    daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
        const size_t blockSize  = (iBlock == nBlocks - 1) ? nElements % blockSizeDefault : blockSizeDefault;
        const size_t blockBegin = iBlock * blockSizeDefault;
        for (size_t i = 0; i < blockSize; ++i)
        {
            predict[blockBegin + i].value = testPredictionPtr[blockBegin + i];
            predict[blockBegin + i].index = blockBegin + i;
        }
    });

    sort<FPType>(predict.get(), predict.get() + nElements);

    size_t rank            = 1;
    size_t elementsInBlock = 1;
    size_t i               = 0;
    while (i < nElements)
    {
        size_t j = i;
        while ((j < (nElements - 1)) && (predict[j].value == predict[j + 1].value))
        {
            j++;
        }
        elementsInBlock = j - i + 1;

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < elementsInBlock; ++j)
        {
            size_t idx         = predict[i + j].index;
            predictedRank[idx] = static_cast<FPType>(rank) + ((static_cast<FPType>(elementsInBlock) - FPType(1.0)) * FPType(0.5));
        }
        rank += elementsInBlock;
        i += elementsInBlock;
    }
    return s;
}

template <typename FPType>
DAAL_EXPORT void calculateRankData(FPType * predictedRank, const NumericTablePtr & testPrediction, const size_t & nElements)
{
#define DAAL_CALC_RANK_DATA(cpuId, ...) calculateRankDataImpl<FPType, cpuId>(__VA_ARGS__);

    DAAL_DISPATCH_FUNCTION_BY_CPU_SAFE(DAAL_CALC_RANK_DATA, predictedRank, testPrediction, nElements);

#undef DAAL_CALC_RANK_DATA
}

template DAAL_EXPORT void calculateRankData<float>(float * predictedRank, const NumericTablePtr & testPrediction, const size_t & nElements);
template DAAL_EXPORT void calculateRankData<double>(double * predictedRank, const NumericTablePtr & testPrediction, const size_t & nElements);

template <typename FPType, daal::CpuType cpu>
services::Status rocAucScoreImpl(const FPType * const predictedRank, const NumericTablePtr & truePrediction, const size_t & nElements, FPType & score)
{
    services::Status s;
    ReadRows<FPType, cpu> truePredictionBlock(truePrediction.get(), 0, 1);
    const FPType * const truePredictionPtr = truePredictionBlock.get();
    DAAL_CHECK_BLOCK_STATUS(truePredictionBlock);
    FPType sum = FPType(0);

    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < nElements; ++i)
    {
        sum += truePredictionPtr[i];
    }
    const FPType nPos = sum;
    const FPType nNeg = static_cast<FPType>(nElements) - nPos;

    FPType filteredRankSum = FPType(0);

    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < nElements; ++i) // parallel this
    {
        if (truePredictionPtr[i] == FPType(1))
        {
            filteredRankSum += predictedRank[i];
        }
    }

    score = (filteredRankSum - (nPos * (nPos + FPType(1.0)) * FPType(0.5))) / (nPos * nNeg);
    return s;
}

template <typename FPType>
DAAL_EXPORT FPType rocAucScore(const FPType * const predictedRank, const NumericTablePtr & truePrediction, const size_t & nElements)
{
    FPType score = FPType(0);
#define DAAL_ROC_AUC_SCORE(cpuId, ...) rocAucScoreImpl<FPType, cpuId>(__VA_ARGS__);

    DAAL_DISPATCH_FUNCTION_BY_CPU_SAFE(DAAL_ROC_AUC_SCORE, predictedRank, truePrediction, nElements, score);

#undef DAAL_ROC_AUC_SCORE
    return score;
}

template DAAL_EXPORT float rocAucScore<float>(const float * const predictedRank, const NumericTablePtr & truePrediction, const size_t & nElements);
template DAAL_EXPORT double rocAucScore<double>(const double * const predictedRank, const NumericTablePtr & truePrediction, const size_t & nElements);

} // namespace internal
} // namespace data_management
} // namespace daal
