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
#include "src/services/service_data_utils.h"

#define DAAL_DISPATCH_FUNCTION_BY_CPU_SAFE(func, ...)                                                                           \
    int cpuid = daal::sse2;                                                                                                     \
    DAAL_SAFE_CPU_CALL((cpuid = daal::services::Environment::getInstance()->getCpuId()), (cpuid = daal::sse2))                  \
    switch (static_cast<daal::CpuType>(cpuid))                                                                                  \
    {                                                                                                                           \
        DAAL_KERNEL_SSSE3_ONLY_CODE(case daal::CpuType::ssse3 : func(daal::CpuType::ssse3, __VA_ARGS__); break;)                \
        DAAL_KERNEL_SSE42_ONLY_CODE(case daal::CpuType::sse42 : func(daal::CpuType::sse42, __VA_ARGS__); break;)                \
        DAAL_KERNEL_AVX_ONLY_CODE(case daal::CpuType::avx : func(daal::CpuType::avx, __VA_ARGS__); break;)                      \
        DAAL_KERNEL_AVX2_ONLY_CODE(case daal::CpuType::avx2 : func(daal::CpuType::avx2, __VA_ARGS__); break;)                   \
        DAAL_KERNEL_AVX512_ONLY_CODE(case daal::CpuType::avx512 : func(daal::CpuType::avx512, __VA_ARGS__); break;)             \
        DAAL_KERNEL_AVX512_MIC_ONLY_CODE(case daal::CpuType::avx512_mic : func(daal::CpuType::avx512_mic, __VA_ARGS__); break;) \
        DAAL_EXPAND(default : func(daal::CpuType::sse2, __VA_ARGS__); break;)                                                   \
    }

namespace daal
{
namespace data_management
{
namespace internal
{
template <typename FPType, daal::CpuType cpu>
void calculateRankDataImpl(FPType * predictedRank, NumericTablePtr & prediction_numpy, const int & size)
{
    ReadRows<FPType, cpu> numpyBlock(prediction_numpy.get(), 0, 1);
    const FPType * const numpyPtr = numpyBlock.get();

    TArray<FPType, cpu> values(size);
    TArray<size_t, cpu> indexes(size);

    for (size_t i = 0; i < size; ++i)
    {
        values[i]  = numpyPtr[i];
        indexes[i] = i;
    }

    daal::algorithms::internal::qSort<FPType, size_t, cpu>(size, values.get(), indexes.get());

    int r    = 1;
    int n    = 1;
    size_t i = 0;
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
            int idx            = indexes[i + j];
            predictedRank[idx] = r + ((n - 1) * 0.5);
        }
        r += n;
        i += n;
    }
}

template <typename FPType>
DAAL_EXPORT void calculateRankData(FPType * predictedRank, NumericTablePtr & prediction_numpy, const int & size)
{
#define DAAL_CALC_RANK_DATA(cpuId, ...) calculateRankDataImpl<FPType, cpuId>(__VA_ARGS__);

    DAAL_DISPATCH_FUNCTION_BY_CPU_SAFE(DAAL_CALC_RANK_DATA, predictedRank, prediction_numpy, size);

#undef DAAL_CALC_RANK_DATA
}

template DAAL_EXPORT void calculateRankData<float>(float * predictedRank, NumericTablePtr & prediction_numpy, const int & size);
template DAAL_EXPORT void calculateRankData<double>(double * predictedRank, NumericTablePtr & prediction_numpy, const int & size);

template <typename FPType, daal::CpuType cpu>
void rocAucScoreImpl(const FPType * predictedRank, NumericTablePtr & actual_numpy, const int & size, FPType * score)
{
    ReadRows<FPType, cpu> numpyBlock(actual_numpy.get(), 0, 1);
    const FPType * const numpyPtr = numpyBlock.get();

    FPType sum = 0;

    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < size; ++i) sum += numpyPtr[i];

    FPType nPos = sum;
    FPType nNeg = size - nPos;

    FPType filteredRankSum = 0;

    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < size; ++i) // parallel this
    {
        if (numpyPtr[i] == 1)
        {
            filteredRankSum += predictedRank[i];
        }
    }

    *score = (filteredRankSum - (nPos * (nPos + 1) / 2)) / (nPos * nNeg);
}

template <typename FPType>
DAAL_EXPORT FPType rocAucScore(const FPType * predictedRank, NumericTablePtr & actual_numpy, const int & size)
{
    FPType score = 0;
#define DAAL_ROC_AUC_SCORE(cpuId, ...) rocAucScoreImpl<FPType, cpuId>(__VA_ARGS__);

    DAAL_DISPATCH_FUNCTION_BY_CPU_SAFE(DAAL_ROC_AUC_SCORE, predictedRank, actual_numpy, size, &score);

#undef DAAL_ROC_AUC_SCORE
    return score;
}

template DAAL_EXPORT float rocAucScore<float>(const float * predictedRank, NumericTablePtr & actual_numpy, const int & size);
template DAAL_EXPORT double rocAucScore<double>(const double * predictedRank, NumericTablePtr & actual_numpy, const int & size);

} // namespace internal
} // namespace data_management
} // namespace daal
