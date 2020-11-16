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
#include "src/algorithms/service_kernel_math.h"
#include "src/algorithms/service_sort.h"
#include "src/externals/service_dispatch.h"
#include "src/externals/service_math.h"
#include "src/services/service_data_utils.h"

namespace daal
{
namespace data_management
{
namespace internal
{
template <typename DataType, daal::CpuType cpu>
void calculateRankDataImpl(DataType * predictedRank, NumericTablePtr & prediction_numpy, const int & size)
{
    ReadRows<DataType, cpu> numpyBlock(prediction_numpy.get(), 0, 1);
    const DataType * const numpyPtr = numpyBlock.get();
    DataType * const values         = new DataType[size];
    size_t * const indexes          = new size_t[size];

    for (size_t i = 0; i < size; ++i)
    {
        values[i]  = numpyPtr[i];
        indexes[i] = i;
    }

    daal::algorithms::internal::qSort<DataType, size_t, cpu>(size, values, indexes);

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
    delete[] indexes;
}

template <typename DataType>
void calculateRankDataDispImpl(DataType * predictedRank, NumericTablePtr & prediction_numpy, const int & size)
{
#define DAAL_CALC_RANK_DATA(cpuId, ...) calculateRankDataImpl<DataType, cpuId>(__VA_ARGS__);

    DAAL_DISPATCH_FUNCTION_BY_CPU(DAAL_CALC_RANK_DATA, predictedRank, prediction_numpy, size);

#undef DAAL_CALC_RANK_DATA
}

template <typename DataType>
DAAL_EXPORT void calculateRankData(DataType * predictedRank, NumericTablePtr & prediction_numpy, const int & size)
{
    DAAL_SAFE_CPU_CALL((calculateRankDataDispImpl<DataType>(predictedRank, prediction_numpy, size)),
                       (calculateRankDataImpl<DataType, daal::CpuType::sse2>(predictedRank, prediction_numpy, size)));
}

template DAAL_EXPORT void calculateRankData<float>(float * predictedRank, NumericTablePtr & prediction_numpy, const int & size);
template DAAL_EXPORT void calculateRankData<double>(double * predictedRank, NumericTablePtr & prediction_numpy, const int & size);

template <typename DataType, daal::CpuType cpu>
void rocAucScoreImpl(const DataType * predictedRank, NumericTablePtr & actual_numpy, const int & size, DataType * score)
{
    ReadRows<DataType, cpu> numpyBlock(actual_numpy.get(), 0, 1);
    const DataType * const numpyPtr = numpyBlock.get();

    DataType sum = 0;

    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < size; ++i) sum += numpyPtr[i];

    DataType nPos = sum;
    DataType nNeg = size - nPos;

    DataType filteredRankSum = 0;

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

template <typename DataType>
void rocAucScoreDispImpl(const DataType * predictedRank, NumericTablePtr & actual_numpy, const int & size, DataType * score)
{
#define DAAL_ROC_AUC_SCORE(cpuId, ...) rocAucScoreImpl<DataType, cpuId>(__VA_ARGS__);

    DAAL_DISPATCH_FUNCTION_BY_CPU(DAAL_ROC_AUC_SCORE, predictedRank, actual_numpy, size, score);

#undef DAAL_ROC_AUC_SCORE
}

template <typename DataType>
DAAL_EXPORT DataType rocAucScore(const DataType * predictedRank, NumericTablePtr & actual_numpy, const int & size)
{
    DataType score = 0;
    DAAL_SAFE_CPU_CALL((rocAucScoreDispImpl<DataType>(predictedRank, actual_numpy, size, &score)),
                       (rocAucScoreImpl<DataType, daal::CpuType::sse2>(predictedRank, actual_numpy, size, &score)));
    return score;
}

template DAAL_EXPORT float rocAucScore<float>(const float * predictedRank, NumericTablePtr & actual_numpy, const int & size);
template DAAL_EXPORT double rocAucScore<double>(const double * predictedRank, NumericTablePtr & actual_numpy, const int & size);

} // namespace internal
} // namespace data_management
} // namespace daal
