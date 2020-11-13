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
#include "data_management/data/data_dictionary.h"
#include "services/env_detect.h"
#include "src/externals/service_dispatch.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_utils.h"
#include "src/services/service_defines.h"
#include "src/threading/threading.h"
#include "src/data_management/service_numeric_table.h"
#include "data_management/features/defines.h"
#include "src/algorithms/service_error_handling.h"
#include "src/externals/service_rng.h"
#include "src/externals/service_rng_mkl.h"

namespace daal
{
namespace data_management
{
namespace internal
{
/*
template <typename DataType>
struct Sum 
{
    double value;

    Sum() : value(0) {}
    Sum(Sum& s, tbb::split) : value(0) {}

    void operator()(const tbb::blocked_range<typename std::vector<DataType>::iterator>& r) {
        value = std::accumulate(r.begin(), r.end(), value);
    }

    void join(Sum& rhs) { value += rhs.value; }
};
*/
template <typename DataType>
void calculateRankData(double * predictedRank, NumericTablePtr & prediction_numpy, const int& size)
{
    data_management::BlockDescriptor<DataType> numpyBlock;
    prediction_numpy->getBlockOfRows(0, 1, data_management::readOnly, numpyBlock);
    DataType * numpyPtr = numpyBlock.getBlockPtr();

    std::pair<DataType, size_t>* v_sort = new std::pair<DataType, size_t>[size];

    for (size_t i = 0; i < size; ++i) {
        v_sort[i] = std::make_pair(numpyPtr[i], i);
    }

    for(size_t i = 0;i < size;++i){ //temporarily, bubble sort
        for(size_t j = i + 1;j < size;++j){
            if (v_sort[j] < v_sort[i]){
                std::swap(v_sort[i], v_sort[j]);
            }
        }
    }

    /*
    tbb::parallel_sort(v_sort.begin(), v_sort.end(), [](auto &left, auto &right) {
        return left.first < right.first;
    });
    */

    int r = 1;
    int n = 1;
    size_t i = 0;

    while (i < size) {
        size_t j = i;
        while ((j < (size - 1)) && (v_sort[j].first == v_sort[j + 1].first)) {
            j++;
        }
        n = j - i + 1;
        for (size_t j = 0; j < n; ++j) { // parallel this
            int idx = v_sort[i+j].second;
            predictedRank[idx] = r + ((n - 1) * 0.5);
        }
        r += n;
        i += n;
    }
    prediction_numpy->releaseBlockOfRows(numpyBlock);
}

template DAAL_EXPORT void calculateRankData<float> (double * predictedRank, NumericTablePtr & prediction_numpy, const int& size);
template DAAL_EXPORT void calculateRankData<double>(double * predictedRank, NumericTablePtr & prediction_numpy, const int& size);

template <typename DataType>
double rocAucScore(double * predictedRank, NumericTablePtr & actual_numpy, const int& size)
{
    data_management::BlockDescriptor<DataType> numpyBlock;
    actual_numpy->  getBlockOfRows(0, 1, data_management::readOnly, numpyBlock);
    DataType * numpyPtr = numpyBlock.getBlockPtr();

    DataType sum(0);
    for(size_t i = 0;i < size;++i) sum += numpyPtr[i];
    printf("actual_numpy\n");
    for(size_t i = 0;i < size;++i) printf("%.5lf ", numpyPtr[i]);
    printf("\n");
    DataType nPos = sum;
    DataType nNeg = size - nPos;

    DataType filteredRankSum = 0;
    
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < size; ++i) { // parallel this
        if (numpyPtr[i] == 1) {
            filteredRankSum += predictedRank[i];
        }
    }

    actual_numpy->releaseBlockOfRows(numpyBlock);

    DataType score = (filteredRankSum - (nPos*(nPos+1)/2)) / (nPos * nNeg);
    return score;
}

template DAAL_EXPORT double rocAucScore<float> (double * predictedRank, NumericTablePtr & actual_numpy, const int& size);
template DAAL_EXPORT double rocAucScore<double>(double * predictedRank, NumericTablePtr & actual_numpy, const int& size);

} // namespace internal
} // namespace data_management
} // namespace daal
