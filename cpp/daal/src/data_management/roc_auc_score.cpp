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
#include "tbb/parallel_sort.h"
#include "tbb/parallel_reduce.h"

namespace daal
{
namespace data_management
{
namespace internal
{
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

template <typename DataType>
DataType roc_auc_score_(const std::vector<DataType> &predictedRank, PyArrayObject* actual_numpy, int size)
{
    DataType* actual = (DataType*)PyArray_DATA(actual_numpy);
    std::vector<DataType> actualVec;

    actualVec.assign(actual, actual + size);
    Sum<DataType> actualSum;

    parallel_reduce(tbb::blocked_range<typename std::std::vector<DataType>::iterator>(actualVec.begin(), actualVec.end()), actualSum);

    DataType nPos = actualSum.value;
    DataType nNeg = size-nPos;

    DataType filteredRankSum = 0;
    
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < size; ++i) { // parallel this
        if (actual[i] == 1) {
            filteredRankSum = filteredRankSum + predictedRank[i];
        }
    }

    DataType score = (filteredRankSum - (nPos*(nPos+1)/2)) / (nPos * nNeg);
    return score;
}

} // namespace internal
} // namespace data_management
} // namespace daal
