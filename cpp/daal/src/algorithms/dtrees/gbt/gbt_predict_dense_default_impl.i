/* file: gbt_predict_dense_default_impl.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

/*
//++
//  Implementation of auxiliary functions for gradient boosted trees prediction
//  (defaultDense) method.
//--
*/

#ifndef __GBT_PREDICT_DENSE_DEFAULT_IMPL_I__
#define __GBT_PREDICT_DENSE_DEFAULT_IMPL_I__

#include "src/algorithms/dtrees/dtrees_model_impl.h"
#include "src/algorithms/dtrees/dtrees_train_data_helper.i"
#include "src/algorithms/dtrees/dtrees_predict_dense_default_impl.i"
#include "src/algorithms/dtrees/dtrees_feature_type_helper.h"
#include "src/algorithms/dtrees/gbt/gbt_internal.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace prediction
{
namespace internal
{
typedef float ModelFPType;
typedef uint32_t FeatureIndexType;
const FeatureIndexType VECTOR_BLOCK_SIZE = 64;

template <bool hasUnorderedFeatures, bool hasAnyMissing>
struct PredictDispatcher
{
    typedef PredictDispatcher<hasUnorderedFeatures, hasAnyMissing> type;
};

template <typename algorithmFPType>
inline FeatureIndexType updateIndex(FeatureIndexType idx, algorithmFPType valueFromDataSet, const ModelFPType * splitPoints, const int * defaultLeft,
                                    const FeatureTypes & featTypes, FeatureIndexType splitFeature, const PredictDispatcher<false, false> & dispatcher)
{
    return idx * 2 + (valueFromDataSet > splitPoints[idx]);
}

template <typename algorithmFPType>
inline FeatureIndexType updateIndex(FeatureIndexType idx, algorithmFPType valueFromDataSet, const ModelFPType * splitPoints, const int * defaultLeft,
                                    const FeatureTypes & featTypes, FeatureIndexType splitFeature, const PredictDispatcher<true, false> & dispatcher)
{
    return idx * 2 + (featTypes.isUnordered(splitFeature) ? valueFromDataSet != splitPoints[idx] : valueFromDataSet > splitPoints[idx]);
}

template <typename algorithmFPType>
inline FeatureIndexType updateIndex(FeatureIndexType idx, algorithmFPType valueFromDataSet, const ModelFPType * splitPoints, const int * defaultLeft,
                                    const FeatureTypes & featTypes, FeatureIndexType splitFeature, const PredictDispatcher<false, true> & dispatcher)
{
    if (isnan(valueFromDataSet))
    {
        return idx * 2 + (defaultLeft[idx] != 1);
    }
    else
    {
        return idx * 2 + (valueFromDataSet > splitPoints[idx]);
    }
}

template <typename algorithmFPType>
inline FeatureIndexType updateIndex(FeatureIndexType idx, algorithmFPType valueFromDataSet, const ModelFPType * splitPoints, const int * defaultLeft,
                                    const FeatureTypes & featTypes, FeatureIndexType splitFeature, const PredictDispatcher<true, true> & dispatcher)
{
    if (isnan(valueFromDataSet))
    {
        return idx * 2 + (defaultLeft[idx] != 1);
    }
    else
    {
        return idx * 2 + (featTypes.isUnordered(splitFeature) ? valueFromDataSet != splitPoints[idx] : valueFromDataSet > splitPoints[idx]);
    }
}

template <typename algorithmFPType, typename DecisionTreeType, CpuType cpu, bool hasUnorderedFeatures, bool hasAnyMissing>
inline void predictForTreeVector(const DecisionTreeType & t, const FeatureTypes & featTypes, const algorithmFPType * x, algorithmFPType v[],
                                 const PredictDispatcher<hasUnorderedFeatures, hasAnyMissing> & dispatcher)
{
    const ModelFPType * const values        = t.getSplitPoints() - 1;
    const FeatureIndexType * const fIndexes = t.getFeatureIndexesForSplit() - 1;
    const int * const defaultLeft           = t.getdefaultLeftForSplit() - 1;
    const FeatureIndexType nFeat            = featTypes.getNumberOfFeatures();

    FeatureIndexType i[VECTOR_BLOCK_SIZE];
    services::internal::service_memset_seq<FeatureIndexType, cpu>(i, FeatureIndexType(1), VECTOR_BLOCK_SIZE);

    const FeatureIndexType maxLvl = t.getMaxLvl();

    for (FeatureIndexType itr = 0; itr < maxLvl; itr++)
    {
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (FeatureIndexType k = 0; k < VECTOR_BLOCK_SIZE; k++)
        {
            const FeatureIndexType idx          = i[k];
            const FeatureIndexType splitFeature = fIndexes[idx];
            i[k] = updateIndex(idx, x[splitFeature + k * nFeat], values, defaultLeft, featTypes, splitFeature, dispatcher);
        }
    }

    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (FeatureIndexType k = 0; k < VECTOR_BLOCK_SIZE; k++)
    {
        v[k] = values[i[k]];
    }
}

template <typename algorithmFPType, typename DecisionTreeType, CpuType cpu, bool hasUnorderedFeatures, bool hasAnyMissing>
inline algorithmFPType predictForTree(const DecisionTreeType & t, const FeatureTypes & featTypes, const algorithmFPType * x,
                                      const PredictDispatcher<hasUnorderedFeatures, hasAnyMissing> & dispatcher)
{
    const ModelFPType * const values        = (const ModelFPType *)t.getSplitPoints() - 1;
    const FeatureIndexType * const fIndexes = t.getFeatureIndexesForSplit() - 1;
    const int * const defaultLeft           = t.getdefaultLeftForSplit() - 1;

    const FeatureIndexType maxLvl = t.getMaxLvl();

    FeatureIndexType i = 1;

    for (FeatureIndexType itr = 0; itr < maxLvl; itr++)
    {
        const FeatureIndexType splitFeature = fIndexes[i];
        i                                   = updateIndex(i, x[splitFeature], values, defaultLeft, featTypes, splitFeature, dispatcher);
    }
    return values[i];
}

template <typename algorithmFPType>
struct TileDimensions
{
    size_t nRowsTotal    = 0;
    size_t nTreesTotal   = 0;
    size_t nCols         = 0;
    size_t nRowsInBlock  = 0;
    size_t nTreesInBlock = 0;
    size_t nDataBlocks   = 0;
    size_t nTreeBlocks   = 0;

    TileDimensions(const NumericTable & data, size_t nTrees)
        : nTreesTotal(nTrees), nRowsTotal(data.getNumberOfRows()), nCols(data.getNumberOfColumns())
    {
        nRowsInBlock = nRowsTotal;

        if (nRowsTotal > 2 * VECTOR_BLOCK_SIZE)
        {
            nRowsInBlock = 2 * VECTOR_BLOCK_SIZE;

            if (daal::threader_get_threads_number() > nRowsTotal / nRowsInBlock)
            {
                nRowsInBlock = VECTOR_BLOCK_SIZE;
            }
        }
        nDataBlocks = nRowsTotal / nRowsInBlock;

        nTreesInBlock = nTreesTotal;
        nTreeBlocks   = 1;
    }
};

} /* namespace internal */
} /* namespace prediction */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif
