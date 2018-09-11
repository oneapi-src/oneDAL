/* file: dtrees_predict_dense_default_impl.i */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of auxiliary functions for decision forest predict algorithms
//  (defaultDense) method.
//--
*/

#ifndef __DTREES_PREDICT_DENSE_DEFAULT_IMPL_I__
#define __DTREES_PREDICT_DENSE_DEFAULT_IMPL_I__

#include "dtrees_model_impl.h"
#include "service_data_utils.h"
#include "dtrees_feature_type_helper.h"
#include "service_environment.h"

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace dtrees
{
namespace prediction
{
namespace internal
{

using namespace dtrees::internal;
//////////////////////////////////////////////////////////////////////////////////////////
// Common service function. Finds node corresponding to the given observation
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, typename TreeType, CpuType cpu>
const typename TreeType::NodeType::Base* findNode(const dtrees::internal::Tree& t, const algorithmFPType* x)
{
    const TreeType& tree = static_cast<const TreeType&>(t);
    const typename TreeType::NodeType::Base* pNode = tree.top();
    if(tree.hasUnorderedFeatureSplits())
    {
        for(; pNode && pNode->isSplit();)
        {
            auto pSplit = TreeType::NodeType::castSplit(pNode);
            const int sn = (pSplit->featureUnordered ? (int(x[pSplit->featureIdx]) != int(pSplit->featureValue)) :
                daal::services::internal::SignBit<algorithmFPType, cpu>::get(pSplit->featureValue - x[pSplit->featureIdx]));
            pNode = pSplit->kid[sn];
        }
    }
    else
    {
        for(; pNode && pNode->isSplit();)
        {
            auto pSplit = TreeType::NodeType::castSplit(pNode);
            const int sn = daal::services::internal::SignBit<algorithmFPType, cpu>::get(pSplit->featureValue - x[pSplit->featureIdx]);
            pNode = pSplit->kid[sn];
        }
    }
    return pNode;
}

//////////////////////////////////////////////////////////////////////////////////////////
// Common service function. Finds a node corresponding to the given observation
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, typename TreeType, CpuType cpu>
const DecisionTreeNode* findNode(const dtrees::internal::DecisionTreeTable& t,
    const FeatureTypes& featTypes, const algorithmFPType* x)
{
    const DecisionTreeNode* aNode = (const DecisionTreeNode*)t.getArray();
    if(!aNode)
        return nullptr;
    const DecisionTreeNode* pNode = aNode;
    if(featTypes.hasUnorderedFeatures())
    {
        for(; pNode->isSplit();)
        {
            const int sn = (featTypes.isUnordered(pNode->featureIndex) ? (int(x[pNode->featureIndex]) != int(pNode->featureValue())) :
                daal::services::internal::SignBit<algorithmFPType, cpu>::get(algorithmFPType(pNode->featureValue()) - x[pNode->featureIndex]));
            DAAL_ASSERT(pNode->leftIndexOrClass > 0);
            DAAL_ASSERT(sn == 0 || sn == 1);
            DAAL_ASSERT(pNode->leftIndexOrClass + sn > 0 && pNode->leftIndexOrClass + sn < t.getNumberOfRows());
            pNode = aNode + (pNode->leftIndexOrClass + sn);
        }
    }
    else
    {
        for(; pNode->isSplit();)
        {
            const int sn = daal::services::internal::SignBit<algorithmFPType, cpu>::get(algorithmFPType(pNode->featureValue()) - x[pNode->featureIndex]);
            DAAL_ASSERT(pNode->leftIndexOrClass > 0);
            DAAL_ASSERT(sn == 0 || sn == 1);
            DAAL_ASSERT(pNode->leftIndexOrClass + sn > 0 && pNode->leftIndexOrClass + sn < t.getNumberOfRows());
            pNode = aNode + (pNode->leftIndexOrClass + sn);
        }
    }
    return pNode;
}

template <typename algorithmFPType>
struct TileDimensions
{
    size_t nRowsTotal = 0;
    size_t nTreesTotal = 0;
    size_t nCols = 0;
    size_t nRowsInBlock = 0;
    size_t nTreesInBlock = 0;
    size_t nDataBlocks = 0;
    size_t nTreeBlocks = 0;

    TileDimensions(const NumericTable& data, size_t nTrees, size_t treeSize, size_t nYPerRow = 1) :
        nTreesTotal(nTrees), nRowsTotal(data.getNumberOfRows()), nCols(data.getNumberOfColumns())
    {
        nRowsInBlock = services::internal::getNumElementsFitInMemory(services::internal::getL1CacheSize()*0.8,
            (nCols + nYPerRow)*sizeof(algorithmFPType), nRowsInBlockDefault);
        nTreesInBlock = services::internal::getNumElementsFitInMemory(services::internal::getLLCacheSize()*0.8, treeSize, nTrees);
        nDataBlocks = nRowsTotal / nRowsInBlock + !!(nRowsTotal%nRowsInBlock);
        nTreeBlocks = nTreesTotal / nTreesInBlock + !!(nTreesTotal%nTreesInBlock);
    }
    static const size_t nRowsInBlockDefault = 500;
};

} /* namespace internal */
} /* namespace prediction */
} /* namespace dtrees */
} /* namespace algorithms */
} /* namespace daal */

#endif
