/* file: treeshap.h */
/*******************************************************************************
* Copyright 2023 Intel Corporation
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

/**
 * Original TreeSHAP algorithm by Scott Lundberg, 2018
 * https://arxiv.org/abs/1802.03888
 *
 * Fast TreeSHAP algorithm v1 and v2 by Jilei Yang, 2021
 * https://arxiv.org/abs/2109.09847.
 */

/*
//++
//  Implementation of the treeShap algorithm
//--
*/

#ifndef __TREESHAP_H__
#define __TREESHAP_H__

#include "services/daal_defines.h"
#include "src/algorithms/dtrees/dtrees_feature_type_helper.h"
#include "src/algorithms/dtrees/gbt/gbt_model_impl.h"
#include "stdint.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace treeshap
{
using gbt::internal::FeatureIndexType;
using FeatureTypes = algorithms::dtrees::internal::FeatureTypes;

// data we keep about our decision path
// note that partialWeight is included for convenience and is not tied with the other attributes
// the partialWeight of the i'th path element is the permutation weight of paths with i-1 ones in them
struct PathElement
{
    // featureIndex -1 underflows, we use it as a reserved value for the initial node
    FeatureIndexType featureIndex    = -1;
    float zeroFraction               = 0;
    float oneFraction                = 0;
    float partialWeight              = 0;
    PathElement()                    = default;
    PathElement(const PathElement &) = default;
};

namespace internal
{

void treeShapExtendPath(PathElement * uniquePath, size_t uniqueDepth, float zeroFraction, float oneFraction, FeatureIndexType featureIndex);
void treeShapUnwindPath(PathElement * uniquePath, size_t uniqueDepth, size_t pathIndex);
float treeShapUnwoundPathSum(const PathElement * uniquePath, size_t uniqueDepth, size_t pathIndex);

/** Extension of
 * \param nodeIndex the index of the current node in the tree, counted from 1
 * \param depth how deep are we in the tree
 * \param uniqueDepth how many unique features are above the current node in the tree
 * \param parentZeroFraction what fraction of the parent path weight is coming as 0 (integrated)
 * \param parentOneFraction what fraction of the parent path weight is coming as 1 (fixed)
 * \param parentFeatureIndex what feature the parent node used to split
 * \param conditionFraction what fraction of the current weight matches our conditioning feature
 */
template <typename algorithmFPType, bool hasUnorderedFeatures, bool hasAnyMissing>
void treeShap(const gbt::internal::GbtDecisionTree * tree, const algorithmFPType * x, algorithmFPType * phi, size_t nFeatures,
              FeatureTypes * featureHelper, size_t nodeIndex, size_t depth, size_t uniqueDepth, PathElement * parentUniquePath,
              float parentZeroFraction, float parentOneFraction, FeatureIndexType parentFeatureIndex, int condition,
              FeatureIndexType conditionFeature, float conditionFraction)
{
    DAAL_ASSERT(parentUniquePath);

    // stop if we have no weight coming down to us
    if (conditionFraction < FLT_EPSILON) return;

    const size_t nNodes = tree->getNumberOfNodes();
    // splitValues contain
    //   - the feature value that is used for the split for internal nodes
    //   - the tree prediction for leaf nodes
    // we are accounting for the fact that nodeIndex is counted from 1 (not from 0) as required by helper functions
    const gbt::prediction::internal::ModelFPType * const splitValues     = tree->getSplitPoints() - 1;
    const gbt::prediction::internal::FeatureIndexType * const fIndexes   = tree->getFeatureIndexesForSplit() - 1;
    const gbt::prediction::internal::ModelFPType * const nodeCoverValues = tree->getNodeCoverValues() - 1;
    const int * const defaultLeft                                        = tree->getDefaultLeftForSplit() - 1;

    PathElement * uniquePath = parentUniquePath + uniqueDepth + 1;
    const size_t nBytes      = uniqueDepth * sizeof(PathElement);
    const int copy_status    = daal::services::internal::daal_memcpy_s(uniquePath, nBytes, parentUniquePath, nBytes);
    DAAL_ASSERT(copy_status == 0);

    if (condition == 0 || conditionFeature != parentFeatureIndex)
    {
        treeShapExtendPath(uniquePath, uniqueDepth, parentZeroFraction, parentOneFraction, parentFeatureIndex);
    }

    printf("--------------------------------------------------------------------------------\n");
    printf("depth                    = %lu\n", depth);
    printf("uniqueDepth              = %lu\n", uniqueDepth);
    printf("uniquePath               = %p\n", uniquePath);
    printf("uniquePath + uniqueDepth = %p\n", uniquePath + uniqueDepth);

    const bool isLeaf = gbt::internal::ModelImpl::nodeIsLeaf(nodeIndex, *tree, depth);

    // leaf node
    if (isLeaf)
    {
        for (size_t i = 1; i <= uniqueDepth; ++i)
        {
            const float w          = treeShapUnwoundPathSum(uniquePath, uniqueDepth, i);
            const PathElement & el = uniquePath[i];
            phi[el.featureIndex] += w * (el.oneFraction - el.zeroFraction) * splitValues[nodeIndex] * conditionFraction;
        }

        return;
    }

    const FeatureIndexType splitFeature = fIndexes[nodeIndex];
    const auto dataValue                = x[splitFeature];

    gbt::prediction::internal::PredictDispatcher<hasUnorderedFeatures, hasAnyMissing> dispatcher;
    FeatureIndexType hotIndex        = updateIndex(nodeIndex, dataValue, splitValues, defaultLeft, *featureHelper, splitFeature, dispatcher);
    const FeatureIndexType coldIndex = 2 * nodeIndex + (hotIndex == (2 * nodeIndex));

    const float w = nodeCoverValues[nodeIndex];
    DAAL_ASSERT(w > 0);
    const float hotZeroFraction  = nodeCoverValues[hotIndex] / w;
    const float coldZeroFraction = nodeCoverValues[coldIndex] / w;
    float incomingZeroFraction   = 1.0f;
    float incomingOneFraction    = 1.0f;

    DAAL_ASSERT(hotZeroFraction < 1.0f);
    DAAL_ASSERT(coldZeroFraction < 1.0f);

    // see if we have already split on this feature,
    // if so we undo that split so we can redo it for this node
    size_t previousSplitPathIndex = 0;
    for (; previousSplitPathIndex <= uniqueDepth; ++previousSplitPathIndex)
    {
        if (uniquePath[previousSplitPathIndex].featureIndex == splitFeature)
        {
            break;
        }
    }
    if (previousSplitPathIndex != uniqueDepth + 1)
    {
        incomingZeroFraction = uniquePath[previousSplitPathIndex].zeroFraction;
        incomingOneFraction  = uniquePath[previousSplitPathIndex].oneFraction;
        treeShapUnwindPath(uniquePath, uniqueDepth, previousSplitPathIndex);
        printf("previousSplitPathIndex != uniqueDepth + 1 : %lu != %lu\n", previousSplitPathIndex, uniqueDepth + 1);
        uniqueDepth -= 1;
    }

    // divide up the conditionFraction among the recursive calls
    float hotConditionFraction  = conditionFraction;
    float coldConditionFraction = conditionFraction;
    if (condition > 0 && splitFeature == conditionFeature)
    {
        coldConditionFraction = 0;
        uniqueDepth -= 1;
    }
    else if (condition < 0 && splitFeature == conditionFeature)
    {
        hotConditionFraction *= hotZeroFraction;
        coldConditionFraction *= coldZeroFraction;
        uniqueDepth -= 1;
    }

    treeShap<algorithmFPType, hasUnorderedFeatures, hasAnyMissing>(tree, x, phi, nFeatures, featureHelper, hotIndex, depth + 1, uniqueDepth + 1,
                                                                   uniquePath, hotZeroFraction * incomingZeroFraction, incomingOneFraction,
                                                                   splitFeature, condition, conditionFeature, hotConditionFraction);
    treeShap<algorithmFPType, hasUnorderedFeatures, hasAnyMissing>(tree, x, phi, nFeatures, featureHelper, coldIndex, depth + 1, uniqueDepth + 1,
                                                                   uniquePath, coldZeroFraction * incomingZeroFraction, 0, splitFeature, condition,
                                                                   conditionFeature, coldConditionFraction);
}
} // namespace internal

/**
 * \brief Recursive function that computes the feature attributions for a single tree.
 * \param tree current tree
 * \param x dense data matrix
 * \param phi dense output matrix of feature attributions
 * \param nFeatures number features, i.e. length of feat and phi vectors
 * \param featureHelper pointer to a FeatureTypes object (required to traverse tree)
 * \param parentUniquePath a vector of statistics about our current path through the tree
 * \param condition fix one feature to either off (-1) on (1) or not fixed (0 default)
 * \param conditionFeature the index of the feature to fix
 */
template <typename algorithmFPType, bool hasUnorderedFeatures, bool hasAnyMissing>
void treeShap(const gbt::internal::GbtDecisionTree * tree, const algorithmFPType * x, algorithmFPType * phi, size_t nFeatures,
              FeatureTypes * featureHelper, PathElement * parentUniquePath, int condition, FeatureIndexType conditionFeature)
{
    DAAL_ASSERT(x);
    DAAL_ASSERT(phi);
    DAAL_ASSERT(featureHelper);

    // parentFeatureIndex -1 underflows, we use it as a reserved value for the initial node
    treeshap::internal::treeShap<algorithmFPType, hasUnorderedFeatures, hasAnyMissing>(tree, x, phi, nFeatures, featureHelper, 1, 0, 0,
                                                                                       parentUniquePath, 1, 1, -1, condition, conditionFeature, 1);
}
} // namespace treeshap
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif // __TREESHAP_H__
