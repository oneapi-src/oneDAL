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
#include "services/error_handling.h"
#include "src/algorithms/dtrees/dtrees_feature_type_helper.h"
#include "src/algorithms/dtrees/gbt/gbt_model_impl.h"
#include "src/services/service_arrays.h"
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

/**
 * Determine the requested version of the TreeSHAP algorithm set in the
 * environment variable SHAP_VERSION.
 * Defaults to 0 if SHAP_VERSION is not set.
*/
uint8_t getRequestedAlgorithmVersion();

/**
 * Decision Path context
*/
struct PathElement
{
    int featureIndex                 = 0;
    float zeroFraction               = 0;
    float oneFraction                = 0;
    float partialWeight              = 0;
    PathElement()                    = default;
    PathElement(const PathElement &) = default;
};

/**
 * Model details required for TreeSHAP algorithms
*/
template <typename algorithmFPType>
struct ModelDetails
{
    size_t maxDepth                  = 0;
    size_t maxLeafs                  = 0;
    size_t maxNodes                  = 0;
    size_t maxCombinations           = 0;
    size_t nTreesToUse               = 0;
    bool requiresPrecompute          = false;
    algorithmFPType * combinationSum = nullptr;
    int * duplicatedNode             = nullptr;
    ModelDetails()                   = default;
    ModelDetails(const gbt::internal::GbtDecisionTree ** trees, size_t firstTreeIndex, size_t nTrees)
    {
        const uint8_t shapVersion = getRequestedAlgorithmVersion();
        requiresPrecompute        = shapVersion == 2;
        if (!requiresPrecompute)
        {
            // only Fast TreeSHAP v2.2 requires what we do here
            return;
        }

        nTreesToUse = nTrees;
        for (size_t i = firstTreeIndex; i < firstTreeIndex + nTreesToUse; ++i)
        {
            const gbt::internal::GbtDecisionTree * tree = trees[i];
            const size_t nNodes                         = tree->getNumberOfNodes();
            const size_t tDepth                         = tree->getMaxLvl();
            // this is over-estimating number of leafs, but that's okay because
            // we're only reserving memory
            // TODO: Add nLeafs to the tree structure
            //       (allocating space for the theoretical max is wasting space for sparse trees)
            const size_t nLeafs = static_cast<size_t>(1 << tDepth);

            maxDepth = maxDepth > tDepth ? maxDepth : tDepth;
            maxLeafs = maxLeafs > nLeafs ? maxLeafs : nLeafs;
            maxNodes = maxNodes > nNodes ? maxNodes : nNodes;
        }

        maxCombinations = static_cast<int>(1 << maxDepth);

        // allocate combinationSum buffer for Fast TreeSHAP v2.2
        combinationSum = static_cast<algorithmFPType *>(daal_calloc(sizeof(algorithmFPType) * maxLeafs * maxCombinations * nTreesToUse));
        DAAL_ASSERT(combinationSum);

        // allocate duplicatedNode buffer for Fast TreeSHAP v2.2
        duplicatedNode = static_cast<int *>(daal_calloc(sizeof(int) * maxNodes * nTreesToUse));
        DAAL_ASSERT(duplicatedNode);
    }
    ~ModelDetails()
    {
        if (combinationSum)
        {
            daal_free(combinationSum);
            combinationSum = nullptr;
        }
        if (duplicatedNode)
        {
            daal_free(duplicatedNode);
            duplicatedNode = nullptr;
        }
    }
};

namespace internal
{

namespace v0
{

void extendPath(PathElement * uniquePath, size_t uniqueDepth, float zeroFraction, float oneFraction, int featureIndex);
void unwindPath(PathElement * uniquePath, size_t uniqueDepth, size_t pathIndex);
float unwoundPathSum(const PathElement * uniquePath, size_t uniqueDepth, size_t pathIndex);

/** Recursive treeShap function
 * \param nodeIndex the index of the current node in the tree, counted from 1
 * \param depth how deep are we in the tree
 * \param uniqueDepth how many unique features are above the current node in the tree
 * \param parentUniquePath a vector of statistics about our current path through the tree
 * \param parentZeroFraction what fraction of the parent path weight is coming as 0 (integrated)
 * \param parentOneFraction what fraction of the parent path weight is coming as 1 (fixed)
 * \param parentFeatureIndex what feature the parent node used to split
 * \param conditionFraction what fraction of the current weight matches our conditioning feature
 */
template <typename algorithmFPType, bool hasUnorderedFeatures, bool hasAnyMissing>
inline void treeShap(const gbt::internal::GbtDecisionTree * tree, const algorithmFPType * x, algorithmFPType * phi, FeatureTypes * featureHelper,
                     size_t nodeIndex, size_t depth, size_t uniqueDepth, PathElement * parentUniquePath, float parentZeroFraction,
                     float parentOneFraction, int parentFeatureIndex, int condition, FeatureIndexType conditionFeature, float conditionFraction)
{
    DAAL_ASSERT(parentUniquePath);

    // stop if we have no weight coming down to us
    if (conditionFraction < FLT_EPSILON) return;

    const gbt::prediction::internal::ModelFPType * const splitValues     = tree->getSplitPoints() - 1;
    const gbt::prediction::internal::FeatureIndexType * const fIndexes   = tree->getFeatureIndexesForSplit() - 1;
    const gbt::prediction::internal::ModelFPType * const nodeCoverValues = tree->getNodeCoverValues() - 1;
    const int * const defaultLeft                                        = tree->getDefaultLeftForSplit() - 1;

    PathElement * uniquePath = parentUniquePath + uniqueDepth + 1;
    const size_t nBytes      = (uniqueDepth + 1) * sizeof(PathElement);
    const int copyStatus     = daal::services::internal::daal_memcpy_s(uniquePath, nBytes, parentUniquePath, nBytes);
    DAAL_ASSERT(copyStatus == 0);

    if (condition == 0 || conditionFeature != static_cast<FeatureIndexType>(parentFeatureIndex))
    {
        extendPath(uniquePath, uniqueDepth, parentZeroFraction, parentOneFraction, parentFeatureIndex);
    }

    const bool isLeaf = gbt::internal::ModelImpl::nodeIsLeaf(nodeIndex, *tree, depth);

    // leaf node
    if (isLeaf)
    {
        for (size_t i = 1; i <= uniqueDepth; ++i)
        {
            const float w          = unwoundPathSum(uniquePath, uniqueDepth, i);
            const PathElement & el = uniquePath[i];
            phi[el.featureIndex] += w * (el.oneFraction - el.zeroFraction) * splitValues[nodeIndex] * conditionFraction;
        }

        return;
    }

    const FeatureIndexType splitIndex = fIndexes[nodeIndex];
    const algorithmFPType dataValue   = x[splitIndex];

    gbt::prediction::internal::PredictDispatcher<hasUnorderedFeatures, hasAnyMissing> dispatcher;
    FeatureIndexType hotIndex        = updateIndex(nodeIndex, x[splitIndex], splitValues, defaultLeft, *featureHelper, splitIndex, dispatcher);
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
        const FeatureIndexType castIndex = static_cast<FeatureIndexType>(uniquePath[previousSplitPathIndex].featureIndex);

        // It cannot be that a feature that is ignored is in the uniquePath
        DAAL_ASSERT((condition == 0) || (castIndex != conditionFeature));

        if (castIndex == splitIndex)
        {
            break;
        }
    }
    if (previousSplitPathIndex != uniqueDepth + 1)
    {
        incomingZeroFraction = uniquePath[previousSplitPathIndex].zeroFraction;
        incomingOneFraction  = uniquePath[previousSplitPathIndex].oneFraction;
        unwindPath(uniquePath, uniqueDepth, previousSplitPathIndex);
        uniqueDepth -= 1;
    }

    // divide up the conditionFraction among the recursive calls
    float hotConditionFraction  = conditionFraction;
    float coldConditionFraction = conditionFraction;
    if (condition > 0 && splitIndex == conditionFeature)
    {
        coldConditionFraction = 0;
        uniqueDepth -= 1;
    }
    else if (condition < 0 && splitIndex == conditionFeature)
    {
        hotConditionFraction *= hotZeroFraction;
        coldConditionFraction *= coldZeroFraction;
        uniqueDepth -= 1;
    }

    treeShap<algorithmFPType, hasUnorderedFeatures, hasAnyMissing>(tree, x, phi, featureHelper, hotIndex, depth + 1, uniqueDepth + 1, uniquePath,
                                                                   hotZeroFraction * incomingZeroFraction, incomingOneFraction, splitIndex, condition,
                                                                   conditionFeature, hotConditionFraction);
    treeShap<algorithmFPType, hasUnorderedFeatures, hasAnyMissing>(tree, x, phi, featureHelper, coldIndex, depth + 1, uniqueDepth + 1, uniquePath,
                                                                   coldZeroFraction * incomingZeroFraction, 0, splitIndex, condition,
                                                                   conditionFeature, coldConditionFraction);
}

/**
 * \brief Version 0, i.e. the original TreeSHAP algorithm to compute feature attributions for a single tree
 * \param tree current tree
 * \param x dense data matrix
 * \param phi dense output matrix of feature attributions
 * \param featureHelper pointer to a FeatureTypes object (required to traverse tree)
 * \param condition fix one feature to either off (-1) on (1) or not fixed (0 default)
 * \param conditionFeature the index of the feature to fix
 */
template <typename algorithmFPType, bool hasUnorderedFeatures, bool hasAnyMissing>
inline services::Status treeShap(const gbt::internal::GbtDecisionTree * tree, const algorithmFPType * x, algorithmFPType * phi,
                                 FeatureTypes * featureHelper, int condition, FeatureIndexType conditionFeature)
{
    services::Status st;
    const int depth              = tree->getMaxLvl() + 2;
    const size_t bufferSize      = sizeof(PathElement) * ((depth * (depth + 1)) / 2);
    PathElement * uniquePathData = static_cast<PathElement *>(daal_calloc(bufferSize));
    DAAL_CHECK_MALLOC(uniquePathData)

    treeShap<algorithmFPType, hasUnorderedFeatures, hasAnyMissing>(tree, x, phi, featureHelper, 1, 0, 0, uniquePathData, 1, 1, -1, condition,
                                                                   conditionFeature, 1);

    daal_free(uniquePathData);

    return st;
}

} // namespace v0

namespace v1
{

void extendPath(PathElement * uniquePath, float * pWeights, unsigned uniqueDepth, unsigned uniqueDepthPWeights, float zeroFraction, float oneFraction,
                int featureIndex);
void unwindPath(PathElement * uniquePath, float * pWeights, unsigned uniqueDepth, unsigned uniqueDepthPWeights, unsigned pathIndex);
float unwoundPathSum(const PathElement * uniquePath, const float * pWeights, unsigned uniqueDepth, unsigned uniqueDepthPWeights, unsigned pathIndex);
float unwoundPathSumZero(const float * pWeights, unsigned uniqueDepth, unsigned uniqueDepthPWeights);

/**
 * Recursive Fast TreeSHAP version 1
 * Important: nodeIndex is counted from 0 here!
*/
template <typename algorithmFPType, bool hasUnorderedFeatures, bool hasAnyMissing>
inline void treeShap(const gbt::internal::GbtDecisionTree * tree, const algorithmFPType * x, algorithmFPType * phi, FeatureTypes * featureHelper,
                     size_t nodeIndex, size_t depth, size_t uniqueDepth, size_t uniqueDepthPWeights, PathElement * parentUniquePath,
                     float * parentPWeights, algorithmFPType pWeightsResidual, float parentZeroFraction, float parentOneFraction,
                     int parentFeatureIndex, int condition, FeatureIndexType conditionFeature, float conditionFraction)
{
    // stop if we have no weight coming down to us
    if (conditionFraction < FLT_EPSILON) return;

    const size_t numOutputs                                              = 1; // TODO: support multi-output models
    const gbt::prediction::internal::ModelFPType * const splitValues     = tree->getSplitPoints() - 1;
    const int * const defaultLeft                                        = tree->getDefaultLeftForSplit() - 1;
    const gbt::prediction::internal::FeatureIndexType * const fIndexes   = tree->getFeatureIndexesForSplit() - 1;
    const gbt::prediction::internal::ModelFPType * const nodeCoverValues = tree->getNodeCoverValues() - 1;

    // extend the unique path
    PathElement * uniquePath = parentUniquePath + uniqueDepth + 1;
    size_t nBytes            = (uniqueDepth + 1) * sizeof(PathElement);
    int copyStatus           = daal::services::internal::daal_memcpy_s(uniquePath, nBytes, parentUniquePath, nBytes);
    DAAL_ASSERT(copyStatus == 0);
    // extend pWeights
    float * pWeights = parentPWeights + uniqueDepthPWeights + 1;
    nBytes           = (uniqueDepthPWeights + 1) * sizeof(float);
    copyStatus       = daal::services::internal::daal_memcpy_s(pWeights, nBytes, parentPWeights, nBytes);
    DAAL_ASSERT(copyStatus == 0);

    if (condition == 0 || conditionFeature != static_cast<unsigned>(parentFeatureIndex))
    {
        extendPath(uniquePath, pWeights, uniqueDepth, uniqueDepthPWeights, parentZeroFraction, parentOneFraction, parentFeatureIndex);
        // update pWeightsResidual if the feature of the last split does not satisfy the threshold
        if (parentOneFraction != 1)
        {
            pWeightsResidual *= parentZeroFraction;
            uniqueDepthPWeights -= 1;
        }
    }

    const bool isLeaf = gbt::internal::ModelImpl::nodeIsLeaf(nodeIndex, *tree, depth);

    if (isLeaf)
    {
        // +1 to account for -1 in splitValues array
        const unsigned valuesOffset = nodeIndex * numOutputs;
        unsigned valuesNonZeroInd   = 0;
        unsigned valuesNonZeroCount = 0;
        for (unsigned j = 0; j < numOutputs; ++j)
        {
            if (splitValues[valuesOffset + j] != 0)
            {
                valuesNonZeroInd = j;
                valuesNonZeroCount++;
            }
        }
        // pre-calculate wZero for all features not satisfying the thresholds
        const algorithmFPType wZero     = unwoundPathSumZero(pWeights, uniqueDepth, uniqueDepthPWeights);
        const algorithmFPType scaleZero = -wZero * pWeightsResidual * conditionFraction;
        algorithmFPType scale;
        for (unsigned i = 1; i <= uniqueDepth; ++i)
        {
            const PathElement & el   = uniquePath[i];
            const unsigned phiOffset = el.featureIndex * numOutputs;
            // update contributions to SHAP values for features satisfying the thresholds and not satisfying the thresholds separately
            if (el.oneFraction != 0)
            {
                const algorithmFPType w = unwoundPathSum(uniquePath, pWeights, uniqueDepth, uniqueDepthPWeights, i);
                scale                   = w * pWeightsResidual * (1 - el.zeroFraction) * conditionFraction;
            }
            else
            {
                scale = scaleZero;
            }
            if (valuesNonZeroCount == 1)
            {
                phi[phiOffset + valuesNonZeroInd] += scale * splitValues[valuesOffset + valuesNonZeroInd];
            }
            else
            {
                for (unsigned j = 0; j < numOutputs; ++j)
                {
                    phi[phiOffset + j] += scale * splitValues[valuesOffset + j];
                }
            }
        }

        return;
    }

    const unsigned splitIndex = fIndexes[nodeIndex];
    gbt::prediction::internal::PredictDispatcher<hasUnorderedFeatures, hasAnyMissing> dispatcher;
    FeatureIndexType hotIndex        = updateIndex(nodeIndex, x[splitIndex], splitValues, defaultLeft, *featureHelper, splitIndex, dispatcher);
    const FeatureIndexType coldIndex = 2 * nodeIndex + (hotIndex == (2 * nodeIndex));

    const algorithmFPType w                = nodeCoverValues[nodeIndex];
    const algorithmFPType hotZeroFraction  = nodeCoverValues[hotIndex] / w;
    const algorithmFPType coldZeroFraction = nodeCoverValues[coldIndex] / w;
    algorithmFPType incomingZeroFraction   = 1;
    algorithmFPType incomingOneFraction    = 1;

    // see if we have already split on this feature,
    // if so we undo that split so we can redo it for this node
    unsigned pathIndex = 0;
    for (; pathIndex <= uniqueDepth; ++pathIndex)
    {
        if (static_cast<unsigned>(uniquePath[pathIndex].featureIndex) == splitIndex) break;
    }
    if (pathIndex != uniqueDepth + 1)
    {
        incomingZeroFraction = uniquePath[pathIndex].zeroFraction;
        incomingOneFraction  = uniquePath[pathIndex].oneFraction;
        unwindPath(uniquePath, pWeights, uniqueDepth, uniqueDepthPWeights, pathIndex);
        uniqueDepth -= 1;
        // update pWeightsResidual iff the duplicated feature does not satisfy the threshold
        if (incomingOneFraction != 0.)
        {
            uniqueDepthPWeights -= 1;
        }
        else
        {
            pWeightsResidual /= incomingZeroFraction;
        }
    }

    // divide up the conditionFraction among the recursive calls
    algorithmFPType hotConditionFraction  = conditionFraction;
    algorithmFPType coldConditionFraction = conditionFraction;
    if (condition > 0 && splitIndex == conditionFeature)
    {
        coldConditionFraction = 0;
        uniqueDepth -= 1;
        uniqueDepthPWeights -= 1;
    }
    else if (condition < 0 && splitIndex == conditionFeature)
    {
        hotConditionFraction *= hotZeroFraction;
        coldConditionFraction *= coldZeroFraction;
        uniqueDepth -= 1;
        uniqueDepthPWeights -= 1;
    }

    treeShap<algorithmFPType, hasUnorderedFeatures, hasAnyMissing>(
        tree, x, phi, featureHelper, hotIndex, depth + 1, uniqueDepth + 1, uniqueDepthPWeights + 1, uniquePath, pWeights, pWeightsResidual,
        hotZeroFraction * incomingZeroFraction, incomingOneFraction, splitIndex, condition, conditionFeature, hotConditionFraction);

    treeShap<algorithmFPType, hasUnorderedFeatures, hasAnyMissing>(
        tree, x, phi, featureHelper, coldIndex, depth + 1, uniqueDepth + 1, uniqueDepthPWeights + 1, uniquePath, pWeights, pWeightsResidual,
        coldZeroFraction * incomingZeroFraction, 0, splitIndex, condition, conditionFeature, coldConditionFraction);
}

/**
 * \brief Version 1, i.e. first Fast TreeSHAP algorithm
 * \param tree current tree
 * \param x dense data matrix
 * \param phi dense output matrix of feature attributions
 * \param featureHelper pointer to a FeatureTypes object (required to traverse tree)
 * \param condition fix one feature to either off (-1) on (1) or not fixed (0 default)
 * \param conditionFeature the index of the feature to fix
 */
template <typename algorithmFPType, bool hasUnorderedFeatures, bool hasAnyMissing>
inline services::Status treeShap(const gbt::internal::GbtDecisionTree * tree, const algorithmFPType * x, algorithmFPType * phi,
                                 FeatureTypes * featureHelper, int condition, FeatureIndexType conditionFeature)
{
    services::Status st;

    // pre-allocate space for the unique path data and pWeights
    const int depth              = tree->getMaxLvl() + 2;
    const size_t nElements       = (depth * (depth + 1)) / 2;
    PathElement * uniquePathData = static_cast<PathElement *>(daal_calloc(sizeof(PathElement) * nElements));
    DAAL_CHECK_MALLOC(uniquePathData)

    float * pWeights = static_cast<float *>(daal_calloc(sizeof(float) * nElements));
    DAAL_CHECK_MALLOC(pWeights)

    treeShap<algorithmFPType, hasUnorderedFeatures, hasAnyMissing>(tree, x, phi, featureHelper, 1, 0, 0, 0, uniquePathData, pWeights, 1, 1, 1, -1,
                                                                   condition, conditionFeature, 1);

    daal_free(uniquePathData);
    daal_free(pWeights);

    return st;
}
} // namespace v1

namespace v2
{
template <typename algorithmFPType>
inline void computeCombinationSum(const gbt::internal::GbtDecisionTree * tree, algorithmFPType * combinationSum, int * duplicatedNode,
                                  unsigned nodeIndex, unsigned depth, unsigned uniqueDepth, int * parentUniqueDepthPWeights,
                                  PathElement * parentUniquePath, float * parentPWeights, float parentZeroFraction, int parentFeatureIndex,
                                  int * leafCount, size_t maxDepth)
{
    const gbt::prediction::internal::FeatureIndexType * const fIndexes   = tree->getFeatureIndexesForSplit() - 1;
    const gbt::prediction::internal::ModelFPType * const nodeCoverValues = tree->getNodeCoverValues() - 1;

    // extend the unique path
    PathElement * uniquePath = parentUniquePath + uniqueDepth;
    size_t nBytes            = uniqueDepth * sizeof(PathElement);
    int copyStatus           = daal::services::internal::daal_memcpy_s(uniquePath, nBytes, parentUniquePath, nBytes);
    DAAL_ASSERT(copyStatus == 0);

    uniquePath[uniqueDepth].featureIndex = parentFeatureIndex;
    uniquePath[uniqueDepth].zeroFraction = parentZeroFraction;

    unsigned l;
    int * uniqueDepthPWeights;
    float * pWeights;
    float * tPWeights;

    // extend pWeights and update uniqueDepthPWeights
    if (uniqueDepth == 0)
    {
        l                      = 1;
        uniqueDepthPWeights    = parentUniqueDepthPWeights;
        uniqueDepthPWeights[0] = 0;
        pWeights               = parentPWeights;
        pWeights[0]            = 1.0f;
    }
    else
    {
        l                   = static_cast<int>(1 << (uniqueDepth - 1));
        uniqueDepthPWeights = parentUniqueDepthPWeights + l;
        nBytes              = l * sizeof(int);
        copyStatus          = daal::services::internal::daal_memcpy_s(uniqueDepthPWeights, nBytes, parentUniqueDepthPWeights, nBytes);
        DAAL_ASSERT(copyStatus == 0);
        copyStatus = daal::services::internal::daal_memcpy_s(uniqueDepthPWeights + l, nBytes, parentUniqueDepthPWeights, nBytes);
        DAAL_ASSERT(copyStatus == 0);
        std::copy(parentUniqueDepthPWeights, parentUniqueDepthPWeights + l, uniqueDepthPWeights);
        std::copy(parentUniqueDepthPWeights, parentUniqueDepthPWeights + l, uniqueDepthPWeights + l);

        pWeights   = parentPWeights + l * (maxDepth + 1);
        nBytes     = l * (maxDepth + 1) * sizeof(float);
        copyStatus = daal::services::internal::daal_memcpy_s(pWeights, nBytes, parentPWeights, nBytes);
        DAAL_ASSERT(copyStatus == 0);
        copyStatus = daal::services::internal::daal_memcpy_s(pWeights + l * (maxDepth + 1), nBytes, parentPWeights, nBytes);
        DAAL_ASSERT(copyStatus == 0);

        for (unsigned t = 0; t < l; t++)
        {
            tPWeights = pWeights + t * (maxDepth + 1);
            for (int i = uniqueDepthPWeights[t] - 1; i >= 0; i--)
            {
                tPWeights[i] *= (uniqueDepth - i) / static_cast<algorithmFPType>(uniqueDepth + 1);
            }
            uniqueDepthPWeights[t] -= 1;
        }
        for (unsigned t = l; t < 2 * l; t++)
        {
            tPWeights                         = pWeights + t * (maxDepth + 1);
            tPWeights[uniqueDepthPWeights[t]] = 0.0f;
            for (int i = uniqueDepthPWeights[t] - 1; i >= 0; i--)
            {
                tPWeights[i + 1] += tPWeights[i] * (i + 1) / static_cast<algorithmFPType>(uniqueDepth + 1);
                tPWeights[i] *= parentZeroFraction * (uniqueDepth - i) / static_cast<algorithmFPType>(uniqueDepth + 1);
            }
        }
    }

    const bool isLeaf = gbt::internal::ModelImpl::nodeIsLeaf(nodeIndex, *tree, depth);
    if (isLeaf)
    {
        // calculate one row of combinationSum for the current path
        algorithmFPType * leafCombinationSum = combinationSum + leafCount[0] * static_cast<int>(1 << maxDepth);
        for (unsigned t = 0; t < 2 * l - 1; t++)
        {
            leafCombinationSum[t] = 0;
            tPWeights             = pWeights + t * (maxDepth + 1);
            for (int i = uniqueDepthPWeights[t]; i >= 0; i--)
            {
                leafCombinationSum[t] += tPWeights[i] / static_cast<algorithmFPType>(uniqueDepth - i);
            }
            leafCombinationSum[t] *= (uniqueDepth + 1);
        }
        leafCount[0] += 1;

        return;
    }

    const FeatureIndexType splitIndex = fIndexes[nodeIndex];

    const unsigned leftIndex                = 2 * nodeIndex;
    const unsigned rightIndex               = 2 * nodeIndex + 1;
    const algorithmFPType w                 = nodeCoverValues[nodeIndex];
    const algorithmFPType leftZeroFraction  = nodeCoverValues[leftIndex] / w;
    const algorithmFPType rightZeroFraction = nodeCoverValues[rightIndex] / w;
    algorithmFPType incomingZeroFraction    = 1;

    // see if we have already split on this feature,
    // if so we undo that split so we can redo it for this node
    unsigned pathIndex = 0;
    for (; pathIndex <= uniqueDepth; ++pathIndex)
    {
        if (static_cast<unsigned>(uniquePath[pathIndex].featureIndex) == splitIndex) break;
    }
    if (pathIndex != uniqueDepth + 1)
    {
        duplicatedNode[nodeIndex] = pathIndex; // record node index of duplicated feature
        incomingZeroFraction      = uniquePath[pathIndex].zeroFraction;

        // shrink pWeights and uniquePath, and update uniqueDepthPWeights, given the duplicated feature
        unsigned p = static_cast<int>(1 << (pathIndex - 1));
        unsigned t = 0;
        float * kPWeights;
        for (unsigned j = 0; j < 2 * l; j += 2 * p)
        {
            for (unsigned k = j; k < j + p; k++)
            {
                tPWeights = pWeights + t * (maxDepth + 1);
                kPWeights = pWeights + k * (maxDepth + 1);
                for (int i = uniqueDepthPWeights[k]; i >= 0; i--)
                {
                    tPWeights[i] = kPWeights[i] * (uniqueDepth + 1) / static_cast<algorithmFPType>(uniqueDepth - i);
                }
                uniqueDepthPWeights[t] = uniqueDepthPWeights[k];
                t += 1;
            }
        }
        for (unsigned i = pathIndex; i < uniqueDepth; ++i)
        {
            uniquePath[i].featureIndex = uniquePath[i + 1].featureIndex;
            uniquePath[i].zeroFraction = uniquePath[i + 1].zeroFraction;
        }
        uniqueDepth -= 1;
    }
    else
    {
        duplicatedNode[nodeIndex] = -1;
    }

    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (unsigned t = 0; t < 2 * l; ++t)
    {
        ++(uniqueDepthPWeights[t]);
    }

    computeCombinationSum<algorithmFPType>(tree, combinationSum, duplicatedNode, leftIndex, depth + 1, uniqueDepth + 1, uniqueDepthPWeights,
                                           uniquePath, pWeights, incomingZeroFraction * leftZeroFraction, splitIndex, leafCount, maxDepth);

    computeCombinationSum<algorithmFPType>(tree, combinationSum, duplicatedNode, rightIndex, depth + 1, uniqueDepth + 1, uniqueDepthPWeights,
                                           uniquePath, pWeights, incomingZeroFraction * rightZeroFraction, splitIndex, leafCount, maxDepth);
}

template <typename algorithmFPType>
inline services::Status computeCombinationSum(const gbt::internal::GbtDecisionTree * tree, algorithmFPType * combinationSum, int * duplicatedNode,
                                              size_t maxDepth)
{
    services::Status st;

    // const size_t maxDepth             = tree->getMaxLvl();
    // const size_t maxCombinations      = 1 << maxDepth;
    // const size_t nuniqueDepthPWeights = 2 * maxCombinations;
    // const size_t nPWeights            = 2 * maxCombinations * (maxDepth + 1);
    // const size_t nUniquePath          = (maxDepth + 1) * (maxDepth + 2) / 2;

    // for (unsigned maxNode = 0; maxNode < tree->getNumberOfNodes(); ++maxNode)
    // {
    // // Pre-allocate space for the unique path data, pWeights and uniqueDepthPWeights
    // int * uniqueDepthPWeights = static_cast<int *>(daal_malloc(sizeof(int) * nuniqueDepthPWeights));
    // DAAL_CHECK_MALLOC(uniqueDepthPWeights)
    // for (size_t i = 0; i < nuniqueDepthPWeights; ++i)
    // {
    //     uniqueDepthPWeights[i] = 0;
    // }
    // printf("Allocated uniqueDepthPWeights @ %p\n", uniqueDepthPWeights);

    // float * pWeights = static_cast<float *>(daal_malloc(sizeof(float) * nPWeights));
    // DAAL_CHECK_MALLOC(pWeights)
    // for (size_t i = 0; i < nPWeights; ++i)
    // {
    //     pWeights[i] = 0.0f;
    // }
    // printf("Allocated pWeights            @ %p\n", pWeights);

    // PathElement * uniquePathData = static_cast<PathElement *>(daal_malloc(sizeof(PathElement) * nUniquePath));
    // DAAL_CHECK_MALLOC(uniquePathData)
    // PathElement init;
    // for (size_t i = 0; i < nUniquePath; ++i)
    // {
    //     DAAL_ASSERT(0 == daal::services::internal::daal_memcpy_s(uniquePathData + i, sizeof(PathElement), &init, sizeof(PathElement)));
    // }
    // printf("Allocated uniquePathData            @ %p\n", uniquePathData);
    // int leafCount = 0;

    const unsigned maxCombinations = static_cast<int>(1 << maxDepth);
    int * uniqueDepthPWeights      = new int[2 * maxCombinations];
    float * pWeights               = new float[2 * maxCombinations * (maxDepth + 1)];
    PathElement * uniquePathData   = new PathElement[(maxDepth + 1) * (maxDepth + 2) / 2];
    int * leafCount                = new int[1];
    leafCount[0]                   = 0;

    computeCombinationSum<algorithmFPType>(tree, combinationSum, duplicatedNode, 1, 0, 0, uniqueDepthPWeights, uniquePathData, pWeights, 1, -1,
                                           leafCount, maxDepth);

    delete[] uniqueDepthPWeights;
    delete[] pWeights;
    delete[] uniquePathData;
    delete[] leafCount;

    // printf("Free uniquePathData            @ %p", uniquePathData);
    // daal_free(uniquePathData);
    // printf("...success\n");
    // printf("Free pWeights            @ %p", pWeights);
    // daal_free(pWeights);
    // printf("...success\n");
    // printf("Free uniqueDepthPWeights @ %p", uniqueDepthPWeights);
    // daal_free(uniqueDepthPWeights);
    // printf("...success\n");
    // }

    return st;
}

template <typename algorithmFPType, bool hasUnorderedFeatures, bool hasAnyMissing>
inline void treeShap(const gbt::internal::GbtDecisionTree * tree, const algorithmFPType * x, algorithmFPType * phi, FeatureTypes * featureHelper,
                     size_t nodeIndex, size_t depth, size_t uniqueDepth, size_t uniqueDepthPWeights, PathElement * parentUniquePath,
                     float * parentPWeights, algorithmFPType pWeightsResidual, float parentZeroFraction, float parentOneFraction,
                     int parentFeatureIndex, int condition, FeatureIndexType conditionFeature, float conditionFraction)
{}

/**
 * \brief Version 2, i.e. second Fast TreeSHAP algorithm
 * \param tree current tree
 * \param x dense data matrix
 * \param phi dense output matrix of feature attributions
 * \param featureHelper pointer to a FeatureTypes object (required to traverse tree)
 * \param condition fix one feature to either off (-1) on (1) or not fixed (0 default)
 * \param conditionFeature the index of the feature to fix
 */
template <typename algorithmFPType, bool hasUnorderedFeatures, bool hasAnyMissing>
inline services::Status treeShap(const gbt::internal::GbtDecisionTree * tree, const algorithmFPType * x, algorithmFPType * phi,
                                 FeatureTypes * featureHelper, int condition, FeatureIndexType conditionFeature,
                                 const ModelDetails<algorithmFPType> & modelDetails)
{
    services::Status st;
    const int depth              = tree->getMaxLvl() + 2;
    const size_t nElements       = (depth * (depth + 1)) / 2;
    PathElement * uniquePathData = static_cast<PathElement *>(daal_calloc(sizeof(PathElement) * nElements));
    DAAL_CHECK_MALLOC(uniquePathData)
    int leafCount = 0;

    // treeShap(tree, combinationSum, duplicatedNode, data.X, data.X_missing, out_contribs, 0, 0, uniquePathData, 1, 1, 1, -1, leafCount);

    daal_free(uniquePathData);
    return st;
}
} // namespace v2
} // namespace internal

/**
 * \brief Return the combination sum, required for Fast TreeSHAP v2
*/
template <typename algorithmFPType>
services::Status computeCombinationSum(const gbt::internal::GbtDecisionTree * tree, const size_t treeIndex,
                                       const ModelDetails<algorithmFPType> & modelDetails)
{
    if (!modelDetails.requiresPrecompute)
    {
        // nothing to be done
        return services::Status();
    }
    if (!modelDetails.combinationSum || !modelDetails.duplicatedNode)
    {
        // buffer wasn't properly allocated
        return services::Status(ErrorMemoryAllocationFailed);
    }

    algorithmFPType * combinationSum = modelDetails.combinationSum + treeIndex * modelDetails.maxLeafs * modelDetails.maxCombinations;
    int * duplicatedNode             = modelDetails.duplicatedNode + treeIndex * modelDetails.maxNodes;
    return treeshap::internal::v2::computeCombinationSum<algorithmFPType>(tree, combinationSum, duplicatedNode, modelDetails.maxDepth);
}

/**
 * \brief Recursive function that computes the feature attributions for a single tree.
 * \param tree current tree
 * \param x dense data matrix
 * \param phi dense output matrix of feature attributions
 * \param featureHelper pointer to a FeatureTypes object (required to traverse tree)
 * \param condition fix one feature to either off (-1) on (1) or not fixed (0 default)
 * \param conditionFeature the index of the feature to fix
 */
template <typename algorithmFPType, bool hasUnorderedFeatures, bool hasAnyMissing>
inline services::Status treeShap(const gbt::internal::GbtDecisionTree * tree, const algorithmFPType * x, algorithmFPType * phi,
                                 FeatureTypes * featureHelper, int condition, FeatureIndexType conditionFeature,
                                 const ModelDetails<algorithmFPType> & modelDetails)
{
    DAAL_ASSERT(x);
    DAAL_ASSERT(phi);
    DAAL_ASSERT(featureHelper);

    uint8_t shapVersion = getRequestedAlgorithmVersion();

    switch (shapVersion)
    {
    case 0:
        return treeshap::internal::v0::treeShap<algorithmFPType, hasUnorderedFeatures, hasAnyMissing>(tree, x, phi, featureHelper, condition,
                                                                                                      conditionFeature);
    case 1:
        return treeshap::internal::v1::treeShap<algorithmFPType, hasUnorderedFeatures, hasAnyMissing>(tree, x, phi, featureHelper, condition,
                                                                                                      conditionFeature);
    case 2:
        return treeshap::internal::v2::treeShap<algorithmFPType, hasUnorderedFeatures, hasAnyMissing>(tree, x, phi, featureHelper, condition,
                                                                                                      conditionFeature, modelDetails);
    default: return services::Status(ErrorMethodNotImplemented);
    }
}

} // namespace treeshap
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif // __TREESHAP_H__
