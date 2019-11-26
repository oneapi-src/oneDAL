/* file: dtrees_regression_model.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of the regression model methods common for dtrees
//--
*/

#include "dtrees_model_impl_common.h"
#include "algorithms/regression/tree_traverse.h"
#include "algorithms/tree_utils/tree_utils_regression.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::algorithms::dtrees::internal;

namespace daal
{
namespace algorithms
{
typedef daal::algorithms::dtrees::internal::TreeNodeRegression<RegressionFPType>::Leaf TLeaf;

namespace dtrees
{
namespace internal
{
template <>
void writeLeaf(const TLeaf & l, DecisionTreeNode & row)
{
    row.featureValueOrResponse = l.response;
}

template <>
bool visitSplit(size_t iRowInTable, size_t level, const DecisionTreeNode * aNode, algorithms::regression::TreeNodeVisitor & visitor)
{
    const DecisionTreeNode & n = aNode[iRowInTable];
    return visitor.onSplitNode(level, n.featureIndex, n.featureValueOrResponse);
}

template <>
bool visitLeaf(size_t iRowInTable, size_t level, const DecisionTreeNode * aNode, algorithms::regression::TreeNodeVisitor & visitor)
{
    const DecisionTreeNode & n = aNode[iRowInTable];
    return visitor.onLeafNode(level, n.featureValueOrResponse);
}

template <>
bool visitSplit(size_t iRowInTable, size_t level, tree_utils::SplitNodeDescriptor & descSplit, const DecisionTreeNode * aNode, const double * imp,
                const int * nodeSamplesCount, tree_utils::regression::TreeNodeVisitor & visitor)
{
    const DecisionTreeNode & n = aNode[iRowInTable];
    if (imp) descSplit.impurity = imp[iRowInTable];
    if (nodeSamplesCount) descSplit.nNodeSampleCount = (size_t)(nodeSamplesCount[iRowInTable]);
    descSplit.featureIndex = n.featureIndex;
    descSplit.featureValue = n.featureValueOrResponse;
    descSplit.level        = level;
    return visitor.onSplitNode(descSplit);
}

template <>
bool visitLeaf(size_t iRowInTable, size_t level, tree_utils::regression::LeafNodeDescriptor & descLeaf, const DecisionTreeNode * aNode,
               const double * imp, const int * nodeSamplesCount, tree_utils::regression::TreeNodeVisitor & visitor)
{
    const DecisionTreeNode & n = aNode[iRowInTable];
    if (imp) descLeaf.impurity = imp[iRowInTable];
    if (nodeSamplesCount) descLeaf.nNodeSampleCount = (size_t)(nodeSamplesCount[iRowInTable]);
    descLeaf.level    = level;
    descLeaf.response = n.featureValueOrResponse;
    return visitor.onLeafNode(descLeaf);
}

} // namespace internal
} // namespace dtrees
} // namespace algorithms
} // namespace daal
