/* file: dtrees_regression_model.cpp */
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
template<>
void writeLeaf(const TLeaf& l, DecisionTreeNode& row)
{
    row.featureValueOrResponse = l.response;
}

template<>
bool visitSplit(size_t iRowInTable, size_t level, const DecisionTreeNode* aNode, algorithms::regression::TreeNodeVisitor& visitor)
{
    const DecisionTreeNode& n = aNode[iRowInTable];
    return visitor.onSplitNode(level, n.featureIndex, n.featureValueOrResponse);
}

template<>
bool visitLeaf(size_t iRowInTable, size_t level, const DecisionTreeNode* aNode, algorithms::regression::TreeNodeVisitor& visitor)
{
    const DecisionTreeNode& n = aNode[iRowInTable];
    return visitor.onLeafNode(level, n.featureValueOrResponse);
}

template<>
bool visitSplit(size_t iRowInTable, size_t level, tree_utils::SplitNodeDescriptor& descSplit, const DecisionTreeNode* aNode, const double *imp,
    const int *nodeSamplesCount, tree_utils::regression::TreeNodeVisitor& visitor)
{
    const DecisionTreeNode& n                       = aNode[iRowInTable];
    if(imp)              descSplit.impurity         = imp[iRowInTable];
    if(nodeSamplesCount) descSplit.nNodeSampleCount = (size_t)(nodeSamplesCount[iRowInTable]);
    descSplit.featureIndex                          = n.featureIndex;
    descSplit.featureValue                          = n.featureValueOrResponse;
    descSplit.level                                 = level;
    return visitor.onSplitNode(descSplit);
}

template<>
bool visitLeaf(size_t iRowInTable, size_t level, tree_utils::regression::LeafNodeDescriptor& descLeaf, const DecisionTreeNode* aNode, const double *imp,
    const int *nodeSamplesCount, tree_utils::regression::TreeNodeVisitor& visitor)
{
    const DecisionTreeNode& n                      = aNode[iRowInTable];
    if(imp)              descLeaf.impurity         = imp[iRowInTable];
    if(nodeSamplesCount) descLeaf.nNodeSampleCount = (size_t)(nodeSamplesCount[iRowInTable]);
    descLeaf.level                                 = level;
    descLeaf.response                              = n.featureValueOrResponse;
    return visitor.onLeafNode(descLeaf);
}

} // namespace dtrees
} // namespace internal
} // namespace algorithms
} // namespace daal
