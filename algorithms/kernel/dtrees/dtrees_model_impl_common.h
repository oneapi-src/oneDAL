/* file: dtrees_model_impl_common.h */
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
//  Implementation of common and template functions for decision forest model
//--
*/

#ifndef __DTREES_MODEL_IMPL_COMMON__
#define __DTREES_MODEL_IMPL_COMMON__

#include "dtrees_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace dtrees
{
namespace internal
{

template <typename NodeLeaf>
void writeLeaf(const NodeLeaf& l, DecisionTreeNode& row);

template <typename NodeType, typename NodeBase>
void nodeToTable(const NodeBase& node, size_t iRow, size_t& iCur, DecisionTreeNode* aRow, double *impVals, int *nNodeSamplesVals)
{
    DecisionTreeNode& row = aRow[iRow];
    impVals[iRow] = node.impurity;
    nNodeSamplesVals[iRow] = (int)node.count;

    if(node.isSplit())
    {
        const typename NodeType::Split& s = *NodeType::castSplit(&node);
        row.featureIndex = s.featureIdx;
        DAAL_ASSERT(row.featureIndex >= 0);
        row.featureValueOrResponse = s.featureValue;
        row.leftIndexOrClass = iCur++; //+1 for left kid
        ++iCur;//+1 for right kid
        nodeToTable<NodeType, NodeBase>(*s.kid[0], row.leftIndexOrClass, iCur, aRow, impVals, nNodeSamplesVals);
        nodeToTable<NodeType, NodeBase>(*s.kid[1], row.leftIndexOrClass + 1, iCur, aRow, impVals, nNodeSamplesVals);
    }
    else
    {
        const typename NodeType::Leaf& l = *NodeType::castLeaf(&node);
        row.featureIndex = -1;
        writeLeaf<typename NodeType::Leaf>(l, row);
    }
}

template <typename TNodeType, typename TAllocator>
void TreeImpl<TNodeType, TAllocator>::convertToTable(DecisionTreeTable *treeTable,
    data_management::HomogenNumericTable<double> *impurities,
    data_management::HomogenNumericTable<int> *nNodeSamples) const
{
    const size_t nNode    = treeTable->getNumberOfRows();
    double *impVals       = impurities->getArray();
    int *nNodeSamplesVals = nNodeSamples->getArray();
    if(nNode)
    {
        DecisionTreeNode* aNode = (DecisionTreeNode*)treeTable->getArray();
        size_t iRow = 0; //index of the current available row in the table
        nodeToTable<TNodeType, typename TNodeType::Base>(*top(), iRow++, iRow, aNode, impVals, nNodeSamplesVals);
    }
}

template<typename TVisitor>
bool visitSplit(size_t iRowInTable, size_t level, const DecisionTreeNode* aNode, TVisitor& visitor);

template<typename TVisitor>
bool visitLeaf(size_t iRowInTable, size_t level, const DecisionTreeNode* aNode, TVisitor& visitor);

template<typename TVisitor, typename TDescriptor>
bool visitSplit(size_t iRowInTable, size_t level, TDescriptor& descSplit, const DecisionTreeNode* aNode, const double *imp,
    const int *nodeSamplesCount, TVisitor& visitor);

template<typename TVisitor, typename TDescriptor>
bool visitLeaf(size_t iRowInTable, size_t level, TDescriptor& descLeaf, const DecisionTreeNode* aNode, const double *imp,
    const int *nodeSamplesCount, TVisitor& visitor);

template <typename OnSplitFunctor, typename OnLeafFunctor>
bool traverseNodeDF(size_t level, size_t iRowInTable, const DecisionTreeNode* aNode, OnSplitFunctor &visitSplit, OnLeafFunctor &visitLeaf)
{
    const dtrees::internal::DecisionTreeNode& n = aNode[iRowInTable];
    if(n.isSplit())
    {
        if(!visitSplit(iRowInTable, level))
            return false; //do not continue traversing
        ++level;
        if(n.leftIndexOrClass && !traverseNodeDF(level, n.leftIndexOrClass, aNode, visitSplit, visitLeaf))
            return false; //do not continue traversing
        return (n.leftIndexOrClass ? traverseNodeDF(level, n.leftIndexOrClass + 1, aNode, visitSplit, visitLeaf) : true);
    }
    return visitLeaf(iRowInTable, level);
}

typedef services::Collection<size_t> NodeIdxArray;

template <typename OnSplitFunctor, typename OnLeafFunctor>
static bool traverseNodesBF(size_t level, NodeIdxArray& aCur,
    NodeIdxArray& aNext, const DecisionTreeNode* aNode, OnSplitFunctor &visitSplit, OnLeafFunctor &visitLeaf)
{
    for(size_t i = 0; i < aCur.size(); ++i)
    {
        for(size_t j = 0; j < (level ? 2 : 1); ++j)
        {
            size_t iRowInTable = aCur[i] + j;
            const DecisionTreeNode& n = aNode[iRowInTable];//right is next to left
            if(n.isSplit())
            {
                if(!visitSplit(iRowInTable, level))
                    return false; //do not continue traversing
                if(n.leftIndexOrClass)
                    aNext.push_back(n.leftIndexOrClass);
            }
            else
            {
                if(!visitLeaf(iRowInTable, level))
                    return false; //do not continue traversing
            }
        }
    }
    aCur.clear();
    if(!aNext.size())
        return true;//done
    return traverseNodesBF(level + 1, aNext, aCur, aNode, visitSplit, visitLeaf);
}

} // namespace internal
} // namespace dtrees
} // namespace algorithms
} // namespace daal

#endif
