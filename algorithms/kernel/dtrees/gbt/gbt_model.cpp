/* file: gbt_model.cpp */
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
//  Implementation of the class defining the gradient boosted trees model
//--
*/

#include "daal_defines.h"
#include "gbt_model_impl.h"
#include "dtrees_model_impl_common.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::algorithms::dtrees::internal;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace internal
{
typedef gbt::internal::ModelImpl::TreeType::NodeType::Leaf TLeaf;

int GbtDecisionTree::serializationTag() { return _desc.tag(); }
int GbtDecisionTree::getSerializationTag() const { return _desc.tag(); }
static data_management::SerializationIface* creatorGbtDecisionTree() { return new GbtDecisionTree(); }
data_management::SerializationDesc GbtDecisionTree::_desc(creatorGbtDecisionTree, SERIALIZATION_GBT_DECISION_TREE_ID);

size_t ModelImpl::numberOfTrees() const
{
    return ImplType::size();
}

void ModelImpl::traverseDF(size_t iTree, algorithms::regression::TreeNodeVisitor& visitor) const
{
    if(iTree >= size())
        return;
    const GbtDecisionTree& gbtTree = *at(iTree);
    SharedPtr<DecisionTreeTable> t = gbtTreeToDecisionTree(gbtTree);

    const DecisionTreeNode* aNode = (const DecisionTreeNode*)t->getArray();
    if(aNode)
    {
        auto onSplitNodeFunc = [&aNode, &visitor](size_t iRowInTable, size_t level) -> bool {
            return visitSplit(iRowInTable, level, aNode, visitor);
        };

        auto onLeafNodeFunc = [&aNode, &visitor](size_t iRowInTable, size_t level) -> bool {
            return visitLeaf(iRowInTable, level, aNode, visitor);
        };

        traverseNodeDF(0, 0, aNode, onSplitNodeFunc, onLeafNodeFunc);
    }
}

void ModelImpl::traverseBF(size_t iTree, algorithms::regression::TreeNodeVisitor& visitor) const
{
    if(iTree >= size())
        return;

    const GbtDecisionTree& gbtTree = *at(iTree);
    SharedPtr<DecisionTreeTable> t = gbtTreeToDecisionTree(gbtTree);
    const DecisionTreeNode* aNode = (const DecisionTreeNode*)t->getArray();

    NodeIdxArray aCur;//nodes of current layer
    NodeIdxArray aNext;//nodes of next layer
    if(aNode)
    {
        aCur.push_back(0);

        auto onSplitNodeFunc = [&aNode, &visitor](size_t iRowInTable, size_t level) -> bool {
            return visitSplit(iRowInTable, level, aNode, visitor);
        };

        auto onLeafNodeFunc = [&aNode, &visitor](size_t iRowInTable, size_t level) -> bool {
            return visitLeaf(iRowInTable, level, aNode, visitor);
        };

        traverseNodesBF(0, aCur, aNext, aNode, onSplitNodeFunc, onLeafNodeFunc);
    }
}

void ModelImpl::traverseDFS(size_t iTree, tree_utils::regression::TreeNodeVisitor& visitor) const
{
    if(iTree >= size())
        return;

    const GbtDecisionTree& gbtTree = *at(iTree);
    SharedPtr<DecisionTreeTable> t = gbtTreeToDecisionTree(gbtTree);
    const DecisionTreeNode* aNode = (const DecisionTreeNode*)t->getArray();

    const double *imp = getImpVals(iTree);
    const int *nodeSamplesCount = getNodeSampleCount(iTree);
    if(aNode)
    {
        tree_utils::SplitNodeDescriptor descSplit;
        tree_utils::regression::LeafNodeDescriptor descLeaf;

        auto onSplitNodeFunc = [&descSplit, &aNode, &imp, &nodeSamplesCount, &visitor](size_t iRowInTable, size_t level) -> bool {
            return visitSplit(iRowInTable, level, descSplit, aNode, imp, nodeSamplesCount, visitor);
        };

        auto onLeafNodeFunc = [&descLeaf, &aNode, &imp, &nodeSamplesCount, &visitor](size_t iRowInTable, size_t level) -> bool {
            return visitLeaf(iRowInTable, level, descLeaf, aNode, imp, nodeSamplesCount, visitor);
        };

        traverseNodeDF(0, 0, aNode, onSplitNodeFunc, onLeafNodeFunc);
    }
}

void ModelImpl::traverseBFS(size_t iTree, tree_utils::regression::TreeNodeVisitor& visitor) const
{
    if(iTree >= size())
        return;

    const GbtDecisionTree& gbtTree = *at(iTree);
    SharedPtr<DecisionTreeTable> t = gbtTreeToDecisionTree(gbtTree);
    const DecisionTreeNode* aNode = (const DecisionTreeNode*)t->getArray();

    const double *imp = getImpVals(iTree);
    const int *nodeSamplesCount = getNodeSampleCount(iTree);
    NodeIdxArray aCur;  //nodes of current layer
    NodeIdxArray aNext; //nodes of next layer
    if(aNode)
    {
        tree_utils::SplitNodeDescriptor descSplit;
        tree_utils::regression::LeafNodeDescriptor descLeaf;

        auto onSplitNodeFunc = [&descSplit, &aNode, &imp, &nodeSamplesCount, &visitor](size_t iRowInTable, size_t level) -> bool {
            return visitSplit(iRowInTable, level, descSplit, aNode, imp, nodeSamplesCount, visitor);
        };

        auto onLeafNodeFunc = [&descLeaf, &aNode, &imp, &nodeSamplesCount, &visitor](size_t iRowInTable, size_t level) -> bool {
            return visitLeaf(iRowInTable, level, descLeaf, aNode, imp, nodeSamplesCount, visitor);
        };

        aCur.push_back(0);
        traverseNodesBF(0, aCur, aNext, aNode, onSplitNodeFunc, onLeafNodeFunc);
    }
}

void ModelImpl::treeToTable(TreeType& t,
    gbt::internal::GbtDecisionTree** pTbl, HomogenNumericTable<double>** pTblImp, HomogenNumericTable<int>** pTblSmplCnt)
{
    const size_t nNode = t.getNumberOfNodes();



    t.convertGbtTreeToTable(pTbl, pTblImp, pTblSmplCnt);
}

void ModelImpl::add(gbt::internal::GbtDecisionTree* pTbl, HomogenNumericTable<double>* pTblImp, HomogenNumericTable<int>* pTblSmplCnt)
{
    DAAL_ASSERT(pTbl);
    DAAL_ASSERT(pTblImp);
    DAAL_ASSERT(pTblSmplCnt);

    _nTree.inc();

    _serializationData->push_back(SerializationIfacePtr(pTbl));
    _impurityTables->push_back(SerializationIfacePtr(pTblImp));
    _nNodeSampleTables->push_back(SerializationIfacePtr(pTblSmplCnt));
}

ModelImpl::~ModelImpl()
{
    destroy();
}

size_t ModelImpl::size() const
{
    return super::size();
}

bool ModelImpl::reserve(const size_t nTrees)
{
    return super::reserve(nTrees);
}

bool ModelImpl::resize(const size_t nTrees)
{
    return super::resize(nTrees);
}

void ModelImpl::clear()
{
    super::clear();
}

void ModelImpl::destroy()
{
    super::destroy();
}


SharedPtr<DecisionTreeTable> ModelImpl::gbtTreeToDecisionTree(const GbtDecisionTree& gbtTree)
{
    const gbt::prediction::internal::ModelFPType* splitPoints = gbtTree.getSplitPoints();
    const gbt::prediction::internal::FeatureIndexType* splitFeatures = gbtTree.getFeatureIndexesForSplit();

    SharedPtr<DecisionTreeTable> newTree(new DecisionTreeTable(gbtTree.getSourceNumOfNodes()));

    size_t idxInTable = 0;
    convertNode(gbtTree, *newTree, 0, 0, idxInTable);

    return newTree;
}

void ModelImpl::convertNode(const GbtDecisionTree& gbtTree, DecisionTreeTable& newTree, const size_t idx, const size_t lvl, size_t& idxInTable)
{
    const gbt::prediction::internal::ModelFPType* splitPoints = gbtTree.getSplitPoints();
    const gbt::prediction::internal::FeatureIndexType* splitFeatures = gbtTree.getFeatureIndexesForSplit();

    DecisionTreeNode* const arr = (DecisionTreeNode*)newTree.getArray();

    if (nodeIsLeaf(idx, gbtTree, lvl))
    {
        arr[idxInTable].featureValueOrResponse = splitPoints[idx];
        arr[idxInTable].featureIndex = -1;
        arr[idxInTable].leftIndexOrClass = 0;
        idxInTable++;
    }
    else
    {
        arr[idxInTable].featureValueOrResponse = splitPoints[idx];
        arr[idxInTable].featureIndex = splitFeatures[idx];
        arr[idxInTable].leftIndexOrClass = idxInTable + 1;
        idxInTable++;

        convertNode(gbtTree, newTree, idx*2+1, lvl + 1, idxInTable);
        convertNode(gbtTree, newTree, idx*2+2, lvl + 1, idxInTable);
    }
    return;
}

bool ModelImpl::nodeIsDummyLeaf(size_t idx, const GbtDecisionTree& gbtTree)
{
    const gbt::prediction::internal::ModelFPType* splitPoints        = gbtTree.getSplitPoints();
    const gbt::prediction::internal::FeatureIndexType* splitFeatures = gbtTree.getFeatureIndexesForSplit();

    if(idx)
    {
        const size_t parent = getIdxOfParent(idx);
        return (splitPoints[parent] == splitPoints[idx] && splitFeatures[parent] == splitFeatures[idx]);
    }
    else
    {
        return false;
    }
}

bool ModelImpl::nodeIsLeaf(size_t idx, const GbtDecisionTree& gbtTree, const size_t lvl)
{
    if (lvl == gbtTree.getMaxLvl())
    {
        return true;
    }
    else if (nodeIsDummyLeaf(2 * idx + 1, gbtTree)) // check, that left son is dummy
    {
        return true;
    }
    return false;
}

size_t ModelImpl::getIdxOfParent(const size_t sonIdx)
{
    return sonIdx ? (sonIdx - 1) / 2 : 0;
}


void ModelImpl::decisionTreeToGbtTree(const DecisionTreeTable& tree, GbtDecisionTree& newTree)
{
    const size_t nSourceNodes = tree.getNumberOfRows();
    const size_t nLvls = newTree.getMaxLvl();

    using NodeType = const dtrees::internal::DecisionTreeNode*;
    services::Collection<NodeType> sonsArr(newTree.getNumberOfNodes() + 1);
    services::Collection<NodeType> parentsArr(newTree.getNumberOfNodes() + 1);
    NodeType arr = (const NodeType)tree.getArray();

    NodeType* sons = sonsArr.data();
    NodeType* parents = parentsArr.data();

    gbt::prediction::internal::ModelFPType* const spitPoints = newTree.getSplitPoints();
    gbt::prediction::internal::FeatureIndexType* const featureIndexes = newTree.getFeatureIndexesForSplit();

    for(size_t i = 0; i < nSourceNodes; ++i)
    {
        sons[i] = nullptr;
        parents[i] = nullptr;
    }

    size_t nParents = 1;
    parents[0] = arr;
    size_t idxInTable = 0;

    for(size_t lvl = 0; lvl < nLvls + 1; ++lvl)
    {
        size_t nSons = 0;
        for(size_t iParent = 0; iParent < nParents; ++iParent)
        {
            const dtrees::internal::DecisionTreeNode* p = parents[iParent];

            if(p->isSplit())
            {
                sons[nSons++] = arr + p->leftIndexOrClass;
                sons[nSons++] = arr + p->leftIndexOrClass + 1;
                featureIndexes[idxInTable] = p->featureIndex;
                DAAL_ASSERT(featureIndexes[idxInTable] >= 0);
                spitPoints[idxInTable] = p->featureValueOrResponse;
            }
            else
            {
                sons[nSons++] = p;
                sons[nSons++] = p;
                featureIndexes[idxInTable] = 0;
                DAAL_ASSERT(featureIndexes[idxInTable] >= 0);
                spitPoints[idxInTable] = p->featureValueOrResponse;
            }

            idxInTable++;
        }
        swap(parents, sons);
        nParents = nSons;
    }
}

void ModelImpl::getMaxLvl(const dtrees::internal::DecisionTreeNode* const arr, const size_t idx, size_t& maxLvl, size_t curLvl)
{
    curLvl++;

    if(arr->isSplit())
    {
        getMaxLvl(arr, (arr+idx)->leftIndexOrClass,     maxLvl, curLvl);
        getMaxLvl(arr, (arr+idx)->leftIndexOrClass + 1, maxLvl, curLvl);
    }
    else
    {
        if (maxLvl < curLvl)
            maxLvl = curLvl;
    }
}

const GbtDecisionTree* ModelImpl::at(const size_t idx) const
{
    return (const GbtDecisionTree*)(*super::_serializationData)[idx].get();
}

} // namespace internal
} // namespace gbt
} // namespace algorithms
} // namespace daal
