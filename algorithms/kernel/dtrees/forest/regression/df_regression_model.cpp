/* file: df_regression_model.cpp */
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
//  Implementation of the class defining the decision forest model
//--
*/

#include "algorithms/decision_forest/decision_forest_regression_model.h"
#include "serialization_utils.h"
#include "df_regression_model_impl.h"
#include "dtrees_model_impl_common.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::algorithms::dtrees::internal;

namespace daal
{
namespace algorithms
{

typedef decision_forest::regression::internal::ModelImpl::TreeType::NodeType::Leaf TLeaf;

namespace decision_forest
{

namespace regression
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS2(Model, internal::ModelImpl, SERIALIZATION_DECISION_FOREST_REGRESSION_MODEL_ID);

Model::Model(){}

}

namespace internal
{

size_t ModelImpl::numberOfTrees() const
{
    return ImplType::size();
}

void ModelImpl::traverseDF(size_t iTree, algorithms::regression::TreeNodeVisitor& visitor) const
{
    if(iTree >= size())
        return;
    const DecisionTreeTable& t = *at(iTree);
    const DecisionTreeNode* aNode = (const DecisionTreeNode*)t.getArray();
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
    const DecisionTreeTable& t = *at(iTree);
    const DecisionTreeNode* aNode = (const DecisionTreeNode*)t.getArray();
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
    const DecisionTreeTable& t = *at(iTree);
    const DecisionTreeNode* aNode = (const DecisionTreeNode*)t.getArray();
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
    const DecisionTreeTable& t = *at(iTree);
    const DecisionTreeNode* aNode = (const DecisionTreeNode*)t.getArray();
    const double *imp = getImpVals(iTree);
    const int *nodeSamplesCount = getNodeSampleCount(iTree);
    NodeIdxArray aCur;//nodes of current layer
    NodeIdxArray aNext;//nodes of next layer
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

services::Status ModelImpl::serializeImpl(data_management::InputDataArchive  * arch)
{
    auto s = RegressionImplType::serialImpl<data_management::InputDataArchive, false>(arch);
    return s.add(ImplType::serialImpl<data_management::InputDataArchive, false>(arch));
}

services::Status ModelImpl::deserializeImpl(const data_management::OutputDataArchive * arch)
{
    auto s = RegressionImplType::serialImpl<const data_management::OutputDataArchive, true>(arch);
    return s.add(ImplType::serialImpl<const data_management::OutputDataArchive, true>(arch));
}

bool ModelImpl::add(const TreeType& tree)
{
    if(size() >= _serializationData->size())
        return false;
    size_t i = _nTree.inc();
    const size_t nNode = tree.getNumberOfNodes();

    Status st;

    auto pTbl           = new DecisionTreeTable(nNode);
    auto impTbl         = new HomogenNumericTable<double>(1, nNode, NumericTable::doAllocate);
    auto nodeSamplesTbl = new HomogenNumericTable<int>(1, nNode, NumericTable::doAllocate);

    tree.convertToTable(pTbl, impTbl, nodeSamplesTbl);

    (*_serializationData)[i - 1].reset(pTbl);
    (*_impurityTables)[i - 1].reset(impTbl);
    (*_nNodeSampleTables)[i - 1].reset(nodeSamplesTbl);

    return true;
}

} // namespace interface1
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
