/* file: decision_tree_regression_model_impl.h */
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
//  Implementation of the class defining the Decision tree model
//--
*/

#ifndef __DECISION_TREE_REGRESSION_MODEL_IMPL_
#define __DECISION_TREE_REGRESSION_MODEL_IMPL_

#include "algorithms/decision_tree/decision_tree_regression_model.h"
#include "src/algorithms/regression/regression_model_impl.h"
#include "src/services/service_defines.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace regression
{
struct DecisionTreeNode
{
    size_t dimension;
    size_t leftIndex;
    double cutPointOrDependantVariable;
};

class DecisionTreeTable : public data_management::AOSNumericTable
{
public:
    DecisionTreeTable(size_t rowCount, services::Status & st) : data_management::AOSNumericTable(sizeof(DecisionTreeNode), 3, rowCount, st)
    {
        setFeature<size_t>(0, DAAL_STRUCT_MEMBER_OFFSET(DecisionTreeNode, dimension));
        setFeature<size_t>(1, DAAL_STRUCT_MEMBER_OFFSET(DecisionTreeNode, leftIndex));
        setFeature<double>(2, DAAL_STRUCT_MEMBER_OFFSET(DecisionTreeNode, cutPointOrDependantVariable));
        st |= allocateDataMemory();
    }
    DecisionTreeTable(services::Status & st) : DecisionTreeTable(0, st) {}
};

typedef services::SharedPtr<DecisionTreeTable> DecisionTreeTablePtr;
typedef services::SharedPtr<const DecisionTreeTable> DecisionTreeTableConstPtr;

class Model::ModelImpl : public algorithms::regression::internal::ModelImpl
{
    typedef services::Collection<size_t> NodeIdxArray;

    static bool visitSplit(size_t iRowInTable, size_t level, const DecisionTreeNode * aNode, algorithms::regression::TreeNodeVisitor & visitor)
    {
        const DecisionTreeNode & n = aNode[iRowInTable];
        return visitor.onSplitNode(level, n.dimension, n.cutPointOrDependantVariable);
    }

    static bool visitLeaf(size_t iRowInTable, size_t level, const DecisionTreeNode * aNode, algorithms::regression::TreeNodeVisitor & visitor)
    {
        const DecisionTreeNode & n = aNode[iRowInTable];
        return visitor.onLeafNode(level, n.cutPointOrDependantVariable);
    }

    static bool visitSplit(size_t iRowInTable, size_t level, tree_utils::SplitNodeDescriptor & descSplit, const DecisionTreeNode * aNode,
                           const double * imp, const int * nodeSamplesCount, tree_utils::regression::TreeNodeVisitor & visitor)
    {
        const DecisionTreeNode & n = aNode[iRowInTable];
        if (imp) descSplit.impurity = imp[iRowInTable];
        if (nodeSamplesCount)
        {
            DAAL_ASSERT(nodeSamplesCount[iRowInTable] >= 0)
            descSplit.nNodeSampleCount = (size_t)(nodeSamplesCount[iRowInTable]);
        }
        descSplit.featureIndex = n.dimension;
        descSplit.featureValue = n.cutPointOrDependantVariable;
        descSplit.level        = level;
        return visitor.onSplitNode(descSplit);
    }

    static bool visitLeaf(size_t iRowInTable, size_t level, tree_utils::regression::LeafNodeDescriptor & descLeaf, const DecisionTreeNode * aNode,
                          const double * imp, const int * nodeSamplesCount, tree_utils::regression::TreeNodeVisitor & visitor)
    {
        const DecisionTreeNode & n = aNode[iRowInTable];
        if (imp) descLeaf.impurity = imp[iRowInTable];
        if (nodeSamplesCount)
        {
            DAAL_ASSERT(nodeSamplesCount[iRowInTable] >= 0)
            descLeaf.nNodeSampleCount = (size_t)(nodeSamplesCount[iRowInTable]);
        }
        descLeaf.level    = level;
        descLeaf.response = n.cutPointOrDependantVariable;
        return visitor.onLeafNode(descLeaf);
    }

public:
    /**
     * Constructs decision tree model with the specified number of features
     * \param[in] featureCount Number of features
     */
    ModelImpl() : _TreeTable() {}

    /**
     * Returns the decision tree table
     * \return decision tree table
     */
    DecisionTreeTablePtr getTreeTable() { return _TreeTable; }

    /**
     * Returns the decision tree table
     * \return decision tree table
     */
    DecisionTreeTableConstPtr getTreeTable() const { return _TreeTable; }

    /**
     *  Sets a decision tree table
     *  \param[in]  value  decision tree table
     */
    void setTreeTable(const DecisionTreeTablePtr & value) { _TreeTable = value; }

    void setImpTable(const services::SharedPtr<data_management::HomogenNumericTable<double> > & value) { _impTable = value; }

    void setNodeSmplCntTable(const services::SharedPtr<data_management::HomogenNumericTable<int> > & value) { _nodeSampleCountTable = value; }

    const double * getImpVals() const { return _impTable ? _impTable->getArray() : nullptr; }

    const int * getNodeSampleCount() const { return _nodeSampleCountTable ? _nodeSampleCountTable->getArray() : nullptr; }

    void traverseDF(algorithms::regression::TreeNodeVisitor & visitor) const
    {
        const DecisionTreeNode * aNode = (const DecisionTreeNode *)_TreeTable->getArray();
        if (aNode)
        {
            auto onSplitNodeFunc = [&aNode, &visitor](size_t iRowInTable, size_t level) -> bool {
                return visitSplit(iRowInTable, level, aNode, visitor);
            };

            auto onLeafNodeFunc = [&aNode, &visitor](size_t iRowInTable, size_t level) -> bool {
                return visitLeaf(iRowInTable, level, aNode, visitor);
            };

            traverseNodesDF(0, 0, aNode, onSplitNodeFunc, onLeafNodeFunc);
        }
    }

    void traverseBF(algorithms::regression::TreeNodeVisitor & visitor) const
    {
        const DecisionTreeNode * aNode = (const DecisionTreeNode *)_TreeTable->getArray();
        NodeIdxArray aCur;  //nodes of current layer
        NodeIdxArray aNext; //nodes of next layer
        if (aNode)
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

    void traverseDFS(tree_utils::regression::TreeNodeVisitor & visitor) const
    {
        const DecisionTreeNode * aNode = (const DecisionTreeNode *)_TreeTable->getArray();
        const double * imp             = getImpVals();
        const int * nodeSamplesCount   = getNodeSampleCount();
        if (aNode)
        {
            tree_utils::SplitNodeDescriptor descSplit;
            tree_utils::regression::LeafNodeDescriptor descLeaf;

            auto onSplitNodeFunc = [&descSplit, &aNode, &imp, &nodeSamplesCount, &visitor](size_t iRowInTable, size_t level) -> bool {
                return visitSplit(iRowInTable, level, descSplit, aNode, imp, nodeSamplesCount, visitor);
            };

            auto onLeafNodeFunc = [&descLeaf, &aNode, &imp, &nodeSamplesCount, &visitor](size_t iRowInTable, size_t level) -> bool {
                return visitLeaf(iRowInTable, level, descLeaf, aNode, imp, nodeSamplesCount, visitor);
            };

            traverseNodesDF(0, 0, aNode, onSplitNodeFunc, onLeafNodeFunc);
        }
    }

    void traverseBFS(tree_utils::regression::TreeNodeVisitor & visitor) const
    {
        const DecisionTreeNode * aNode = (const DecisionTreeNode *)_TreeTable->getArray();
        const double * imp             = getImpVals();
        const int * nodeSamplesCount   = getNodeSampleCount();
        NodeIdxArray aCur;  //nodes of current layer
        NodeIdxArray aNext; //nodes of next layer
        if (aNode)
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

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch, int daalVersion = INTEL_DAAL_VERSION)
    {
        algorithms::regression::internal::ModelImpl::serialImpl<Archive, onDeserialize>(arch);
        arch->setSharedPtrObj(_TreeTable);

        if ((daalVersion >= COMPUTE_DAAL_VERSION(2019, 0, 0)))
        {
            arch->setSharedPtrObj(_impTable);
            arch->setSharedPtrObj(_nodeSampleCountTable);
        }

        return services::Status();
    }

private:
    DecisionTreeTablePtr _TreeTable;
    services::SharedPtr<data_management::HomogenNumericTable<double> > _impTable;
    services::SharedPtr<data_management::HomogenNumericTable<int> > _nodeSampleCountTable;

    template <typename OnSplitFunctor, typename OnLeafFunctor>
    bool traverseNodesDF(size_t level, size_t iRowInTable, const DecisionTreeNode * aNode, OnSplitFunctor & visitSplit,
                         OnLeafFunctor & visitLeaf) const
    {
        const DecisionTreeNode & n = aNode[iRowInTable];
        if (n.dimension != static_cast<size_t>(-1))
        {
            DAAL_CHECK_STATUS_VAR((visitSplit(iRowInTable, level)));
            ++level;
            size_t leftIdx  = n.leftIndex;
            size_t rightIdx = leftIdx + 1;
            DAAL_CHECK_STATUS_VAR((traverseNodesDF(level, leftIdx, aNode, visitSplit, visitLeaf)));
            return traverseNodesDF(level, rightIdx, aNode, visitSplit, visitLeaf);
        }
        return visitLeaf(iRowInTable, level);
    }

    template <typename OnSplitFunctor, typename OnLeafFunctor>
    bool traverseNodesBF(size_t level, NodeIdxArray & aCur, NodeIdxArray & aNext, const DecisionTreeNode * aNode, OnSplitFunctor & visitSplit,
                         OnLeafFunctor & visitLeaf) const
    {
        for (size_t i = 0; i < aCur.size(); ++i)
        {
            for (size_t j = 0; j < (level ? 2 : 1); ++j)
            {
                size_t iRowInTable         = aCur[i] + j;
                const DecisionTreeNode & n = aNode[iRowInTable];
                if (n.dimension != static_cast<size_t>(-1))
                {
                    DAAL_CHECK_STATUS_VAR((visitSplit(iRowInTable, level)));
                    aNext.push_back(n.leftIndex);
                }
                else
                {
                    DAAL_CHECK_STATUS_VAR((visitLeaf(iRowInTable, level)));
                }
            }
        }
        aCur.clear();
        if (!aNext.size()) return true; //done
        return traverseNodesBF(level + 1, aNext, aCur, aNode, visitSplit, visitLeaf);
    }
};

} // namespace regression
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
