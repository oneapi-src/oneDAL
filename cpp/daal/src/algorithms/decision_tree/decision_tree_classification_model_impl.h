/* file: decision_tree_classification_model_impl.h */
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

#ifndef __DECISION_TREE_CLASSIFICATION_MODEL_IMPL_
#define __DECISION_TREE_CLASSIFICATION_MODEL_IMPL_

#include "algorithms/decision_tree/decision_tree_classification_model.h"
#include "src/algorithms/classifier/classifier_model_impl.h"
#include "src/services/service_defines.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace classification
{
struct DecisionTreeNode
{
    size_t dimension;
    size_t leftIndexOrClass;
    double cutPoint;
};

class DecisionTreeTable : public data_management::AOSNumericTable
{
public:
    DecisionTreeTable(size_t rowCount, services::Status & st) : data_management::AOSNumericTable(sizeof(DecisionTreeNode), 3, rowCount, st)
    {
        setFeature<size_t>(0, DAAL_STRUCT_MEMBER_OFFSET(DecisionTreeNode, dimension));
        setFeature<size_t>(1, DAAL_STRUCT_MEMBER_OFFSET(DecisionTreeNode, leftIndexOrClass));
        setFeature<double>(2, DAAL_STRUCT_MEMBER_OFFSET(DecisionTreeNode, cutPoint));
        st |= allocateDataMemory();
    }
    DecisionTreeTable(services::Status & st) : DecisionTreeTable(0, st) {}
};

typedef services::SharedPtr<DecisionTreeTable> DecisionTreeTablePtr;
typedef services::SharedPtr<const DecisionTreeTable> DecisionTreeTableConstPtr;

class Model::ModelImpl : public classifier::internal::ModelImpl
{
    typedef classifier::internal::ModelImpl super;
    typedef services::Collection<size_t> NodeIdxArray;

    static bool visitSplit(size_t iRowInTable, size_t level, const DecisionTreeNode * aNode, classifier::TreeNodeVisitor & visitor)
    {
        const DecisionTreeNode & n = aNode[iRowInTable];
        return visitor.onSplitNode(level, n.dimension, n.cutPoint);
    }

    static bool visitLeaf(size_t iRowInTable, size_t level, const DecisionTreeNode * aNode, classifier::TreeNodeVisitor & visitor)
    {
        const DecisionTreeNode & n = aNode[iRowInTable];
        return visitor.onLeafNode(level, n.leftIndexOrClass);
    }

    template <typename InternalSplitNodeDescriptor, typename InternalTreeNodeVisitor>
    static bool visitSplit(size_t iRowInTable, size_t level, InternalSplitNodeDescriptor & descSplit, const DecisionTreeNode * aNode,
                           const double * imp, const int * nodeSamplesCount, InternalTreeNodeVisitor & visitor)
    {
        const DecisionTreeNode & n = aNode[iRowInTable];
        if (imp) descSplit.impurity = imp[iRowInTable];
        if (nodeSamplesCount)
        {
            DAAL_ASSERT(nodeSamplesCount[iRowInTable] >= 0)
            descSplit.nNodeSampleCount = (size_t)(nodeSamplesCount[iRowInTable]);
        }
        descSplit.featureIndex = n.dimension;
        descSplit.featureValue = n.cutPoint;
        descSplit.level        = level;
        return visitor.onSplitNode(descSplit);
    }

    template <typename InternalLeafNodeDescriptor, typename InternalTreeNodeVisitor>
    static bool visitLeaf(size_t iRowInTable, size_t level, InternalLeafNodeDescriptor & descLeaf, const DecisionTreeNode * aNode, const double * imp,
                          const int * nodeSamplesCount, InternalTreeNodeVisitor & visitor)
    {
        const DecisionTreeNode & n = aNode[iRowInTable];
        if (imp) descLeaf.impurity = imp[iRowInTable];
        if (nodeSamplesCount)
        {
            DAAL_ASSERT(nodeSamplesCount[iRowInTable] >= 0)
            descLeaf.nNodeSampleCount = (size_t)(nodeSamplesCount[iRowInTable]);
        }
        descLeaf.level = level;
        descLeaf.label = n.leftIndexOrClass;
        return visitor.onLeafNode(descLeaf);
    }

public:
    /**
     * Empty constructor for deserialization
     */
    ModelImpl(size_t nFeatures = 0) : super(nFeatures), _TreeTable() {}

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
     * Sets a decision tree table
     * \param[in]  value  decision tree table
     */
    void setTreeTable(const DecisionTreeTablePtr & value) { _TreeTable = value; }

    void setImpTable(const services::SharedPtr<data_management::HomogenNumericTable<double> > & value) { _impTable = value; }

    void setNodeSmplCntTable(const services::SharedPtr<data_management::HomogenNumericTable<int> > & value) { _nodeSampleCountTable = value; }

    void setProbTable(const services::SharedPtr<data_management::HomogenNumericTable<double> > & value) { _probTable = value; }

    void setProbIndicesTable(const services::SharedPtr<data_management::HomogenNumericTable<int> > & value) { _probIndicesTable = value; }

    const double * getImpVals() const { return _impTable ? _impTable->getArray() : nullptr; }

    const int * getNodeSampleCount() const { return _nodeSampleCountTable ? _nodeSampleCountTable->getArray() : nullptr; }

    const double * getProbabilities(size_t index) const
    {
        if (_probTable)
        {
            DAAL_ASSERT(_probIndicesTable);
            DAAL_ASSERT(index < _probIndicesTable->getNumberOfRows());
            const int probTableIndex = *(_probIndicesTable->getArray() + index);
            DAAL_ASSERT(probTableIndex < _probTable->getNumberOfRows());
            // operator[] of homogen numeric table does not have const version
            return _probTable->getArray() + probTableIndex * _probTable->getNumberOfColumns();
        }
        else
        {
            return nullptr;
        }
    }

    void traverseDF(classifier::TreeNodeVisitor & visitor) const
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

    void traverseBF(classifier::TreeNodeVisitor & visitor) const
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

    template <typename InternalLeafNodeDescriptor, typename InternalTreeNodeVisitor>
    void traverseDFS(InternalTreeNodeVisitor & visitor) const
    {
        const DecisionTreeNode * aNode = (const DecisionTreeNode *)_TreeTable->getArray();
        const double * imp             = getImpVals();
        const int * nodeSamplesCount   = getNodeSampleCount();
        if (aNode)
        {
            tree_utils::SplitNodeDescriptor descSplit;
            InternalLeafNodeDescriptor descLeaf;

            auto onSplitNodeFunc = [&descSplit, &aNode, &imp, &nodeSamplesCount, &visitor](size_t iRowInTable, size_t level) -> bool {
                return visitSplit(iRowInTable, level, descSplit, aNode, imp, nodeSamplesCount, visitor);
            };

            auto onLeafNodeFunc = [&descLeaf, &aNode, &imp, &nodeSamplesCount, &visitor](size_t iRowInTable, size_t level) -> bool {
                return visitLeaf(iRowInTable, level, descLeaf, aNode, imp, nodeSamplesCount, visitor);
            };

            traverseNodesDF(0, 0, aNode, onSplitNodeFunc, onLeafNodeFunc);
        }
    }

    template <typename InternalLeafNodeDescriptor, typename InternalTreeNodeVisitor>
    void traverseBFS(InternalTreeNodeVisitor & visitor) const
    {
        const DecisionTreeNode * aNode = (const DecisionTreeNode *)_TreeTable->getArray();
        const double * imp             = getImpVals();
        const int * nodeSamplesCount   = getNodeSampleCount();
        NodeIdxArray aCur;  //nodes of current layer
        NodeIdxArray aNext; //nodes of next layer
        if (aNode)
        {
            tree_utils::SplitNodeDescriptor descSplit;
            InternalLeafNodeDescriptor descLeaf;

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
        super::serialImpl<Archive, onDeserialize>(arch);
        arch->setSharedPtrObj(_TreeTable);

        if ((daalVersion >= COMPUTE_DAAL_VERSION(2019, 0, 0)))
        {
            arch->setSharedPtrObj(_impTable);
            arch->setSharedPtrObj(_nodeSampleCountTable);

            if (daalVersion >= COMPUTE_DAAL_VERSION(2019, 0, 3))
            {
                arch->setSharedPtrObj(_probIndicesTable);
                arch->setSharedPtrObj(_probTable);
            }
        }

        return services::Status();
    }

private:
    DecisionTreeTablePtr _TreeTable;
    services::SharedPtr<data_management::HomogenNumericTable<double> > _impTable;
    services::SharedPtr<data_management::HomogenNumericTable<int> > _nodeSampleCountTable;
    services::SharedPtr<data_management::HomogenNumericTable<int> > _probIndicesTable;
    services::SharedPtr<data_management::HomogenNumericTable<double> > _probTable;

    template <typename OnSplitFunctor, typename OnLeafFunctor>
    bool traverseNodesDF(size_t level, size_t iRowInTable, const DecisionTreeNode * aNode, OnSplitFunctor & visitSplit,
                         OnLeafFunctor & visitLeaf) const
    {
        const DecisionTreeNode & n = aNode[iRowInTable];
        if (n.dimension != static_cast<size_t>(-1))
        {
            DAAL_CHECK_STATUS_VAR((visitSplit(iRowInTable, level)));
            ++level;
            size_t leftIdx  = n.leftIndexOrClass;
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
                    aNext.push_back(n.leftIndexOrClass);
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

} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
