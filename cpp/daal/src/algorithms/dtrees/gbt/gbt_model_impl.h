/* file: gbt_model_impl.h */
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
//  Implementation of the class defining the gradient boosted trees model
//--
*/

#ifndef __GBT_MODEL_IMPL__
#define __GBT_MODEL_IMPL__

#include "src/algorithms/dtrees/dtrees_model_impl.h"
#include "algorithms/regression/tree_traverse.h"
#include "src/algorithms/dtrees/gbt/gbt_predict_dense_default_impl.i"
#include "algorithms/tree_utils/tree_utils_regression.h"
#include "src/algorithms/dtrees/dtrees_model_impl_common.h"
#include "src/services/service_arrays.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace internal
{
static inline size_t getNumberOfNodesByLvls(const size_t nLvls)
{
    size_t nNodes = 2; // nNodes = pow(2, nLvl+1) - 1
    for (size_t i = 0; i < nLvls; ++i) nNodes *= 2;
    nNodes--;
    return nNodes;
}

template <typename T>
void swap(T & t1, T & t2)
{
    T tmp = t1;
    t1    = t2;
    t2    = tmp;
}

class GbtDecisionTree : public SerializationIface
{
public:
    DECLARE_SERIALIZABLE();
    using SplitPointType             = HomogenNumericTable<gbt::prediction::internal::ModelFPType>;
    using FeatureIndexesForSplitType = HomogenNumericTable<gbt::prediction::internal::FeatureIndexType>;
    using defaultLeftForSplitType    = HomogenNumericTable<int>;

    GbtDecisionTree(const size_t nNodes, const size_t maxLvl, const size_t sourceNumOfNodes)
        : _nNodes(nNodes),
          _maxLvl(maxLvl),
          _sourceNumOfNodes(sourceNumOfNodes),
          _splitPoints(SplitPointType::create(1, nNodes, NumericTableIface::doAllocate)),
          _featureIndexes(FeatureIndexesForSplitType::create(1, nNodes, NumericTableIface::doAllocate)),
          _defaultLeft(defaultLeftForSplitType::create(1, nNodes, NumericTableIface::doAllocate))
    {}

    // for serailization only
    GbtDecisionTree() : _nNodes(0), _maxLvl(0), _sourceNumOfNodes(0) {}

    gbt::prediction::internal::ModelFPType * getSplitPoints() { return _splitPoints->getArray(); }

    gbt::prediction::internal::FeatureIndexType * getFeatureIndexesForSplit() { return _featureIndexes->getArray(); }

    int * getdefaultLeftForSplit() { return _defaultLeft->getArray(); }

    const gbt::prediction::internal::ModelFPType * getSplitPoints() const { return _splitPoints->getArray(); }

    const gbt::prediction::internal::FeatureIndexType * getFeatureIndexesForSplit() const { return _featureIndexes->getArray(); }

    const int * getdefaultLeftForSplit() const { return _defaultLeft->getArray(); }

    size_t getNumberOfNodes() const { return _nNodes; }

    size_t * getArrayNumSplitFeature() { return nNodeSplitFeature.data(); }

    size_t * getArrayCoverFeature() { return CoverFeature.data(); }

    double * getArrayGainFeature() { return GainFeature.data(); }

    gbt::prediction::internal::FeatureIndexType getMaxLvl() const { return _maxLvl; }

    // recursive build of tree (breadth-first)
    template <typename NodeType, typename NodeBase>
    static services::Status internalTreeToGbtDecisionTree(const NodeBase & root, const size_t nNodes, const size_t nLvls, GbtDecisionTree * tree,
                                                          double * impVals, int * nNodeSamplesVals, size_t countFeature)
    {
        using SplitType = const typename NodeType::Split *;
        services::Collection<SplitType> sonsArr(nNodes + 1);
        services::Collection<SplitType> parentsArr(nNodes + 1);

        SplitType * sons    = sonsArr.data();
        SplitType * parents = parentsArr.data();

        int result = 0;

        gbt::prediction::internal::ModelFPType * const spitPoints          = tree->getSplitPoints();
        gbt::prediction::internal::FeatureIndexType * const featureIndexes = tree->getFeatureIndexesForSplit();

        for (size_t i = 0; i < nNodes; ++i)
        {
            sons[i]    = nullptr;
            parents[i] = nullptr;
        }

        tree->nNodeSplitFeature.resize(countFeature);
        tree->CoverFeature.resize(countFeature);
        tree->GainFeature.resize(countFeature);

        for (size_t i = 0; i < countFeature; ++i)
        {
            tree->nNodeSplitFeature[i] = 0;
            tree->CoverFeature[i]      = 0;
            tree->GainFeature[i]       = 0;
        }

        size_t nParents   = 1;
        parents[0]        = NodeType::castSplit(&root);
        size_t idxInTable = 0;

        for (size_t lvl = 0; lvl < nLvls + 1; ++lvl)
        {
            size_t nSons = 0;
            for (size_t iParent = 0; iParent < nParents; ++iParent)
            {
                const typename NodeType::Split * p = parents[iParent];

                if (p->isSplit())
                {
                    tree->nNodeSplitFeature[p->featureIdx] += 1;
                    tree->CoverFeature[p->featureIdx] += p->count;
                    tree->GainFeature[p->featureIdx] -= p->impurity - p->left()->impurity - p->right()->impurity;

                    sons[nSons++]              = NodeType::castSplit(p->left());
                    sons[nSons++]              = NodeType::castSplit(p->right());
                    featureIndexes[idxInTable] = p->featureIdx;
                }
                else
                {
                    sons[nSons++]              = p;
                    sons[nSons++]              = p;
                    featureIndexes[idxInTable] = 0;
                }
                DAAL_ASSERT(featureIndexes[idxInTable] >= 0);
                nNodeSamplesVals[idxInTable] = (int)p->count;
                impVals[idxInTable]          = p->impurity;
                spitPoints[idxInTable]       = p->featureValue;

                idxInTable++;
            }

            const size_t size = nSons * sizeof(SplitType);
            result |= daal::services::internal::daal_memcpy_s(parents, size, sons, size);

            nParents = nSons;
        }

        return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
    }

protected:
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        arch->set(_nNodes);
        arch->set(_maxLvl);
        arch->set(_sourceNumOfNodes);

        arch->setSharedPtrObj(_splitPoints);
        arch->setSharedPtrObj(_featureIndexes);
        arch->setSharedPtrObj(_defaultLeft);

        return services::Status();
    }

protected:
    size_t _nNodes;
    gbt::prediction::internal::FeatureIndexType _maxLvl;
    size_t _sourceNumOfNodes;
    services::SharedPtr<SplitPointType> _splitPoints;
    services::SharedPtr<FeatureIndexesForSplitType> _featureIndexes;
    services::SharedPtr<defaultLeftForSplitType> _defaultLeft;
    services::Collection<size_t> nNodeSplitFeature;
    services::Collection<size_t> CoverFeature;
    services::Collection<double> GainFeature;
};

template <typename TNodeType, typename TAllocator = dtrees::internal::ChunkAllocator<TNodeType> >
class GbtTreeImpl : public dtrees::internal::TreeImpl<TNodeType, TAllocator>
{
private:
    typedef dtrees::internal::TreeImpl<TNodeType, TAllocator> super;

public:
    typedef TAllocator Allocator;
    typedef TNodeType NodeType;

    services::Status convertGbtTreeToTable(GbtDecisionTree ** pTbl, HomogenNumericTable<double> ** pTblImp, HomogenNumericTable<int> ** pTblSmplCnt,
                                           size_t nFeature) const
    {
        size_t nLvls = 1;
        services::Status status;
        getMaxLvl(*super::top(), nLvls, static_cast<size_t>(-1));
        const size_t nNodes = getNumberOfNodesByLvls(nLvls);

        *pTbl        = new GbtDecisionTree(nNodes, nLvls, super::top()->numChildren() + 1);
        *pTblImp     = new HomogenNumericTable<double>(1, nNodes, NumericTable::doAllocate);
        *pTblSmplCnt = new HomogenNumericTable<int>(1, nNodes, NumericTable::doAllocate);

        if (!(*pTbl) || !(*pTblImp) || !(*pTblSmplCnt))
        {
            status = services::Status(services::ErrorMemoryAllocationFailed);
        }

        if (super::top())
        {
            status |= GbtDecisionTree::internalTreeToGbtDecisionTree<TNodeType, typename TNodeType::Base>(
                *super::top(), nNodes, nLvls, *pTbl, (*pTblImp)->getArray(), (*pTblSmplCnt)->getArray(), nFeature);
        }

        return status;
    }

protected:
    void getMaxLvl(const typename TNodeType::Base & node, size_t & maxLvl, size_t curLvl = 0) const
    {
        curLvl++;
        const auto p = TNodeType::castSplit(&node);

        if (p->isSplit())
        {
            getMaxLvl(*static_cast<const typename NodeType::Split *>(p->left()), maxLvl, curLvl);
            getMaxLvl(*static_cast<const typename NodeType::Split *>(p->right()), maxLvl, curLvl);
        }
        else
        {
            if (maxLvl < curLvl) maxLvl = curLvl;
        }
    }
};

template <typename Allocator = dtrees::internal::ChunkAllocator<dtrees::internal::TreeNodeRegression<RegressionFPType> > >
using TreeImpRegression = GbtTreeImpl<dtrees::internal::TreeNodeRegression<RegressionFPType>, Allocator>;

template <typename Allocator = dtrees::internal::ChunkAllocator<dtrees::internal::TreeNodeClassification<ClassificationFPType> > >
using TreeImpClassification = GbtTreeImpl<dtrees::internal::TreeNodeClassification<ClassificationFPType>, Allocator>;

class ModelImpl : protected dtrees::internal::ModelImpl
{
public:
    using ImplType = dtrees::internal::ModelImpl;
    using TreeType = gbt::internal::TreeImpRegression<>;
    using super    = dtrees::internal::ModelImpl;

    ~ModelImpl() DAAL_C11_OVERRIDE;
    size_t size() const;
    bool reserve(const size_t nTrees);
    bool resize(const size_t nTrees);
    void clear();

    const GbtDecisionTree * at(const size_t idx) const;

    static void decisionTreeToGbtTree(const DecisionTreeTable & tree, GbtDecisionTree & gbtTree);
    static services::Status convertDecisionTreesToGbtTrees(data_management::DataCollectionPtr & serializationData);

    // Methods common for regression or classification model, not virtual!!!
    size_t numberOfTrees() const;
    void traverseDF(size_t iTree, algorithms::regression::TreeNodeVisitor & visitor) const;
    void traverseBF(size_t iTree, algorithms::regression::TreeNodeVisitor & visitor) const;
    void add(gbt::internal::GbtDecisionTree * pTbl, HomogenNumericTable<double> * pTblImp, HomogenNumericTable<int> * pTblSmplCnt);
    void traverseDFS(size_t iTree, tree_utils::regression::TreeNodeVisitor & visitor) const;
    void traverseBFS(size_t iTree, tree_utils::regression::TreeNodeVisitor & visitor) const;
    static services::Status treeToTable(TreeType & t, gbt::internal::GbtDecisionTree ** pTbl, HomogenNumericTable<double> ** pTblImp,
                                        HomogenNumericTable<int> ** pTblSmplCnt, size_t nFeature);

protected:
    static bool nodeIsDummyLeaf(size_t idx, const GbtDecisionTree & gbtTree);
    static bool nodeIsLeaf(size_t idx, const GbtDecisionTree & gbtTree, const size_t lvl);
    static size_t getIdxOfParent(const size_t sonIdx);
    static void getMaxLvl(const dtrees::internal::DecisionTreeNode * const arr, const size_t idx, size_t & maxLvl, size_t curLvl = 0);

    static GbtDecisionTree * allocateGbtTree(const DecisionTreeTable & tree)
    {
        const dtrees::internal::DecisionTreeNode * const arr = (const dtrees::internal::DecisionTreeNode *)tree.getArray();

        size_t nLvls = 1;
        getMaxLvl(arr, 0, nLvls, static_cast<size_t>(-1));
        const size_t nNodes = getNumberOfNodesByLvls(nLvls);

        return new GbtDecisionTree(nNodes, nLvls, tree.getNumberOfRows());
    }

    template <typename OnSplitFunctor, typename OnLeafFunctor>
    static void traverseGbtDF(size_t level, size_t iRowInTable, const GbtDecisionTree & gbtTree, OnSplitFunctor & visitSplit,
                              OnLeafFunctor & visitLeaf)
    {
        if (!nodeIsLeaf(iRowInTable, gbtTree, level))
        {
            if (!visitSplit(iRowInTable, level)) return; //do not continue traversing

            traverseGbtDF(level + 1, iRowInTable * 2 + 1, gbtTree, visitSplit, visitLeaf);
            traverseGbtDF(level + 1, iRowInTable * 2 + 2, gbtTree, visitSplit, visitLeaf);
        }
        else if (!nodeIsDummyLeaf(iRowInTable, gbtTree))
        {
            if (!visitLeaf(iRowInTable, level)) return; //do not continue traversing
        }
    }

    template <typename OnSplitFunctor, typename OnLeafFunctor>
    static void traverseGbtBF(size_t level, NodeIdxArray & aCur, NodeIdxArray & aNext, const GbtDecisionTree & gbtTree, OnSplitFunctor & visitSplit,
                              OnLeafFunctor & visitLeaf)
    {
        for (size_t i = 0; i < aCur.size(); ++i)
        {
            for (size_t j = 0; j < (level ? 2 : 1); ++j)
            {
                size_t iRowInTable = aCur[i] + j;
                if (!nodeIsLeaf(iRowInTable, gbtTree, level))
                {
                    if (!visitSplit(iRowInTable, level)) return; //do not continue traversing

                    aNext.push_back(iRowInTable * 2 + 1);
                }
                else if (!nodeIsDummyLeaf(iRowInTable, gbtTree))
                {
                    if (!visitLeaf(iRowInTable, level)) return; //do not continue traversing
                }
            }
        }
        aCur.clear();
        if (!aNext.size()) return; //done
        traverseGbtBF(level + 1, aNext, aCur, gbtTree, visitSplit, visitLeaf);
    }

    void destroy();

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch, int daalVersion = INTEL_DAAL_VERSION)
    {
        if ((daalVersion >= COMPUTE_DAAL_VERSION(2019, 0, 0)))
        {
            arch->setSharedPtrObj(_serializationData);
            arch->setSharedPtrObj(_impurityTables);
            arch->setSharedPtrObj(_nNodeSampleTables);
        }
        else
        {
            arch->setSharedPtrObj(_serializationData);
            convertDecisionTreesToGbtTrees(_serializationData);
        }

        if (onDeserialize) _nTree.set(_serializationData->size());

        return services::Status();
    }
};

} // namespace internal
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif
