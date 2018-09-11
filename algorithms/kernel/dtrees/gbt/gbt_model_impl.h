/* file: gbt_model_impl.h */
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

#ifndef __GBT_MODEL_IMPL__
#define __GBT_MODEL_IMPL__

#include "dtrees_model_impl.h"
#include "algorithms/regression/tree_traverse.h"
#include "gbt_predict_dense_default_impl.i"
#include "algorithms/tree_utils/tree_utils_regression.h"
#include "dtrees_model_impl_common.h"
#include "service_arrays.h"

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
    for(size_t i = 0; i < nLvls; ++i) nNodes *= 2;
    nNodes--;
    return nNodes;
}

template<typename T>
void swap(T& t1, T& t2)
{
    T tmp = t1;
    t1 = t2;
    t2 = tmp;
}

class GbtDecisionTree: public SerializationIface
{
public:
    DECLARE_SERIALIZABLE();
    using SplitPointType = HomogenNumericTable<gbt::prediction::internal::ModelFPType>;
    using FeatureIndexesForSplitType = HomogenNumericTable<gbt::prediction::internal::FeatureIndexType>;

    GbtDecisionTree(const size_t nNodes, const size_t maxLvl, const size_t sourceNumOfNodes):
        _nNodes(nNodes), _maxLvl(maxLvl), _sourceNumOfNodes(sourceNumOfNodes),
        _splitPoints(SplitPointType::create(1, nNodes, NumericTableIface::doAllocate)),
        _featureIndexes(FeatureIndexesForSplitType::create(1, nNodes, NumericTableIface::doAllocate))
    {
    }

    // for serailization only
    GbtDecisionTree(): _nNodes(0), _maxLvl(0), _sourceNumOfNodes(0)
    {
    }

    gbt::prediction::internal::ModelFPType* getSplitPoints()
    {
        return _splitPoints->getArray();
    }

    gbt::prediction::internal::FeatureIndexType* getFeatureIndexesForSplit()
    {
        return _featureIndexes->getArray();
    }

    const gbt::prediction::internal::ModelFPType* getSplitPoints() const
    {
        return _splitPoints->getArray();
    }

    const gbt::prediction::internal::FeatureIndexType* getFeatureIndexesForSplit() const
    {
        return _featureIndexes->getArray();
    }

    size_t getNumberOfNodes() const
    {
        return _nNodes;
    }

    gbt::prediction::internal::FeatureIndexType getMaxLvl() const
    {
        return _maxLvl;
    }

    size_t getSourceNumOfNodes() const
    {
        return _sourceNumOfNodes;
    }

    // recursive build of tree (breadth-first)
    template <typename NodeType, typename NodeBase>
    static void internalTreeToGbtDecisionTree(const NodeBase& root, const size_t nNodes, const size_t nLvls, GbtDecisionTree* tree, double* impVals, int* nNodeSamplesVals)
    {
        using SplitType = const typename NodeType::Split*;
        services::Collection<SplitType> sonsArr(nNodes + 1);
        services::Collection<SplitType> parentsArr(nNodes + 1);

        SplitType* sons = sonsArr.data();
        SplitType* parents = parentsArr.data();

        gbt::prediction::internal::ModelFPType* const spitPoints = tree->getSplitPoints();
        gbt::prediction::internal::FeatureIndexType* const featureIndexes = tree->getFeatureIndexesForSplit();

        for(size_t i = 0; i < nNodes; ++i)
        {
            sons[i] = nullptr;
            parents[i] = nullptr;
        }

        size_t nParents = 1;
        parents[0] = NodeType::castSplit(&root);
        size_t idxInTable = 0;

        for(size_t lvl = 0; lvl < nLvls + 1; ++lvl)
        {
            size_t nSons = 0;
            for(size_t iParent = 0; iParent < nParents; ++iParent)
            {
                const typename NodeType::Split* p = parents[iParent];

                if(p->isSplit())
                {
                    sons[nSons++] = NodeType::castSplit(p->left());
                    sons[nSons++] = NodeType::castSplit(p->right());
                    featureIndexes[idxInTable] = p->featureIdx;
                }
                else
                {
                    sons[nSons++] = p;
                    sons[nSons++] = p;
                    featureIndexes[idxInTable] = 0;
                }

                DAAL_ASSERT(featureIndexes[idxInTable] >= 0);
                spitPoints[idxInTable] = p->featureValue;
                impVals[idxInTable] = p->impurity;
                nNodeSamplesVals[idxInTable] = (int)p->count;

                idxInTable++;
            }

            const size_t size = nSons*sizeof(SplitType);
            daal::services::daal_memcpy_s(parents, size, sons, size);

            nParents = nSons;
        }
    }

protected:
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive* arch)
    {
        arch->set(_nNodes);
        arch->set(_maxLvl);
        arch->set(_sourceNumOfNodes);

        arch->setSharedPtrObj(_splitPoints);
        arch->setSharedPtrObj(_featureIndexes);

        return services::Status();
    }

protected:
    size_t _nNodes;
    gbt::prediction::internal::FeatureIndexType _maxLvl;
    size_t _sourceNumOfNodes;
    services::SharedPtr<SplitPointType> _splitPoints;
    services::SharedPtr<FeatureIndexesForSplitType> _featureIndexes;
};

template <typename TNodeType, typename TAllocator = dtrees::internal::ChunkAllocator<TNodeType> >
class GbtTreeImpl : public dtrees::internal::TreeImpl<TNodeType, TAllocator>
{
private:
    typedef dtrees::internal::TreeImpl<TNodeType, TAllocator> super;


public:
    typedef TAllocator Allocator;
    typedef TNodeType NodeType;

    void convertGbtTreeToTable(GbtDecisionTree** pTbl, HomogenNumericTable<double>** pTblImp, HomogenNumericTable<int>** pTblSmplCnt) const
    {
        size_t nLvls = 1;
        getMaxLvl(*super::top(), nLvls);
        const size_t nNodes = getNumberOfNodesByLvls(nLvls);

        *pTblImp     = new HomogenNumericTable<double>(1, nNodes, NumericTable::doAllocate);
        *pTblSmplCnt = new HomogenNumericTable<int>(1, nNodes, NumericTable::doAllocate);

        *pTbl = new GbtDecisionTree(nNodes, nLvls, super::top()->numChildren() + 1);
        if(super::top())
        {
            GbtDecisionTree::internalTreeToGbtDecisionTree<TNodeType, typename TNodeType::Base>(*super::top(), nNodes, nLvls,
                    *pTbl, (*pTblImp)->getArray(), (*pTblSmplCnt)->getArray());
        }
    }


protected:
    void getMaxLvl(const typename TNodeType::Base& node, size_t& maxLvl, size_t curLvl  = 0) const
    {
        curLvl++;
        const auto p = TNodeType::castSplit(&node);

        if(p->isSplit())
        {
            getMaxLvl(*static_cast<const typename NodeType::Split*>(p->left()), maxLvl, curLvl);
            getMaxLvl(*static_cast<const typename NodeType::Split*>(p->right()), maxLvl, curLvl);
        }
        else
        {
            if (maxLvl < curLvl)
                maxLvl = curLvl;
        }
    }
};

template<typename Allocator = dtrees::internal::ChunkAllocator<dtrees::internal::TreeNodeRegression<RegressionFPType> > >
using TreeImpRegression = GbtTreeImpl<dtrees::internal::TreeNodeRegression<RegressionFPType>, Allocator>;

template<typename Allocator = dtrees::internal::ChunkAllocator<dtrees::internal::TreeNodeClassification<ClassificationFPType> > >
using TreeImpClassification = GbtTreeImpl<dtrees::internal::TreeNodeClassification<ClassificationFPType>, Allocator>;

class ModelImpl : private dtrees::internal::ModelImpl
{
public:
    using ImplType = dtrees::internal::ModelImpl;
    using TreeType = gbt::internal::TreeImpRegression<>;
    using super    = dtrees::internal::ModelImpl;

    ~ModelImpl();
    size_t size() const;
    bool reserve(const size_t nTrees);
    bool resize(const size_t nTrees);
    void clear();

    const GbtDecisionTree* at(const size_t idx) const;

    static SharedPtr<DecisionTreeTable> gbtTreeToDecisionTree(const GbtDecisionTree& gbtTree);
    static void decisionTreeToGbtTree(const DecisionTreeTable& tree, GbtDecisionTree& gbtTree);

    // Methods common for regression or classification model, not virtual!!!
    size_t numberOfTrees() const;
    void traverseDF(size_t iTree, algorithms::regression::TreeNodeVisitor& visitor) const;
    void traverseBF(size_t iTree, algorithms::regression::TreeNodeVisitor& visitor) const;
    void add(gbt::internal::GbtDecisionTree* pTbl, HomogenNumericTable<double>* pTblImp, HomogenNumericTable<int>* pTblSmplCnt);
    void traverseDFS(size_t iTree, tree_utils::regression::TreeNodeVisitor& visitor) const;
    void traverseBFS(size_t iTree, tree_utils::regression::TreeNodeVisitor& visitor) const;
    static void treeToTable(TreeType& t, gbt::internal::GbtDecisionTree** pTbl, HomogenNumericTable<double>** pTblImp, HomogenNumericTable<int>** pTblSmplCnt);

protected:
    static void convertNode(const GbtDecisionTree& gbtTree, DecisionTreeTable& newTree, const size_t idx, const size_t lvl, size_t& idxInTable);
    static bool nodeIsDummyLeaf(size_t idx, const GbtDecisionTree& gbtTree);
    static bool nodeIsLeaf(size_t idx, const GbtDecisionTree& gbtTree, const size_t lvl);
    static size_t getIdxOfParent(const size_t sonIdx);
    static void getMaxLvl(const dtrees::internal::DecisionTreeNode* const arr, const size_t idx, size_t& maxLvl, size_t curLvl = 0);

    static GbtDecisionTree* allocateGbtTree(const DecisionTreeTable& tree)
    {
        const dtrees::internal::DecisionTreeNode* const arr = (const dtrees::internal::DecisionTreeNode*)tree.getArray();

        size_t nLvls = 1;
        getMaxLvl(arr, 0, nLvls);
        const size_t nNodes = getNumberOfNodesByLvls(nLvls);

        return new GbtDecisionTree(nNodes, nLvls, tree.getNumberOfRows());
    }

    void destroy();

    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch, int daalVersion = INTEL_DAAL_VERSION)
    {
        if((daalVersion >= COMPUTE_DAAL_VERSION(2019, 0, 0)))
        {
            arch->setSharedPtrObj(_serializationData);
            arch->setSharedPtrObj(_impurityTables);
            arch->setSharedPtrObj(_nNodeSampleTables);
        }
        else
        {
            arch->setSharedPtrObj(_serializationData);
            const size_t size = _serializationData->size();

            data_management::DataCollection* newTrees = new data_management::DataCollection();
            for(size_t i = 0; i < size; ++i)
            {
                const DecisionTreeTable& tree = *(DecisionTreeTable*)(*_serializationData)[i].get();
                GbtDecisionTree* newTree = allocateGbtTree(tree);
                decisionTreeToGbtTree(tree, *newTree);
                newTrees->push_back(SerializationIfacePtr(newTree));
            }
            _serializationData.reset(newTrees);
        }

        if(onDeserialize)
            _nTree.set(_serializationData->size());

        return services::Status();
    }
};

} // namespace internal
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif
