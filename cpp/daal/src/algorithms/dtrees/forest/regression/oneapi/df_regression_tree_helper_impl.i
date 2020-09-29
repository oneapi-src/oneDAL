/* file: df_regression_tree_helper_impl.i */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  Implementation of the class defining the decision forest regression tree
//--
*/

#ifndef __DF_REGRESSION_TREE_HELPER_IMPL__
#define __DF_REGRESSION_TREE_HELPER_IMPL__

//#include "data_management/data/aos_numeric_table.h"
#include "src/services/service_arrays.h"
#include "src/algorithms/dtrees/dtrees_predict_dense_default_impl.i"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
{
namespace internal
{
using namespace daal::algorithms::dtrees::internal;
using namespace daal::services::internal;

template <typename algorithmFPType, CpuType cpu>
class RegressionTreeHelperOneAPI
{
public:
    typedef dtrees::internal::TreeImpRegression<> TreeType;
    typedef typename TreeType::NodeType NodeType;

    typename NodeType::Leaf * makeLeaf(size_t n, algorithmFPType response, algorithmFPType impurity)
    {
        typename NodeType::Leaf * pNode = _tree.allocator().allocLeaf(0);
        DAAL_ASSERT(n > 0);
        pNode->response = response;
        pNode->count    = n;
        pNode->impurity = impurity;

        return pNode;
    }

    typename NodeType::Split * makeSplit(size_t iFeature, algorithmFPType featureValue, bool bUnordered, algorithmFPType impurity,
                                         typename NodeType::Base * left, typename NodeType::Base * right)
    {
        typename NodeType::Split * pNode = _tree.allocator().allocSplit();
        pNode->set(iFeature, featureValue, bUnordered);
        pNode->kid[0]   = left;
        pNode->kid[1]   = right;
        pNode->impurity = impurity;

        return pNode;
    }

    algorithmFPType predict(const dtrees::internal::Tree & t, const algorithmFPType * x) const
    {
        const typename NodeType::Base * pNode = dtrees::prediction::internal::findNode<algorithmFPType, TreeType, cpu>(t, x);
        DAAL_ASSERT(pNode);
        return pNode ? NodeType::castLeaf(pNode)->response : 0.0;
    }

    TreeType _tree;
};

template <typename algorithmFPType>
struct TreeLevelRecord
{
    TreeLevelRecord() : _nodeList(nullptr), _impInfo(nullptr), _nNodes(0) {}
    services::Status init(oneapi::internal::UniversalBuffer & nodeList, oneapi::internal::UniversalBuffer & impInfo, size_t nNodes)
    {
        _nNodes = nNodes;
        DAAL_ASSERT(nNodes * _nNodeSplitProps == nodeList.template get<int>().size());
        DAAL_ASSERT(nNodes * _nNodeImpProps == impInfo.template get<algorithmFPType>().size());

        auto nodeListHost = nodeList.template get<int>().toHost(ReadWriteMode::readOnly);
        _nodeList         = nodeListHost.get();
        DAAL_CHECK_MALLOC(_nodeList);

        auto impInfoHost = impInfo.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly);
        _impInfo         = impInfoHost.get();
        DAAL_CHECK_MALLOC(_impInfo);

        return services::Status();
    }

    size_t getNodesNum() { return _nNodes; }
    int getRowsNum(size_t nodeIdx) { return _nodeList[nodeIdx * _nNodeSplitProps + 1]; }
    int getFtrIdx(size_t nodeIdx) { return _nodeList[nodeIdx * _nNodeSplitProps + 2]; }
    int getFtrVal(size_t nodeIdx) { return _nodeList[nodeIdx * _nNodeSplitProps + 3]; }
    algorithmFPType getImpurity(size_t nodeIdx) { return _impInfo[nodeIdx * _nNodeImpProps + 0]; }
    algorithmFPType getResponse(size_t nodeIdx) { return _impInfo[nodeIdx * _nNodeImpProps + 1]; }
    bool hasUnorderedFtr(size_t nodeIdx) { return false; }

    constexpr static int _nNodeImpProps   = 2;
    constexpr static int _nNodeSplitProps = 5;

    int * _nodeList;
    algorithmFPType * _impInfo;
    size_t _nNodes;
};

template <typename algorithmFPType, CpuType cpu>
struct DFTreeConverter
{
    typedef RegressionTreeHelperOneAPI<algorithmFPType, cpu> TreeHelperType;

    services::Status convertToDFDecisionTree(Collection<TreeLevelRecord<algorithmFPType> > & treeLevelsList, algorithmFPType ** binValues,
                                             TreeHelperType & treeBuilder)
    {
        services::Status status;
        typedef TArray<typename TreeHelperType::NodeType::Base *, cpu> DFTreeNodesArr;
        typedef SharedPtr<DFTreeNodesArr> DFTreeNodesArrPtr;

        DFTreeNodesArrPtr dfTreeLevelNodesPrev;
        bool unorderedFeaturesUsed = false;
        const int notFoundVal      = -1;

        TreeLevelRecord<algorithmFPType> & r0 = treeLevelsList[0];

        size_t level = treeLevelsList.size();
        DAAL_ASSERT(level);

        do
        {
            level--;
            TreeLevelRecord<algorithmFPType> & record = treeLevelsList[level];
            DFTreeNodesArrPtr dfTreeLevelNodes(new DFTreeNodesArr(record.getNodesNum()));
            DAAL_CHECK_MALLOC(dfTreeLevelNodes.get());
            DAAL_CHECK_MALLOC(dfTreeLevelNodes->get());

            size_t nSplits = 0;
            // nSplits is used to calculate index of child nodes on next level
            for (size_t nodeIdx = 0; nodeIdx < record.getNodesNum(); nodeIdx++)
            {
                if (record.getFtrIdx(nodeIdx) == notFoundVal)
                {
                    // leaf node
                    dfTreeLevelNodes->get()[nodeIdx] =
                        treeBuilder.makeLeaf(record.getRowsNum(nodeIdx), record.getResponse(nodeIdx), record.getImpurity(nodeIdx));
                }
                else
                {
                    DAAL_ASSERT(dfTreeLevelNodesPrev->get());
                    //split node
                    dfTreeLevelNodes->get()[nodeIdx] =
                        treeBuilder.makeSplit(record.getFtrIdx(nodeIdx), binValues[record.getFtrIdx(nodeIdx)][record.getFtrVal(nodeIdx)],
                                              static_cast<bool>(record.hasUnorderedFtr(nodeIdx)), record.getImpurity(nodeIdx),
                                              dfTreeLevelNodesPrev->get()[nSplits * 2], dfTreeLevelNodesPrev->get()[nSplits * 2 + 1]);
                    nSplits++;
                }
            }

            dfTreeLevelNodesPrev = dfTreeLevelNodes;
        } while (level > 0);

        treeBuilder._tree.reset(dfTreeLevelNodesPrev->get()[0], unorderedFeaturesUsed);
        return status;
    }
};

} // namespace internal
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
