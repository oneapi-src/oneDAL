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

    RegressionTreeHelperOneAPI() = delete;
    explicit RegressionTreeHelperOneAPI(size_t nTrees) : _allocator(_cNumNodesHint) { _tree_list.reset(nTrees); }
    ~RegressionTreeHelperOneAPI() {}

    typename NodeType::Leaf * makeLeaf(size_t n, algorithmFPType response, algorithmFPType impurity)
    {
        typename NodeType::Leaf * pNode = _allocator.allocLeaf(0);
        DAAL_ASSERT(n > 0);
        pNode->response = response;
        pNode->count    = n;
        pNode->impurity = impurity;

        return pNode;
    }

    typename NodeType::Split * makeSplit(size_t n, size_t iFeature, algorithmFPType featureValue, bool bUnordered, algorithmFPType impurity,
                                         typename NodeType::Base * left, typename NodeType::Base * right)
    {
        typename NodeType::Split * pNode = _allocator.allocSplit();
        pNode->set(iFeature, featureValue, bUnordered);
        pNode->kid[0]   = left;
        pNode->kid[1]   = right;
        pNode->impurity = impurity;
        pNode->count    = n;

        return pNode;
    }

    static algorithmFPType predict(const dtrees::internal::Tree & t, const algorithmFPType * x)
    {
        const typename NodeType::Base * pNode = dtrees::prediction::internal::findNode<algorithmFPType, TreeType, cpu>(t, x);
        DAAL_ASSERT(pNode);
        return pNode ? NodeType::castLeaf(pNode)->response : 0.0;
    }

    static const size_t _cNumNodesHint = 512; //number of nodes as a hint for allocator to grow by
    TreeType::Allocator _allocator;
    TArray<TreeType, DAAL_BASE_CPU> _tree_list;
};

template <typename algorithmFPType>
struct TreeLevelRecord
{
    TreeLevelRecord() : _isInitialized(false), _nNodes(0) {}
    services::Status init(services::internal::sycl::UniversalBuffer & nodeList, services::internal::sycl::UniversalBuffer & impInfo, size_t nNodes)
    {
        services::Status status;

        _nNodes = nNodes;
        DAAL_ASSERT_UNIVERSAL_BUFFER(nodeList, int32_t, nNodes * _nNodeSplitProps);
        DAAL_ASSERT_UNIVERSAL_BUFFER(impInfo, algorithmFPType, nNodes * _nNodeImpProps);

        auto nodeListHost = nodeList.template get<int>().toHost(ReadWriteMode::readOnly, status);
        auto impInfoHost  = impInfo.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly, status);
        DAAL_CHECK_STATUS_VAR(status);

        _nodeList = nodeListHost;
        _impInfo  = impInfoHost;

        _isInitialized = true;

        return status;
    }

    bool isInitialized() const { return _isInitialized; }
    size_t getNodesNum() { return _nNodes; }
    int getRowsNum(size_t nodeIdx) { return _nodeList.get()[nodeIdx * _nNodeSplitProps + 1]; }
    int getFtrIdx(size_t nodeIdx) { return _nodeList.get()[nodeIdx * _nNodeSplitProps + 2]; }
    int getFtrVal(size_t nodeIdx) { return _nodeList.get()[nodeIdx * _nNodeSplitProps + 3]; }
    algorithmFPType getImpurity(size_t nodeIdx) { return _impInfo.get()[nodeIdx * _nNodeImpProps + 0]; }
    algorithmFPType getResponse(size_t nodeIdx) { return _impInfo.get()[nodeIdx * _nNodeImpProps + 1]; }
    bool hasUnorderedFtr(size_t nodeIdx) { return false; }

    constexpr static int _nNodeImpProps   = 2;
    constexpr static int _nNodeSplitProps = 5;

    SharedPtr<int> _nodeList;
    SharedPtr<algorithmFPType> _impInfo;
    size_t _nNodes;

    bool _isInitialized;
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

        size_t level = treeLevelsList.size();
        DAAL_ASSERT(level);

        do
        {
            level--;
            TreeLevelRecord<algorithmFPType> & record = treeLevelsList[level];
            DAAL_ASSERT(record.isInitialized());

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
                    dfTreeLevelNodes->get()[nodeIdx] = treeBuilder.makeSplit(
                        record.getRowsNum(nodeIdx), record.getFtrIdx(nodeIdx), binValues[record.getFtrIdx(nodeIdx)][record.getFtrVal(nodeIdx)],
                        static_cast<bool>(record.hasUnorderedFtr(nodeIdx)), record.getImpurity(nodeIdx), dfTreeLevelNodesPrev->get()[nSplits * 2],
                        dfTreeLevelNodesPrev->get()[nSplits * 2 + 1]);
                    nSplits++;
                }
            }

            dfTreeLevelNodesPrev = dfTreeLevelNodes;
        } while (level > 0);

        for (size_t tree = 0; tree < treeBuilder._tree_list.size(); tree++)
        {
            treeBuilder._tree_list[tree].reset(dfTreeLevelNodesPrev->get()[tree], unorderedFeaturesUsed);
        }
        return status;
    }
};

} // namespace internal
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
