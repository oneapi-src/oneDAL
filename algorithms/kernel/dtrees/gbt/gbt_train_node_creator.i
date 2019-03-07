/* file: gbt_train_node_creator.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of leaf/nodes creation for gradient boosted trees training
//  (defaultDense) method.
//--
*/

#ifndef __GBT_TRAIN_NODE_CREATOR_I__
#define __GBT_TRAIN_NODE_CREATOR_I__

#include "dtrees_model_impl.h"
#include "dtrees_train_data_helper.i"
#include "dtrees_predict_dense_default_impl.i"
#include "gbt_train_aux.i"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace training
{
namespace internal
{

template<typename PartialResult, CpuType cpu>
struct MergedResult;

template<typename algorithmFPType, typename IndexType, typename UpdaterType, CpuType cpu>
class DefaultNodesCreator
{
public:
    using NodeInfoType     = SplitJob<algorithmFPType,cpu>;
    using ImpurityType     = ImpurityData<algorithmFPType, cpu>;
    using SplitDataType    = SplitData<algorithmFPType, ImpurityType>;
    using DataType         = SharedDataForTree<algorithmFPType, IndexType, cpu>;
    using NodeType         = typename TreeBuilder<algorithmFPType, cpu>::NodeType;
    using ResultType       = typename UpdaterType::ResultType;
    using MergedResultType = MergedResult<ResultType, cpu>;

    DefaultNodesCreator(DataType& data, SplitDataType& split, NodeInfoType& parent, MergedResultType* prevRes):
        _data(data), _split(split), _node(parent), _prevRes(prevRes)
    {
    }

    void create(IndexType iFeature, GbtTask** newTasks, size_t& nTask)
    {
        if (iFeature >= 0)
        {
            typename NodeType::Split* res = makeSplit(iFeature, _split.featureValue, _split.featureUnordered);
            _node.res = res;
            res->kid[0] = buildLeaf(_node.iStart, _split.nLeft, _node.level + 1, _split.left);

            ImpurityType impRight;
            impRight.g = _node.imp.g - _split.left.g;
            impRight.h = _node.imp.h - _split.left.h;

            res->kid[1] = buildLeaf(_node.iStart + _split.nLeft, _node.n - _split.nLeft, _node.level + 1, impRight);

            res->count = _node.n;
            res->impurity = _node.imp.value(_data.ctx.par().lambda);

            if (!res->kid[0] && !res->kid[1])
            {
                build2nodes(newTasks, nTask, res, impRight);
            }
            else if( !res->kid[0] )
            {
                buildLeftnode(newTasks, nTask, res);
            }
            else if( !res->kid[1] )
            {
                buildRightnode(newTasks, nTask, res, impRight);
            }
            else
            {
                if (_prevRes) { _prevRes->release(_data); _prevRes = nullptr; }
            }
        }
        else
        {
            _node.res = makeLeaf(_data.aIdx + _node.iStart, _node.n, _node.imp);
            if (_prevRes) { _prevRes->release(_data); _prevRes = nullptr; }
        }
    }

protected:
    virtual void build2nodes(GbtTask** newTasks, size_t& nTask, typename NodeType::Split* res, ImpurityType& impRight)
    {
        buildLeftnode(newTasks, nTask, res);
        buildRightnode(newTasks, nTask, res, impRight);
    }

    void buildLeftnode(GbtTask** newTasks, size_t& nTask, typename NodeType::Split* res)
    {
        NodeInfoType node(_node.iStart, _split.nLeft, _node.level + 1, _split.left, res->kid[0]);
        newTasks[nTask++] = new (services::internal::service_scalable_malloc<UpdaterType, cpu>(1))UpdaterType(_data, node);
        if (_prevRes) { _prevRes->release(_data); _prevRes = nullptr; }
    }

    void buildRightnode(GbtTask** newTasks, size_t& nTask, typename NodeType::Split* res, ImpurityType& impRight)
    {
        NodeInfoType node(_node.iStart + _split.nLeft, _node.n - _split.nLeft, _node.level + 1, impRight, res->kid[1]);
        newTasks[nTask++] = new (services::internal::service_scalable_malloc<UpdaterType, cpu>(1)) UpdaterType(_data, node);

        if (_prevRes) { _prevRes->release(_data); _prevRes = nullptr; }
    }

    typename NodeType::Base* buildLeaf(size_t iStart, size_t n, size_t level, const ImpurityType& imp)
    {
        return _data.ctx.terminateCriteria(n, level, imp) ? makeLeaf(_data.aIdx + iStart, n, imp) : nullptr;
    }

    typename NodeType::Leaf* makeLeaf(const IndexType* idx, size_t n, const ImpurityType& imp)
    {
        typename NodeType::Leaf* pNode = nullptr;
        if(_data.ctx.isThreaded())
        {
            _data.mtAlloc.lock();
            pNode = _data.tree.allocator().allocLeaf();
            _data.mtAlloc.unlock();
        }
        else
            pNode = _data.tree.allocator().allocLeaf();
        pNode->response = _data.ctx.computeLeafWeightUpdateF(idx, n, imp, _data.iTree);
        pNode->count = n;
        pNode->impurity = imp.value(_data.ctx.par().lambda);
        return pNode;
    }

    typename NodeType::Split* makeSplit(size_t iFeature, algorithmFPType featureValue, bool bUnordered)
    {
        typename NodeType::Split* pNode = nullptr;
        if(_data.ctx.isThreaded())
        {
            _data.mtAlloc.lock();
            pNode = _data.tree.allocator().allocSplit();
            _data.mtAlloc.unlock();
        }
        else
            pNode = _data.tree.allocator().allocSplit();
        pNode->set(iFeature, featureValue, bUnordered);
        return pNode;
    }

    DataType& _data;
    SplitDataType& _split;
    NodeInfoType& _node;
    MergedResultType* _prevRes;
};


template<typename algorithmFPType, typename IndexType, typename UpdaterType, typename MergedUpdaterType, CpuType cpu>
class MergedNodesCreator: public DefaultNodesCreator<algorithmFPType, IndexType, UpdaterType, cpu>
{
public:
    using super = DefaultNodesCreator<algorithmFPType, IndexType, UpdaterType, cpu>;

    MergedNodesCreator(typename super::DataType& data, typename super::SplitDataType& split, typename super::NodeInfoType& parent, typename super::MergedResultType* prevRes):
        super(data, split, parent, prevRes) { }

protected:
    virtual void build2nodes(GbtTask** newTasks, size_t& nTask, typename super::NodeType::Split* res, typename super::ImpurityType& impRight)
    {
        typename super::NodeInfoType node1(super::_node.iStart, super::_split.nLeft, super::_node.level + 1, super::_split.left, res->kid[0]);
        typename super::NodeInfoType node2(super::_node.iStart + super::_split.nLeft, super::_node.n - super::_split.nLeft, super::_node.level + 1, impRight, res->kid[1]);
        newTasks[nTask++] = new (services::internal::service_scalable_malloc<MergedUpdaterType, cpu>(1)) MergedUpdaterType(super::_data, node1, node2, super::_prevRes);
    }

    using super::_data;
    using super::_split;
    using super::_node;
    using super::_prevRes;
};

} /* namespace internal */
} /* namespace training */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif
