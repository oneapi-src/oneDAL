/* file: dtrees_model.cpp */
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
//  Implementation of the class defining the decision trees model
//--
*/

#include "dtrees_model_impl.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace dtrees
{

namespace internal
{
Tree::~Tree()
{
}

ModelImpl::ModelImpl() : _nTree(0)
{
}

ModelImpl::~ModelImpl()
{
    destroy();
}

void ModelImpl::destroy()
{
    _serializationData.reset();
}

bool ModelImpl::reserve(const size_t nTrees)
{
    if(_serializationData.get())
        return false;
    _nTree.set(0);
    _serializationData.reset(new DataCollection());
    _serializationData->resize(nTrees);

    _impurityTables.reset(new DataCollection());
    _impurityTables->resize(nTrees);

    _nNodeSampleTables.reset(new DataCollection());
    _nNodeSampleTables->resize(nTrees);

    return _serializationData.get();
}

bool ModelImpl::resize(const size_t nTrees)
{
    if(_serializationData.get())
        return false;
    _nTree.set(0);
    _serializationData.reset(new DataCollection(nTrees));
    _impurityTables.reset(new DataCollection(nTrees));
    _nNodeSampleTables.reset(new DataCollection(nTrees));
    return _serializationData.get();
}

void ModelImpl::clear()
{
    if(_serializationData.get())
        _serializationData.reset();

    if(_impurityTables.get())
        _impurityTables.reset();

    if(_nNodeSampleTables.get())
        _nNodeSampleTables.reset();

    _nTree.set(0);
}

void MemoryManager::destroy()
{
    for(size_t i = 0; i < _aChunk.size(); ++i)
        daal_free(_aChunk[i]);
    _aChunk.clear();
    _posInChunk = 0;
    _iCurChunk = -1;
}

void* MemoryManager::alloc(size_t nBytes)
{
    DAAL_ASSERT(nBytes <= _chunkSize);
    size_t pos = 0; //pos in the chunk to allocate from
    if((_iCurChunk >= 0) && (_posInChunk + nBytes <= _chunkSize))
    {
        //allocate from the current chunk
        pos = _posInChunk;
    }
    else
    {
        if(!_aChunk.size() || _iCurChunk + 1 >= _aChunk.size())
        {
            //allocate a new chunk
            DAAL_ASSERT(_aChunk.size() ? _iCurChunk >= 0 : _iCurChunk < 0);
            byte* ptr = (byte*)services::daal_malloc(_chunkSize);
            if(!ptr)
                return nullptr;
            _aChunk.push_back(ptr);
        }
        //there are free chunks, make next available a current one and allocate from it
        _iCurChunk++;
        pos = 0;
        _posInChunk = 0;
    }
    //allocate from the current chunk
    _posInChunk += nBytes;
    return _aChunk[_iCurChunk] + pos;
}

void MemoryManager::reset()
{
    _iCurChunk = -1;
    _posInChunk = 0;
}

services::Status createTreeInternal(data_management::DataCollectionPtr& serializationData, size_t nNodes, size_t& resId)
{
    if (nNodes == 0)
    {
        return services::Status(services::ErrorID::ErrorIncorrectParameter);
    }
    services::Status s;

    size_t treeId = 0;
    bool isNotEmptyTreeTable = ((*(serializationData))[treeId].get()) != nullptr;
    const size_t nTrees = (*(serializationData)).size();
    while(isNotEmptyTreeTable && (treeId < nTrees))
    {
        isNotEmptyTreeTable = ((*(serializationData))[treeId].get()) != nullptr;
        if(isNotEmptyTreeTable)
            treeId++;
    }
    if (treeId == nTrees)
    {
        return services::Status(services::ErrorID::ErrorIncorrectParameter);
    }

    services::SharedPtr<DecisionTreeTable> treeTablePtr(new DecisionTreeTable(nNodes));
    const size_t nRows = treeTablePtr->getNumberOfRows();
    DecisionTreeNode* const pNodes = (DecisionTreeNode*)treeTablePtr->getArray();
    pNodes[0].featureIndex = __NODE_RESERVED_ID;
    pNodes[0].leftIndexOrClass = 0;
    pNodes[0].featureValueOrResponse = 0;
    for(size_t i = 1; i < nRows; i++)
    {
        pNodes[i].featureIndex = __NODE_FREE_ID;
        pNodes[i].leftIndexOrClass = 0;
        pNodes[i].featureValueOrResponse = 0;
    }
    (*(serializationData))[treeId] = treeTablePtr;
    resId = treeId;
    return s;
}

void setNode(DecisionTreeNode& node, int featureIndex, size_t classLabel)
{
    node.featureIndex = featureIndex;
    node.leftIndexOrClass = classLabel;
    node.featureValueOrResponse = 0;
}

void setNode(DecisionTreeNode& node, int featureIndex, double response)
{
    node.featureIndex = featureIndex;
    node.leftIndexOrClass = 0;
    node.featureValueOrResponse = response;
}

services::Status addSplitNodeInternal(data_management::DataCollectionPtr& serializationData,size_t treeId, size_t parentId, size_t position, size_t featureIndex, double featureValue, size_t& res)
{
    const size_t noParent = static_cast<size_t>(-1);
    services::Status s;

    if ((treeId > (*(serializationData)).size()) || (position != 0 && position != 1 ))
    {
        return services::Status(services::ErrorID::ErrorIncorrectParameter);
    }

    const DecisionTreeTable* const pTreeTable = static_cast<DecisionTreeTable*>((*(serializationData))[treeId].get());
    if (!pTreeTable)
        return services::Status(services::ErrorID::ErrorNullPtr);
    const size_t nRows = pTreeTable->getNumberOfRows();
    DecisionTreeNode* const aNode = (DecisionTreeNode*)pTreeTable->getArray();
    size_t nodeId = 0;
    if (parentId == noParent)
    {
        aNode[0].featureIndex = featureIndex;
        aNode[0].leftIndexOrClass = 0;
        aNode[0].featureValueOrResponse = featureValue;
        nodeId = 0;
    }
    else if (aNode[parentId].featureIndex < 0)
    {
        return services::Status(services::ErrorID::ErrorIncorrectParameter);
    }
    else
    {
        /*if not leaf, and parent has child already*/
        if ((aNode[parentId].leftIndexOrClass > 0) && (position == 1))
        {
            const size_t reservedId = aNode[parentId].leftIndexOrClass + 1;
            nodeId = reservedId;
            if (aNode[reservedId].featureIndex == __NODE_RESERVED_ID)
            {
                aNode[nodeId].featureIndex = featureIndex;
                aNode[nodeId].leftIndexOrClass = 0;
                aNode[nodeId].featureValueOrResponse = featureValue;
            }
        }
        if ((aNode[parentId].leftIndexOrClass > 0) && (position == 0))
        {
            const size_t reservedId = aNode[parentId].leftIndexOrClass;
            nodeId = reservedId;
            if (aNode[reservedId].featureIndex == __NODE_RESERVED_ID)
            {
                aNode[nodeId].featureIndex = featureIndex;
                aNode[nodeId].leftIndexOrClass = 0;
                aNode[nodeId].featureValueOrResponse = featureValue;
            }
        }
        if ((aNode[parentId].leftIndexOrClass == 0) && (position == 0))
        {
            size_t i;
            for(i = parentId + 1; i < nRows; i++)
            {
                if (aNode[i].featureIndex == __NODE_FREE_ID)
                {
                    nodeId = i;
                    break;
                }
            }
            /* no space left */
            if (i == nRows)
            {
                return services::Status(services::ErrorID::ErrorIncorrectParameter);
            }
            aNode[nodeId].featureIndex = featureIndex;
            aNode[nodeId].leftIndexOrClass = 0;
            aNode[nodeId].featureValueOrResponse = featureValue;
            aNode[parentId].leftIndexOrClass = nodeId;
            if (((nodeId + 1) < nRows) && (aNode[nodeId+1].featureIndex == __NODE_FREE_ID))
            {
                    aNode[nodeId+1].featureIndex = __NODE_RESERVED_ID;
            }
            else
            {
                return services::Status(services::ErrorID::ErrorIncorrectParameter);
            }
        }
        if ((aNode[parentId].leftIndexOrClass == 0) && (position == 1))
        {
            size_t leftEmptyId = 0;
            size_t i;
            for(i = parentId + 1; i < nRows; i++)
            {
                if (aNode[i].featureIndex == __NODE_FREE_ID)
                {
                    leftEmptyId = i;
                    break;
                }
            }
            /*if no free nodes leftBound is not initialized and no space left*/
            if (i == nRows)
            {
                return services::Status(services::ErrorID::ErrorIncorrectParameter);
            }
            aNode[leftEmptyId].featureIndex = __NODE_RESERVED_ID;
            aNode[parentId].leftIndexOrClass = leftEmptyId;
            nodeId = leftEmptyId + 1;
            if (nodeId < nRows)
            {
                aNode[nodeId].featureIndex = featureIndex;
                aNode[nodeId].leftIndexOrClass = 0;
                aNode[nodeId].featureValueOrResponse = featureValue;
            }
            else
            {
                return services::Status(services::ErrorID::ErrorIncorrectParameter);
            }
        }
    }
    res = nodeId;
    return s;
}

} // namespace internal
} // namespace dtrees
} // namespace algorithms
} // namespace daal
