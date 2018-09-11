/* file: dtrees_model.cpp */
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

} // namespace internal
} // namespace dtrees
} // namespace algorithms
} // namespace daal
