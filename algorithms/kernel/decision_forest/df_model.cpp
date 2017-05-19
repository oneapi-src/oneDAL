/* file: df_model.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of the class defining the decision forest model
//--
*/

#include "df_model_impl.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{

namespace internal
{
Tree::~Tree()
{
}

} // namespace internal


namespace internal
{

ModelImpl::ModelImpl() : _aTree(nullptr), _nTree(0), _nCapacity(0)
{
}

ModelImpl::~ModelImpl()
{
    destroy();
}

void ModelImpl::destroy()
{
    if(_aTree)
    {
        for(size_t i = 0, n = size(); i < n; ++i)
            delete _aTree[i];
        daal::services::daal_free(_aTree);
        _aTree = nullptr;
        _nTree.set(0);
        _nCapacity = 0;
    }
}

bool ModelImpl::reserve(size_t nTrees)
{
    if(_aTree)
        return false;
    _aTree = (Tree**)(nTrees ? daal::services::daal_malloc(nTrees*sizeof(Tree*)) : nullptr);
    if(!_aTree)
        return false;
    _nCapacity = nTrees;
    for(size_t i = 0; i < _nCapacity; ++i)
        _aTree[i] = nullptr;
    return true;
}

bool ModelImpl::add(Tree* pTree)
{
    if(size() >= _nCapacity)
        return false;
    size_t i = _nTree.inc();
    _aTree[i - 1] = pTree;
    return true;
}

} // namespace internal
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
