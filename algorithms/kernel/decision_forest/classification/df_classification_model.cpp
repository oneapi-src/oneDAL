/* file: df_classification_model.cpp */
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
//  Implementation of the class defining the decision forest classification model
//--
*/

#include "algorithms/decision_forest/decision_forest_classification_model.h"
#include "serialization_utils.h"
#include "df_classification_model_impl.h"
#include "collection.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace interface1
{

__DAAL_REGISTER_SERIALIZATION_CLASS2(Model, internal::ModelImpl, SERIALIZATION_DECISION_FOREST_CLASSIFICATION_MODEL_ID);
}

namespace internal
{

size_t ModelImpl::numberOfTrees() const
{
    return ImplType::size();
}

static bool traverseNodeDF(size_t level, const ModelImpl::TreeType::NodeType::Base& n, decision_forest::classification::NodeVisitor& visitor)
{
    if(n.isSplit())
    {
        const ModelImpl::TreeType::NodeType::Split* s = ModelImpl::TreeType::NodeType::castSplit(&n);
        if(!visitor.onSplitNode(level, s->featureIdx, s->featureValue))
            return false; //do not continue traversing
        ++level;
        if(s->left() && !traverseNodeDF(level, *s->left(), visitor))
            return false; //do not continue traversing
        return (s->right() ? traverseNodeDF(level, *s->right(), visitor) : true);
    }
    const ModelImpl::TreeType::NodeType::Leaf* l = ModelImpl::TreeType::NodeType::castLeaf(&n);
    return visitor.onLeafNode(level, l->response.value);
}

void ModelImpl::traverseDF(size_t iTree, decision_forest::classification::NodeVisitor& visitor) const
{
    if(iTree >= size())
        return;
    const TreeType* t = static_cast<const TreeType*>(this->at(iTree));
    if(t && t->top())
        traverseNodeDF(0, *t->top(), visitor);
}

typedef services::Collection<const ModelImpl::TreeType::NodeType::Base*> NodePtrArray;

static bool traverseNodesBF(size_t level, NodePtrArray& aCur,
    NodePtrArray& aNext, decision_forest::classification::NodeVisitor& visitor)
{
    for(size_t i = 0; i < aCur.size(); ++i)
    {
        const ModelImpl::TreeType::NodeType::Base& n = *aCur[i];
        if(n.isSplit())
        {
            const ModelImpl::TreeType::NodeType::Split* s = ModelImpl::TreeType::NodeType::castSplit(&n);
            if(!visitor.onSplitNode(level, s->featureIdx, s->featureValue))
                return false; //do not continue traversing
            if(s->left())
                aNext.push_back(s->left());
            if(s->right())
                aNext.push_back(s->right());
        }
        else
        {
            const ModelImpl::TreeType::NodeType::Leaf* l = ModelImpl::TreeType::NodeType::castLeaf(&n);
            if(!visitor.onLeafNode(level, l->response.value))
                return false; //do not continue traversing
        }
    }
    aCur.clear();
    if(!aNext.size())
        return true;//done
    return traverseNodesBF(level + 1, aNext, aCur, visitor);
}

void ModelImpl::traverseBF(size_t iTree, NodeVisitor& visitor) const
{
    if(iTree >= size())
        return;
    const TreeType* t = static_cast<const TreeType*>(this->at(iTree));
    NodePtrArray aCur;//nodes of current layer
    NodePtrArray aNext;//nodes of next layer
    if(t && t->top())
    {
        aCur.push_back(t->top());
        traverseNodesBF(0, aCur, aNext, visitor);
    }
}

} // namespace interface1
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
