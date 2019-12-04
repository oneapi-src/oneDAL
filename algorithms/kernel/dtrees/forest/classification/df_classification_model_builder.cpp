/* file: df_classification_model_builder.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of the class defining the decision forest classification model builder
//--
*/

#include "algorithms/decision_forest/decision_forest_classification_model_builder.h"
#include "algorithms/decision_forest/decision_forest_classification_model.h"
#include "../../dtrees_model_impl.h"
#include "df_classification_model_impl.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::algorithms::dtrees::internal;

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
services::Status ModelBuilder::initialize(size_t nClasses, size_t nTrees)
{
    auto modelImpl = new decision_forest::classification::internal::ModelImpl();
    DAAL_CHECK_MALLOC(modelImpl)
    _model.reset(modelImpl);
    decision_forest::classification::internal::ModelImpl & modelImplRef =
        daal::algorithms::dtrees::internal::getModelRef<decision_forest::classification::internal::ModelImpl, ModelPtr>(_model);

    modelImplRef.resize(nTrees);
    modelImplRef._impurityTables.reset();
    modelImplRef._nNodeSampleTables.reset();
    modelImplRef._probTbl.reset();
    modelImplRef._nTree.set(nTrees);
    return Status();
}

services::Status ModelBuilder::createTreeInternal(size_t nNodes, TreeId & resId)
{
    decision_forest::classification::internal::ModelImpl & modelImplRef =
        daal::algorithms::dtrees::internal::getModelRef<decision_forest::classification::internal::ModelImpl, ModelPtr>(_model);
    return daal::algorithms::dtrees::internal::createTreeInternal(modelImplRef._serializationData, nNodes, resId);
}

services::Status ModelBuilder::addLeafNodeInternal(TreeId treeId, NodeId parentId, size_t position, size_t classLabel, NodeId & res)
{
    decision_forest::classification::internal::ModelImpl & modelImplRef =
        daal::algorithms::dtrees::internal::getModelRef<decision_forest::classification::internal::ModelImpl, ModelPtr>(_model);
    return daal::algorithms::dtrees::internal::addLeafNodeInternal<size_t>(modelImplRef._serializationData, treeId, parentId, position, classLabel,
                                                                           res);
}

services::Status ModelBuilder::addSplitNodeInternal(TreeId treeId, NodeId parentId, size_t position, size_t featureIndex, double featureValue,
                                                    NodeId & res)
{
    decision_forest::classification::internal::ModelImpl & modelImplRef =
        daal::algorithms::dtrees::internal::getModelRef<decision_forest::classification::internal::ModelImpl, ModelPtr>(_model);
    return daal::algorithms::dtrees::internal::addSplitNodeInternal(modelImplRef._serializationData, treeId, parentId, position, featureIndex,
                                                                    featureValue, res);
}

} // namespace interface1
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
