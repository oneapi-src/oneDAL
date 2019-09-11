/* file: df_classification_model_builder.cpp */
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
    services::Status s;
    _model.reset(new decision_forest::classification::internal::ModelImpl());
    decision_forest::classification::internal::ModelImpl& modelImplRef = daal::algorithms::dtrees::internal::getModelRef<decision_forest::classification::internal::ModelImpl,ModelPtr>(_model);

    modelImplRef.resize(nTrees);
    modelImplRef._impurityTables.reset();
    modelImplRef._nNodeSampleTables.reset();
    modelImplRef._nTree.set(nTrees);
    return s;
}

services::Status ModelBuilder::createTreeInternal(size_t nNodes, TreeId& resId)
{
    decision_forest::classification::internal::ModelImpl& modelImplRef = daal::algorithms::dtrees::internal::getModelRef<decision_forest::classification::internal::ModelImpl,ModelPtr>(_model);
    return daal::algorithms::dtrees::internal::createTreeInternal(modelImplRef._serializationData, nNodes, resId);
}

services::Status ModelBuilder::addLeafNodeInternal(TreeId treeId, NodeId parentId, size_t position, size_t classLabel, NodeId& res)
{
    decision_forest::classification::internal::ModelImpl& modelImplRef = daal::algorithms::dtrees::internal::getModelRef<decision_forest::classification::internal::ModelImpl,ModelPtr>(_model);
    return daal::algorithms::dtrees::internal::addLeafNodeInternal<size_t>(modelImplRef._serializationData, treeId, parentId, position, classLabel, res);
}

services::Status ModelBuilder::addSplitNodeInternal(TreeId treeId, NodeId parentId, size_t position, size_t featureIndex, double featureValue, NodeId& res)
{
    decision_forest::classification::internal::ModelImpl& modelImplRef = daal::algorithms::dtrees::internal::getModelRef<decision_forest::classification::internal::ModelImpl,ModelPtr>(_model);
    return daal::algorithms::dtrees::internal::addSplitNodeInternal(modelImplRef._serializationData, treeId, parentId, position, featureIndex, featureValue, res);
}

} // namespace interface1
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
