/* file: gbt_regression_model_builder.cpp */
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
//  Implementation of the class defining the gbt regression model builder
//--
*/

#include "algorithms/gradient_boosted_trees/gbt_regression_model_builder.h"
#include "algorithms/gradient_boosted_trees/gbt_regression_model.h"
#include "../../dtrees_model_impl.h"
#include "../gbt_model_impl.h"
#include "gbt_regression_model_impl.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace interface1
{

services::Status ModelBuilder::convertModelInternal()
{
    gbt::regression::internal::ModelImpl& modelImplRef = daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::regression::internal::ModelImpl,ModelPtr>(_model);
    return daal::algorithms::gbt::internal::ModelImpl::convertDecisionTreesToGbtTrees(modelImplRef._serializationData);
}

services::Status ModelBuilder::initialize(size_t nFeatures, size_t nIterations)
{
    services::Status s;
    _model.reset(new gbt::regression::internal::ModelImpl(nFeatures));
    gbt::regression::internal::ModelImpl& modelImplRef = daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::regression::internal::ModelImpl,ModelPtr>(_model);

    modelImplRef.resize(nIterations);
    modelImplRef._impurityTables.reset();
    modelImplRef._nNodeSampleTables.reset();
    modelImplRef._nTree.set(nIterations);
    return s;
}

services::Status ModelBuilder::createTreeInternal(size_t nNodes, TreeId& resId)
{
    gbt::regression::internal::ModelImpl& modelImplRef = daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::regression::internal::ModelImpl,ModelPtr>(_model);
    return daal::algorithms::dtrees::internal::createTreeInternal(modelImplRef._serializationData, nNodes, resId);
}

services::Status ModelBuilder::addLeafNodeInternal(TreeId treeId, NodeId parentId, size_t position, double response, NodeId& res)
{
    gbt::regression::internal::ModelImpl& modelImplRef = daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::regression::internal::ModelImpl,ModelPtr>(_model);
    return daal::algorithms::dtrees::internal::addLeafNodeInternal<double>(modelImplRef._serializationData, treeId, parentId, position, response, res);
}

services::Status ModelBuilder::addSplitNodeInternal(TreeId treeId, NodeId parentId, size_t position, size_t featureIndex, double featureValue, NodeId& res)
{
    gbt::regression::internal::ModelImpl& modelImplRef = daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::regression::internal::ModelImpl,ModelPtr>(_model);
    return daal::algorithms::dtrees::internal::addSplitNodeInternal(modelImplRef._serializationData, treeId, parentId, position, featureIndex, featureValue, res);
}

} // namespace interface1
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
