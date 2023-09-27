/* file: gbt_regression_model_builder.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of the class defining the gbt regression model builder
//--
*/

#include "algorithms/gradient_boosted_trees/gbt_regression_model_builder.h"
#include "algorithms/gradient_boosted_trees/gbt_regression_model.h"
#include "src/algorithms/dtrees/dtrees_model_impl.h"
#include "src/algorithms/dtrees/gbt/gbt_model_impl.h"
#include "src/algorithms/dtrees/gbt/regression/gbt_regression_model_impl.h"

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
    gbt::regression::internal::ModelImpl & modelImplRef =
        daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::regression::internal::ModelImpl, ModelPtr>(_model);
    return daal::algorithms::gbt::internal::ModelImpl::convertDecisionTreesToGbtTrees(modelImplRef._serializationData);
}

ModelBuilder::ModelBuilder() : _model(new gbt::regression::internal::ModelImpl()) {}

services::Status ModelBuilder::initialize(size_t nFeatures, size_t nIterations)
{
    auto modelImpl = new gbt::regression::internal::ModelImpl(nFeatures);
    DAAL_CHECK_MALLOC(modelImpl)
    _model.reset(modelImpl);
    gbt::regression::internal::ModelImpl & modelImplRef =
        daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::regression::internal::ModelImpl, ModelPtr>(_model);

    modelImplRef.resize(nIterations);
    modelImplRef._impurityTables.reset();
    modelImplRef._nNodeSampleTables.reset();
    modelImplRef._nTree.set(nIterations);
    return services::Status();
}

services::Status ModelBuilder::createTreeInternal(size_t nNodes, TreeId & resId)
{
    gbt::regression::internal::ModelImpl & modelImplRef =
        daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::regression::internal::ModelImpl, ModelPtr>(_model);
    return daal::algorithms::dtrees::internal::createTreeInternal(modelImplRef._serializationData, nNodes, resId);
}

services::Status ModelBuilder::addLeafNodeInternal(TreeId treeId, NodeId parentId, size_t position, double response, double cover, NodeId & res)
{
    gbt::regression::internal::ModelImpl & modelImplRef =
        daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::regression::internal::ModelImpl, ModelPtr>(_model);
    return daal::algorithms::dtrees::internal::addLeafNodeInternal<double>(modelImplRef._serializationData, treeId, parentId, position, response,
                                                                           cover, res);
}

services::Status ModelBuilder::addSplitNodeInternal(TreeId treeId, NodeId parentId, size_t position, size_t featureIndex, double featureValue,
                                                    int defaultLeft, double cover, NodeId & res)
{
    gbt::regression::internal::ModelImpl & modelImplRef =
        daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::regression::internal::ModelImpl, ModelPtr>(_model);
    return daal::algorithms::dtrees::internal::addSplitNodeInternal(modelImplRef._serializationData, treeId, parentId, position, featureIndex,
                                                                    featureValue, defaultLeft, cover, res);
}

} // namespace interface1
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal
