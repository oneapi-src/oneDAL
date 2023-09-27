/* file: gbt_classification_model_builder.cpp */
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
//  Implementation of the class defining the gbt classification model builder
//--
*/

#include "algorithms/gradient_boosted_trees/gbt_classification_model_builder.h"
#include "algorithms/gradient_boosted_trees/gbt_classification_model.h"
#include "src/algorithms/dtrees/dtrees_model_impl.h"
#include "src/algorithms/dtrees/gbt/gbt_model_impl.h"
#include "src/algorithms/dtrees/gbt/classification/gbt_classification_model_impl.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::algorithms::dtrees::internal;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace classification
{
namespace interface1
{
services::Status ModelBuilder::convertModelInternal()
{
    gbt::classification::internal::ModelImpl & modelImplRef =
        daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::classification::internal::ModelImpl, ModelPtr>(_model);
    return daal::algorithms::gbt::internal::ModelImpl::convertDecisionTreesToGbtTrees(modelImplRef._serializationData);
}

ModelBuilder::ModelBuilder() : _nClasses(1), _nIterations(1), _model(new gbt::classification::internal::ModelImpl()) {}

services::Status ModelBuilder::initialize(size_t nFeatures, size_t nIterations, size_t nClasses)
{
    services::Status s;
    _nClasses      = (nClasses == 2) ? 1 : nClasses;
    _nIterations   = nIterations;
    auto modelImpl = new gbt::classification::internal::ModelImpl(nFeatures);
    DAAL_CHECK_MALLOC(modelImpl)
    _model.reset(modelImpl);
    gbt::classification::internal::ModelImpl & modelImplRef =
        daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::classification::internal::ModelImpl, ModelPtr>(_model);

    modelImplRef.resize(_nIterations * _nClasses);
    modelImplRef._impurityTables.reset();
    modelImplRef._nNodeSampleTables.reset();
    modelImplRef._nTree.set(_nIterations * _nClasses);
    return s;
}

services::Status ModelBuilder::createTreeInternal(size_t nNodes, size_t clasLabel, TreeId & resId)
{
    gbt::classification::internal::ModelImpl & modelImplRef =
        daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::classification::internal::ModelImpl, ModelPtr>(_model);
    if (_nClasses == 1)
    {
        return daal::algorithms::dtrees::internal::createTreeInternal(modelImplRef._serializationData, nNodes, resId);
    }
    else
    {
        Status s;
        if (nNodes == 0)
        {
            return Status(ErrorID::ErrorIncorrectParameter);
        }
        if (clasLabel > (_nClasses - 1))
        {
            return Status(ErrorID::ErrorIncorrectParameter);
        }
        TreeId treeId                               = clasLabel * _nIterations;
        const SerializationIface * isEmptyTreeTable = (*(modelImplRef._serializationData))[treeId].get();
        const size_t nTrees                         = (clasLabel + 1) * _nIterations;
        while (isEmptyTreeTable && treeId < nTrees)
        {
            treeId++;
            isEmptyTreeTable = (*(modelImplRef._serializationData))[treeId].get();
        }
        if (treeId == nTrees) return Status(ErrorID::ErrorIncorrectParameter);

        services::SharedPtr<DecisionTreeTable> treeTablePtr(
            new DecisionTreeTable(nNodes)); //DecisionTreeTable* const treeTable = new DecisionTreeTable(nNodes);
        const size_t nRows              = treeTablePtr->getNumberOfRows();
        DecisionTreeNode * const pNodes = (DecisionTreeNode *)treeTablePtr->getArray();
        DAAL_CHECK_MALLOC(pNodes)
        pNodes[0].featureIndex           = __NODE_RESERVED_ID;
        pNodes[0].leftIndexOrClass       = 0;
        pNodes[0].featureValueOrResponse = 0;

        for (size_t i = 1; i < nRows; i++)
        {
            pNodes[i].featureIndex           = __NODE_FREE_ID;
            pNodes[i].leftIndexOrClass       = 0;
            pNodes[i].featureValueOrResponse = 0;
        }

        (*(modelImplRef._serializationData))[treeId] = treeTablePtr;

        resId = treeId;
        return s;
    }
}

services::Status ModelBuilder::addLeafNodeInternal(TreeId treeId, NodeId parentId, size_t position, double response, double cover, NodeId & res)
{
    gbt::classification::internal::ModelImpl & modelImplRef =
        daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::classification::internal::ModelImpl, ModelPtr>(_model);
    return daal::algorithms::dtrees::internal::addLeafNodeInternal<double>(modelImplRef._serializationData, treeId, parentId, position, response,
                                                                           cover, res);
}

services::Status ModelBuilder::addSplitNodeInternal(TreeId treeId, NodeId parentId, size_t position, size_t featureIndex, double featureValue,
                                                    int defaultLeft, const double cover, NodeId & res)
{
    gbt::classification::internal::ModelImpl & modelImplRef =
        daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::classification::internal::ModelImpl, ModelPtr>(_model);
    return daal::algorithms::dtrees::internal::addSplitNodeInternal(modelImplRef._serializationData, treeId, parentId, position, featureIndex,
                                                                    featureValue, defaultLeft, cover, res);
}

} // namespace interface1
} // namespace classification
} // namespace gbt
} // namespace algorithms
} // namespace daal
