/* file: gbt_classification_model_builder.cpp */
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
//  Implementation of the class defining the gbt classification model builder
//--
*/

#include "algorithms/gradient_boosted_trees/gbt_classification_model_builder.h"
#include "algorithms/gradient_boosted_trees/gbt_classification_model.h"
#include "../../dtrees_model_impl.h"
#include "../gbt_model_impl.h"
#include "gbt_classification_model_impl.h"

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
    gbt::classification::internal::ModelImpl& modelImplRef = daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::classification::internal::ModelImpl,ModelPtr>(_model);
    return daal::algorithms::gbt::internal::ModelImpl::convertDecisionTreesToGbtTrees(modelImplRef._serializationData);
}

services::Status ModelBuilder::initialize(size_t nFeatures, size_t nIterations, size_t nClasses)
{
    services::Status s;
    _nClasses = (nClasses == 2) ? 1 : nClasses;
    _nIterations = nIterations;
    _model.reset(new gbt::classification::internal::ModelImpl(nFeatures));
    gbt::classification::internal::ModelImpl& modelImplRef = daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::classification::internal::ModelImpl,ModelPtr>(_model);

    modelImplRef.resize(_nIterations*_nClasses);
    modelImplRef._impurityTables.reset();
    modelImplRef._nNodeSampleTables.reset();
    modelImplRef._nTree.set(_nIterations*_nClasses);
    return s;
}

services::Status ModelBuilder::createTreeInternal(size_t nNodes, size_t clasLabel, TreeId& resId)
{
    gbt::classification::internal::ModelImpl& modelImplRef = daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::classification::internal::ModelImpl,ModelPtr>(_model);
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
        TreeId treeId = clasLabel * _nIterations;
        const SerializationIface* isEmptyTreeTable = (*(modelImplRef._serializationData))[treeId].get();
        const size_t nTrees = (clasLabel + 1)* _nIterations;
        while(isEmptyTreeTable && treeId < nTrees)
        {
            treeId++;
            isEmptyTreeTable = (*(modelImplRef._serializationData))[treeId].get();
        }
        if (treeId == nTrees)
            return Status(ErrorID::ErrorIncorrectParameter);

        services::SharedPtr<DecisionTreeTable> treeTablePtr(new DecisionTreeTable(nNodes));//DecisionTreeTable* const treeTable = new DecisionTreeTable(nNodes);
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

        (*(modelImplRef._serializationData))[treeId] = treeTablePtr;

        resId = treeId;
        return s;
    }
}

services::Status ModelBuilder::addLeafNodeInternal(TreeId treeId, NodeId parentId, size_t position, double response, NodeId& res)
{
    gbt::classification::internal::ModelImpl& modelImplRef = daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::classification::internal::ModelImpl,ModelPtr>(_model);
    return daal::algorithms::dtrees::internal::addLeafNodeInternal<double>(modelImplRef._serializationData, treeId, parentId, position, response, res);;
}

services::Status ModelBuilder::addSplitNodeInternal(TreeId treeId, NodeId parentId, size_t position, size_t featureIndex, double featureValue, NodeId& res)
{
    gbt::classification::internal::ModelImpl& modelImplRef = daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::classification::internal::ModelImpl,ModelPtr>(_model);
    return daal::algorithms::dtrees::internal::addSplitNodeInternal(modelImplRef._serializationData, treeId, parentId, position, featureIndex, featureValue, res);
}

} // namespace interface1
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
