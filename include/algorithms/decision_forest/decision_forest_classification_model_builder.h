/* file: decision_forest_classification_model_builder.h */
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
//  Implementation of class defining decision_forest classification model builder.
//--
*/
#ifndef __DECISION_FOREST_CLASSIFICATION_MODEL_BUILDER_H__
#define __DECISION_FOREST_CLASSIFICATION_MODEL_BUILDER_H__

#include "algorithms/decision_forest/decision_forest_classification_model.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
/**
 * @defgroup decision_forest_classification Decision Forest Classification
 * \copydoc daal::algorithms::decision_forest::classification
 * @ingroup classification
 */
/**
 * \brief Contains classes for the decision_forest classification algorithm
 */
namespace classification
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @ingroup decision_forest_classification
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST_CLASSIFICATION_MODEL_BUILDER"></a>
 * \brief %Model Builder class for Decision Forest Classification Model algorithm
 *
 * \par References
 *      - \ref classification::interface1::Model "classification::Model" class
 */
class DAAL_EXPORT ModelBuilder
{
public:
    /**
     * \brief %Node identifier type is size_t
     */
    typedef size_t NodeId;

    /**
     * \brief %Tree identifier type is size_t
     */
    typedef size_t TreeId;

    static const NodeId noParent = static_cast<size_t>(-1); /*!< %Reserved value for root nodes */

    /**
     * Constructs the Decision forest classification model builder
     * \param[in] nClasses  Number of classes
     * \param[in] nTrees  Number of trees in model
     */
    ModelBuilder(size_t nClasses, size_t nTrees)
    {
        _status |= initialize(nClasses, nTrees);
        services::throwIfPossible(_status);
    }

    /**
    *  Create certain tree in the decision forest model
    *  \param[in] nNodes  Number of nodes in created tree
    *  \return Tree identifier
    */
    TreeId createTree(size_t nNodes)
    {
        TreeId resId;
        _status |= createTreeInternal(nNodes, resId);
        services::throwIfPossible(_status);
        return resId;
    }

    /**
    *  Create Leaf node and add it to certain tree
    *  \param[in] treeId          Tree to which new node is added
    *  \param[in] parentId        Parent node to which new node is added (use noParent for root node)
    *  \param[in] position        Position in parent (e.g. 0 for left and 1 for right child in a binary tree)
    *  \param[in] classLabel      Class label to be predicted
    *  \return Node identifier
    */
    NodeId addLeafNode(TreeId treeId, NodeId parentId, size_t position, size_t classLabel)
    {
        NodeId resId;
        _status |= addLeafNodeInternal(treeId, parentId, position, classLabel, resId);
        services::throwIfPossible(_status);
        return resId;
    }

    /**
    *  Create Split node and add it to certain tree
    *  \param[in] treeId          Tree to which new node is added
    *  \param[in] parentId        Parent node to which new node is added (use noParent for root node)
    *  \param[in] position        Position in parent (e.g. 0 for left and 1 for right child in a binary tree)
    *  \param[in] featureIndex    Feature index for spliting
    *  \param[in] featureValue    Feature value for spliting
    *  \return Node identifier
    */
    NodeId addSplitNode(TreeId treeId, NodeId parentId, size_t position, size_t featureIndex, double featureValue)
    {
        NodeId resId;
        _status |= addSplitNodeInternal(treeId, parentId, position, featureIndex, featureValue, resId);
        services::throwIfPossible(_status);
        return resId;
    }

    /**
    *  Get built model
    *  \return Model pointer
    */
    ModelPtr getModel()
    {
        services::throwIfPossible(_status);
        return _model;
    }

    /**
    *  Get status of model building
    *  \return Status
    */
    services::Status getStatus() const { return _status; }

protected:
    ModelPtr _model;
    services::Status _status;
    services::Status initialize(size_t nClasses, size_t nTrees);
    services::Status createTreeInternal(size_t nNodes, TreeId & resId);
    services::Status addLeafNodeInternal(TreeId treeId, NodeId parentId, size_t position, size_t classLabel, NodeId & res);
    services::Status addSplitNodeInternal(TreeId treeId, NodeId parentId, size_t position, size_t featureIndex, double featureValue, NodeId & res);
};
/** @} */
} // namespace interface1
using interface1::ModelBuilder;

} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
#endif
