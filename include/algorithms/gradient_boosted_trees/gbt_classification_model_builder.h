/* file: gbt_classification_model_builder.h */
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
//  Implementation of class defining gradient boosting trees classification model builder.
//--
*/
#ifndef __GBT_CLASSIFICATION_MODEL_BUILDER_H__
#define __GBT_CLASSIFICATION_MODEL_BUILDER_H__

#include "algorithms/gradient_boosted_trees/gbt_classification_model.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
/**
 * @defgroup gbt_classification Gradient Boosted Trees Classification
 * \copydoc daal::algorithms::gbt::classification
 * @ingroup classification
 */
/**
 * \brief Contains classes for the gbt classification algorithm
 */
namespace classification
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @ingroup gbt_classification
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__CLASSIFICATION__MODEL_BUILDER"></a>
 * \brief %Model Builder class for gradient boosted trees classification model.
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
     * Constructs the gradient boosted trees classification model builder
     * \param[in] nFeatures    Number of features in training data
     * \param[in] nIterations  Number of trees in model for each class
     * \param[in] nClasses     Number of classes in model
     */
    ModelBuilder(size_t nFeatures, size_t nIterations, size_t nClasses = 2)
    {
        _status |= initialize(nFeatures, nIterations, nClasses);
        services::throwIfPossible(_status);
    }

    /**
    *  Create certain tree in the gradient boosted trees classification model for certain class
    *  \param[in] nNodes     Number of nodes in created tree
    *  \param[in] classLabel Label of class for which tree is created. classLabel bellows  interval from 0 to (nClasses - 1)
    *  \return Tree identifier
    */
    TreeId createTree(size_t nNodes, size_t classLabel)
    {
        TreeId resId;
        _status |= createTreeInternal(nNodes, classLabel, resId);
        services::throwIfPossible(_status);
        return resId;
    }

    /**
    *  Create Leaf node and add it to certain tree
    *  \param[in] treeId          Tree to which new node is added
    *  \param[in] parentId        Parent node to which new node is added (use noParent for root node)
    *  \param[in] position        Position in parent (e.g. 0 for left and 1 for right child in a binary tree)
    *  \param[in] response        Response value for leaf node to be predicted
    *  \return Node identifier
    */
    NodeId addLeafNode(TreeId treeId, NodeId parentId, size_t position, double response)
    {
        NodeId resId;
        _status |= addLeafNodeInternal(treeId, parentId, position, response, resId);
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
        _status |= convertModelInternal();
        services::throwIfPossible(_status);
        return _model;
    }

    /**
    *  Get status of model building
    *  \return Status
    */
    services::Status getStatus()
    {
        return _status;
    }

protected:
    ModelPtr _model;
    services::Status _status;
    services::Status initialize(size_t nFeatures, size_t nIterations, size_t nClasses);
    services::Status createTreeInternal(size_t nNodes, size_t classLabel, TreeId& resId);
    services::Status addLeafNodeInternal(TreeId treeId, NodeId parentId, size_t position, double response, NodeId& res);
    services::Status addSplitNodeInternal(TreeId treeId, NodeId parentId, size_t position, size_t featureIndex, double featureValue, NodeId& res);
    services::Status convertModelInternal();
    size_t _nClasses;
    size_t _nIterations;

};
/** @} */
} // namespace interface1
using interface1::ModelBuilder;

} // namespace classification
} // namespace gbt
} // namespace algorithms
} // namespace daal
#endif
