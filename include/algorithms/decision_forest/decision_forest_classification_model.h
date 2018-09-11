/* file: decision_forest_classification_model.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
//  Implementation of class defining decision_forest classification model.
//--
*/

#ifndef __DECISION_FOREST_CLASSIFICATION_MODEL_H__
#define __DECISION_FOREST_CLASSIFICATION_MODEL_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_model.h"
#include "algorithms/classifier/tree_traverse.h"
#include "algorithms/tree_utils/tree_utils_classification.h"

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
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__MODEL"></a>
 * \brief %Model of the classifier trained by the decision_forest::training::Batch algorithm.
 *
 * \par References
 *      - \ref classification::training::interface1::Batch "training::Batch" class
 *      - \ref classification::prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public classifier::Model
{
public:
    DECLARE_MODEL(Model, classifier::Model)

    /**
    *  Get number of trees in the decision forest model
    *  \return number of trees
    */
    virtual size_t numberOfTrees() const = 0;

    /**
    *  Perform Depth First Traversal of i-th tree
    *  \param[in] iTree    Index of the tree to traverse
    *  \param[in] visitor  This object gets notified when tree nodes are visited
    *  \DAAL_DEPRECATED_USE{ Model::traverseDFS }
    */
    virtual void traverseDF(size_t iTree, classifier::TreeNodeVisitor& visitor) const = 0;

    /**
    *  Perform Breadth First Traversal of i-th tree
    *  \param[in] iTree    Index of the tree to traverse
    *  \param[in] visitor  This object gets notified when tree nodes are visited
    *  \DAAL_DEPRECATED_USE{ Model::traverseBFS }
    */
    virtual void traverseBF(size_t iTree, classifier::TreeNodeVisitor& visitor) const = 0;

    /**
     *  Removes all trees from the model
     */
    virtual void clear() = 0;

    /**
    *  Perform Depth First Traversal of i-th tree
    *  \param[in] iTree    Index of the tree to traverse
    *  \param[in] visitor  This object gets notified when tree nodes are visited
    */
    virtual void traverseDFS(size_t iTree, tree_utils::classification::TreeNodeVisitor& visitor) const = 0;

    /**
    *  Perform Breadth First Traversal of i-th tree
    *  \param[in] iTree    Index of the tree to traverse
    *  \param[in] visitor  This object gets notified when tree nodes are visited
    */
    virtual void traverseBFS(size_t iTree, tree_utils::classification::TreeNodeVisitor& visitor) const = 0;

protected:
    Model() : classifier::Model()
    {}
};
/** @} */
typedef services::SharedPtr<Model> ModelPtr;
} // namespace interface1
using interface1::Model;
using interface1::ModelPtr;

} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
#endif
