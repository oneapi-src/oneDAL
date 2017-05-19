/* file: decision_forest_classification_model.h */
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
//  Implementation of class defining decision_forest classification model.
//--
*/

#ifndef __DECISION_FOREST_CLASSIFICATION_MODEL_H__
#define __DECISION_FOREST_CLASSIFICATION_MODEL_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_model.h"

using namespace daal::services;

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
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__MODEL_VISITOR"></a>
 * \brief %Interface of abstract visitor used in model traversal methods.
 *
 * \par References
 *      - \ref decision_forest::classification::interface1::Model "Model" class
 */
class DAAL_EXPORT NodeVisitor
{
public:
    virtual ~NodeVisitor(){}
    /**
    *  This method is called by traversal method when a leaf node is visited.
    *  \param[in] level    Level in the tree where this node is located
    *  \param[in] response The value of response given by that node
    *  \return This method should return false to cancel further search and true otherwise
    */
    virtual bool onLeafNode(size_t level, size_t response) = 0;
    /**
    *  This method is called by traversal method when a split node is visited.
    *  \param[in] level        Index of the feature used in a split criteria
    *  \param[in] featureIndex Index of the feature used as a split criteria in this node
    *  \param[in] featureValue Feature value used as a split criteria in this node
    *  \return This method should return false to cancel further search and true otherwise
    */
    virtual bool onSplitNode(size_t level, size_t featureIndex, double featureValue) = 0;
};

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
    DECLARE_SERIALIZABLE();
    DAAL_DOWN_CAST_OPERATOR(Model, classifier::Model)

    /**
     * Empty constructor for deserialization
     */
    Model() : classifier::Model()
    {}

    /**
    *  Get number of trees in the decision forest model
    *  \return number of trees
    */
    virtual size_t numberOfTrees() const = 0;

    /**
    *  Perform Depth First Traversal of i-th tree
    *  \param[in] iTree    Index of the tree to traverse
    *  \param[in] visitor  This object gets notified when tree nodes are visited
    */
    virtual void traverseDF(size_t iTree, NodeVisitor& visitor) const = 0;

    /**
    *  Perform Breadth First Traversal of i-th tree
    *  \param[in] iTree    Index of the tree to traverse
    *  \param[in] visitor  This object gets notified when tree nodes are visited
    */
    virtual void traverseBF(size_t iTree, NodeVisitor& visitor) const = 0;

protected:
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        classifier::Model::serialImpl<Archive, onDeserialize>(arch);
    }

    void serializeImpl(data_management::InputDataArchive *archive) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(archive);}

    void deserializeImpl(data_management::OutputDataArchive *archive) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(archive);}
};
/** @} */
typedef services::SharedPtr<Model> ModelPtr;
} // namespace interface1
using interface1::Model;
using interface1::ModelPtr;
using interface1::NodeVisitor;

} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
#endif
