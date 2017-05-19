/* file: decision_forest_regression_model.h */
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
//  Implementation of the class defining the decision forest regression model
//--
*/

#ifndef __DECISION_FOREST_REGRESSION_MODEL_H__
#define __DECISION_FOREST_REGRESSION_MODEL_H__

#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/regression/regression_model.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
/**
 * @defgroup decision_forest_regression Decision Forest Regression
 * \copydoc daal::algorithms::decision_forest::regression
 * @ingroup regression
 */
/**
 * \brief Contains classes for decision forest regression algorithm
 */
namespace regression
{
/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * @ingroup decision_forest_regression
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__REGRESSION__MODEL_VISITOR"></a>
 * \brief %Interface of abstract visitor used in model traversal methods.
 *
 * \par References
 *      - \ref decision_forest::regression::interface1::Model "Model" class
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
    virtual bool onLeafNode(size_t level, double response) = 0;
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
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST_REGRESSION_MODEL"></a>
 * \brief %Base class for models trained with the decision forest regression algorithm
 *
 * \tparam modelFPType  Data type to store decision forest model data, double or float
 *
 * \par References
 *      - \ref regression::training::interface1::Batch "regression::training::Batch" class
 *      - \ref regression::prediction::interface1::Batch "regression::prediction::Batch" class
 */
class DAAL_EXPORT Model : public algorithms::regression::Model
{
public:
    DECLARE_SERIALIZABLE();
    DAAL_CAST_OPERATOR(Model);

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
    Model();
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        algorithms::regression::Model::serialImpl<Archive, onDeserialize>(arch);
    }

    void serializeImpl(data_management::InputDataArchive *archive) DAAL_C11_OVERRIDE {}

    void deserializeImpl(data_management::OutputDataArchive *archive) DAAL_C11_OVERRIDE {}
};
typedef services::SharedPtr<Model> ModelPtr;
typedef services::SharedPtr<const Model> ModelConstPtr;

/** @} */
} // namespace interface1
using interface1::Model;
using interface1::ModelPtr;
using interface1::ModelConstPtr;
using interface1::NodeVisitor;

}
}
}
}
#endif
