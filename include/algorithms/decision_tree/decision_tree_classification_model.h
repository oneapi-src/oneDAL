/* file: decision_tree_classification_model.h */
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
//  Implementation of the class defining the Decision tree classification model
//--
*/

#ifndef __DECISION_TREE_CLASSIFICATION_MODEL_H__
#define __DECISION_TREE_CLASSIFICATION_MODEL_H__

#include "algorithms/classifier/classifier_model.h"
#include "data_management/data/aos_numeric_table.h"
#include "data_management/data/soa_numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "decision_tree_model.h"

using namespace daal::services;

namespace daal
{
namespace algorithms
{

/**
 * @defgroup decision_tree_classification Decision Tree Classification
 * \copydoc daal::algorithms::decision_tree::classification
 * @ingroup classification
 */

/**
 * \brief Contains classes for Decision tree algorithm
 */
namespace decision_tree
{

/**
 * \brief Contains classes for Decision tree classification algorithm
 */
namespace classification
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_TREE__CLASSIFICATION__SPLITCRITERION"></a>
 * \brief Split criterion for Decision tree classification algorithm
 */
enum SplitCriterion
{
    gini     = 0, /*!< Gini index */
    infoGain = 1  /*!< Information gain */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @ingroup decision_tree_classification
 * @{
 */
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__DECISION_TREE__CLASSIFICATION__PARAMETER"></a>
 * \brief Decision tree algorithm parameters
 *
 * \snippet decision_tree/decision_tree_classification_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::classifier::Parameter
{
    /**
     *  Main constructor
     *  \param[in] nClasses                         Number of classes
     */
    Parameter(size_t nClasses = 2) : daal::algorithms::classifier::Parameter(nClasses),
                                     pruning(reducedErrorPruning), maxTreeDepth(0), minObservationsInLeafNodes(1),
                                     splitCriterion(infoGain) {}

    /**
     * Checks a parameter of the Decision tree algorithm
     */
    services::Status check() const DAAL_C11_OVERRIDE;

    SplitCriterion splitCriterion;      /*!< Split criterion for Decision tree classification */
    Pruning pruning;                    /*!< Pruning method for Decision tree */
    size_t maxTreeDepth;                /*!< Maximum tree depth. 0 means unlimited depth. */
    size_t minObservationsInLeafNodes;  /*!< Minimum number of observations in the leaf node. Can be any positive number. */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__CLASSIFICATION__MODEL"></a>
 * \brief %Base class for models trained with the Decision tree algorithm
 *
 * \par References
 *      - Parameter class
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public daal::algorithms::classifier::Model
{
public:
    DAAL_DOWN_CAST_OPERATOR(Model, classifier::Model)

    DECLARE_SERIALIZABLE();

    /**
     * Empty constructor for deserialization
     */
    Model();

    virtual ~Model();

    class ModelImpl;

    /**
     * Returns actual model implementation
     * \return Model implementation
     */
    const ModelImpl * impl() const { return _impl.get(); }

    /**
     * Returns actual model implementation
     * \return Model implementation
     */
    ModelImpl * impl() { return _impl.get(); }

protected:
    void serializeImpl(data_management::InputDataArchive * arch) DAAL_C11_OVERRIDE;

    void deserializeImpl(data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE;

private:
    services::SharedPtr<ModelImpl> _impl;  /*!< Model implementation */
};

typedef services::SharedPtr<Model> ModelPtr;
typedef services::SharedPtr<const Model> ModelConstPtr;

} // namespace interface1

using interface1::Parameter;
using interface1::Model;
using interface1::ModelPtr;
using interface1::ModelConstPtr;

/** @} */
} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
